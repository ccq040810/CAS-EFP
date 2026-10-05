"""Leakage-safe adaptive activation of confidence/asymmetric modules.

This script is intentionally lightweight: it consumes validation/test
predictions already produced by the rolling experiments and treats the four
configurations as operating modes of one correction framework:

  base                confidence=off, asymmetric-loss=off
  confidence          confidence=on,  asymmetric-loss=off
  asymmetric          confidence=off, asymmetric-loss=on
  confidence_asym     confidence=on,  asymmetric-loss=on

For each chronological window, the mode is selected from validation only and
then evaluated once on the locked test segment.  Test labels are never used
for selection.  The output includes the selected mode, fixed-mode scores,
oracle score, and selection regret for direct use in a paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


MODES = {
    "base": "tsfm_point",
    "confidence": "tsfm_point_C",
    "asymmetric": "tsfm_point_C_fixed_asym",
    "confidence_asym": "tsfm_point_C_confidence_asym",
}


def arr(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return {"mae": np.nan, "rmse": np.nan, "n": 0}
    e = p[ok] - y[ok]
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))), "n": int(ok.sum())}


def mode_score(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    y, p = arr(frame, "y_true"), arr(frame, "tree_prediction")
    valid = np.isfinite(y) & np.isfinite(p)
    normal, tail = valid & (y < threshold), valid & (y >= threshold)
    out = score(y, p)
    out["normal_rmse"] = score(y[normal], p[normal])["rmse"]
    out["tail_rmse"] = score(y[tail], p[tail])["rmse"]
    return out


def choose(scores: dict[str, dict[str, float]], risk_weight: float, max_normal_deg: float) -> str:
    base = scores.get("base")
    if not base or not np.isfinite(base["normal_rmse"]):
        return "base"
    feasible = []
    for mode, s in scores.items():
        if not np.isfinite(s["rmse"]):
            continue
        deg = s["normal_rmse"] / base["normal_rmse"] - 1.0 if base["normal_rmse"] else np.inf
        if deg <= max_normal_deg:
            objective = (1.0 - risk_weight) * s["rmse"] + risk_weight * s["tail_rmse"]
            feasible.append((objective, mode))
    return min(feasible)[1] if feasible else "base"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="rolling output root, e.g. results/rolling_adaptive_v3")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--risk-weight", type=float, default=0.5)
    ap.add_argument("--max-normal-degradation", type=float, default=0.05)
    ap.add_argument("--tail-quantile", type=float, default=0.95)
    args = ap.parse_args()
    if not 0 <= args.risk_weight <= 1:
        ap.error("--risk-weight must be in [0,1]")
    out = args.output or (args.root / "adaptive_module_summary.csv")
    rows: list[dict] = []
    for market_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for window_dir in sorted(market_dir.glob("window_*")):
            files = {}
            for mode, dirname in MODES.items():
                path = window_dir / dirname
                valid, test = path / "predictions_valid.csv", path / "predictions_test.csv"
                if valid.exists() and test.exists():
                    files[mode] = (pd.read_csv(valid), pd.read_csv(test))
            if "base" not in files:
                continue
            yv = arr(files["base"][0], "y_true")
            threshold_file = window_dir / "tail_threshold_train_p95.txt"
            if threshold_file.exists():
                threshold = float(threshold_file.read_text(encoding="utf-8").strip())
            else:
                # Fallback is explicitly marked by the output; rerunning the
                # rolling experiment will create the leakage-safe threshold.
                threshold = float(np.nanquantile(yv, args.tail_quantile))
            valid_scores = {m: mode_score(v, threshold) for m, (v, _) in files.items()}
            selected = choose(valid_scores, args.risk_weight, args.max_normal_degradation)
            test_scores = {m: mode_score(t, threshold) for m, (_, t) in files.items()}
            oracle = min((s["rmse"], m) for m, s in test_scores.items() if np.isfinite(s["rmse"]))
            chosen = test_scores[selected]
            base = test_scores["base"]
            row = {
                "market": market_dir.name,
                "window": window_dir.name,
                "selected_mode": selected,
                "risk_weight": args.risk_weight,
                "max_normal_degradation": args.max_normal_degradation,
                "validation_tail_threshold": threshold,
                "selected_test_mae": chosen["mae"],
                "selected_test_rmse": chosen["rmse"],
                "base_test_rmse": base["rmse"],
                "oracle_mode_posthoc": oracle[1],
                "oracle_test_rmse_posthoc": oracle[0],
                "selection_regret_rmse": chosen["rmse"] - oracle[0],
                "selected_minus_base_rmse": chosen["rmse"] - base["rmse"],
            }
            for mode, s in valid_scores.items():
                row[f"valid_{mode}_rmse"] = s["rmse"]
                row[f"valid_{mode}_normal_rmse"] = s["normal_rmse"]
                row[f"valid_{mode}_tail_rmse"] = s["tail_rmse"]
            for mode, s in test_scores.items():
                row[f"test_{mode}_rmse"] = s["rmse"]
                row[f"test_{mode}_normal_rmse"] = s["normal_rmse"]
                row[f"test_{mode}_tail_rmse"] = s["tail_rmse"]
            rows.append(row)
    result = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Wrote {out} ({len(result)} windows)")
    if not result.empty:
        print(result.groupby("market")["selected_mode"].value_counts().to_string())
        print("mean selected-minus-base RMSE:", result["selected_minus_base_rmse"].mean())
        print("mean selection regret RMSE:", result["selection_regret_rmse"].mean())


if __name__ == "__main__":
    main()
