"""Select the forecasting configuration from historical out-of-sample predictions.

This is a leakage-safe *configuration selector*, not another model.  It reads
predictions produced by the existing pipeline, scores each candidate on a
validation split, and writes the configuration to use for the locked test
period.  If only predictions_test.csv exists, the script can be run in
``--allow-test-diagnostic`` mode, but the output is explicitly labelled
post-hoc and must not be used as a deployment claim.

Example (on the remote checkout):
  python scripts/auto_select_mechanism.py \
    --results-root results/electricity \
    --output-dir results/auto_selector \
    --data guangdong=data/GD/guangdong_aligned.csv:ds:y \
    --data shanxi=data/SHANXI/shanxi_aligned.csv:ds:y \
    --data reale_delu=data/REALE/DE-LU_aligned.csv:time:'y_Day-ahead Price [EUR/MWh]' \
    --data reale_fr=data/REALE/FR_aligned.csv:time:'y_Day-ahead Price [EUR/MWh]'

The selector supports a family of risk preferences. ``risk_weight=0`` is
ordinary RMSE; larger values increasingly emphasize errors above the
training-period P95 threshold.  A candidate must also stay within
``max_normal_rmse_degradation`` on the non-tail region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE = "tsfm_point"
CANDIDATES = [
    "tsfm_point",
    "tsfm_point_C_weighted_mse",
    "tsfm_point_C_fixed_asym",
    "tsfm_point_C_confidence_asym",
    "tsfm_point_fixed_asym_noC",
]


def parse_data(value: str) -> tuple[str, str, str, str]:
    # Split from the right so local Windows paths such as ``D:\\data\\x.csv``
    # remain valid when the same script is tested off-cluster.
    name, rest = value.split("=", 1)
    path, time_col, target = rest.rsplit(":", 2)
    return name, path, time_col, target


def score(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(y) & np.isfinite(pred)
    if not ok.any():
        return np.nan, np.nan, 0
    e = pred[ok] - y[ok]
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e * e))), int(ok.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results/electricity"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/auto_selector"))
    ap.add_argument("--data", action="append", required=True, help="name=path:time_column:target_column")
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--risk-weight", type=float, default=None, help="0..1; default writes 0, .25, .5, .75, 1 recommendations")
    ap.add_argument("--max-normal-rmse-degradation", type=float, default=0.05, help="relative tolerance versus baseline")
    ap.add_argument("--allow-test-diagnostic", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test_diagnostic:
        ap.error("test selection is post-hoc; add --allow-test-diagnostic explicitly")
    if not 0 <= args.max_normal_rmse_degradation:
        ap.error("--max-normal-rmse-degradation must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    selections: list[dict] = []
    data_specs = [parse_data(v) for v in args.data]
    risk_weights = [args.risk_weight] if args.risk_weight is not None else [0.0, 0.25, 0.5, 0.75, 1.0]

    for market, data_path, time_col, target_col in data_specs:
        src = pd.read_csv(data_path, parse_dates=[time_col]).sort_values(time_col).reset_index(drop=True)
        n = len(src)
        train_y = pd.to_numeric(src[target_col].iloc[: int(n * 0.7)], errors="coerce").dropna()
        threshold = float(train_y.quantile(0.95))
        method_frames: dict[str, pd.DataFrame] = {}
        chosen_file = f"predictions_{args.split}.csv"
        for method in CANDIDATES:
            path = args.results_root / market / method / chosen_file
            if path.exists():
                method_frames[method] = pd.read_csv(path, parse_dates=["time"])
        if BASELINE not in method_frames:
            rows.append({"market": market, "status": "missing_baseline", "split": args.split})
            continue

        base = method_frames[BASELINE]
        y = pd.to_numeric(base["y_true"], errors="coerce").to_numpy()
        baseline_pred = pd.to_numeric(base["tree_prediction"], errors="coerce").to_numpy()
        tail = y >= threshold
        normal = ~tail
        market_scores: dict[str, dict[str, float]] = {}
        for method, frame in method_frames.items():
            pred = pd.to_numeric(frame["tree_prediction"], errors="coerce").to_numpy()
            if len(pred) != len(y):
                continue
            mae, rmse, count = score(y, pred)
            nmae, nrmse, ncount = score(y[normal], pred[normal])
            tmae, trmse, tcount = score(y[tail], pred[tail])
            market_scores[method] = {"mae": mae, "rmse": rmse, "normal_rmse": nrmse, "tail_rmse": trmse}
            rows.append({"market": market, "method": method, "split": args.split, "threshold_train_p95": threshold, "n": count, "normal_n": ncount, "tail_n": tcount, "mae": mae, "rmse": rmse, "normal_rmse": nrmse, "tail_rmse": trmse})

        b = market_scores.get(BASELINE)
        if not b:
            continue
        for rw in risk_weights:
            feasible = []
            for method, s in market_scores.items():
                normal_degradation = s["normal_rmse"] / b["normal_rmse"] - 1.0 if b["normal_rmse"] else np.inf
                objective = (1.0 - rw) * s["rmse"] + rw * s["tail_rmse"]
                if normal_degradation <= args.max_normal_rmse_degradation:
                    feasible.append((objective, method, normal_degradation))
            if feasible:
                objective, selected, degradation = min(feasible)
            else:
                objective, selected, degradation = np.nan, BASELINE, np.nan
            selections.append({"market": market, "split": args.split, "risk_weight": rw, "selected_method": selected, "objective": objective, "baseline_rmse": b["rmse"], "baseline_normal_rmse": b["normal_rmse"], "baseline_tail_rmse": b["tail_rmse"], "selected_normal_rmse_degradation": degradation, "status": "posthoc_test_diagnostic" if args.split == "test" else "validation_selection"})

    pd.DataFrame(rows).to_csv(args.output_dir / "candidate_scores.csv", index=False)
    pd.DataFrame(selections).to_csv(args.output_dir / "selected_configurations.csv", index=False)
    report = {
        "split": args.split,
        "leakage_safe": args.split == "valid",
        "risk_weights": risk_weights,
        "max_normal_rmse_degradation": args.max_normal_rmse_degradation,
        "baseline": BASELINE,
        "candidates": CANDIDATES,
        "note": "Use selected_configurations.csv from validation to lock the test configuration; do not select from test unless this is explicitly a post-hoc diagnostic.",
    }
    (args.output_dir / "selector_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'candidate_scores.csv'}")
    print(f"Wrote {args.output_dir / 'selected_configurations.csv'}")


if __name__ == "__main__":
    main()
