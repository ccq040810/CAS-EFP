"""Run expanding-origin adaptive-mechanism experiments.

Each window has explicit chronological train/validation/test dates. Candidate
models are selected using validation RMSE plus an optional P95-tail term, then
evaluated once on the locked test segment. The script writes one summary CSV;
it is intended to be launched remotely as a single command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "guangdong": ("data/GD/guangdong_aligned.csv", "ds", "configs/columns/CH/electricity_price_regress_safe.json", "configs/columns/CH/electricity_price_tsfm_safe.json", True),
    "shanxi": ("data/SHANXI/shanxi_aligned.csv", "ds", "configs/columns/shanxi/realtime/regress_realtime_safe.json", "configs/columns/shanxi/realtime/tsfm_realtime_safe.json", True),
    "reale_delu": ("data/REALE/DE-LU_aligned.csv", "time", "configs/columns/REALE/DE/regress_dayahead_safe.json", "configs/columns/REALE/DE/tsfm_dayahead_safe.json", True),
    "reale_fr": ("data/REALE/FR_aligned.csv", "time", "configs/columns/REALE/FR/regress_dayahead_safe.json", "configs/columns/REALE/FR/tsfm_dayahead_safe.json", True),
}
CANDIDATES = {
    "tsfm_point": ["--no-use_tsfm_uncertainty"],
    "tsfm_point_C": ["--use_tsfm_confidence"],
    "tsfm_point_fixed_asym_noC": ["--no-use_tsfm_uncertainty", "--use_asymmetric_loss", "--asymmetric_alpha", "2.0", "--asymmetric_direction", "auto"],
    "tsfm_point_C_fixed_asym": ["--no-use_tsfm_uncertainty", "--use_tsfm_confidence", "--use_asymmetric_loss", "--asymmetric_alpha", "2.0", "--asymmetric_direction", "auto"],
    "tsfm_point_C_confidence_asym": ["--no-use_tsfm_uncertainty", "--use_tsfm_confidence", "--use_asymmetric_loss", "--asymmetric_alpha", "2.0", "--asymmetric_direction", "auto", "--sample_weight_mode", "confidence", "--sample_weight_lambda", "1.0"],
}


def metric(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return np.nan, np.nan
    e = p[ok] - y[ok]
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e * e)))


def split_dates(times: pd.Series, fractions: tuple[float, float, float, float]) -> dict[str, str]:
    t = pd.DatetimeIndex(times.dropna().sort_values().unique())
    idx = [max(0, min(len(t) - 1, int(len(t) * f))) for f in fractions]
    # End boundaries are exclusive; use the next observed timestamp where possible.
    def at(i: int) -> pd.Timestamp:
        return t[min(i, len(t) - 1)]
    end = lambda i: t[min(i, len(t) - 1)] + (t[1] - t[0] if len(t) > 1 else pd.Timedelta(hours=1))
    return {"train_start": str(at(idx[0])), "train_end": str(at(idx[1])), "valid_start": str(at(idx[1])), "valid_end": str(at(idx[2])), "test_start": str(at(idx[2])), "test_end": str(end(idx[3]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("results/rolling_adaptive"))
    ap.add_argument("--cache-root", type=Path, default=Path("results/rolling_adaptive_cache"))
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    ap.add_argument("--model-path", default=os.environ.get("CHRONOS2_MODEL_PATH", "models/chronos-2"))
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--risk-weight", type=float, default=0.5)
    ap.add_argument("--max-normal-rmse-degradation", type=float, default=0.05)
    ap.add_argument("--only", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--skip-run", action="store_true", help="only summarize existing window outputs")
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for market in args.only:
        rel_data, time_col, regress, tsfm, is_std = DATASETS[market]
        data_path = ROOT / rel_data
        frame = pd.read_csv(data_path, parse_dates=[time_col]).sort_values(time_col).reset_index(drop=True)
        target_col = json.loads((ROOT / regress).read_text(encoding="utf-8"))[0]["target_columns"][-1]
        # Expanding origins: 0-50/60/70% train, next 10% validation, next 10% test.
        for w in range(args.windows):
            train_end = 0.50 + 0.10 * w
            valid_end = train_end + 0.10
            test_end = valid_end + 0.10
            split_cfg = {"train": 0.7, "valid": 0.1, "test": 0.2, "time_col": time_col, **split_dates(frame[time_col], (0.0, train_end, valid_end, test_end))}
            wdir = args.output_root / market / f"window_{w+1}"
            wdir.mkdir(parents=True, exist_ok=True)
            split_path = wdir / "data_split.json"
            split_path.write_text(json.dumps([{"data_split": split_cfg}], indent=2), encoding="utf-8")
            if not args.skip_run:
                cache = args.cache_root / market / f"window_{w+1}"
                common = [sys.executable, str(ROOT / "run_pipeline.py"), "--device", args.device, "--model_name", "chronos2-lgbm", "--seed", "42", "--data_path", str(data_path), "--regress_cols_path", str(ROOT / regress), "--tsfm_cols_path", str(ROOT / tsfm), "--data_split_path", str(split_path), "--tag", f"{market}_rolling_w{w+1}", "--is_std", "--enable_tsfm", "1", "--seq_len", "168", "--pred_len", "24", "--batch_size", "16", "--tsfm_num_samples", os.environ.get("TSFM_NUM_SAMPLES", "100"), "--tsfm_cache_path", str(cache), "--tsfm_models", "chronos2", "--tsfm_model_paths", json.dumps({"chronos2": args.model_path}), "--tsfm_use_future_covariates", "0", "--regression_model", "lgbm", "--num_boost_round", "2000", "--early_stopping_rounds", "100", "--lgbm_learning_rate", "0.05", "--num_leaves", "63", "--min_data_in_leaf", "40", "--n_jobs", os.environ.get("N_JOBS", "8"), "--use_ar_features", "--ar_lags", "1,24,168", "--ar_roll", "24", "--disable_shap"]
                for method, extra in CANDIDATES.items():
                    out = wdir / method
                    if (out / "metrics_test.json").exists():
                        continue
                    cmd = common.copy()
                    cmd += ["--save_dir", str(out), *extra]
                    print("RUN", market, w + 1, method, flush=True)
                    subprocess.run(cmd, cwd=ROOT, check=True)

            files = {m: (wdir / m) for m in CANDIDATES}
            if not (files["tsfm_point"] / "predictions_valid.csv").exists():
                continue
            base_valid = pd.read_csv(files["tsfm_point"] / "predictions_valid.csv")
            yv = pd.to_numeric(base_valid.y_true, errors="coerce").to_numpy()
            # Tail threshold is fitted only on the historical training period.
            train_target = pd.to_numeric(frame.iloc[: int(len(frame) * train_end)][target_col], errors="coerce").dropna()
            threshold = float(train_target.quantile(.95))
            (wdir / "tail_threshold_train_p95.txt").write_text(f"{threshold:.12g}\n", encoding="utf-8")
            normal = yv < threshold
            scores: dict[str, dict[str, float]] = {}
            for method, path in files.items():
                pv = path / "predictions_valid.csv"
                pt = path / "predictions_test.csv"
                if not pv.exists() or not pt.exists():
                    continue
                vf = pd.read_csv(pv); tf = pd.read_csv(pt)
                if len(vf) != len(yv):
                    continue
                mae, rmse = metric(yv, pd.to_numeric(vf.tree_prediction, errors="coerce").to_numpy())
                _, nrmse = metric(yv[normal], pd.to_numeric(vf.tree_prediction, errors="coerce").to_numpy()[normal])
                _, trmse = metric(yv[~normal], pd.to_numeric(vf.tree_prediction, errors="coerce").to_numpy()[~normal])
                scores[method] = {"rmse": rmse, "normal_rmse": nrmse, "tail_rmse": trmse}
            if "tsfm_point" not in scores:
                continue
            b = scores["tsfm_point"]
            feasible = []
            for method, s in scores.items():
                degradation = s["normal_rmse"] / b["normal_rmse"] - 1 if b["normal_rmse"] else np.inf
                if degradation <= args.max_normal_rmse_degradation:
                    objective = (1 - args.risk_weight) * s["rmse"] + args.risk_weight * s["tail_rmse"]
                    feasible.append((objective, method, degradation))
            selected = min(feasible)[1] if feasible else "tsfm_point"
            test = pd.read_csv(files[selected] / "predictions_test.csv")
            yt = pd.to_numeric(test.y_true, errors="coerce").to_numpy(); pp = pd.to_numeric(test.tree_prediction, errors="coerce").to_numpy()
            mae, rmse = metric(yt, pp)
            all_rows.append({"market": market, "window": w + 1, "selected_method": selected, "validation_risk_weight": args.risk_weight, "test_mae": mae, "test_rmse": rmse, "baseline_test_rmse": metric(yt, pd.to_numeric(pd.read_csv(files["tsfm_point"] / "predictions_test.csv").tree_prediction, errors="coerce").to_numpy())[1], "status": "validation_selected"})

    pd.DataFrame(all_rows).to_csv(args.output_root / "rolling_adaptive_summary.csv", index=False)
    print(f"Wrote {args.output_root / 'rolling_adaptive_summary.csv'}")


if __name__ == "__main__":
    main()
