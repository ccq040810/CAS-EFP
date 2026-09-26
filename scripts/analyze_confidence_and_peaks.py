"""Analyze confidence/error association and high-price performance.

This is a post-hoc descriptive analysis. The primary high-price thresholds
are computed from each dataset's chronological training segment only. Test-set
top-5%/top-10% rows are also reported as descriptive tail slices, not as
deployable thresholds and never for model selection.

Example (run from repository root):
    python scripts/analyze_confidence_and_peaks.py \
      --dataset 'guangdong|data/GD/guangdong_aligned.csv|ds|y|results/electricity/guangdong' \
      --dataset 'shanxi|data/SHANXI/shanxi_aligned.csv|ds|y|results/electricity/shanxi' \
      --dataset 'reale_delu|data/REALE/DE-LU_aligned.csv|time|y_Day-ahead Price [EUR/MWh]|results/electricity/reale_delu'

Each experiment directory is expected to contain ``predictions_test.csv``
written by Exp_Pipeline. Existing confidence-bin summaries alone are
insufficient for exact tail metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_dataset(spec: str) -> dict[str, str]:
    parts = spec.split("|", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "dataset must be NAME|DATA_CSV|TIME_COL|TARGET_COL|RESULT_ROOT"
        )
    name, data_path, time_col, target_col, result_root = parts
    return {
        "name": name,
        "data_path": data_path,
        "time_col": time_col,
        "target_col": target_col,
        "result_root": result_root,
    }


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if not len(y):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias_pred_minus_true": np.nan, "underprediction_rate": np.nan}
    err = pred - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias_pred_minus_true": float(np.mean(err)),
        "underprediction_rate": float(np.mean(err < 0)),
    }


def spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return float("nan")
    return float(x[valid].corr(y[valid], method="spearman"))


def analyze_dataset(spec: dict[str, str], output_root: Path) -> tuple[list[dict], list[dict], dict]:
    data = pd.read_csv(spec["data_path"], parse_dates=[spec["time_col"]]).sort_values(spec["time_col"])
    n = len(data)
    n_train = int(n * 0.7)
    train_y = pd.to_numeric(data[spec["target_col"]].iloc[:n_train], errors="coerce").dropna()
    if train_y.empty:
        raise ValueError(f"{spec['name']}: no observed training targets")
    train_q90, train_q95 = (float(train_y.quantile(q)) for q in (0.90, 0.95))

    result_root = Path(spec["result_root"])
    files = sorted(result_root.glob("*/predictions_test.csv"))
    if not files:
        raise FileNotFoundError(
            f"No */predictions_test.csv under {result_root}. Re-run the pipeline with the updated exporter first."
        )

    method_rows: list[dict] = []
    bin_rows: list[dict] = []
    plot_entries: list[tuple[str, pd.DataFrame]] = []
    for path in files:
        method = path.parent.name
        frame = pd.read_csv(path, parse_dates=["time"])
        y = pd.to_numeric(frame["y_true"], errors="coerce")
        row: dict = {"dataset": spec["name"], "method": method}
        for pred_col, prefix in (("tree_prediction", "tree"), ("tsfm_point", "tsfm")):
            if pred_col not in frame:
                continue
            for label, mask in (
                ("all", np.ones(len(frame), dtype=bool)),
                ("train_q90_plus", y.to_numpy() >= train_q90),
                ("train_q95_plus", y.to_numpy() >= train_q95),
            ):
                metrics = regression_metrics(
                    y.to_numpy()[mask], pd.to_numeric(frame[pred_col], errors="coerce").to_numpy()[mask]
                )
                for key, value in metrics.items():
                    row[f"{prefix}_{label}_{key}"] = value

            # Test top-10% / top-5% are descriptive slices only; the threshold
            # is based on test labels and must not be used to tune the method.
            for q, label in ((0.90, "test_top10pct_descriptive"), (0.95, "test_top5pct_descriptive")):
                threshold = float(y.quantile(q))
                mask = y.to_numpy() >= threshold
                metrics = regression_metrics(
                    y.to_numpy()[mask], pd.to_numeric(frame[pred_col], errors="coerce").to_numpy()[mask]
                )
                for key, value in metrics.items():
                    row[f"{prefix}_{label}_{key}"] = value

        confidence = pd.to_numeric(frame.get("confidence", np.nan), errors="coerce")
        if isinstance(confidence, pd.Series) and confidence.notna().sum() >= 3:
            tsfm_abs = pd.to_numeric(frame["tsfm_abs_error"], errors="coerce")
            tree_abs = pd.to_numeric(frame["tree_abs_error"], errors="coerce")
            row["spearman_C_vs_Tsfm_abs_error"] = spearman(confidence, tsfm_abs)
            row["spearman_C_vs_tree_abs_error"] = spearman(confidence, tree_abs)
            valid = confidence.notna() & tsfm_abs.notna() & tree_abs.notna()
            binned = pd.DataFrame({"confidence": confidence[valid], "tsfm_abs_error": tsfm_abs[valid], "tree_abs_error": tree_abs[valid]})
            bins = min(10, int(binned["confidence"].nunique()))
            if bins >= 2:
                binned["confidence_bin"] = pd.qcut(binned["confidence"], q=bins, labels=False, duplicates="drop")
                grouped = binned.groupby("confidence_bin", observed=True).agg(
                    n=("confidence", "size"), confidence_mean=("confidence", "mean"),
                    tsfm_mae=("tsfm_abs_error", "mean"), tsfm_rmse=("tsfm_abs_error", lambda x: float(np.sqrt(np.mean(x**2)))),
                    tree_mae=("tree_abs_error", "mean"), tree_rmse=("tree_abs_error", lambda x: float(np.sqrt(np.mean(x**2)))),
                ).reset_index()
                grouped.insert(0, "method", method)
                grouped.insert(0, "dataset", spec["name"])
                bin_rows.extend(grouped.to_dict("records"))
                plot_entries.append((method, grouped))
        else:
            row["spearman_C_vs_Tsfm_abs_error"] = np.nan
            row["spearman_C_vs_tree_abs_error"] = np.nan
        row["train_q90_threshold"] = train_q90
        row["train_q95_threshold"] = train_q95
        method_rows.append(row)

    ds_out = output_root / spec["name"]
    ds_out.mkdir(parents=True, exist_ok=True)
    if plot_entries:
        # Draw binned errors for all confidence-bearing methods. Usually the
        # C feature and its values are identical across these ablations.
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, grouped in plot_entries:
            ax.plot(grouped["confidence_mean"], grouped["tsfm_mae"], marker="o", label=f"TSFM: {method}")
            ax.plot(grouped["confidence_mean"], grouped["tree_mae"], marker=".", linestyle="--", alpha=0.75, label=f"Tree: {method}")
        ax.set(title=f"{spec['name']}: confidence vs absolute error", xlabel="Mean confidence C (higher = more confident)", ylabel="MAE")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(ds_out / "confidence_error_by_bin.png", dpi=180)
        plt.close(fig)
    return method_rows, bin_rows, {"dataset": spec["name"], "train_q90": train_q90, "train_q95": train_q95, "train_targets": int(len(train_y)), "prediction_files": len(files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True,
                        help="NAME|DATA_CSV|TIME_COL|TARGET_COL|RESULT_ROOT; repeat for datasets")
    parser.add_argument("--output-dir", default="results/mechanism_analysis")
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    methods: list[dict] = []
    bins: list[dict] = []
    metadata: list[dict] = []
    for spec in args.dataset:
        current_methods, current_bins, current_meta = analyze_dataset(spec, output_root)
        methods.extend(current_methods)
        bins.extend(current_bins)
        metadata.append(current_meta)
    pd.DataFrame(methods).to_csv(output_root / "overall_peak_and_confidence_metrics.csv", index=False)
    if bins:
        pd.DataFrame(bins).to_csv(output_root / "confidence_error_bins.csv", index=False)
    pd.DataFrame(metadata).to_csv(output_root / "analysis_metadata.csv", index=False)
    print(f"Analysis written to {output_root}")
    print("Primary peak masks use training-set q90/q95. Test-top-percent results are labeled descriptive only.")


if __name__ == "__main__":
    main()
