"""Compare the aligned Guangdong and Real-E electricity datasets.

The analysis uses the same chronological 70/10/20 split as the forecasting
pipeline.  Thresholds (spike threshold and normalization scale) are learned
from the training part only, so the report can be used to motivate
cross-dataset differences without test-set leakage.

Example:
    python scripts/compare_dataset_characteristics.py
    python scripts/compare_dataset_characteristics.py --output-dir results/dataset_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "guangdong": {
        "path": "data/GD/guangdong_aligned.csv",
        "time": "ds",
        "target": "y",
    },
    "reale_delu": {
        "path": "data/REALE/DE-LU_aligned.csv",
        "time": "time",
        "target": "y_Day-ahead Price [EUR/MWh]",
    },
}


def _safe_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _autocorr(y: pd.Series, lag: int) -> float | None:
    if len(y) <= lag or y.std() == 0:
        return None
    return _safe_float(y.autocorr(lag=lag))


def analyze_dataset(name: str, spec: dict[str, str]) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(spec["path"], parse_dates=[spec["time"]])
    frame = frame.sort_values(spec["time"]).reset_index(drop=True)
    time = frame[spec["time"]]
    y = pd.to_numeric(frame[spec["target"]], errors="coerce")
    n = len(frame)
    n_train = int(n * 0.7)
    n_valid = int(n * 0.1)
    split = np.full(n, "test", dtype=object)
    split[:n_train] = "train"
    split[n_train : n_train + n_valid] = "valid"
    frame["_split"] = split
    frame["_y"] = y

    train_y = y.iloc[:n_train].dropna()
    # All thresholds/scales below are learned from train only.
    spike_q95 = train_y.quantile(0.95) if len(train_y) else np.nan
    spike_q99 = train_y.quantile(0.99) if len(train_y) else np.nan
    center = train_y.median() if len(train_y) else np.nan
    scale = train_y.std() if len(train_y) else np.nan

    rows: list[dict] = []
    for part in ("train", "valid", "test"):
        mask = frame["_split"].eq(part)
        yp = y.loc[mask]
        valid = yp.dropna()
        diff = valid.diff().dropna()
        rows.append(
            {
                "dataset": name,
                "split": part,
                "rows": int(mask.sum()),
                "target_valid": int(valid.size),
                "target_missing": int(yp.isna().sum()),
                "missing_rate": _safe_float(yp.isna().mean()),
                "mean": _safe_float(valid.mean()),
                "median": _safe_float(valid.median()),
                "std": _safe_float(valid.std()),
                "iqr": _safe_float(valid.quantile(0.75) - valid.quantile(0.25)),
                "min": _safe_float(valid.min()),
                "max": _safe_float(valid.max()),
                "p95": _safe_float(valid.quantile(0.95)),
                "p99": _safe_float(valid.quantile(0.99)),
                "mean_abs_change": _safe_float(diff.abs().mean()),
                "p95_abs_change": _safe_float(diff.abs().quantile(0.95)),
                "lag1_autocorr": _autocorr(valid, 1),
                "lag24_autocorr": _autocorr(valid, 24),
                "lag168_autocorr": _autocorr(valid, 168),
                "train_q95_spike_rate": _safe_float((valid > spike_q95).mean()),
                "train_q99_spike_rate": _safe_float((valid > spike_q99).mean()),
                "train_centered_abs_z_mean": _safe_float(
                    (valid.sub(center).abs() / scale).mean() if scale else np.nan
                ),
            }
        )

    # Feature availability is part of the task definition and explains why
    # a method may transfer differently even when target statistics are alike.
    feature_missing = frame.drop(columns=["_split", "_y"]).isna().mean().sort_values(ascending=False)
    summary = {
        "dataset": name,
        "path": spec["path"],
        "time_col": spec["time"],
        "target_col": spec["target"],
        "rows": int(n),
        "start": str(time.min()),
        "end": str(time.max()),
        "duplicate_timestamps": int(time.duplicated().sum()),
        "non_hourly_intervals": int((time.diff().dropna() != pd.Timedelta(hours=1)).sum()),
        "train_rows": n_train,
        "valid_rows": n_valid,
        "test_rows": n - n_train - n_valid,
        "train_target_count": int(train_y.size),
        "train_spike_q95": _safe_float(spike_q95),
        "train_spike_q99": _safe_float(spike_q99),
        "feature_columns": int(frame.shape[1] - 2),
        "feature_missing_rate_max": _safe_float(feature_missing.max()),
        "feature_missing_rate_mean": _safe_float(feature_missing.mean()),
        "feature_missing_top10": {
            str(k): _safe_float(v) for k, v in feature_missing.head(10).items()
        },
    }
    return summary, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/dataset_comparison")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    tables: list[pd.DataFrame] = []
    for name, spec in DATASETS.items():
        summary, table = analyze_dataset(name, spec)
        summaries.append(summary)
        tables.append(table)

    (root / "dataset_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.concat(tables, ignore_index=True).to_csv(root / "split_statistics.csv", index=False)

    # A compact markdown report is convenient for directly quoting in the
    # paper and preserves the exact thresholds used by the analysis.
    lines = [
        "# Dataset characteristic comparison",
        "",
        "All splits are chronological 70% train / 10% validation / 20% test.",
        "Spike thresholds and normalization scales are learned from train only.",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['dataset']}",
                f"- Time: {summary['start']} to {summary['end']} ({summary['rows']} rows)",
                f"- Target: `{summary['target_col']}`; features: {summary['feature_columns']}",
                f"- Duplicate timestamps: {summary['duplicate_timestamps']}; non-hourly intervals: {summary['non_hourly_intervals']}",
                f"- Train-only spike thresholds: q95={summary['train_spike_q95']}, q99={summary['train_spike_q99']}",
                f"- Mean/max feature missing rate: {summary['feature_missing_rate_mean']}, {summary['feature_missing_rate_max']}",
                "",
            ]
        )
    stats = pd.concat(tables, ignore_index=True)
    cols = [
        "dataset", "split", "target_valid", "missing_rate", "mean", "std", "iqr",
        "min", "max", "mean_abs_change", "p95_abs_change", "lag1_autocorr",
        "lag24_autocorr", "lag168_autocorr", "train_q95_spike_rate", "train_q99_spike_rate",
    ]
    lines.append("## Split statistics")
    lines.append("")
    # Avoid an optional ``tabulate`` dependency: the repository's minimal
    # environment should be sufficient to produce this report.
    display = stats[cols].copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    lines.extend([header, divider, *body])
    (root / "dataset_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {root / 'dataset_summary.json'}")
    print(f"Wrote {root / 'split_statistics.csv'}")
    print(f"Wrote {root / 'dataset_comparison.md'}")


if __name__ == "__main__":
    main()
