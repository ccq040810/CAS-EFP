"""Create leakage-safe descriptive comparisons for the four electricity markets.

Run from the repository root:
    python scripts/analyze_market_regimes.py

All target-distribution cutoffs are fitted on the chronological training split
(first 70%); validation is the next 10%, and test is the final 20%.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "market_regime_analysis"
DATASETS = {
    "Guangdong": ("data/GD/guangdong_aligned.csv", "ds", "y"),
    "Shanxi": ("data/SHANXI/shanxi_aligned.csv", "ds", "y"),
    "Real-E DE-LU": ("data/REALE/DE-LU_aligned.csv", "time", "y_Day-ahead Price [EUR/MWh]"),
    "Real-E France": ("data/REALE/FR_aligned.csv", "time", "y_Day-ahead Price [EUR/MWh]"),
}


def autocorr(series: pd.Series, lag: int) -> float:
    return float(series.autocorr(lag=lag)) if len(series) > lag and series.std() else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    split_rows: list[dict] = []
    plot_data: dict[str, tuple[pd.Series, dict[str, pd.Series]]] = {}

    for market, (relpath, time_col, target_col) in DATASETS.items():
        frame = pd.read_csv(ROOT / relpath, parse_dates=[time_col]).sort_values(time_col).reset_index(drop=True)
        y = pd.to_numeric(frame[target_col], errors="coerce")
        n = len(frame)
        n_train, n_valid = int(n * 0.7), int(n * 0.1)
        cuts = {
            "train": y.iloc[:n_train],
            "valid": y.iloc[n_train:n_train + n_valid],
            "test": y.iloc[n_train + n_valid:],
        }
        train = cuts["train"].dropna()
        q95, q99 = float(train.quantile(.95)), float(train.quantile(.99))
        feature_cols = [c for c in frame.columns if c not in (time_col, target_col)]
        feature_missing = frame[feature_cols].isna().mean() if feature_cols else pd.Series(dtype=float)
        summaries.append({
            "market": market,
            "source_file": relpath,
            "target": target_col,
            "rows": n,
            "start": str(frame[time_col].min()),
            "end": str(frame[time_col].max()),
            "duplicate_timestamps": int(frame[time_col].duplicated().sum()),
            "non_hourly_intervals": int((frame[time_col].diff().dropna() != pd.Timedelta(hours=1)).sum()),
            "feature_count": len(feature_cols),
            "target_missing_rate": float(y.isna().mean()),
            "feature_missing_mean": float(feature_missing.mean()) if len(feature_missing) else 0.0,
            "feature_missing_max": float(feature_missing.max()) if len(feature_missing) else 0.0,
            "train_q95": q95,
            "train_q99": q99,
        })
        plot_data[market] = (frame[time_col], cuts)
        for split, part in cuts.items():
            valid = part.dropna()
            # Do not compact away missing timestamps before calculating changes
            # or autocorrelation. That would incorrectly treat observations
            # separated by gaps as consecutive hourly values.
            delta = part.diff().abs().dropna()
            split_rows.append({
                "market": market,
                "split": split,
                "rows": len(part),
                "valid_targets": len(valid),
                "missing_rate": float(part.isna().mean()),
                "mean": float(valid.mean()),
                "median": float(valid.median()),
                "std": float(valid.std()),
                "iqr": float(valid.quantile(.75) - valid.quantile(.25)),
                "p95": float(valid.quantile(.95)),
                "p99": float(valid.quantile(.99)),
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean_abs_change": float(delta.mean()),
                "p95_abs_change": float(delta.quantile(.95)),
                "acf_lag1": autocorr(part, 1),
                "acf_lag24": autocorr(part, 24),
                "acf_lag168": autocorr(part, 168),
                "above_train_q95_rate": float((valid > q95).mean()),
                "above_train_q99_rate": float((valid > q99).mean()),
            })

    summary_df = pd.DataFrame(summaries)
    splits_df = pd.DataFrame(split_rows)
    summary_df.to_csv(OUT / "market_summary.csv", index=False)
    splits_df.to_csv(OUT / "split_statistics.csv", index=False)
    (OUT / "market_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    colors = {"train": "#3465a4", "valid": "#e6a23c", "test": "#bd4b4b"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, market in zip(axes.flat, DATASETS):
        time, cuts = plot_data[market]
        for split, part in cuts.items():
            idx = {"train": 0, "valid": 1, "test": 2}[split]
            xs = time.iloc[part.index]
            ax.scatter(xs, part, s=2, alpha=.28, color=colors[split], label=split if ax is axes.flat[0] else None)
        ax.set_title(market)
        ax.set_ylabel("Price (source unit)")
        ax.grid(alpha=.2)
    axes.flat[0].legend(frameon=False, markerscale=4)
    fig.suptitle("Hourly spot prices by chronological split (70/10/20)")
    fig.savefig(OUT / "price_regimes_by_split.png", dpi=180)
    plt.close(fig)

    # The normalized panel compares within-market distribution shift without
    # treating different currency / price scales as directly comparable.
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    markets = list(DATASETS)
    width = .23
    for j, split in enumerate(("train", "valid", "test")):
        vals = []
        for market in markets:
            row = splits_df[(splits_df.market == market) & (splits_df.split == split)].iloc[0]
            tr = splits_df[(splits_df.market == market) & (splits_df.split == "train")].iloc[0]
            vals.append((row["mean"] - tr["mean"]) / tr["std"])
        ax.bar(np.arange(len(markets)) + (j - 1) * width, vals, width, label=split, color=colors[split])
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(np.arange(len(markets)), markets)
    ax.set_ylabel("Split mean minus train mean / train SD")
    ax.set_title("Target-level temporal shift (standardized within each market)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "target_shift_by_split.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Cross-market regime comparison",
        "",
        "Computed from local aligned CSVs with a chronological 70/10/20 split. Thresholds use train targets only. Price units and market definitions differ, so absolute levels are not compared across markets.",
        "",
        "## Dataset integrity and coverage",
        "",
        "| Market | Rows | Period | Target missing | Feature count | Feature missing (mean/max) | Train P95/P99 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(f"| {s['market']} | {s['rows']} | {s['start'][:10]} to {s['end'][:10]} | {s['target_missing_rate']:.1%} | {s['feature_count']} | {s['feature_missing_mean']:.1%}/{s['feature_missing_max']:.1%} | {s['train_q95']:.3f}/{s['train_q99']:.3f} |")
    lines += ["", "## Split diagnostics", "", "| Market | Split | Mean | SD | P95 | P99 | Min | Max | P95 abs Δ | ACF(1/24/168) | Above train P95 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in split_rows:
        lines.append(f"| {r['market']} | {r['split']} | {r['mean']:.3f} | {r['std']:.3f} | {r['p95']:.3f} | {r['p99']:.3f} | {r['min']:.3f} | {r['max']:.3f} | {r['p95_abs_change']:.3f} | {r['acf_lag1']:.3f}/{r['acf_lag24']:.3f}/{r['acf_lag168']:.3f} | {r['above_train_q95_rate']:.1%} |")
    lines += [
        "",
        "## Reading the evidence",
        "",
        "- Guangdong and Shanxi are Chinese market datasets, but they are not matched replications: their geography, price formation, sampling history, covariates, target definition, and effective test windows differ. Shanxi is downsampled from 15-minute prices to hourly means; Guangdong is a separate hourly aligned product.",
        "- DE-LU has a severe chronological regime break in this split: its train target is mostly 2015–2021 data with a large missing block, while validation/test prices are substantially higher and much more volatile. A train-fitted P95/P99 threshold therefore marks a large share of later samples as spikes. This makes the experiment a stress test of regime transfer, not a stationary average-case benchmark.",
        "- France is also a Real-E hourly day-ahead series, but the weak near-zero test R² and large train/validation RMSE gap in the provided run indicate that benchmark-family similarity alone does not ensure useful transfer. The split-level target statistics below should be read alongside the exact validation/test dates before attributing its outcome to a confidence mechanism.",
        "- The numerical score differences across datasets do not by themselves identify the cause. Potential mediators include temporal distribution shift, TSFM error/calibration, confidence–error association, covariate availability/missingness, and asymmetric residual direction. Confidence bins or correlations are descriptive; use paired bootstrap by day/week and report confidence intervals before making inferential claims.",
        "",
        "## Observed model outcomes from experiment logs",
        "",
        "These are user-provided experiment logs, not recomputed from result files in this checkout. The Shanxi exact metrics and FR per-sample forecasts were not present among the local result artifacts inspected.",
        "",
        "| Market | Baseline/point result | Confidence/asymmetric result | Interpretation limited to current run |",
        "|---|---|---|---|",
        "| Guangdong | TSFM point MSE 7759.98, R² 0.5134; best among reported variants | C + weighted MSE MSE 8168.33, R² 0.4878; confidence-asymmetric variants worse | Added confidence/loss did not improve this test window |",
        "| DE-LU | AR-LGBM MSE 1302.95, R² 0.3192; TSFM point MSE 1329.43, R² 0.3054 | C + confidence-asym MSE 866.38, R² 0.5473 | Large gain in this nonstationary later-period window; mechanism and uncertainty need paired/time-block inference |",
        "| France | TSFM point MSE 16289.09, R² 0.0043 | C + confidence-asym MSE 16024.80, R² 0.0204 | Small absolute improvement, weak overall predictive skill; train/valid RMSE gap is extreme |",
        "| Shanxi | User reported TSFM point best overall | Exact stage-3 metrics not available here | Do not quantify or claim significance until metrics/predictions are collected |",
        "",
        "## Paper-safe current conclusion",
        "",
        "The defensible claim is conditional: a frozen TSFM–tree correction architecture is a useful experimental framework, while confidence-guided asymmetric correction is not universally beneficial. It improved the reported DE-LU run, gave only a small gain on France, and was not beneficial on Guangdong; Shanxi currently has only a qualitative result in the local evidence. The candidate boundary is whether the TSFM uncertainty/confidence signal is stable and informative under the target market's temporal regime, and whether the downstream loss direction matches the residual risk. This boundary remains a hypothesis until tested with aligned per-sample diagnostics, rolling-origin splits, and uncertainty intervals.",
        "",
        "## Next analyses needed before submission",
        "",
        "1. Export timestamp-aligned test predictions for all methods and markets. Analyze TSFM absolute error by confidence decile, with Spearman correlation and day/week-block bootstrap intervals.",
        "2. Define peak events from training-only thresholds (P95/P99), then compare each method's MAE/RMSE and peak-event error on the same test timestamps. Also report non-peak performance and paired confidence intervals.",
        "3. Add rolling-origin or multiple-year windows for DE-LU and France, and matched forecast horizons where possible. The current single 70/10/20 split confounds dataset identity with calendar regime.",
        "4. Audit timestamp/time-zone/DST handling and feature availability at forecast time, especially the 15-minute-to-hourly Shanxi aggregation and DE-LU's missing early targets.",
        "",
        "Figures: `price_regimes_by_split.png` and `target_shift_by_split.png`. Machine-readable tables: `market_summary.csv`, `split_statistics.csv`.",
    ]
    (OUT / "cross_market_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
