"""Build Chinese paper tables and figures from saved electricity experiments.

This is read-only with respect to experiment outputs and writes a compact
paper_artifacts directory containing CSV/Markdown summaries and PNG figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "ar_lgbm": "AR-LGBM",
    "tsfm_point": "TSFM点预测+树",
    "tsfm_point_C": "TSFM点预测+C",
    "tsfm_point_rawU": "TSFM点预测+原始不确定性",
    "tsfm_point_C_weighted_mse": "C+置信度加权MSE",
    "tsfm_point_C_fixed_asym": "C+固定非对称损失",
    "tsfm_point_C_confidence_asym": "C+置信度非对称损失",
    "tsfm_point_fixed_asym_noC": "无C非对称损失",
}
MARKET_LABELS = {
    "guangdong": "广东",
    "shanxi": "山西",
    "reale_delu": "Real-E DE-LU",
    "reale_fr": "Real-E France",
}


def read_json_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for market_dir in sorted(root.iterdir()):
        if not market_dir.is_dir():
            continue
        for method_dir in sorted(market_dir.iterdir()):
            f = method_dir / "metrics_test.json"
            if not f.exists():
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append({
                "market": market_dir.name,
                "market_cn": MARKET_LABELS.get(market_dir.name, market_dir.name),
                "method": method_dir.name,
                "method_cn": METHOD_LABELS.get(method_dir.name, method_dir.name),
                "MSE": d.get("MSE"), "MAE": d.get("MAE"), "R2": d.get("R2"),
                "best_iteration": d.get("best_iteration"),
            })
    return pd.DataFrame(rows)


def read_rolling(root: Path) -> pd.DataFrame:
    rows = []
    methods = ["tsfm_point", "tsfm_point_C_fixed_asym", "tsfm_point_C_confidence_asym"]
    for market_dir in sorted(root.iterdir()):
        if not market_dir.is_dir():
            continue
        for window_dir in sorted(market_dir.glob("window_*")):
            for method in methods:
                f = window_dir / method / "metrics_test.json"
                if not f.exists():
                    continue
                d = json.loads(f.read_text(encoding="utf-8"))
                rows.append({"market": market_dir.name, "market_cn": MARKET_LABELS.get(market_dir.name, market_dir.name),
                             "window": window_dir.name, "method": method,
                             "method_cn": METHOD_LABELS.get(method, method),
                             "MSE": d.get("MSE"), "MAE": d.get("MAE"), "R2": d.get("R2")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    base = df[df.method == "tsfm_point"][["market", "window", "MSE", "MAE", "R2"]].rename(columns={"MSE":"base_MSE", "MAE":"base_MAE", "R2":"base_R2"})
    return df.merge(base, on=["market", "window"], how="left").assign(
        delta_MSE_pct=lambda x: 100 * (x.MSE / x.base_MSE - 1),
        delta_MAE_pct=lambda x: 100 * (x.MAE / x.base_MAE - 1),
        delta_R2=lambda x: x.R2 - x.base_R2,
    )


def make_figures(static: pd.DataFrame, rolling: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "figure.dpi": 160})
    colors = {"TSFM点预测+树":"#4C78A8", "C+固定非对称损失":"#F58518", "C+置信度非对称损失":"#54A24B", "AR-LGBM":"#B279A2"}

    core = static[static.method_cn.isin(["TSFM点预测+树", "C+固定非对称损失", "C+置信度非对称损失", "AR-LGBM"])].copy()
    if not core.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        piv = core.pivot_table(index="market_cn", columns="method_cn", values="MSE")
        cols = [c for c in ["AR-LGBM", "TSFM点预测+树", "C+固定非对称损失", "C+置信度非对称损失"] if c in piv]
        piv[cols].plot.bar(ax=ax, color=[colors[c] for c in cols], width=.78)
        ax.set_title("四个电力市场测试集 MSE 对比")
        ax.set_xlabel(""); ax.set_ylabel("MSE（越低越好）"); ax.grid(axis="y", alpha=.25); ax.legend(title="方法", ncol=2)
        fig.tight_layout(); fig.savefig(out / "图1_四市场总体MSE对比.png", bbox_inches="tight"); plt.close(fig)

    if not rolling.empty:
        sub = rolling[rolling.method_cn.isin(["TSFM点预测+树", "C+固定非对称损失", "C+置信度非对称损失"])].copy()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
        for ax, market in zip(axes.flat, sorted(sub.market.unique())):
            z = sub[sub.market == market]
            for method, g in z.groupby("method_cn"):
                ax.plot(g.window, g.delta_MSE_pct, marker="o", label=method, color=colors.get(method))
            ax.axhline(0, color="black", lw=.8); ax.set_title(MARKET_LABELS.get(market, market)); ax.set_ylabel("相对点预测 MSE变化（%）"); ax.grid(alpha=.25)
        axes[-1,0].set_xlabel("滚动窗口"); axes[-1,1].set_xlabel("滚动窗口")
        handles, labels = axes[0,0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.02))
        fig.suptitle("滚动窗口中机制相对点预测基线的变化", y=1.07); fig.tight_layout(); fig.savefig(out / "图2_滚动窗口机制收益.png", bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        g = sub.groupby("method_cn").agg(平均MSE变化百分比=("delta_MSE_pct", "mean"), 改善窗口数=("delta_MSE_pct", lambda x: int((x < 0).sum())), 总窗口数=("delta_MSE_pct", "size")).reset_index()
        bars = ax.bar(g.method_cn, g.平均MSE变化百分比, color=[colors.get(x, "#999999") for x in g.method_cn])
        ax.axhline(0, color="black", lw=.8); ax.set_ylabel("相对点预测的平均 MSE变化（%）"); ax.set_title("滚动实验的平均机制收益与稳定性"); ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25)
        for b, (_, row) in zip(bars, g.iterrows()): ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"改善 {row.改善窗口数}/{row.总窗口数}", ha="center", va="bottom" if b.get_height() >= 0 else "top", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "图3_机制平均收益与稳定性.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-root", type=Path, required=True)
    ap.add_argument("--rolling-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    static = read_json_metrics(args.static_root); rolling = read_rolling(args.rolling_root)
    static.to_csv(args.output / "表1_全量消融指标.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(args.output / "表2_滚动窗口指标.csv", index=False, encoding="utf-8-sig")
    if not static.empty:
        static.groupby(["market_cn", "method_cn"], as_index=False)[["MSE", "MAE", "R2"]].mean().to_csv(args.output / "表3_各市场平均指标.csv", index=False, encoding="utf-8-sig")
    if not rolling.empty:
        rolling.groupby(["market_cn", "method_cn"], as_index=False)[["MSE", "MAE", "R2", "delta_MSE_pct", "delta_MAE_pct", "delta_R2"]].mean().to_csv(args.output / "表4_滚动实验平均指标.csv", index=False, encoding="utf-8-sig")
    make_figures(static, rolling, args.output)
    missing_valid = not any(args.rolling_root.rglob("predictions_valid.csv"))
    report = ["# 论文结果审计", "", f"原始全量指标记录：{len(static)} 条。", f"滚动窗口指标记录：{len(rolling)} 条。", "", "## 当前缺口", ""]
    report.append("- 滚动结果未保存 predictions_valid.csv，因此 rolling_adaptive_summary.csv 为空，严格的验证集自动门控尚未完成。" if missing_valid else "- 已发现验证集预测文件，可继续核验自动门控汇总。")
    report += ["- 当前结果足以支持跨市场横向比较、三类机制消融和滚动状态依赖分析。", "- 论文中不能把离线逐窗口最优方法称为自动选择结果，必须标注为事后机制比较。", "", "## 建议论文图表", "", "- 图1：四市场总体 MSE。", "- 图2：滚动窗口相对点预测的 MSE 变化。", "- 图3：机制平均收益与改善窗口数。"]
    (args.output / "论文结果审计.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote artifacts to {args.output}")


if __name__ == "__main__":
    main()
