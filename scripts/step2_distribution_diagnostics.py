# -*- coding: utf-8 -*-
"""Step-2：TSFM 原生预测分布质量诊断（CAS-EPF 2.0）。

在把冻结 TSFM 的预测分布接进下游 LightGBM 之前，先验证分布本身是否可信。
不引入任何树模型 / 非对称损失，只检查 TSFM 零样本的原生预测分布：

  1. 原生分位数单调性（是否存在分位穿越）
  2. 中位数点预测 Q(0.5) 与模型返回 point_forecast 的一致性
  3. 逐原生分位覆盖率 P(Y<=Q(τ)) vs 名义 τ + pinball loss
  4. 可靠性图 + PIT 直方图
  5. 离散度 U 的范围 / NaN / 极端值；U 积分近似的稳定性
  6. 置信度 C=exp(-U/(σ_U+ε))（σ_U 仅用训练窗计算）的范围与按 C 分箱的 MAE
  7. U 与 |误差| 的秩/线相关；尾部不对称（上尾 q90-q50 vs 下尾 q50-q10）
  8. 误差方向 err=y_true-y_hat（>0 = 低估，与 regressor.py 的 residual 约定一致）

关键约定（对齐 P0 修正）：
  - Chronos-2 输出最后一维是「原生分位节点」而非独立样本；直接消费
    res['quantile_levels'] + res['forecast']（原生分位数值），不再对其做 percentile。
  - 点预测 = Q(0.5)（中位数）；离散度 U = (1/(τ_hi-τ_lo)) ∫|Q(τ)-Q(0.5)|dτ（梯形积分）。
  - 覆盖率/分位损失仅用于检验分布质量，不代表用测试集调置信度公式。

用法：
  python scripts/step2_distribution_diagnostics.py --model chronos2 --device cuda
  python scripts/step2_distribution_diagnostics.py --model chronos2 --device cpu --max_windows 200
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

DEFAULT_MODEL_PATHS = {
    "chronos2": "models/chronos-2/chronos-2",
    "timesfm": "models/TimesFM",
}

# 采样型模型（无原生分位水平）的退化固定网格，仅在非 chronos2 时使用
FALLBACK_GRID = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_model(model_name: str, model_path: str, device: str):
    from ts_models import TimeSeriesModelFactory
    model = TimeSeriesModelFactory.create_model(model_name, model_path=model_path, model_type=model_name)
    if hasattr(model, "to"):
        model.to(device)
    return model


def _extract_mean(res) -> np.ndarray:
    if isinstance(res, dict):
        m = res.get("mean")
        if m is None:
            m = res.get("point_forecast")
    else:
        m = res
    m = _as_numpy(m).astype(np.float32)
    if m.ndim == 2:
        m = m[:, None, :]
    elif m.ndim == 4:
        m = np.nanmean(m, axis=-1)
    return m  # [B, C, H]


def _extract_native(res):
    """返回 (point [B,C,H], qvals [B,C,H,Q], levels [Q])；无原生分位时 qvals/levels 为 None。"""
    point = _extract_mean(res)
    qvals = levels = None
    if isinstance(res, dict) and res.get("quantile_levels") is not None:
        levels = [float(x) for x in res["quantile_levels"]]
        vals = res.get("forecast")
        if vals is None:
            vals = res.get("quantile_values")
        if vals is not None:
            vals = _as_numpy(vals).astype(np.float32)
            if vals.ndim == 3:
                vals = vals[:, None, :, :]
            if vals.ndim == 4:
                qvals = vals
    return point, qvals, levels


def _trapz_abs_dev(qvals: np.ndarray, point: np.ndarray, levels) -> np.ndarray:
    """U = (1/(τ_hi-τ_lo)) ∫|Q(τ)-Q(0.5)|dτ（梯形积分）。qvals [.., Q]，point 同前维。"""
    levels = np.asarray(levels, dtype=np.float64)
    ad = np.abs(qvals - point[..., None])
    d = levels[1:] - levels[:-1]
    return ((ad[..., :-1] + ad[..., 1:]) * 0.5 * d).sum(-1) / (levels[-1] - levels[0])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(1, len(x) + 1)
        return r
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def run_model(args, model_name, model_path, device, X, seq_len, horizon, anchors, num_samples):
    model = _load_model(model_name, model_path, device)
    n = len(anchors)
    points, qvals_list, targets = [], [], []
    native_levels = None
    for i in range(0, n, args.batch_size):
        batch = anchors[i:i + args.batch_size]
        ctx = np.stack([X[a - seq_len:a].T for a in batch], axis=0)  # [B, C, L]
        ctx_t = torch.from_numpy(ctx).to(device)
        with torch.no_grad():
            res = model.predict(context=ctx_t, forecast_horizon=horizon, num_samples=num_samples)

        point, qv, lv = _extract_native(res)  # point [B,C,H], qv [B,C,H,Q], lv [Q]
        yb = np.stack([X[a:a + horizon, -1] for a in batch], axis=0)  # [B, H]

        points.append(point[:, -1, :])  # y 通道
        targets.append(yb)
        if qv is not None:
            qvals_list.append(qv[:, -1, :, :])  # [B, H, Q]
            if native_levels is None:
                native_levels = np.asarray(lv, dtype=np.float64)
        if args.max_windows and (i + args.batch_size) >= args.max_windows:
            break

    points = np.concatenate(points, axis=0)          # [N, H]
    targets = np.concatenate(targets, axis=0)        # [N, H]
    if qvals_list:
        qvals = np.concatenate(qvals_list, axis=0)   # [N, H, Q]
    else:
        qvals = None
    return points, qvals, native_levels, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"data/GD/guangdong_aligned.csv")
    ap.add_argument("--time_col", default="ds")
    ap.add_argument("--model", default="chronos2")
    ap.add_argument("--model_paths", default="", help='JSON: {"chronos2": "...", "timesfm": "..."}')
    ap.add_argument("--mv_cols", default="y", help="进 TSFM 的列，逗号分隔，最后一列必须是 y")
    ap.add_argument("--seq_len", type=int, default=168)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--train_ratio", type=float, default=0.7, help="用于估计 σ_U 的训练窗比例（按时间）")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="诊断评估的测试窗比例（尾部）")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_windows", type=int, default=0, help=">0 时限制窗口数（冒烟）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_dir", default="exp_results/step2_distribution")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] 未检测到 GPU，回退到 cpu")
        args.device = "cpu"
    print(f"[step2] device={args.device}  torch={torch.__version__}")

    models = [m.strip().lower() for m in args.model.split(",") if m.strip()]
    model_paths = json.loads(args.model_paths) if args.model_paths else {}
    for m in models:
        if m not in model_paths:
            model_paths[m] = DEFAULT_MODEL_PATHS[m]

    df = pd.read_csv(args.data)
    df[args.time_col] = pd.to_datetime(df[args.time_col])
    df = df.sort_values(args.time_col).set_index(args.time_col)
    mv_cols = [c.strip() for c in args.mv_cols.split(",") if c.strip()]
    missing = [c for c in mv_cols if c not in df.columns]
    if missing:
        raise KeyError(f"列缺失: {missing}；可用列: {list(df.columns)}")
    X = df[mv_cols].to_numpy(np.float32)
    X = X[np.isfinite(X).all(axis=1)]  # 整行含 NaN 则丢弃

    n = X.shape[0]
    anchors = list(range(args.seq_len, n - args.horizon, args.stride))
    if args.max_windows:
        anchors = anchors[: args.max_windows]
    n_w = len(anchors)
    # 时间顺序切分：前 train_ratio 估计 σ_U，尾 test_ratio 评估
    n_test = int(n_w * args.test_ratio)
    n_train = int(n_w * args.train_ratio)
    print(f"[step2] 有效 {X.shape[0]} 行；窗口 {n_w} 个（train={n_train}, test={n_test}）")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for model_name in models:
        print(f"[step2] 推理 {model_name} @ {model_paths[model_name]} ...")
        points, qvals, levels, targets = run_model(
            args, model_name, model_paths[model_name], args.device, X,
            args.seq_len, args.horizon, anchors, args.num_samples,
        )
        N, H = points.shape
        y = targets.reshape(-1)
        yhat = points.reshape(-1)
        err = y - yhat
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))

        if qvals is not None and levels is not None:
            Q = qvals.reshape(-1, qvals.shape[-1])  # [N*H, Q]
            # 单调性：沿升序水平的原生分位值是否非降
            monotone = float(np.mean(np.all(np.diff(Q, axis=1) >= -1e-9, axis=1)))
            # 中位数一致性：point 应等于 τ=0.5 最近节点
            med_idx = int(np.argmin(np.abs(levels - 0.5)))
            median_consistency = float(np.max(np.abs(yhat - Q[:, med_idx])))
            # 逐原生水平覆盖率 + pinball
            emp_cov = {}
            pin = {}
            for j, tau in enumerate(levels):
                q = Q[:, j]
                emp_cov[f"cov_tau{tau:g}"] = float(np.mean(y <= q))
                e = y - q
                pin[f"pinball_tau{tau:g}"] = float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))
            pinball_avg = float(np.mean(list(pin.values())))
            # U：重算（全节点）并与 res 侧一致性、粗节点稳定性
            U = _trapz_abs_dev(qvals.reshape(N, H, qvals.shape[-1]), points, levels).reshape(-1)
            coarse_idx = np.arange(0, len(levels), 2)
            U_coarse = _trapz_abs_dev(qvals.reshape(N, H, qvals.shape[-1])[:, :, coarse_idx],
                                      points, levels[coarse_idx]).reshape(-1)
            integ_stability = float(np.corrcoef(U, U_coarse)[0, 1]) if np.std(U) > 0 else float("nan")
        else:
            Q = None
            monotone = float("nan")
            median_consistency = float("nan")
            emp_cov = {}
            pin = {}
            pinball_avg = float("nan")
            # 无原生分位：退化为 |预测残差| 作为粗糙离散度
            U = np.abs(err)
            integ_stability = float("nan")

        # σ_U 仅用训练窗；C = exp(-U/(σ_U+ε))
        U_full = U.reshape(N, H)
        n_flat = N * H
        flat_idx = np.arange(n_flat)
        is_test = (flat_idx // H) >= (N - n_test)
        is_train = (flat_idx // H) < n_train
        sigma_U = float(np.std(U_full.reshape(-1)[is_train])) + 1e-8
        C = np.exp(-U / (sigma_U + 1e-8))

        # U / C 范围与 NaN
        u_stat = dict(u_min=float(np.nanmin(U)), u_max=float(np.nanmax(U)),
                      u_mean=float(np.nanmean(U)), u_nan=int(np.isnan(U).sum()))
        c_stat = dict(c_min=float(np.nanmin(C)), c_max=float(np.nanmax(C)),
                      c_mean=float(np.nanmean(C)), c_nan=int(np.isnan(C).sum()))

        # 尾部不对称（原生 q10/q50/q90）
        tail_asym = float("nan")
        if Q is not None:
            def qi(tau):
                return Q[:, int(np.argmin(np.abs(levels - tau)))]
            tail_asym = float(np.mean((qi(0.90) - qi(0.50)) - (qi(0.50) - qi(0.10))))

        # U 与 |误差| 的相关
        abs_err = np.abs(err)
        spear = _spearman(U[is_test], abs_err[is_test]) if is_test.sum() > 3 else float("nan")
        pear = float(np.corrcoef(U[is_test], abs_err[is_test])[0, 1]) if is_test.sum() > 3 else float("nan")

        # 按 C 分箱 MAE（测试集）
        c_bin = {}
        if is_test.sum() > 20:
            c_test = C[is_test]
            qs = np.quantile(c_test, [0, 0.25, 0.5, 0.75, 1.0])
            qs = np.unique(qs)
            if len(qs) >= 2:
                ids = np.digitize(c_test, qs[1:])
                for b in range(len(qs) - 1):
                    m = ids == b
                    if m.sum() == 0:
                        continue
                    c_bin[f"Cbin{b}_mae"] = float(np.mean(abs_err[is_test][m]))
                    c_bin[f"Cbin{b}_n"] = int(m.sum())

        # 误差方向（测试集）
        p_under = float(np.mean(err[is_test] > 0))
        mean_err = float(np.mean(err[is_test]))

        # 保存逐样本
        pred_df = pd.DataFrame({"y": y, "yhat": yhat, "U": U, "C": C, "err": err})
        if Q is not None:
            for j, tau in enumerate(levels):
                pred_df[f"q{tau:g}"] = Q[:, j]
        pred_df.to_parquet(os.path.join(args.out_dir, f"{model_name}_preds.parquet"), index=False)

        rows.append({
            "model": model_name, "n_samples": n_flat, "sigma_U": sigma_U,
            "mae": mae, "rmse": rmse, "pinball_avg": pinball_avg,
            "mean_err": mean_err, "p_underestimate": p_under,
            "native_monotone_frac": monotone, "median_consistency_maxabs": median_consistency,
            "integ_stability_corr": integ_stability, "tail_asymmetry": tail_asym,
            "spearman_U_abserr": spear, "pearson_U_abserr": pear,
            **u_stat, **c_stat, **emp_cov, **c_bin,
        })
        print(f"[step2][{model_name}] MAE={mae:.3f} RMSE={rmse:.3f} pinball_avg={pinball_avg:.3f} "
              f"monotone={monotone:.4f} median_consist={median_consistency:.4f}")
        print(f"       σ_U={sigma_U:.3f} tail_asym={tail_asym:.3f} spearman(U,|e|)={spear:.3f} "
              f"p_under={p_under:.3f} mean_err={mean_err:.3f}")

        _plot(model_name, args.out_dir, levels, y, Q, err, U, C, abs_err, is_test)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n[step2] 完成。汇总: {summary_path}")
    print(summary.to_string(index=False))


def _plot(model_name, out_dir, levels, y, Q, err, U, C, abs_err, is_test):
    base = os.path.join(out_dir, model_name)
    # 可靠性图
    if Q is not None and levels is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        emp = [float(np.mean(y <= Q[:, j])) for j in range(Q.shape[1])]
        ax.plot(levels, emp, "o-", label="empirical")
        ax.plot([0, 1], [0, 1], "k--", label="ideal")
        ax.set_xlabel("nominal quantile level")
        ax.set_ylabel("empirical coverage P(Y<=Q)")
        ax.set_title(f"{model_name}: reliability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{base}_reliability.png", dpi=120)
        plt.close(fig)

        # PIT
        pit = np.full(y.shape, np.nan)
        order = np.argsort(levels)
        lv = np.asarray(levels)[order]
        for i in range(len(y)):
            qv = Q[i][order]
            m = np.concatenate(([True], np.diff(qv) > 0))
            pit[i] = np.interp(y[i], qv[m], lv[m], left=0.0, right=1.0)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(pit, bins=20, range=(0, 1), density=True, alpha=0.7)
        ax.axhline(1.0, color="k", linestyle="--")
        ax.set_xlabel("PIT"); ax.set_ylabel("density")
        ax.set_title(f"{model_name}: PIT histogram")
        fig.tight_layout()
        fig.savefig(f"{base}_pit.png", dpi=120)
        plt.close(fig)

    # U vs |误差|（测试集散点/分箱）
    if is_test.sum() > 20:
        fig, ax = plt.subplots(figsize=(5, 4))
        ut, at = U[is_test], abs_err[is_test]
        # 按 U 十分位分箱画均值
        qs = np.quantile(ut, np.linspace(0, 1, 11))
        qs = np.unique(qs)
        ids = np.digitize(ut, qs[1:])
        cx, cy = [], []
        for b in range(len(qs) - 1):
            m = ids == b
            if m.sum() > 0:
                cx.append(np.mean(ut[m])); cy.append(np.mean(at[m]))
        ax.plot(cx, cy, "o-")
        ax.set_xlabel("U (MAD integral)"); ax.set_ylabel("mean |error|")
        ax.set_title(f"{model_name}: |error| vs U")
        fig.tight_layout()
        fig.savefig(f"{base}_u_vs_error.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
