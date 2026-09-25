from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import chinese_calendar as calendar
import numpy as np
import pandas as pd


POINTS_PER_DAY = 96
NOON_OFFSET = 48
CHINA_EU_STYLE_SYNTHETIC_COLS = [
    "actual_total_load_proxy",
    "day_ahead_price_proxy",
    "hour_of_day",
    "day_of_week",
    "is_holiday",
    "rpr_proxy",
    "tas_proxy",
]


def project_unified_sources_to_legacy(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()
    if "source_1_historical_price" in dff.columns:
        for c in ["clearing price (CNY/MWh)", "target", "OT", "y_Day-ahead Price [EUR/MWh]", "Day-ahead Price [EUR/MWh]"]:
            if c in dff.columns:
                dff[c] = dff["source_1_historical_price"]
    if "source_2_dynamic_exogenous_load" in dff.columns:
        for c in ["demand", "Ampirion Load Forecast", "Ampirion zonal load forecast", "total_Actual Total Load [MW] - BZN|DE-LU", "Actual Total Load [MW]"]:
            if c in dff.columns:
                dff[c] = dff["source_2_dynamic_exogenous_load"]
    return dff


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return project_unified_sources_to_legacy(pd.read_csv(path))
    if path.endswith(".parquet"):
        return project_unified_sources_to_legacy(pd.read_parquet(path))
    raise ValueError(path)


def _load_single_json(path: str) -> dict:
    obj = json.load(open(path, "r", encoding="utf-8"))
    return obj[0] if isinstance(obj, list) else obj


def _to_shanghai(ts):
    t = pd.to_datetime(ts, errors="raise")
    return t.tz_localize("Asia/Shanghai") if t.tzinfo is None else t.tz_convert("Asia/Shanghai")


def _to_shanghai_series(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="raise")
    return t.dt.tz_localize("Asia/Shanghai") if t.dt.tz is None else t.dt.tz_convert("Asia/Shanghai")


def slice_df(df: pd.DataFrame, time_col: str, start, end) -> pd.DataFrame:
    t = _to_shanghai_series(df[time_col])
    s = _to_shanghai(start)
    e = _to_shanghai(end)
    return df.loc[(t >= s) & (t <= e)].copy()


def _attach_tsfm_features(
    df: pd.DataFrame,
    *,
    time_col: str,
    y_col: str,
    pred_table: pd.DataFrame,
    tsfm_models: str,
    tsfm_vars: List[str],
    use_uncertainty: bool = True,
) -> tuple[pd.DataFrame, List[str]]:
    if pred_table is None or len(pred_table) == 0:
        return df, []

    pt = pred_table.copy()
    pt[time_col] = _to_shanghai_series(pt["time"])
    if time_col != "time":
        pt = pt.drop(columns=["time"])
    pt = pt.sort_values(time_col)

    dff = df.copy()
    dff[time_col] = _to_shanghai_series(dff[time_col])

    models = [m.strip().lower() for m in tsfm_models.split(",") if m.strip()]
    want_vars = list(tsfm_vars)
    if y_col not in want_vars:
        want_vars.append(y_col)

    feature_cols: List[str] = []
    for model_name in models:
        for var_name in want_vars:
            base_col = f"pred_{model_name}_{var_name}"
            unc_col = f"{base_col}_uncertainty"
            if base_col in pt.columns:
                feature_cols.append(base_col)
            if use_uncertainty and unc_col in pt.columns:
                feature_cols.append(unc_col)
    feature_cols = list(dict.fromkeys(feature_cols))

    if not feature_cols:
        return dff, []

    dff = dff.merge(pt[[time_col] + feature_cols], on=time_col, how="left")
    return dff, feature_cols


def make_xy(
    df: pd.DataFrame,
    *,
    time_col: str,
    cov_cols: List[str],
    y_col: str,
    target_hour: int,
) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    if len(df) == 0:
        return np.zeros((0, len(cov_cols)), np.float32), np.zeros((0,), np.float32), 0, []

    dff = df.copy()
    dff["_t"] = pd.to_datetime(dff[time_col])
    dff["_d"] = dff["_t"].dt.date

    X_days, y_days, times = [], [], []

    for _, g in dff.groupby("_d"):
        g = g.sort_values("_t")
        if len(g) != POINTS_PER_DAY:
            continue
        if g["_t"].iloc[0].hour != target_hour:
            continue
        if not calendar.is_workday(g["_t"].iloc[NOON_OFFSET].date()):
            continue

        X_days.append(g[cov_cols].to_numpy(np.float32))
        y_days.append(g[y_col].to_numpy(np.float32))
        times.extend(g["_t"].astype(str).tolist())

    if not X_days:
        return np.zeros((0, len(cov_cols)), np.float32), np.zeros((0,), np.float32), 0, []

    return (
        np.concatenate(X_days, axis=0),
        np.concatenate(y_days, axis=0),
        len(X_days),
        times,
    )


def _build_epf_benchmark_datasets(
    args,
    split_cfg: Dict[str, Any] | None = None,
    cols_cfg: List[str] | None = None,
    *,
    scale: bool | None = None,
):
    from data_provider.data_loader import CovariateDatasetBenchmark

    if split_cfg is None:
        split_cfg = _load_single_json(args.data_split_path)["data_split"]
    if cols_cfg is None:
        cols_cfg = list(_load_single_json(args.regress_cols_path)["target_columns"])

    if scale is None:
        scale_flag = bool(getattr(args, "scale", False))
    else:
        scale_flag = bool(scale)

    size = [
        int(args.seq_len),
        int(getattr(args, "input_token_len", 16)),
        int(getattr(args, "output_token_len", 720)),
        int(args.pred_len),
    ]

    common_kwargs = dict(
        size=size,
        scale=scale_flag,
        data_path=str(args.data_path),
        target_columns=list(cols_cfg),
        data_split={
            "train": float(split_cfg["train"]),
            "valid": float(split_cfg["valid"]),
            "test": float(split_cfg["test"]),
        },
        clean=bool(getattr(args, "clean", False)),
        shift=int(getattr(args, "shift", 0)),
    )

    ds_tr = CovariateDatasetBenchmark(flag="train", **common_kwargs)
    ds_va = CovariateDatasetBenchmark(flag="val", **common_kwargs)
    ds_te = CovariateDatasetBenchmark(flag="test", **common_kwargs)

    print(
        f"[epf_benchmark] scale={scale_flag}, "
        f"ds_tr={len(ds_tr)}, ds_va={len(ds_va)}, ds_te={len(ds_te)}"
    )
    return ds_tr, ds_va, ds_te, split_cfg, cols_cfg


def build_std_key_grid(ds) -> pd.DataFrame:
    cols = ["anchor_time", "time", "h"]

    if len(ds) == 0:
        return pd.DataFrame(columns=cols)
    if getattr(ds, "time_split", None) is None:
        raise ValueError("CovariateDatasetBenchmark.time_split is None (need date column).")

    pred_len = int(ds.pred_len)
    seq_len = int(ds.seq_len)
    t_split = pd.to_datetime(ds.time_split, errors="raise").reset_index(drop=True)
    rows: List[dict] = []

    for i in range(len(ds)):
        anchor_idx = i + seq_len
        anchor_time = pd.to_datetime(t_split.iloc[anchor_idx]).tz_localize(None)
        for h in range(pred_len):
            rows.append(
                {
                    "anchor_time": anchor_time,
                    "time": pd.to_datetime(t_split.iloc[anchor_idx + h]).tz_localize(None),
                    "h": np.int32(h + 1),
                }
            )

    return pd.DataFrame(rows, columns=cols)


def build_std_cov_target_table(ds, *, cov_cols: List[str]) -> pd.DataFrame:
    cols = ["anchor_time", "time", "h", "y"] + list(cov_cols)

    if len(ds) == 0:
        return pd.DataFrame(columns=cols)
    if getattr(ds, "time_split", None) is None:
        raise ValueError("CovariateDatasetBenchmark.time_split is None (need date column).")

    pred_len = int(ds.pred_len)
    need_cov_dim = len(cov_cols)
    key_grid = build_std_key_grid(ds)
    rows: List[dict] = []
    seq_len = int(ds.seq_len)

    for i in range(len(ds)):
        _, _, seq_y, seq_y_cov = ds[i]
        seq_y = np.asarray(seq_y, dtype=np.float32).reshape(-1)
        seq_y_cov = np.asarray(seq_y_cov, dtype=np.float32)

        if seq_y.shape[0] != pred_len:
            raise ValueError(
                f"[std table] seq_y len={seq_y.shape[0]} != pred_len={pred_len} for flag={getattr(ds, 'flag', 'unknown')}"
            )
        if seq_y_cov.ndim != 2 or seq_y_cov.shape[0] != pred_len:
            raise ValueError(
                f"[std table] seq_y_cov shape={seq_y_cov.shape} incompatible with pred_len={pred_len} "
                f"for flag={getattr(ds, 'flag', 'unknown')}"
            )
        if seq_y_cov.shape[1] != need_cov_dim:
            raise ValueError(
                f"[std table] seq_y_cov dim={seq_y_cov.shape[1]} != len(cov_cols)={need_cov_dim} "
                f"for flag={getattr(ds, 'flag', 'unknown')}"
            )

        anchor_idx = i + seq_len
        anchor_time = key_grid.iloc[i * pred_len]["anchor_time"]
        for h in range(pred_len):
            t = key_grid.iloc[i * pred_len + h]["time"]
            row = {
                "anchor_time": anchor_time,
                "time": t,
                "h": np.int32(h + 1),
                "y": float(seq_y[h]),
            }
            row.update({cov_cols[k]: float(seq_y_cov[h, k]) for k in range(need_cov_dim)})
            rows.append(row)

    return pd.DataFrame(rows, columns=cols)


def attach_tsfm_preds_std(
    base: pd.DataFrame,
    *,
    tsfm_pred_table: pd.DataFrame | None,
    tsfm_models: str,
    tsfm_vars: List[str],
    use_point: bool = True,
    use_uncertainty: bool = True,
) -> tuple[pd.DataFrame, List[str]]:
    if tsfm_pred_table is None or len(tsfm_pred_table) == 0:
        return base, []

    models = [m.strip().lower() for m in tsfm_models.split(",") if m.strip()]

    pt = tsfm_pred_table.copy()
    pt["anchor_time"] = pd.to_datetime(pt["anchor_time"]).dt.tz_localize(None)
    pt["time"] = pd.to_datetime(pt["time"]).dt.tz_localize(None)
    pt["h"] = pt["h"].astype(np.int32)

    feature_cols: List[str] = []
    for model_name in models:
        for var_name in tsfm_vars:
            base_col = f"pred_{model_name}_{var_name}"
            unc_col = f"{base_col}_uncertainty"
            if use_point and base_col in pt.columns:
                feature_cols.append(base_col)
            if use_uncertainty and unc_col in pt.columns:
                feature_cols.append(unc_col)
    feature_cols = list(dict.fromkeys(feature_cols))

    if not feature_cols:
        return base, []

    merged = base.merge(
        pt[["anchor_time", "time", "h"] + feature_cols],
        on=["anchor_time", "time", "h"],
        how="left",
    )
    return merged, feature_cols


def _conf_col(unc_col: str) -> str:
    return unc_col.replace("_uncertainty", "_conf")


def add_confidence_columns(
    df: pd.DataFrame,
    *,
    pred_cols: List[str],
    sigma_map: Dict[str, float] | None = None,
) -> tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """为每个 uncertainty 列添加 C = exp(-U/(σ_U+ε))。

    sigma_map=None 时用本表（应为训练集）计算 σ_U=std(U) 并返回；否则用给定 σ_U。
    返回 (df, 新增 conf 列名, {unc_col: sigma_u})。
    """
    eps = 1e-6
    unc_cols = [c for c in pred_cols if c.endswith("_uncertainty")]
    if not unc_cols:
        return df.copy(), [], (sigma_map or {})

    dff = df.copy()
    new_cols: List[str] = []
    out_map: Dict[str, float] = {}
    for uc in unc_cols:
        u = pd.to_numeric(dff[uc], errors="coerce")
        if sigma_map is None:
            s = float(u.std(skipna=True))
            if not np.isfinite(s) or s <= 0:
                s = 1.0
        else:
            s = float(sigma_map.get(uc, 1.0))
        out_map[uc] = s
        dff[_conf_col(uc)] = np.exp(-u / (s + eps))
        new_cols.append(_conf_col(uc))
    return dff, new_cols, out_map


def build_ar_features(
    df: pd.DataFrame,
    *,
    y_col: str,
    lags=(1, 24, 168),
    roll=(24,),
) -> tuple[pd.DataFrame, List[str]]:
    """因果 AR 旁路特征（只用目标时刻之前的 y）。

    返回 (以 'time' 为主键的 DataFrame, 特征名列表)。
    lags 单位 = 数据频率步长（小时级：1=1h, 24=1d, 168=1w）。shift(正) 只回看过去，
    rolling 作用于 shift(1) 后，因此 roll_{stat}_{w}h 只统计 [t-w, t-1] 的过去值，无未来泄漏。
    """
    dff = df.sort_values(df.columns[0]).reset_index(drop=True).copy()
    time_s = pd.to_datetime(dff[df.columns[0]])
    y = pd.to_numeric(dff[y_col], errors="coerce")

    out = pd.DataFrame({"time": time_s})
    feats: List[str] = []
    for lag in lags:
        c = f"lag_{lag}h"
        out[c] = y.shift(lag)
        feats.append(c)
    for w in roll:
        r = y.shift(1).rolling(w, min_periods=1)
        for stat in ("mean", "std", "max", "min"):
            c = f"roll_{stat}_{w}h"
            out[c] = getattr(r, stat)()
            feats.append(c)
    return out, feats


def build_ar_features_for_anchors(
    base: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    time_col: str,
    y_col: str,
    lags=(1, 24, 168),
    roll=(24,),
) -> tuple[pd.DataFrame, List[str]]:
    """Build AR features at each forecast origin, not at each future target.

    Target-time shifting leaks realized intermediate targets when horizon > 1.
    Every row sharing an anchor therefore receives features from history ending
    immediately before that anchor.
    """
    raw_times = pd.to_datetime(raw[time_col]).dt.tz_localize(None)
    y = pd.to_numeric(raw[y_col], errors="coerce").to_numpy(dtype=float)
    lookup = pd.Series(y, index=raw_times).groupby(level=0).last()
    anchors = pd.DatetimeIndex(pd.to_datetime(base["anchor_time"]).dt.tz_localize(None).drop_duplicates())
    out = pd.DataFrame({"anchor_time": anchors})
    feats: List[str] = []
    for lag in lags:
        name = f"lag_{lag}h"
        out[name] = lookup.reindex(anchors - pd.Timedelta(hours=int(lag))).to_numpy()
        feats.append(name)
    for window in roll:
        hist_index = [anchors - pd.Timedelta(hours=i) for i in range(1, int(window) + 1)]
        hist = np.column_stack([lookup.reindex(idx).to_numpy() for idx in hist_index])
        for stat in ("mean", "std", "max", "min"):
            name = f"roll_{stat}_{window}h"
            with np.errstate(invalid="ignore", divide="ignore"):
                out[name] = getattr(np, stat)(hist, axis=1)
            feats.append(name)
    return out, feats


def table_to_xy_std(
    df: pd.DataFrame,
    *,
    cov_cols: List[str],
    pred_cols: List[str],
) -> tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    feature_cols = list(cov_cols) + list(pred_cols)

    if len(df) == 0:
        return (
            np.zeros((0, len(feature_cols)), np.float32),
            np.zeros((0,), np.float32),
            [],
            feature_cols,
        )

    dff = df.copy()
    before = len(dff)
    dff = dff.replace([np.inf, -np.inf], np.nan)

    for c in feature_cols:
        if c not in dff.columns:
            dff[c] = 0.0
            print(f"[std xy] missing feature column -> filled with 0: {c}")
            continue
        ser = pd.to_numeric(dff[c], errors="coerce")
        if ser.notna().any():
            # 因果填充：只向前（用过去值），不向后（避免用未来值补前段）
            dff[c] = ser.ffill().fillna(0.0)
        else:
            dff[c] = 0.0
            print(f"[std xy] feature column all-missing -> filled with 0: {c}")

    # 目标缺失 -> 删除对应标签，不用未来标签插值补出
    dff["y"] = pd.to_numeric(dff["y"], errors="coerce")
    dff = dff.dropna(subset=["y"])
    dropped = before - len(dff)
    if dropped:
        print(f"[std xy] dropped rows with NaN/Inf after repair: {dropped}/{before}")
    X = dff[feature_cols].to_numpy(np.float32)
    y = dff["y"].to_numpy(np.float32)
    times = dff["time"].astype(str).tolist()
    print("[std xy] X shape:", X.shape, "y shape:", y.shape)
    return X, y, times, feature_cols


def build_features(
    *,
    args,
    pred_table=None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
    int,
    Dict[str, Any],
]:
    split_cfg = _load_single_json(args.data_split_path)["data_split"]
    cols_cfg = list(_load_single_json(args.regress_cols_path)["target_columns"])
    tsfm_vars = list(_load_single_json(args.tsfm_cols_path)["target_columns"])

    time_col = split_cfg.get("time_col", "time")
    cov_cols, y_col = cols_cfg[:-1], cols_cfg[-1]
    if y_col not in tsfm_vars:
        tsfm_vars.append(y_col)
    tsfm_vars = list(dict.fromkeys(tsfm_vars))

    if bool(getattr(args, "is_std", False)):
        ds_tr, ds_va, ds_te, _, _ = _build_epf_benchmark_datasets(
            args,
            split_cfg=split_cfg,
            cols_cfg=cols_cfg,
        )

        tr_base = build_std_cov_target_table(ds_tr, cov_cols=cov_cols)
        va_base = build_std_cov_target_table(ds_va, cov_cols=cov_cols)
        te_base = build_std_cov_target_table(ds_te, cov_cols=cov_cols)

        # AR 旁路特征（因果，从原始整表按过去值构造；按 time 对齐回 std 表）
        ar_cols: List[str] = []
        if bool(getattr(args, "use_ar_features", False)):
            raw = read_table(args.data_path)
            lags = tuple(int(x) for x in str(getattr(args, "ar_lags", "1,24,168")).split(",") if x.strip())
            roll = tuple(int(x) for x in str(getattr(args, "ar_roll", "24")).split(",") if x.strip())
            all_base = pd.concat([tr_base, va_base, te_base], ignore_index=True)
            ar_df, ar_cols = build_ar_features_for_anchors(
                all_base, raw, time_col=time_col, y_col=y_col, lags=lags, roll=roll
            )
            tr_base = tr_base.merge(ar_df, on="anchor_time", how="left")
            va_base = va_base.merge(ar_df, on="anchor_time", how="left")
            te_base = te_base.merge(ar_df, on="anchor_time", how="left")

        use_point = bool(getattr(args, "use_tsfm_point", True))
        use_uncertainty = bool(getattr(args, "use_tsfm_uncertainty", True))
        use_confidence = bool(getattr(args, "use_tsfm_confidence", False))
        tr_df, pred_cols = attach_tsfm_preds_std(
            tr_base,
            tsfm_pred_table=pred_table,
            tsfm_models=args.tsfm_models,
            tsfm_vars=tsfm_vars,
            use_point=use_point,
            use_uncertainty=(use_uncertainty or use_confidence),
        )
        va_df, _ = attach_tsfm_preds_std(
            va_base,
            tsfm_pred_table=pred_table,
            tsfm_models=args.tsfm_models,
            tsfm_vars=tsfm_vars,
            use_point=use_point,
            use_uncertainty=(use_uncertainty or use_confidence),
        )
        te_df, _ = attach_tsfm_preds_std(
            te_base,
            tsfm_pred_table=pred_table,
            tsfm_models=args.tsfm_models,
            tsfm_vars=tsfm_vars,
            use_point=use_point,
            use_uncertainty=(use_uncertainty or use_confidence),
        )

        # 置信度 C = exp(-U/σ_U)，σ_U 仅用训练集估计
        if use_confidence:
            tr_df, conf_cols, sigma_map = add_confidence_columns(tr_df, pred_cols=pred_cols)
            va_df, _, _ = add_confidence_columns(va_df, pred_cols=pred_cols, sigma_map=sigma_map)
            te_df, _, _ = add_confidence_columns(te_df, pred_cols=pred_cols, sigma_map=sigma_map)
            if not use_uncertainty:
                pred_cols = [c for c in pred_cols if not c.endswith("_uncertainty")]
            pred_cols = pred_cols + conf_cols

        all_cov = list(cov_cols) + ["h"] + ar_cols
        X_tr, y_tr, tr_time, feature_names = table_to_xy_std(tr_df, cov_cols=all_cov, pred_cols=pred_cols)
        X_va, y_va, va_time, _ = table_to_xy_std(va_df, cov_cols=all_cov, pred_cols=pred_cols)
        X_te, y_te, te_time, _ = table_to_xy_std(te_df, cov_cols=all_cov, pred_cols=pred_cols)

        N_tr = len(ds_tr)
        N_va = len(ds_va)
        N_te = len(ds_te)

        if X_tr.shape[1] != X_va.shape[1]:
            all_feature_names = list(dict.fromkeys(list(feature_names)))
            def _pad_to(X, cols):
                if X.shape[1] == len(all_feature_names):
                    return X
                out = np.zeros((X.shape[0], len(all_feature_names)), dtype=np.float32)
                idx = {c:i for i,c in enumerate(cols)}
                for j,c in enumerate(all_feature_names):
                    if c in idx:
                        out[:, j] = X[:, idx[c]]
                return out
            X_tr = _pad_to(X_tr, feature_names)
            X_va = _pad_to(X_va, feature_names)
            X_te = _pad_to(X_te, feature_names)

        # 训练/验证不合并：默认 X_tr 仅含训练集，X_va 作为干净的早停集。
        # 仅当显式开启 std_merge_valid_to_train 时才把 valid 并入 train
        # （此时 valid 已在训练数据中，早停会失效，调用方应自行关闭早停）。
        if bool(getattr(args, "std_merge_valid_to_train", False)):
            X_tr = np.concatenate([X_tr, X_va], axis=0)
            y_tr = np.concatenate([y_tr, y_va], axis=0)
            N_tr += N_va
            tr_time = tr_time + va_time
            print(
                f"[feature_select][standard] merged valid into train (std_merge_valid_to_train=1): "
                f"N_tr={N_tr}, X_tr.shape={X_tr.shape}, y_tr.shape={y_tr.shape}"
            )

        print(f"[feature_select][standard] windows: train={N_tr}, valid={N_va}, test={N_te}")
        print(f"[feature_select][standard] #features: {len(feature_names)}")
        if pred_cols:
            print(f"[feature_select][standard] TSFM features: {len(pred_cols)}")

        meta = {
            "L": int(args.pred_len),
            "N_tr": N_tr,
            "N_va": N_va,
            "N_te": N_te,
            "feature_names": feature_names,
            "tr_time": tr_time,
            "va_time": va_time,
            "te_time": te_time,
            "target_mean": float(np.asarray(ds_tr.mean_target, dtype=float).reshape(-1)[0]) if getattr(ds_tr, "mean_target", None) is not None else None,
            "target_std": float(np.asarray(ds_tr.std_target, dtype=float).reshape(-1)[0]) if getattr(ds_tr, "std_target", None) is not None else None,
            "target_scaled": bool(getattr(args, "scale", False)),
        }
        return X_tr, y_tr, N_tr, X_va, y_va, N_va, X_te, y_te, N_te, meta

    df = read_table(args.data_path)
    df, tsfm_cols = _attach_tsfm_features(
        df,
        time_col=time_col,
        y_col=y_col,
        pred_table=pred_table,
        tsfm_models=args.tsfm_models,
        tsfm_vars=tsfm_vars,
    )

    feature_names = cov_cols + tsfm_cols
    dtr = slice_df(df, time_col, split_cfg["train_start"], split_cfg["train_end"])
    dva = slice_df(df, time_col, split_cfg["valid_start"], split_cfg["valid_end"])
    dte = slice_df(df, time_col, split_cfg["test_start"], split_cfg["test_end"])

    X_tr, y_tr, N_tr, tr_time = make_xy(
        dtr,
        time_col=time_col,
        cov_cols=feature_names,
        y_col=y_col,
        target_hour=args.target_hour,
    )
    X_va, y_va, N_va, va_time = make_xy(
        dva,
        time_col=time_col,
        cov_cols=feature_names,
        y_col=y_col,
        target_hour=args.target_hour,
    )
    X_te, y_te, N_te, te_time = make_xy(
        dte,
        time_col=time_col,
        cov_cols=feature_names,
        y_col=y_col,
        target_hour=args.target_hour,
    )

    print(f"[feature_select][shanxi] days: train={N_tr}, valid={N_va}, test={N_te}")
    print(f"[feature_select][shanxi] #features: {len(feature_names)}")
    if tsfm_cols:
        print(f"[feature_select][shanxi] TSFM features: {len(tsfm_cols)}")

    meta = {
        "L": POINTS_PER_DAY,
        "N_tr": N_tr,
        "N_va": N_va,
        "N_te": N_te,
        "feature_names": feature_names,
        "tr_time": tr_time,
        "va_time": va_time,
        "te_time": te_time,
    }
    return X_tr, y_tr, N_tr, X_va, y_va, N_va, X_te, y_te, N_te, meta
