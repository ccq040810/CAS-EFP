from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from .pipeline.tsfm_infer import get_tsfm_target_col, load_compare_series_from_cache, run_tsfm_rollout
from .pipeline.feature_select import build_features
from .pipeline.regressor import lgbm_regression, Linear_regression, diagnose_error_direction
from .pipeline.evaluator import evaluate
from .pipeline.eff_profile import profile_stage
from .pipeline.shap_explain import run_model_shap_explain
from .pipeline.shap_case import export_shap_casebook_low_high

class Exp_Pipeline:
    def __init__(self, args):
        self.args = args

    def _build_sample_weight(self, X_tr: np.ndarray, meta: dict) -> np.ndarray | None:
        """按配置构造训练样本权重（默认 uniform → None，即不使用权重管线）。

        confidence 模式：w = 1 + λ(1-C)，C 为 *_conf 置信度特征列；仅作为可选实验，
        不作为默认/真理式设定。
        """
        mode = str(getattr(self.args, "sample_weight_mode", "uniform")).strip().lower()
        if mode == "uniform":
            return None
        feature_names = list(meta.get("feature_names", []))
        if mode == "confidence":
            conf_col = next((c for c in feature_names if str(c).endswith("_conf")), None)
            if conf_col is None:
                raise ValueError(
                    "[sample_weight] confidence 模式需要 *_conf 特征列（需开启 use_tsfm_confidence）"
                )
            idx = feature_names.index(conf_col)
            C = np.asarray(X_tr[:, idx], dtype=np.float64)
            lam = float(getattr(self.args, "sample_weight_lambda", 1.0))
            w = 1.0 + lam * (1.0 - np.clip(C, 0.0, 1.0))
            print(f"[sample_weight] confidence mode, lambda={lam}, col={conf_col}, w in [{w.min():.3f}, {w.max():.3f}]")
            return w
        raise ValueError(f"[sample_weight] unknown sample_weight_mode={mode}")

    def _resolve_asymmetric_direction(self, X_tr: np.ndarray, y_tr: np.ndarray, meta: dict) -> int:
        """Diagnose the penalty direction from training-period TSFM errors only."""
        use_asym = bool(getattr(self.args, "use_asymmetric_loss", False))
        mode = str(getattr(self.args, "asymmetric_direction", "1")).strip().lower()
        if not use_asym or mode in ("+1", "1", "positive", "under"):
            return 1
        if mode in ("-1", "negative", "over"):
            return -1
        if mode == "auto":
            feature_names = list(meta.get("feature_names", []))
            point_col = next(
                (
                    c for c in feature_names
                    if str(c).startswith("pred_")
                    and not str(c).endswith("_uncertainty")
                    and not str(c).endswith("_conf")
                ),
                None,
            )
            if point_col is None:
                print("[direction] auto 模式未找到 TSFM 点预测列，回退 direction=+1")
                return 1
            idx = feature_names.index(point_col)
            direction = diagnose_error_direction(y_tr, X_tr[:, idx])
            print(f"[direction] auto diagnosed on training period ({point_col}): direction={direction}")
            return direction
        raise ValueError(f"[direction] unknown asymmetric_direction={mode}")

    @staticmethod
    def _write_confidence_diagnostics(
        *, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
        feature_names: list[str], save_dir: str | Path, split: str,
    ) -> None:
        """Test-only descriptive analysis; never feeds model selection or training."""
        names = list(feature_names)
        conf_col = next((n for n in names if n.endswith("_conf")), None)
        if conf_col is None:
            return
        pred_col = conf_col.removesuffix("_conf")
        if pred_col not in names:
            return
        conf = np.asarray(X[:, names.index(conf_col)], dtype=float)
        raw_point = np.asarray(X[:, names.index(pred_col)], dtype=float)
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        valid = np.isfinite(conf) & np.isfinite(raw_point) & np.isfinite(yt) & np.isfinite(yp)
        if valid.sum() < 10 or np.unique(conf[valid]).size < 2:
            return
        frame = pd.DataFrame({
            "confidence": conf[valid],
            "tsfm_abs_error": np.abs(raw_point[valid] - yt[valid]),
            "tree_abs_error": np.abs(yp[valid] - yt[valid]),
            "tree_abs_deviation_from_tsfm": np.abs(yp[valid] - raw_point[valid]),
        })
        n_bins = min(10, int(frame["confidence"].nunique()))
        frame["confidence_bin"] = pd.qcut(frame["confidence"], q=n_bins, labels=False, duplicates="drop")
        result = frame.groupby("confidence_bin", observed=True).agg(
            n=("confidence", "size"),
            confidence_min=("confidence", "min"),
            confidence_max=("confidence", "max"),
            confidence_mean=("confidence", "mean"),
            tsfm_mae=("tsfm_abs_error", "mean"),
            tree_mae=("tree_abs_error", "mean"),
            tree_deviation_from_tsfm=("tree_abs_deviation_from_tsfm", "mean"),
        ).reset_index()
        result.to_csv(Path(save_dir) / f"confidence_diagnostics_{split}.csv", index=False)

    @staticmethod
    def _write_train_direction_diagnostics(
        *, X: np.ndarray, y_true: np.ndarray, feature_names: list[str],
        time_points: list[str], save_dir: str | Path,
    ) -> None:
        """Export train-only direction stability; this is not a test metric."""
        point_col = next((n for n in feature_names if n.startswith("pred_") and not n.endswith(("_conf", "_uncertainty"))), None)
        if point_col is None or len(time_points) != len(y_true):
            return
        residual = np.asarray(y_true, dtype=float) - np.asarray(X[:, feature_names.index(point_col)], dtype=float)
        frame = pd.DataFrame({"time": pd.to_datetime(time_points, errors="coerce"), "residual_y_minus_tsfm": residual})
        frame = frame.dropna()
        if frame.empty:
            return
        frame["period"] = frame["time"].dt.to_period("Q").astype(str)
        report = frame.groupby("period", as_index=False).agg(
            n=("residual_y_minus_tsfm", "size"),
            median_residual=("residual_y_minus_tsfm", "median"),
            underprediction_rate=("residual_y_minus_tsfm", lambda x: float((x > 0).mean())),
            overprediction_rate=("residual_y_minus_tsfm", lambda x: float((x < 0).mean())),
        )
        report.to_csv(Path(save_dir) / "train_direction_diagnostics.csv", index=False)

    def run(self, setting=None):
        os.makedirs(self.args.save_dir, exist_ok=True)
        eff_csv = Path(self.args.save_dir).parent / "metrics_efficiency_cache.csv"

        with profile_stage(stage="tsfm_rollout", args=self.args, eff_csv=eff_csv):
            pred_table = run_tsfm_rollout(args=self.args)

        with profile_stage(stage="build_features", args=self.args, eff_csv=eff_csv):
            X_tr, y_tr, N_tr, X_va, y_va, N_va, X_te, y_te, N_te, meta = build_features(
                args=self.args,
                pred_table=pred_table,
            )

        self._write_train_direction_diagnostics(
            X=X_tr, y_true=y_tr, feature_names=meta["feature_names"],
            time_points=meta.get("tr_time", []), save_dir=self.args.save_dir,
        )

        regression_model = str(getattr(self.args, "regression_model", "lgbm")).strip().lower()
        is_std = bool(getattr(self.args, "is_std", False))
        std_step = getattr(self.args, "std_freq", None) if is_std else None
        plot_target_mean = meta.get("target_mean") if is_std and bool(meta.get("target_scaled", False)) else None
        plot_target_std = meta.get("target_std") if is_std and bool(meta.get("target_scaled", False)) else None
        compare_series = {}
        compare_series = load_compare_series_from_cache(
            args=self.args,
            time_points=meta["te_time"],
        )

        if regression_model == "lgbm+linear":
            run_modes = ["lgbm", "linear"]
        elif regression_model in {"lgbm", "linear"}:
            run_modes = [regression_model]
        else:
            raise ValueError(
                f"Unknown regression_model={regression_model}, expected one of: lgbm, linear, lgbm+linear"
            )

        summary_csv = os.path.join(os.path.dirname(self.args.save_dir), "metrics_all.csv")
        linear_method = str(getattr(self.args, "linear_method", "ridge")).strip().lower()

        for run_mode in run_modes:
            if len(run_modes) == 1:
                run_save_dir = self.args.save_dir
            elif run_mode == "lgbm":
                run_save_dir = str(Path(self.args.save_dir) / "lgbm")
            else:
                run_save_dir = str(Path(self.args.save_dir) / f"linear_{linear_method}")

            os.makedirs(run_save_dir, exist_ok=True)

            if run_mode == "lgbm":
                sample_weight = self._build_sample_weight(X_tr, meta)
                direction = self._resolve_asymmetric_direction(X_tr, y_tr, meta)
                with profile_stage(stage="train_regressor_lgbm", args=self.args, eff_csv=eff_csv):
                    model = lgbm_regression(
                        X_tr, y_tr, X_va, y_va, self.args,
                        sample_weight=sample_weight,
                        direction=direction,
                    )

                with profile_stage(stage="test_predict_eval_lgbm", args=self.args, eff_csv=eff_csv):
                    y_va_pred = model.predict(X_va, num_iteration=model.best_iteration)
                    self._write_confidence_diagnostics(
                        X=X_va, y_true=y_va, y_pred=y_va_pred,
                        feature_names=meta["feature_names"], save_dir=run_save_dir,
                        split="valid",
                    )
                    y_te_pred = model.predict(X_te, num_iteration=model.best_iteration)
                    self._write_confidence_diagnostics(
                        X=X_te, y_true=y_te, y_pred=y_te_pred,
                        feature_names=meta["feature_names"], save_dir=run_save_dir,
                        split="test",
                    )
                    test_metrics = evaluate(
                        y_point_pred=y_te_pred,
                        y_point_true=y_te,
                        N=N_te,
                        L=meta["L"],
                        split="test",
                        tag=self.args.tag,
                        save_dir=run_save_dir,
                        best_iteration=int(model.best_iteration),
                        time_points=meta["te_time"],
                        summary_csv=summary_csv,
                        model=model,
                        feature_names=meta["feature_names"],
                        topk=20,
                        compare_series=compare_series,
                        is_std=is_std,
                        std_step=std_step,
                        regression_model="lgbm",
                        summary_meta={"model_name": getattr(self.args, "model_name", "pipeline-lgbm")},
                        plot_target_mean=plot_target_mean,
                        plot_target_std=plot_target_std,
                        compare_series_is_raw=True,
                    )
                    print("[test][lgbm]", test_metrics)
            else:
                with profile_stage(stage=f"train_regressor_linear_{linear_method}", args=self.args, eff_csv=eff_csv):
                    model = Linear_regression(
                        X_tr=X_tr,
                        y_tr=y_tr,
                        method=linear_method,
                        alpha=float(getattr(self.args, "linear_alpha", 0.001)),
                        l1_ratio=float(getattr(self.args, "linear_l1_ratio", 0.5)),
                        fit_intercept=bool(getattr(self.args, "linear_fit_intercept", False)),
                        max_iter=int(getattr(self.args, "linear_max_iter", 20000)),
                        tol=float(getattr(self.args, "linear_tol", 1e-4)),
                        random_state=int(getattr(self.args, "seed", 0)),
                    )
                with profile_stage(stage=f"test_predict_eval_linear_{linear_method}", args=self.args, eff_csv=eff_csv):
                    if (not np.isfinite(X_te).all()) or (not np.isfinite(y_te).all()):
                        raise ValueError("[linear pipeline] test contains NaN/Inf.")
                    y_te_pred = model.predict(X_te)
                    y_tr_pred = model.predict(X_tr) if bool(getattr(self.args, "linear_plot_train", False)) else None
                    summary_meta = {
                        "alpha": float(getattr(self.args, "linear_alpha", 0.001)),
                        "fit_intercept": bool(getattr(self.args, "linear_fit_intercept", False)),
                    }
                    if linear_method == "elasticnet":
                        summary_meta["l1_ratio"] = float(getattr(self.args, "linear_l1_ratio", 0.5))

                    test_metrics = evaluate(
                        y_point_pred=y_te_pred,
                        y_point_true=y_te,
                        N=N_te,
                        L=meta["L"],
                        split="test",
                        tag=self.args.tag,
                        save_dir=run_save_dir,
                        best_iteration=None,
                        time_points=meta["te_time"],
                        summary_csv=summary_csv,
                        model=model,
                        feature_names=meta["feature_names"],
                        topk=int(getattr(self.args, "linear_topk_feat", 30)),
                        compare_series=compare_series,
                        is_std=is_std,
                        std_step=std_step,
                        regression_model="linear",
                        method=linear_method,
                        summary_meta=summary_meta,
                        plot_train=bool(getattr(self.args, "linear_plot_train", False)),
                        train_point_pred=y_tr_pred,
                        train_point_true=y_tr if y_tr_pred is not None else None,
                        train_time_points=meta.get("tr_time") if y_tr_pred is not None else None,
                        train_N=meta.get("N_tr") if y_tr_pred is not None else None,
                        plot_target_mean=plot_target_mean,
                        plot_target_std=plot_target_std,
                        compare_series_is_raw=True,
                    )
                    print(f"[test][{linear_method}]", test_metrics)

            if not getattr(self.args, "disable_shap", False):
                with profile_stage(stage=f"shap_explain_{run_mode}", args=self.args, eff_csv=eff_csv):
                    try:
                        run_model_shap_explain(
                            model=model,
                            X=X_te,
                            feature_names=meta.get("feature_names", []),
                            save_dir=run_save_dir,
                            split="test",
                            tag=getattr(self.args, "tag", "pipeline"),
                            max_samples=int(getattr(self.args, "shap_max_samples", 5000)),
                            seed=int(getattr(self.args, "seed", 2021)),
                            topk=int(getattr(self.args, "shap_topk", 10)),
                            n_dependence_plots=int(getattr(self.args, "shap_n_dependence", 2)),
                            n_waterfall_samples=int(getattr(self.args, "shap_n_waterfall", 0)),
                            n_decision_samples=int(getattr(self.args, "shap_n_decision", 0)),
                            enable_interaction=bool(getattr(self.args, "shap_enable_interaction", False)),
                            enable_heatmap=bool(getattr(self.args, "shap_enable_heatmap", False)),
                        )
                    except Exception as e:
                        print(f"[shap] skipped due to error: {type(e).__name__}: {e}")

                with profile_stage(stage=f"shap_casebook_{run_mode}", args=self.args, eff_csv=eff_csv):
                    try:
                        save_dir = Path(run_save_dir)
                        shap_dir = save_dir / "shap"
                        shap_values = np.load(shap_dir / "shap_values_test.npy")
                        with open(shap_dir / "shap_expected_value_test.json", "r", encoding="utf-8") as f:
                            shap_meta = json.load(f)
                        expected_value = float(shap_meta["expected_value"])

                        feature_names = list(meta.get("feature_names", []))
                        tsfm_feature_name = str(getattr(self.args, "casebook_tsfm_feature_name", "")).strip()
                        if not tsfm_feature_name:
                            target_col = get_tsfm_target_col(args=self.args)
                            models = [m.strip().lower() for m in str(getattr(self.args, "tsfm_models", "")).split(",") if m.strip()]
                            preferred_names = [f"pred_{m}_{target_col}" for m in models]
                            tsfm_feature_name = next(
                                (name for name in preferred_names if name in feature_names),
                                "",
                            )
                        if not tsfm_feature_name:
                            target_col = get_tsfm_target_col(args=self.args)
                            tsfm_feature_name = next(
                                (name for name in feature_names if name.startswith("pred_") and name.endswith(f"_{target_col}")),
                                "",
                            )
                        if not tsfm_feature_name:
                            tsfm_feature_name = next((name for name in feature_names if name.startswith("pred_")), "")
                        if not tsfm_feature_name:
                            raise ValueError("no TSFM feature found for casebook")

                        casebook_idx = shap_meta.get("sampled_idx")
                        if casebook_idx is not None:
                            casebook_idx = np.asarray(casebook_idx, dtype=int)
                            X_case = X_te[casebook_idx]
                            y_true_case = y_te[casebook_idx]
                            y_pred_case = y_te_pred[casebook_idx]
                            time_points_case = np.asarray(meta.get("te_time", []), dtype=object)[casebook_idx]
                            shap_values_case = shap_values
                        else:
                            X_case = X_te
                            y_true_case = y_te
                            y_pred_case = y_te_pred
                            time_points_case = meta.get("te_time", None)
                            shap_values_case = shap_values

                        casebook_group_size = int(meta["L"])
                        if shap_values_case.shape[0] % casebook_group_size != 0:
                            casebook_group_size = None

                        export_shap_casebook_low_high(
                            out_dir=shap_dir / "casebook_low_high_png",
                            shap_values=shap_values_case,
                            expected_value=expected_value,
                            X=X_case,
                            feature_names=feature_names,
                            y_true=y_true_case,
                            y_pred=y_pred_case,
                            L=int(meta["L"]),
                            tsfm_feature_name=tsfm_feature_name,
                            time_points=time_points_case,
                            max_cases=int(getattr(self.args, "casebook_max_cases", 50)),
                            topk_waterfall=int(getattr(self.args, "casebook_topk", 10)),
                            quantile_q=float(getattr(self.args, "casebook_q", 0.10)),
                            seed=int(getattr(self.args, "seed", 0)),
                            group_size=int(meta["L"]),
                        )
                    except Exception as e:
                        print(f"[casebook] skipped due to error: {type(e).__name__}: {e}")
