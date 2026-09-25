from __future__ import annotations

from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge


def asymmetric_l2_objective(alpha: float = 2.0, direction: int = 1):
    alpha = float(alpha)
    direction = 1 if direction >= 0 else -1

    def _objective(y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_true = dataset.get_label()
        residual = y_true - y_pred
        grad = y_pred - y_true
        hess = np.ones_like(grad, dtype=np.float64)

        # direction=+1：对残差>0（欠预测）加大惩罚；direction=-1：对残差<0（过预测）加大惩罚
        mask = (residual * direction) > 0
        grad = grad.astype(np.float64, copy=False)
        hess = hess.astype(np.float64, copy=False)
        grad[mask] *= alpha
        hess[mask] *= alpha
        return grad, hess

    return _objective


def asymmetric_l2_objective_weighted(alpha: float = 2.0, direction: int = 1):
    """加权非对称 L2 目标：在固定非对称缩放的梯度/Hessian 上再乘以样本权重。

    LightGBM 对自定义目标不会自动施加 sample weight，因此这里显式读
    dataset.get_weight() 并一致地缩放梯度与 Hessian（与 lgb.Dataset(weight=...) 配套）。
    """
    alpha = float(alpha)
    direction = 1 if direction >= 0 else -1

    def _objective(y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_true = dataset.get_label()
        w = dataset.get_weight()
        residual = y_true - y_pred
        grad = y_pred - y_true
        hess = np.ones_like(grad, dtype=np.float64)

        mask = (residual * direction) > 0
        grad = grad.astype(np.float64, copy=False)
        hess = hess.astype(np.float64, copy=False)
        grad[mask] *= alpha
        hess[mask] *= alpha

        if w is not None and len(w):
            w = np.asarray(w, dtype=np.float64)
            # Confidence strengthens only the predeclared penalized direction.
            grad[mask] *= w[mask]
            hess[mask] *= w[mask]
        return grad, hess

    return _objective


def asymmetric_rmse(alpha: float = 2.0, direction: int = 1):
    alpha = float(alpha)
    direction = 1 if direction >= 0 else -1

    def _feval(y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple[str, float, bool]:
        y_true = dataset.get_label()
        residual = y_true - y_pred
        err = np.asarray(y_pred - y_true, dtype=np.float64)
        err = np.where((residual * direction) > 0, err * alpha, err)
        return "asymmetric_rmse", float(np.sqrt(np.mean(err ** 2))), False

    return _feval


def asymmetric_rmse_weighted(alpha: float = 2.0, direction: int = 1):
    """带权重的非对称 RMSE（监控/早停用）。"""
    alpha = float(alpha)
    direction = 1 if direction >= 0 else -1

    def _feval(y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple[str, float, bool]:
        y_true = dataset.get_label()
        w = dataset.get_weight()
        residual = y_true - y_pred
        err = np.asarray(y_pred - y_true, dtype=np.float64)
        err = np.where((residual * direction) > 0, err * alpha, err)
        err2 = err ** 2
        if w is not None and len(w):
            w = np.asarray(w, dtype=np.float64)
            penalized = (residual * direction) > 0
            metric_weight = np.where(penalized, w, 1.0)
            return "asymmetric_rmse", float(np.sqrt(np.sum(metric_weight * err2) / np.sum(metric_weight))), False
        return "asymmetric_rmse", float(np.sqrt(np.mean(err2))), False

    return _feval


def diagnose_error_direction(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """训练侧误差方向诊断（仅用于校准，不构成结论）。

    返回 +1 或 -1：
        bias = median(y_true - y_pred)
        bias > 0  → 模型系统性欠预测 → 返回 +1（对欠预测加大惩罚）
        bias < 0  → 模型系统性过预测 → 返回 -1（对过预测加大惩罚）
    该诊断只应在训练/校准侧（非最终测试集）进行，方向与 α/λ 的选择不得来自测试集。
    """
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    bias = float(np.nanmedian(residual))
    return 1 if bias >= 0 else -1


def lgbm_regression(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    args,
    sample_weight: np.ndarray | None = None,
    direction: int = 1,
) -> lgb.Booster:
    """训练 LightGBM 回归器（含固定非对称损失基线与可选样本加权管线）。

    sample_weight: 可选训练权重（长度 = len(y_tr)）。不为 None 时：
        - 通过 lgb.Dataset(weight=sample_weight) 传入；
        - 非对称情形用 asymmetric_l2_objective_weighted 在目标内显式按权重缩放
          梯度/Hessian（LightGBM 对自定义目标不自动施加 weight）；
        - 非非对称情形直接用内置 "regression"（LightGBM 自动施加 weight）。
    权重的构造（如 w = 1 + λ(1-C)）由调用方决定，此处不内置任何固定公式。
    direction: 非对称惩罚方向（+1=欠预测、-1=过预测）。调用方负责用训练/校准侧
        诊断决定，禁止用最终测试集选择。
    """
    dtr = lgb.Dataset(X_tr, label=y_tr, **({"weight": sample_weight} if sample_weight is not None else {}))
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

    alpha = float(getattr(args, "asymmetric_alpha", 2.0))
    use_asym = bool(getattr(args, "use_asymmetric_loss", False))
    weighted = sample_weight is not None

    if use_asym:
        objective = (
            asymmetric_l2_objective_weighted(alpha=alpha, direction=direction) if weighted
            else asymmetric_l2_objective(alpha=alpha, direction=direction)
        )
        feval = (
            asymmetric_rmse_weighted(alpha=alpha, direction=direction) if weighted
            else asymmetric_rmse(alpha=alpha, direction=direction)
        )
        metric = "None"
    else:
        objective = "regression"
        feval = None
        metric = "rmse"

    params = dict(
        objective=objective,
        metric=metric,
        boosting_type="gbdt",
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_data_in_leaf=args.min_data_in_leaf,
        min_sum_hessian_in_leaf=args.min_sum_hessian_in_leaf,
        min_gain_to_split=args.min_gain_to_split,
        linear_tree=args.linear_tree,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l1=args.reg_alpha,
        lambda_l2=args.reg_lambda,
        verbose=-1,
        seed=args.seed,
        num_threads=args.n_jobs,
    )

    return lgb.train(
        params=params,
        train_set=dtr,
        valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        num_boost_round=args.num_boost_round,
        feval=feval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=args.log_period),
        ],
    )


def _make_linear_model(
    *,
    method: str,
    alpha: float,
    l1_ratio: float,
    fit_intercept: bool,
    max_iter: int,
    tol: float,
    random_state: int,
):
    name = method.strip().lower()
    if name == "ridge":
        return Ridge(alpha=float(alpha), fit_intercept=fit_intercept, random_state=random_state)
    if name == "lasso":
        return Lasso(
            alpha=float(alpha),
            fit_intercept=fit_intercept,
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=random_state,
        )
    if name == "elasticnet":
        return ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=fit_intercept,
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=random_state,
        )
    raise ValueError(f"Unknown method={method}, expected ridge|lasso|elasticnet")


def Linear_regression(
    *,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    method: str = "ridge",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    fit_intercept: bool = True,
    max_iter: int = 20000,
    tol: float = 1e-4,
    random_state: int = 0,
):
    if not np.isfinite(X_tr).all() or not np.isfinite(y_tr).all():
        raise ValueError("[Linear_regression] train contains NaN/Inf.")

    model = _make_linear_model(
        method=method,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    model.fit(X_tr, y_tr)
    return model
