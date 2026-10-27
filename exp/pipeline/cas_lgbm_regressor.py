"""LightGBM on exactly the same prepared arrays as the MLP transfer experiment."""
import numpy as np
import lightgbm as lgb
from .regressor import asymmetric_l2_objective_weighted, asymmetric_rmse


class CASLGBMRegressor:
    def __init__(self, *, seed, threads, alpha, direction, asymmetric, selection,
                 rounds=2000, patience=100):
        self.params = dict(objective=asymmetric_l2_objective_weighted(alpha, direction)
                           if asymmetric else "regression", metric="None", learning_rate=.05,
                           num_leaves=63, min_data_in_leaf=40, seed=seed, num_threads=threads,
                           deterministic=True, force_col_wise=True, verbosity=-1)
        self.feval = asymmetric_rmse(alpha, direction) if asymmetric and selection == "legacy" else self.rmse
        self.rounds, self.patience = rounds, patience

    @staticmethod
    def rmse(pred, dataset):
        return "rmse", float(np.sqrt(np.mean((pred-dataset.get_label())**2))), False

    def fit(self, X, y, X_valid, y_valid, sample_weight=None):
        train = lgb.Dataset(X, label=y, weight=sample_weight)
        valid = lgb.Dataset(X_valid, label=y_valid, reference=train)
        history = {}
        self.model = lgb.train(self.params, train, num_boost_round=self.rounds,
                               valid_sets=[valid], valid_names=["valid"], feval=self.feval,
                               callbacks=[lgb.early_stopping(self.patience, verbose=False),
                                          lgb.record_evaluation(history)])
        self.best_iteration = self.model.best_iteration
        scores = next(iter(history["valid"].values()))
        self.history = [dict(iteration=i+1, valid_selection=v) for i,v in enumerate(scores)]
        return self

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.best_iteration)

    def save(self, path):
        self.model.save_model(str(path))
