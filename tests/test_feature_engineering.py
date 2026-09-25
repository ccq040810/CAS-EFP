import unittest
import sys
import types

import numpy as np
import pandas as pd
import lightgbm as lgb

# Calendar generation is outside the feature logic covered here.
sys.modules.setdefault("chinese_calendar", types.SimpleNamespace(is_workday=lambda _: True))

from exp.pipeline.feature_select import (
    add_confidence_columns,
    build_ar_features_for_anchors,
    table_to_xy_std,
)
from exp.pipeline.regressor import asymmetric_l2_objective_weighted


class FeatureEngineeringTests(unittest.TestCase):
    def test_ar_features_are_origin_keyed_and_do_not_use_future_targets(self):
        times = pd.date_range("2024-01-01", periods=220, freq="h")
        raw = pd.DataFrame({"time": times, "y": np.arange(len(times), dtype=float)})
        anchor = times[200]
        base = pd.DataFrame({
            "anchor_time": [anchor, anchor],
            "time": [anchor, anchor + pd.Timedelta(hours=1)],
            "h": [1, 2],
        })
        features, _ = build_ar_features_for_anchors(
            base, raw, time_col="time", y_col="y", lags=(1, 24), roll=(3,)
        )
        self.assertEqual(float(features.loc[0, "lag_1h"]), 199.0)
        self.assertEqual(float(features.loc[0, "lag_24h"]), 176.0)
        changed = raw.copy()
        changed.loc[changed["time"] >= anchor, "y"] = -9999.0
        features_changed, _ = build_ar_features_for_anchors(
            base, changed, time_col="time", y_col="y", lags=(1, 24), roll=(3,)
        )
        np.testing.assert_allclose(features.iloc[:, 1:].to_numpy(), features_changed.iloc[:, 1:].to_numpy())

    def test_confidence_uses_train_scale(self):
        frame = pd.DataFrame({"pred_m_y_uncertainty": [1.0, 2.0]})
        result, cols, scales = add_confidence_columns(frame, pred_cols=["pred_m_y_uncertainty"])
        self.assertEqual(cols, ["pred_m_y_conf"])
        self.assertAlmostEqual(scales["pred_m_y_uncertainty"], np.std([1.0, 2.0], ddof=1))
        self.assertTrue(np.all((result[cols[0]] > 0) & (result[cols[0]] <= 1)))

    def test_std_xy_does_not_forward_fill_missing_tsfm_forecasts(self):
        frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="h"),
            "anchor_time": pd.date_range("2023-12-31 23:00", periods=3, freq="h"),
            "h": [1, 1, 1],
            "load": [10.0, np.nan, 12.0],
            "y": [1.0, 2.0, 3.0],
            "pred_chronos_y": [100.0, np.nan, 300.0],
        })
        X, y, _, names = table_to_xy_std(
            frame, cov_cols=["load", "h"], pred_cols=["pred_chronos_y"]
        )
        self.assertEqual(names, ["load", "h", "pred_chronos_y"])
        self.assertEqual(y.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(X[1, 0], 10.0)  # causal covariate forward-fill remains
        self.assertTrue(np.isnan(X[1, 2]))  # forecast gap is not stale-filled

    def test_std_xy_drops_missing_labels_but_keeps_covariate_rows(self):
        frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="h"),
            "anchor_time": pd.date_range("2023-12-31 23:00", periods=3, freq="h"),
            "h": [1, 1, 1],
            "load": [10.0, 11.0, 12.0],
            "y": [1.0, np.nan, 3.0],
        })
        X, y, times, _ = table_to_xy_std(frame, cov_cols=["load", "h"], pred_cols=[])
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.tolist(), [1.0, 3.0])
        self.assertEqual(len(times), len(y))

    def test_confidence_weight_only_amplifies_penalized_asymmetric_direction(self):
        dataset = lgb.Dataset(
            np.zeros((2, 1)), label=np.array([2.0, 0.0]), weight=np.array([3.0, 3.0])
        )
        grad, hess = asymmetric_l2_objective_weighted(alpha=2.0, direction=1)(
            np.array([1.0, 1.0]), dataset
        )
        np.testing.assert_allclose(grad, [-6.0, 1.0])
        np.testing.assert_allclose(hess, [6.0, 1.0])


if __name__ == "__main__":
    unittest.main()
