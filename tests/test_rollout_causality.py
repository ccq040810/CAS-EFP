"""Regression tests for Pandas ns/us timestamps and causal missing-history fill."""
import sys
import types
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.modules.setdefault("chinese_calendar",types.SimpleNamespace(is_workday=lambda _:True))
from exp.pipeline.tsfm_infer import _predict_multivar, _filter_valid_anchors


class Spy:
    def __init__(self):
        self.contexts = []

    def predict(self, context, forecast_horizon, **kwargs):
        self.contexts.append(context.detach().cpu().numpy().copy())
        point = context[..., -1:].expand(-1,-1,forecast_horizon)
        return dict(mean=point,uncertainty=torch.ones_like(point))


class RolloutCausalityTests(unittest.TestCase):
    def run_rollout(self, unit, mutate_future=False, leading_missing=False):
        index = pd.date_range("2024-01-01",periods=20,freq="h",tz="Asia/Shanghai").as_unit(unit)
        frame = pd.DataFrame({"price":np.arange(20,dtype=float)},index=index)
        frame.iloc[6,0] = np.nan
        if leading_missing:
            frame.iloc[0,0] = np.nan
        if mutate_future:
            frame.iloc[8:,0] = 999999.
        model = Spy()
        anchors = [index[4],index[8]]
        output = _predict_multivar(df=frame,mv_cols=["price"],anchors=anchors,
                                  anchor_offset={anchors[0]:0,anchors[1]:2},model=model,
                                  seq_len=4,horizon=2,batch_size=2,num_samples=100,device="cpu",
                                  use_future_covariates=False,causal_context_fill=True)
        return frame, model.contexts[0], output

    def test_both_time_resolutions_use_exact_origin_history(self):
        expected = np.array([[[0,1,2,3]],[[4,5,5,7]]],dtype=np.float32)
        for unit in ("ns","us"):
            frame, contexts, output = self.run_rollout(unit)
            np.testing.assert_array_equal(contexts,expected)
            np.testing.assert_array_equal(output["price"]["mean"],[3,3,7,7])
            self.assertTrue(np.isnan(frame.iloc[6,0]))  # original missing labels stay missing

    def test_changes_at_or_after_origin_cannot_change_any_context(self):
        for unit in ("ns","us"):
            _, before, _ = self.run_rollout(unit)
            _, after, _ = self.run_rollout(unit,mutate_future=True)
            np.testing.assert_array_equal(before,after)

    def test_no_future_backfill_for_leading_missing_history(self):
        with self.assertRaisesRegex(ValueError,"Incomplete causal history"):
            self.run_rollout("us",leading_missing=True)

    def test_anchor_validation_is_resolution_independent(self):
        for unit in ("ns","us"):
            index = pd.date_range("2024-01-01",periods=20,freq="h",tz="Asia/Shanghai").as_unit(unit)
            candidates = [index[0],index[4],index[18],index[-1]+pd.Timedelta(hours=1)]
            self.assertEqual(_filter_valid_anchors(candidates,index,4,2),[index[4],index[18]])


if __name__ == "__main__":
    unittest.main()
