"""Probability semantics, family bridge, tree backend and matrix alignment checks."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ts_models.cas_distribution import GRID, CASDistributionAdapter, summarize_distribution
from scripts.run_cas_transfer_matrix import commands, verify_alignment


class DistributionTests(unittest.TestCase):
    def test_sundial_uses_legacy_supported_dtype_argument(self):
        from ts_models.adapters.sundial_adapter import SundialAdapter
        captured = {}
        def load(path, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(eval=lambda: None)
        fake = SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *a,**k: object()),
                               AutoModelForCausalLM=SimpleNamespace(from_pretrained=load))
        with patch.dict(sys.modules, {"transformers":fake}):
            SundialAdapter(model_path="fixture", device="cpu").load_model()
        self.assertIs(captured["torch_dtype"], torch.float32)
        self.assertNotIn("dtype", captured)

    def test_median_and_integral_on_common_grid(self):
        result = summarize_distribution(GRID.reshape(1,1,1,9), levels=GRID)
        self.assertAlmostEqual(result["point_forecast"].item(), .5)
        self.assertAlmostEqual(result["uncertainty"].item(), .2)
        # A mean of native quantiles is not generally the median.
        result = summarize_distribution(np.array([0., 2., 20.]).reshape(1,1,1,3), levels=[.1,.5,.9])
        self.assertAlmostEqual(result["point_forecast"].item(), 2.)

    def test_empirical_quantiles_and_invalid_point_only_output(self):
        samples = np.arange(101).reshape(1,1,1,101)
        result = summarize_distribution(samples, samples=True)
        self.assertEqual(result["point_forecast"].item(), 50.)
        self.assertAlmostEqual(result["uncertainty"].item(), 20.)
        with self.assertRaises(ValueError):
            summarize_distribution(samples[..., :1], samples=True)
        with self.assertRaises(ValueError):
            summarize_distribution(np.zeros((1,1,1,3)), levels=[.2,.5,.8])
        with self.assertRaises(ValueError):
            summarize_distribution(np.zeros((1,1,24)))

    def test_each_family_uses_distribution_not_adapter_mean(self):
        for family in ("chronos2", "moirai2", "sundial"):
            bridge = CASDistributionAdapter.__new__(CASDistributionAdapter)
            bridge.family = family
            values = (np.arange(101) if family == "sundial" else GRID).astype(np.float32)
            output = dict(forecast=torch.from_numpy(np.broadcast_to(values,(2,1,24,len(values))).copy()),
                          mean=torch.full((2,1,24), -999.), quantile_levels=GRID.tolist())
            bridge.adapter = SimpleNamespace(_ensure_model_loaded=lambda: None,
                                             module=SimpleNamespace(quantile_levels=GRID.tolist()),
                                             predict=lambda **kw: output)
            result = bridge.predict(torch.zeros(2,1,168), 24)
            self.assertEqual(tuple(result["point_forecast"].shape), (2,1,24))
            np.testing.assert_allclose(result["point_forecast"], 50 if family == "sundial" else .5)

    def test_monotone_rearrangement_is_explicit(self):
        values = GRID.copy(); values[[0,1]] = values[[1,0]]
        result = summarize_distribution(values.reshape(1,1,1,9), levels=GRID)
        self.assertEqual(result["metadata"]["quantile_crossings_rearranged"], 1)
        np.testing.assert_allclose(result["forecast"], GRID.reshape(1,1,1,9))


class BackendTests(unittest.TestCase):
    def test_lightgbm_worker_writes_complete_artifacts(self):
        from scripts.run_mlp_transfer import worker, sha
        scratch = ROOT/"results/test_scratch"; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            folder = root/"prepared/guangdong/window_1"; folder.mkdir(parents=True)
            rng = np.random.default_rng(8)
            arrays = {}
            for split, n in (("train",100),("valid",20),("test",20)):
                X = rng.normal(size=(n,3)); X[:,1] = rng.uniform(.1,.9,n)
                arrays[f"X_{split}"] = X
                arrays[f"y_{split}"] = X[:,0]*2+1
            np.savez(folder/"features.npz",**arrays)
            meta = dict(feature_names=["pred_cas_moirai2_y","pred_cas_moirai2_y_conf","h"],
                        confidence_index=1, point_index=0, direction=-1, upstream="cas_moirai2",
                        arrays_sha256=sha(folder/"features.npz"),valid_time=list(range(20)),test_time=list(range(20)))
            (folder/"manifest.json").write_text(json.dumps(meta))
            args = SimpleNamespace(prepared_root=root/"prepared",output_root=root/"out",market="guangdong",
                                   window=1,method="tsfm_point_C_confidence_asym",seed=42,alpha=2.,weight_lambda=1.,
                                   hidden=[8],lr=.001,batch_size=64,epochs=2,patience=1,selection="rmse",
                                   device="cpu",threads=1,regressor="lightgbm",lgbm_rounds=5,lgbm_patience=2)
            worker(args)
            output = root/"out/guangdong/window_1/tsfm_point_C_confidence_asym/seed_42"
            metrics = json.loads((output/"metrics.json").read_text())
            self.assertEqual(metrics["upstream"], "cas_moirai2")
            self.assertEqual(metrics["regressor"], "lightgbm")
            self.assertTrue((output/"model.txt").is_file())
            self.assertTrue((output/"predictions_test.csv").is_file())

    def test_lightgbm_backend_both_directions_and_selection(self):
        from exp.pipeline.cas_lgbm_regressor import CASLGBMRegressor
        rng = np.random.default_rng(4)
        X = rng.normal(size=(120,3)); y = X[:,0]*2+3
        for direction in (-1, 1):
            for selection in ("rmse", "legacy"):
                model = CASLGBMRegressor(seed=42, threads=1, alpha=2, direction=direction,
                                        asymmetric=True, selection=selection, rounds=10, patience=3)
                model.fit(X[:90],y[:90],X[90:],y[90:],np.linspace(1,2,90))
                self.assertTrue(np.isfinite(model.predict(X[90:])).all())
                self.assertGreater(model.best_iteration, 0)


class MatrixTests(unittest.TestCase):
    def test_commands_use_separate_sources_and_shared_backend_arrays(self):
        config = dict(upstreams={"cas_chronos2":"a", "cas_moirai2":"b"}, prepared_root="prepared",
                      output_root="out",cache_root="cache",split_root="old", markets=["guangdong"],
                      windows=[1],seeds=[42],regressors=["mlp","lightgbm"])
        cmds = list(commands(config, "run"))
        self.assertEqual(len(cmds), 4)
        self.assertEqual(cmds[0][cmds[0].index("--prepared-root")+1], cmds[1][cmds[1].index("--prepared-root")+1])
        self.assertNotEqual(cmds[0][cmds[0].index("--output-root")+1], cmds[1][cmds[1].index("--output-root")+1])

    def test_alignment_rejects_changed_targets_and_forecast_keys(self):
        scratch = ROOT/"results/test_scratch"; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            config = dict(upstreams={"cas_chronos2":"a", "cas_moirai2":"b"},
                          prepared_root=temp, markets=["guangdong"],windows=[1])
            for source in config["upstreams"]:
                folder = Path(temp)/source/"guangdong/window_1"; folder.mkdir(parents=True)
                arrays = {f"y_{s}":np.arange(3.) for s in ("train","valid","test")}
                arrays.update({f"X_{s}":np.ones((3,3)) for s in ("train","valid","test")})
                np.savez(folder/"features.npz",**arrays)
                meta = dict(feature_names=["point","c","h"],point_index=0,confidence_index=1,
                            valid_time=["a","b","c"],test_time=["a","b","c"],
                            distribution_protocol="cas_common_q10_q90_v1")
                (folder/"manifest.json").write_text(json.dumps(meta))
                (folder/"test_sample_keys.csv").write_text("anchor_time,time,h\na,a,1\nb,b,1\nc,c,1\n")
            verify_alignment(config)
            arrays["y_test"] = np.arange(3.)+1
            np.savez(folder/"features.npz",**arrays)
            with self.assertRaises(AssertionError):
                verify_alignment(config)


if __name__ == "__main__":
    unittest.main()
