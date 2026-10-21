"""Meaningful CPU checks; no datasets, GPU, or Chronos weights required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp.pipeline.mlp_regressor import MLPRegressor, directional_loss
from scripts.run_mlp_transfer import sha


class LossTests(unittest.TestCase):
    def test_autograd_matches_existing_lightgbm_objective(self):
        from exp.pipeline.regressor import asymmetric_l2_objective_weighted
        class Dataset:
            def get_label(self):
                return np.array([100., 200., 150.])
            def get_weight(self):
                return np.array([1.2, 1.8, 1.5])
        data = Dataset()
        for direction in (-1, 1):
            values = np.array([130., 170., 160.])
            p = torch.tensor(values, requires_grad=True, dtype=torch.float64)
            target = torch.tensor(data.get_label())
            weights = torch.tensor(data.get_weight())
            loss = directional_loss(p, target, weights, alpha=2, direction=direction)
            grad = torch.autograd.grad(loss, p, create_graph=True)[0]
            hess = torch.stack([torch.autograd.grad(grad[i], p, retain_graph=True)[0][i] for i in range(3)])
            expected_g, expected_h = asymmetric_l2_objective_weighted(direction=direction)(values, data)
            np.testing.assert_allclose(grad.detach().numpy()*3, expected_g)
            np.testing.assert_allclose(hess.numpy()*3, expected_h)

    def test_degenerates_to_l2_and_only_weights_declared_direction(self):
        p = torch.tensor([2., -2.], requires_grad=True)
        target = torch.zeros(2)
        weight = torch.tensor([1.5, 1.5])
        ordinary = directional_loss(p, target, weight, asymmetric=False)
        degenerate = directional_loss(p, target, torch.ones(2), alpha=1, asymmetric=True)
        self.assertEqual(float(ordinary.detach()), float(degenerate.detach()))
        weighted = directional_loss(p, target, weight, alpha=2, direction=1)
        np.testing.assert_allclose(torch.autograd.grad(weighted,p)[0].numpy(), [1., -3.])

    def test_train_only_scaling_and_deterministic_fit(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(160,3)).astype(np.float32)
        y = (5+2*X[:,0]-X[:,1]).astype(np.float32)
        va = X[:30]+0.3
        vy = (5+2*va[:,0]-va[:,1]).astype(np.float32)
        models = [MLPRegressor(hidden=(16,), lr=.02, epochs=35, patience=10, batch_size=80, threads=1, seed=7).fit(X,y,va,vy) for _ in range(2)]
        np.testing.assert_array_equal(models[0].predict(va), models[1].predict(va))
        np.testing.assert_allclose(models[0].mean, X.mean(0), atol=1e-7)
        self.assertLess(np.mean((models[0].predict(va)-vy)**2), np.mean((y.mean()-vy)**2)*.2)


class WorkflowTests(unittest.TestCase):
    def test_prepare_reuses_original_feature_builder(self):
        import pandas as pd
        # This fixture tests feature preparation, not the external calendar.
        import types
        sys.modules.setdefault("chinese_calendar", types.SimpleNamespace(is_workday=lambda _: True))
        from exp.pipeline.feature_select import _build_epf_benchmark_datasets, build_std_cov_target_table, build_features
        from scripts.run_mlp_transfer import prepare
        scratch = ROOT/"results"/"test_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            folder = Path(temp)
            assert folder.resolve().is_relative_to(ROOT.resolve())
            raw = folder/"data.csv"
            times = pd.date_range("2024-01-01",periods=1200,freq="h")
            pd.DataFrame(dict(time=times,load=np.sin(np.arange(1200)/24),y=np.arange(1200)/100)).to_csv(raw,index=False)
            columns = folder/"regress.json"; columns.write_text(json.dumps([dict(target_columns=["load","y"])]))
            tsfm = folder/"tsfm.json"; tsfm.write_text(json.dumps([dict(target_columns=["y"])]))
            splitroot = folder/"old"; splitdir = splitroot/"guangdong"/"window_1"; splitdir.mkdir(parents=True)
            cfg = dict(time_col="time",train=.5,valid=.25,test=.25,
                       train_start=str(times[0]),train_end=str(times[600]),valid_start=str(times[600]),
                       valid_end=str(times[900]),test_start=str(times[900]),test_end=str(times[-1]+pd.Timedelta(hours=1)))
            splitfile = splitdir/"data_split.json"; splitfile.write_text(json.dumps([dict(data_split=cfg)]))
            args = SimpleNamespace(data_split_path=str(splitfile),regress_cols_path=str(columns),
                                   data_path=str(raw),seq_len=168,pred_len=24,scale=False)
            datasets = _build_epf_benchmark_datasets(args)[:3]
            base = pd.concat([build_std_cov_target_table(ds,cov_cols=["load"]) for ds in datasets],ignore_index=True)
            predictions = base[["anchor_time","time","h"]].copy()
            predictions["pred_chronos2_y"] = np.arange(len(base),dtype=float)/100
            predictions["pred_chronos2_y_uncertainty"] = 1+np.arange(len(base))%10
            cache = folder/"cache.parquet"; cache.write_bytes(b"synthetic-cache-fixture")
            command = SimpleNamespace(markets=["guangdong"],windows=[1],prepared_root=folder/"prepared",
                                      split_root=splitroot,cache_root=folder/"cache",model_path="unused",
                                      tsfm_batch_size=16,device="cpu")
            captured = {}
            def frozen_rollout(*,args):
                captured["args"] = args
                return predictions
            definition = (str(raw),"time",str(columns),str(tsfm),True)
            with patch.dict("scripts.run_rolling_adaptive_experiment.DATASETS",{"guangdong":definition},clear=True), \
                 patch("exp.pipeline.tsfm_infer.run_tsfm_rollout",side_effect=frozen_rollout), \
                 patch("exp.pipeline.tsfm_infer._get_cache_path",return_value=str(cache)):
                prepare(command)
            direct = build_features(args=captured["args"],pred_table=predictions)
            with np.load(folder/"prepared/guangdong/window_1/features.npz") as saved:
                for key,index in (("X_train",0),("y_train",1),("X_valid",3),("y_valid",4),("X_test",6),("y_test",7)):
                    np.testing.assert_array_equal(saved[key],direct[index])

    def test_worker_scheduler_summary_resume_and_config_guard(self):
        scratch = ROOT/"results"/"test_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            assert root.resolve().is_relative_to(ROOT.resolve())
            prepared = root/"prepared"/"guangdong"/"window_1"
            prepared.mkdir(parents=True)
            rng = np.random.default_rng(1)
            arrays = {}
            for split,n in (("train",60),("valid",12),("test",12)):
                X = rng.normal(size=(n,3)).astype(np.float32)
                X[:,1] = rng.uniform(.1,.9,n)
                arrays[f"X_{split}"] = X
                arrays[f"y_{split}"] = (X[:,0]+.1*X[:,2]).astype(np.float32)
            np.savez(prepared/"features.npz",**arrays)
            manifest = dict(feature_names=["pred_chronos2_y","pred_chronos2_y_conf","h"],
                            confidence_index=1,point_index=0,direction=1,
                            arrays_sha256=sha(prepared/"features.npz"),
                            valid_time=[str(i) for i in range(12)],test_time=[str(i) for i in range(12)])
            (prepared/"manifest.json").write_text(json.dumps(manifest))
            cmd = [sys.executable,str(ROOT/"scripts/run_mlp_transfer.py"),"run",
                   "--prepared-root",str(root/"prepared"),"--output-root",str(root/"output"),
                   "--markets","guangdong","--windows","1","--seeds","42","--suite","original",
                   "--gpus","cpu","--epochs","2","--patience","1","--threads","1","--hidden","8"]
            result = subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            completion = json.loads((root/"output/completion.json").read_text())
            self.assertEqual(completion["completed"],5)
            self.assertFalse(completion["missing"])
            self.assertTrue((root/"output/paired_cas_vs_fixed.csv").is_file())
            result = subprocess.run(cmd+["--resume"],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(result.stdout.count("SKIP completed"),5)
            result = subprocess.run(cmd+["--resume","--lr","0.02"],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            # Missing jobs must not produce an aggregate report.
            (root/"output/summary.csv").unlink()
            metric = next((root/"output").glob("*/window_*/*/seed_*/metrics.json"))
            metric.unlink()
            summary = [sys.executable,str(ROOT/"scripts/run_mlp_transfer.py"),"summarize","--output-root",str(root/"output")]
            result = subprocess.run(summary,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertFalse((root/"output/summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
