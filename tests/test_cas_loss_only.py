import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import nullcontext
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mlp_transfer import worker, sha, suite_methods
from scripts.run_cas_loss_only import METHOD, checked_result


class LossOnlyTests(unittest.TestCase):
    def test_supplement_reuses_controls_reports_and_resumes(self):
        import pandas as pd
        from scripts import run_cas_loss_only as runner
        from scripts.run_mlp_transfer import job_config, write, read, ROLLOUT_PROTOCOL
        scratch = ROOT/'results/test_scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            settings = SimpleNamespace(regressor='lightgbm', hidden=[128,64],lr=.001,
                batch_size=8192,epochs=300,patience=30,selection='rmse',alpha=2.,
                weight_lambda=1.,lgbm_rounds=2000,lgbm_patience=100)
            for market in ('guangdong','reale_delu'):
                for window in (1,2,3):
                    folder = root/f'prepared/{market}/window_{window}'
                    folder.mkdir(parents=True)
                    (folder/'features.npz').write_bytes(b'fingerprint fixture')
                    manifest = dict(upstream='cas_chronos2',distribution_protocol='cas_common_q10_q90_v1',
                        rollout_protocol=ROLLOUT_PROTOCOL,arrays_sha256=sha(folder/'features.npz'),
                        feature_names=['point','point_conf','h'],confidence_index=1,direction=1)
                    write(folder/'manifest.json',manifest)
                    for seed in (42,43,44):
                        for method in (*runner.CONTROLS, METHOD):
                            base = 'out' if method==METHOD else 'controls'
                            dest = root/f'{base}/{market}/window_{window}/{method}/seed_{seed}'
                            dest.mkdir(parents=True)
                            write(dest/'run_config.json',job_config(settings,manifest,method,seed))
                            write(dest/'metrics.json',dict(market=market,window=window,seed=seed,
                                method=method,upstream='cas_chronos2',regressor='lightgbm',selection='rmse',
                                prepared_sha256=manifest['arrays_sha256'],test_mae=2.,test_rmse=3.,valid_mae=2.,valid_rmse=3.))
            argv = ['run','--prepared-root',str(root/'prepared'),'--control-root',str(root/'controls'),
                    '--output-root',str(root/'out')]
            with patch.object(sys,'argv',argv), patch.object(runner,'run_lock',lambda p:nullcontext()), patch.object(runner.subprocess,'run') as launch:
                runner.main()
                launch.assert_not_called()
            self.assertEqual(len(pd.read_csv(root/'out/ALL_RUNS.csv')),54)
            self.assertEqual(len(pd.read_csv(root/'out/PAIRED_BY_WINDOW.csv')),36)
            self.assertEqual(len(pd.read_csv(root/'out/FINAL_RESULTS.csv')),6)
            self.assertTrue(read(root/'out/completion.json')['complete'])

    def test_worker_drops_c_but_passes_its_weights_to_training(self):
        from exp.pipeline.cas_lgbm_regressor import CASLGBMRegressor
        scratch = ROOT/'results/test_scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            folder = root/'prepared/guangdong/window_1'
            folder.mkdir(parents=True)
            rng = np.random.default_rng(7)
            arrays = {}
            for split,n in (("train",100),("valid",20),("test",20)):
                X = rng.normal(size=(n,3))
                X[:,1] = np.linspace(.1,.9,n)
                arrays['X_'+split] = X
                arrays['y_'+split] = 2*X[:,0]+1
            np.savez(folder/'features.npz', **arrays)
            manifest = dict(feature_names=['pred_cas_chronos2_y','pred_cas_chronos2_y_conf','h'],
                confidence_index=1, point_index=0, direction=1, upstream='cas_chronos2',
                arrays_sha256=sha(folder/'features.npz'), valid_time=list(range(20)), test_time=list(range(20)))
            (folder/'manifest.json').write_text(json.dumps(manifest))
            args = SimpleNamespace(prepared_root=root/'prepared', output_root=root/'out',
                market='guangdong', window=1, method=METHOD, seed=42, alpha=2., weight_lambda=1.,
                hidden=[128,64], lr=.001, batch_size=8192, epochs=300, patience=30,
                selection='rmse', device='cpu', threads=1, regressor='lightgbm', lgbm_rounds=5, lgbm_patience=2)
            original = CASLGBMRegressor.fit
            def capture(model, X, y, Xv, yv, weights):
                np.testing.assert_array_equal(X, arrays['X_train'][:,[0,2]])
                np.testing.assert_array_equal(Xv, arrays['X_valid'][:,[0,2]])
                np.testing.assert_allclose(weights, 2-arrays['X_train'][:,1])
                return original(model,X,y,Xv,yv,weights)
            with patch.object(CASLGBMRegressor,'fit',capture):
                worker(args)
            out = root/f'out/guangdong/window_1/{METHOD}/seed_42'
            config = json.loads((out/'run_config.json').read_text())
            self.assertEqual(config['feature_names'], ['pred_cas_chronos2_y','h'])
            checked_result(out, config, 'guangdong',1,METHOD,42)
            config['prepared_sha256'] = 'different'
            with self.assertRaises(ValueError):
                checked_result(out, config,'guangdong',1,METHOD,42)
        self.assertEqual(len(suite_methods('mechanism')),7)
        self.assertNotIn(METHOD,suite_methods('mechanism'))


if __name__ == '__main__':
    unittest.main()
