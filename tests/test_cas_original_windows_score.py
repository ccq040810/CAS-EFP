from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys,tempfile,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.score_cas_original_windows import score_case,METHODS
from scripts.run_mlp_transfer import write,sha,ROLLOUT_PROTOCOL


class SavedScoringTests(unittest.TestCase):
    def test_saved_models_only_no_training_and_exact_checkpoint(self):
        import lightgbm as lgb
        rng=np.random.default_rng(4)
        with tempfile.TemporaryDirectory(prefix='.saved-score-test-',dir=ROOT) as tmp:
            root=Path(tmp);ref=root/'reference/guangdong/window_1';ref.mkdir(parents=True)
            ext=root/'external/prepared/guangdong/window_1';ext.mkdir(parents=True)
            h=np.tile(np.arange(24),3);ids=np.repeat(np.arange(3),24)
            X=np.column_stack([h+1,rng.normal(size=len(h)),rng.normal(50,5,len(h)),rng.uniform(size=len(h))]).astype(np.float32)
            y=(X[:,2]+rng.normal(size=len(h))).astype(np.float32)
            times=pd.date_range('2024-01-01',periods=3,freq='h',tz='Asia/Shanghai').repeat(24)+pd.to_timedelta(h,unit='h')
            np.savez(ref/'features.npz',X_test=X,y_test=y)
            meta=dict(feature_names=['h','local','point','conf'],point_index=2,confidence_index=3,rollout_protocol=ROLLOUT_PROTOCOL,
                arrays_sha256=sha(ref/'features.npz'),test_time=list(map(str,times)))
            write(ref/'manifest.json',meta)
            np.savez(ext/'inputs.npz',X_test=rng.normal(size=(3,105)).astype(np.float32),steps_test=h,row_ids_test=ids,yrows_test=y)
            em=dict(arrays_sha256=sha(ext/'inputs.npz'),identity={'reference_sha256':meta['arrays_sha256']},covariates=[],test_time=meta['test_time'])
            write(ext/'manifest.json',em)
            args=SimpleNamespace(reference_root=root/'reference',external_root=root/'external',model_root=root/'saved',output_root=root/'scores')
            source_hashes={}
            for seed in [42,43,44]:
                for method in METHODS:
                    p=args.model_root/'guangdong/window_1'/method/f'seed_{seed}';p.mkdir(parents=True)
                    (p/'model.txt').write_text('existing checkpoint')
                    write(p/'validation.json',dict(prepared_sha256=meta['arrays_sha256'],external_sha256=em['arrays_sha256'],center=0.,best_iteration=5))
                    source_hashes[p/'model.txt']=sha(p/'model.txt')
                for method in ['lear_adapted','rnn']:
                    p=args.external_root/'training'/method/'guangdong/window_1'/f'seed_{seed}';p.mkdir(parents=True)
                    write(p/'metrics.json',dict(prepared_sha256=em['arrays_sha256']))
                    pd.DataFrame(dict(time=meta['test_time'],y_true=y,prediction=X[:,2])).to_csv(p/'predictions_test.csv',index=False)
            class SavedModel:
                def __init__(self,model_file):self.n=8 if 'tree_only' in Path(model_file).parts else 13
                def num_feature(self):return self.n
                def predict(self,features,num_iteration,num_threads):
                    assert num_iteration==5 and num_threads==4
                    return np.zeros(len(features))
            with patch.object(lgb,'Booster',SavedModel),patch.object(lgb,'train',side_effect=AssertionError('Training is prohibited')):
                rows=score_case(args,'guangdong',1)
            self.assertEqual(len(rows),27);self.assertEqual(len({(r['method'],r['seed']) for r in rows}),27)
            self.assertTrue(all(not r['retrained'] and not r['calibration_reestimated'] for r in rows))
            self.assertTrue(all(sha(p)==value for p,value in source_hashes.items()))
            self.assertTrue(all(np.isfinite(r['mae']) and np.isfinite(r['rmse']) for r in rows))


if __name__=='__main__':unittest.main()
