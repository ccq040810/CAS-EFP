from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys,tempfile,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_cas_final_period import fresh_split,rank_acceptance,train_market,score_market
from scripts.run_mlp_transfer import write,read,sha,ROLLOUT_PROTOCOL


class FinalPeriodTests(unittest.TestCase):
    def test_fresh_boundary_strictly_excludes_old_final_hour(self):
        times=pd.date_range('2024-01-01',periods=3000,freq='h')
        split,audit=fresh_split(times,[str(times[2700])])
        self.assertEqual(pd.Timestamp(split['test_start']),times[2701])
        self.assertEqual(pd.Timestamp(split['train_end']),times[2400])
        self.assertEqual(pd.Timestamp(split['valid_end']),pd.Timestamp(split['test_start']))
        self.assertEqual(audit['new_observed_hours'],299)
        with self.assertRaises(ValueError):fresh_split(times,[str(times[-1])])
        other,_=fresh_split(times.tz_localize('UTC'),[str(times[2700].tz_localize('UTC'))])
        self.assertEqual(pd.Timestamp(other['test_start']),times[2701].tz_localize('UTC'))

    def test_exact_four_cell_acceptance_rule(self):
        rows=[]
        for market in ['guangdong','reale_delu']:
            for method in ['lear_adapted','rnn','chronos_only','tree_only']:
                rows.append(dict(market=market,method=method,mae_mean=10. if method=='lear_adapted' else 12.,rmse_mean=12.))
            rows.append(dict(market=market,method='cas_10',mae_mean=11. if market=='guangdong' else 9.,rmse_mean=9.))
        table=pd.DataFrame(rows);ranks,passed=rank_acceptance(table)
        self.assertTrue(passed);self.assertEqual(sorted(r['rank'] for r in ranks),[1,1,1,2])
        table.loc[(table.market=='guangdong')&(table.method=='rnn'),'mae_mean']=10.5
        self.assertFalse(rank_acceptance(table)[1])

    def test_training_freeze_and_all_prediction_paths(self):
        import lightgbm as lgb
        rng=np.random.default_rng(4)
        with tempfile.TemporaryDirectory(prefix='.final-period-test-',dir=ROOT) as tmp:
            out=Path(tmp);ref=out/'prepared/guangdong/window_4';ref.mkdir(parents=True)
            ext=out/'external/prepared/guangdong/window_4';ext.mkdir(parents=True)
            meta=dict(rollout_protocol=ROLLOUT_PROTOCOL,feature_names=['h','local','point','conf'],confidence_index=3,point_index=2)
            arrays={};extra={}
            for s,n,start in [('train',100,'2024-01-01'),('valid',12,'2024-06-01'),('test',12,'2024-09-01')]:
                origins=pd.date_range(start,periods=n,freq='h',tz='Asia/Shanghai');h=np.tile(np.arange(24),n);ids=np.repeat(np.arange(n),24)
                point=rng.normal(50,10,len(h));local=rng.normal(size=len(h))
                X=np.column_stack([h+1,local,point,rng.uniform(size=len(h))]).astype(np.float32)
                y=(point+local+rng.normal(size=len(h))).astype(np.float32)
                arrays.update({'X_'+s:X,'y_'+s:y});extra.update({'X_'+s:rng.normal(size=(n,105)).astype(np.float32),
                    'steps_'+s:h,'row_ids_'+s:ids,'yrows_'+s:y})
                meta[s+'_time']=list(map(str,origins.repeat(24)+pd.to_timedelta(h,unit='h')))
            np.savez(ref/'features.npz',**arrays);np.savez(ext/'inputs.npz',**extra)
            meta['arrays_sha256']=sha(ref/'features.npz');write(ref/'manifest.json',meta)
            em=dict(identity={'reference_sha256':meta['arrays_sha256']},arrays_sha256=sha(ext/'inputs.npz'),covariates=[],
                **{s+'_time':meta[s+'_time'] for s in ['train','valid','test']})
            write(ext/'manifest.json',em)
            write(out/'plan.json',dict(tree=dict(learning_rate=.05,num_leaves=7,min_data_in_leaf=10),seeds=[42,43,44]))
            args=SimpleNamespace(output_root=out)
            actual=lgb.train
            def small_train(params,data,**kwargs):
                kwargs['num_boost_round']=min(kwargs.get('num_boost_round',20),20);return actual(params,data,**kwargs)
            with patch.object(lgb,'train',side_effect=small_train):train_market(args,'guangdong')
            with self.assertRaises(ValueError):score_market(args,'guangdong')
            write(out/'TRAINING_FROZEN.json',dict(frozen=True))
            for method in ['lear_adapted','rnn']:
                for seed in ([42] if method=='lear_adapted' else [42,43,44]):
                    dest=out/'external/training'/method/'guangdong/window_4'/f'seed_{seed}';dest.mkdir(parents=True)
                    write(dest/'metrics.json',dict(prepared_sha256=em['arrays_sha256']))
                    pd.DataFrame(dict(time=meta['test_time'],y_true=arrays['y_test'],prediction=arrays['y_test']+1)).to_csv(dest/'predictions_test.csv',index=False)
            rows=score_market(args,'guangdong')
            self.assertEqual(len(rows),27)
            self.assertEqual(len({(r['method'],r['seed']) for r in rows}),27)
            self.assertTrue(all(np.isfinite(r['rmse']) for r in rows))


if __name__=='__main__':unittest.main()
