from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_cas_directional_study import cases,signed_ramp,weights,METHODS,CAS_METHODS


class DirectionalStudyTests(unittest.TestCase):
    def test_end_to_end_alignment_freeze_and_complete_method_coverage(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import lightgbm as lgb
        from scripts import run_cas_directional_study as study
        from scripts.run_mlp_transfer import write,sha,read,ROLLOUT_PROTOCOL
        rng=np.random.default_rng(42)
        with tempfile.TemporaryDirectory(prefix='.directional-test-',dir=ROOT) as tmp:
            root=Path(tmp);ref=root/'reference';ref.mkdir();external=root/'external'
            ep=external/'prepared/guangdong/window_1';ep.mkdir(parents=True)
            arrays={};ea={};meta=dict(rollout_protocol=ROLLOUT_PROTOCOL,confidence_index=3,point_index=2,
                                     feature_names=['h','local','pred_cas_chronos2_y','pred_cas_chronos2_y_conf'])
            for split,count,start in [('train',100,'2024-01-01'),('valid',12,'2024-06-01'),('test',12,'2024-09-01')]:
                origins=pd.date_range(start,periods=count,freq='h',tz='Asia/Shanghai')
                steps=np.tile(np.arange(24),count);ids=np.repeat(np.arange(count),24)
                times=origins.repeat(24)+pd.to_timedelta(steps,unit='h')
                point=50+rng.normal(size=len(ids))*10;local=rng.normal(size=len(ids))
                X=np.column_stack([steps+1,local,point,rng.uniform(size=len(ids))]).astype(np.float32)
                y=(point+2*local+rng.normal(size=len(ids))).astype(np.float32)
                arrays.update({'X_'+split:X,'y_'+split:y})
                ea.update({'X_'+split:rng.normal(size=(count,4)).astype(np.float32),'yrows_'+split:y,
                           'row_ids_'+split:ids,'steps_'+split:steps})
                meta[split+'_time']=list(map(str,times))
            np.savez(ref/'features.npz',**arrays);np.savez(ep/'inputs.npz',**ea)
            meta['arrays_sha256']=sha(ref/'features.npz');write(ref/'manifest.json',meta)
            em=dict(identity={'reference_sha256':meta['arrays_sha256']},arrays_sha256=sha(ep/'inputs.npz'),
                    **{s+'_time':meta[s+'_time'] for s in ['train','valid','test']})
            write(ep/'manifest.json',em)
            for model in ['lear_adapted','rnn']:
                dest=external/'training'/model/'guangdong/window_1/seed_42';dest.mkdir(parents=True)
                write(dest/'metrics.json',dict(prepared_sha256=em['arrays_sha256'],valid_mae=1.,valid_rmse=1.))
                pd.DataFrame(dict(time=meta['test_time'],y_true=arrays['y_test'],prediction=arrays['y_test']+1)).to_csv(dest/'predictions_test.csv',index=False)
            args=SimpleNamespace(output_root=root/'out',reference_root=root/'unused',external_root=external)
            case=dict(id='integration',market='guangdong',window=1,kind='semi_synthetic_signed_ramp',sign=1,strength=.5)
            original_train=lgb.train
            def small_train(params,data,**kwargs):
                kwargs['num_boost_round']=min(kwargs.get('num_boost_round',20),20)
                return original_train(params,data,**kwargs)
            with patch.object(study,'reference',return_value=(ref,external)),patch.object(study.subprocess,'run'),patch.object(lgb,'train',side_effect=small_train):
                study.fit_case(args,case)
            out=args.output_root/'cases/integration'
            self.assertTrue(read(out/'completion.json')['complete'])
            table=pd.read_csv(out/'ALL_RUNS.csv')
            self.assertEqual(len(table),18)
            self.assertEqual(len(set(table.method)),18)
            freeze=read(out/'selection.json')
            chosen=min((r for r in freeze['validation'] if r['method'] in CAS_METHODS),key=lambda r:(r['valid_rmse'],r['method']))['method']
            self.assertEqual(freeze['selected_cas'],chosen)
            self.assertEqual(table.selected_cas.sum(),1)
            self.assertTrue(np.isfinite(table.test_rmse).all())

    def test_preregistered_cases_keep_both_signs_and_all_windows(self):
        plan=cases()
        self.assertEqual(len(plan),18)
        self.assertEqual(len({c['id'] for c in plan}),18)
        self.assertEqual(len(METHODS),15)
        self.assertEqual(set(CAS_METHODS)&set(METHODS),set(CAS_METHODS))
        for strength in [.5,1.]:
            for sign in [-1,1]:
                self.assertEqual({c['window'] for c in plan if c['strength']==strength and c['sign']==sign},{1,2,3})

    def test_ramp_scale_cannot_use_evaluation_targets(self):
        times=pd.date_range('2024-01-01',periods=500,freq='h')
        y=np.sin(np.arange(500.)/12)*10
        first,meta=signed_ramp(times,y,times[0],times[300],1,.5)
        altered=y.copy();altered[300:]+=100000
        second,other=signed_ramp(times,altered,times[0],times[300],1,.5)
        self.assertEqual(meta,other)
        np.testing.assert_allclose(first-y,second-altered,atol=1e-10)
        negative,_=signed_ramp(times,y,times[0],times[300],-1,.5)
        np.testing.assert_allclose(first-y,-(negative-y),atol=1e-12)
        self.assertEqual(first[0],y[0])

    def test_directional_weights_have_correct_gradient_and_hessian(self):
        r=np.array([-2.,3.,-1.,2.]);c=np.array([.1,.7,.5,.9]);d=np.array([1,1,0,-1])
        for v in ['ordinary','fixed','cas','mean','shuffle','gated_fixed','gated_cas','gated_shuffle']:
            b=weights(r,c,v,d)
            self.assertTrue((b>0).all())
            self.assertEqual(b[2],1.)
            eps=1e-4
            def loss(x):return .5*np.sum(weights(x,c,v,d)*x*x)
            for i in range(len(r)):
                plus=r.copy();minus=r.copy();plus[i]+=eps;minus[i]-=eps
                self.assertAlmostEqual((loss(plus)-loss(minus))/(2*eps),b[i]*r[i],places=6)
                self.assertAlmostEqual((loss(plus)-2*loss(r)+loss(minus))/eps**2,b[i],places=5)
        with self.assertRaises(ValueError):weights(r,c+1,'cas',d)


if __name__=='__main__':unittest.main()
