from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pilot_cas_asinh_heads import fit_target,forward,inverse,full_features,calibration


class AsinhHeadTests(unittest.TestCase):
    def test_target_transform_roundtrip_and_no_prediction_clipping(self):
        y=np.array([-200.,0.,50.,300.,1000.]);recipe=fit_target(y)
        np.testing.assert_allclose(inverse(forward(y,recipe),recipe),y,atol=1e-10)
        self.assertGreater(inverse(np.array([10.]),recipe)[0],y.max())
        with self.assertRaises(ValueError):inverse(np.array([10000.]),recipe)
        self.assertEqual(fit_target(np.ones(10))['scale'],1.)

    def test_full_standalone_curves_have_no_upstream_leak(self):
        h=np.array([0,23,5,9]);ids=np.array([0,0,1,1]);context=np.arange(2*105,dtype=np.float32).reshape(2,105)
        X=np.column_stack([h+1,np.arange(4),np.full(4,50.),np.linspace(.1,.9,4)]).astype(np.float32)
        meta=dict(feature_names=['h','local','point','conf'],point_index=2,confidence_index=3)
        A,B=full_features(X,context,ids,h,meta,[])
        changed=X.copy();changed[:,2]=-9999;changed[:,3]=999
        AA,BB=full_features(changed,context,ids,h,meta,[])
        self.assertFalse(np.array_equal(A,AA));np.testing.assert_array_equal(B,BB)
        changed=X.copy();changed[:,3]=999
        AAA,_=full_features(changed,context,ids,h,meta,[]);np.testing.assert_array_equal(A,AAA)

    def test_single_h_calibration_and_custom_loss_coordinate_consistency(self):
        import lightgbm as lgb
        h=np.full(14,7);days=pd.date_range('2024-01-01',periods=14,freq='D')
        d,s,_=calibration(np.full(14,.2),h,days)
        self.assertEqual(d[7],1);self.assertEqual(np.count_nonzero(d),1);self.assertEqual(s[7],.5)
        rng=np.random.default_rng(2);X=rng.normal(size=(200,3));y=300+50*X[:,0]+rng.normal(size=200);point=np.full(200,300.)
        recipe=fit_target(y);base=forward(point,recipe);target=forward(y,recipe)-base;center=target.mean();target-=center
        params=dict(objective='regression',metric='None',seed=42,num_threads=1,num_leaves=7,min_data_in_leaf=10,
                    verbosity=-1,boost_from_average=False)
        native=lgb.train(params,lgb.Dataset(X,label=target),num_boost_round=10)
        def neutral(pred,data):return pred-data.get_label(),np.ones(len(pred))
        custom=lgb.train(dict(params,objective=neutral),lgb.Dataset(X,label=target),num_boost_round=10)
        a=inverse(base+center+native.predict(X,num_threads=1),recipe)
        b=inverse(base+center+custom.predict(X,num_threads=1),recipe)
        np.testing.assert_allclose(a,b,atol=1e-8)


if __name__=='__main__':unittest.main()
