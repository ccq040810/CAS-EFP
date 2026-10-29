from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pilot_cas_optimization import coefficient,chronological_calibration,CANDIDATES


class OptimizationTests(unittest.TestCase):
    def test_strong_controls_match_original_strength(self):
        from scripts.pilot_cas_strength_controls import controlled_coefficient
        r=np.array([-3.,-2.,-1.]);c=np.array([.1,.5,.9])
        original=controlled_coefficient(r,c,'original_cas',1,2.)
        np.testing.assert_array_equal(original,coefficient(r,c,'original_cas',1,2.))
        mean=controlled_coefficient(r,c,'mean_alpha2',1,2.)
        self.assertAlmostEqual(original.mean(),mean.mean())
        shuffled=controlled_coefficient(r,c,'shuffle_alpha2',1,2.)
        np.testing.assert_array_equal(np.sort(original),np.sort(shuffled))
    def test_neutral_custom_and_builtin_have_equal_initialization(self):
        import lightgbm as lgb
        rng=np.random.default_rng(4);X=rng.normal(size=(100,2));y=50+X[:,0]*2
        target=y-y.mean()
        params=dict(objective='regression',metric='None',boost_from_average=False,
            learning_rate=.05,num_leaves=5,min_data_in_leaf=5,num_threads=1,verbosity=-1,
            deterministic=True,force_col_wise=True,seed=42)
        ordinary=lgb.train(params,lgb.Dataset(X,label=target),num_boost_round=10)
        def neutral(pred,data):return pred-data.get_label(),np.ones(len(pred))
        other=lgb.train(dict(params,objective=neutral),lgb.Dataset(X,label=target),num_boost_round=10)
        np.testing.assert_allclose(ordinary.predict(X),other.predict(X),atol=1e-8)
    def test_gradients_match_piecewise_weighted_loss(self):
        r=np.array([-3.,2.]);c=np.array([.1,.9]);eps=1e-4
        for kind in CANDIDATES:
            b=coefficient(r,c,kind,1,1.25)
            def loss(v):return .5*np.sum(coefficient(v,c,kind,1,1.25)*v*v)
            for i in range(2):
                p=r.copy();m=r.copy();p[i]+=eps;m[i]-=eps
                self.assertAlmostEqual((loss(p)-loss(m))/(2*eps),b[i]*r[i],places=6)
                self.assertAlmostEqual((loss(p)-2*loss(r)+loss(m))/(eps*eps),b[i],places=5)
            self.assertTrue((b>0).all())

    def test_neutral_direction_and_control_strength(self):
        r=np.array([-3.,2.]);c=np.array([.1,.9])
        for kind in ['fixed_mild','cas_mild','cas_medium','shuffled','matched_mean']:
            np.testing.assert_array_equal(coefficient(r,c,kind,0,1.),np.ones(2))
        np.testing.assert_array_equal(coefficient(r,c,'fixed_mild',1,1.25),coefficient(r,c,'matched_mean',1,1.25))
        self.assertAlmostEqual(coefficient(r,c,'reliability_symmetric',1,1.25).mean(),1.)

    def test_calibration_never_overlaps_training_labels(self):
        origins=pd.date_range('2024-01-01',periods=100,freq='h')
        h=np.tile(np.arange(1,25),100)
        times=origins.repeat(24)+pd.to_timedelta(h-1,unit='h')
        train,cal=chronological_calibration(times,h)
        self.assertTrue(train.any() and cal.any())
        self.assertLess(times[train].max(),(times-pd.to_timedelta(h-1,unit='h'))[cal].min())
        self.assertFalse((train&cal).any())


if __name__=='__main__':unittest.main()
