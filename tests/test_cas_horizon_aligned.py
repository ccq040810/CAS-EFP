from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pilot_cas_horizon_aligned import aligned_features,calibrate_costs,cost_coefficients,CANDIDATES


class HorizonAlignedTests(unittest.TestCase):
    def test_price_curves_select_target_h_and_exclude_input_confidence(self):
        n=2;context=np.arange(2*(96+72*n+9),dtype=np.float32).reshape(2,-1)
        steps=np.array([0,23,5,9]);ids=np.array([0,0,1,1])
        X=np.column_stack([steps+1,np.arange(4),np.full(4,50.),np.linspace(.1,.9,4),np.full(4,60.)]).astype(np.float32)
        meta=dict(feature_names=['h','local','pred_point','pred_conf','da_price'],point_index=2,confidence_index=3)
        A=aligned_features(X,context,ids,steps,meta,['load','da_price'])
        np.testing.assert_array_equal(A[:,4:8],context[ids[:,None],4*steps[:,None]+np.arange(4)])
        np.testing.assert_array_equal(A[:,8:12],A[:,4:8]-50)
        other=X.copy();other[:,3]=999
        np.testing.assert_array_equal(A,aligned_features(other,context,ids,steps,meta,['load','da_price']))
        with self.assertRaises(ValueError):aligned_features(X,context,ids,steps[::-1],meta,['load','da_price'])

    def test_soft_bias_uses_mean_and_shrinks_uncertain_estimates(self):
        days=pd.date_range('2024-01-01',periods=14,freq='D').repeat(24)
        h=np.tile(np.arange(24),14)
        e=np.ones(len(h))*2
        direction,strength,rows=calibrate_costs(e,h,days)
        np.testing.assert_array_equal(direction,np.ones(24))
        np.testing.assert_allclose(strength,.5)
        mixed=e.copy();mixed[:24]=100
        _,_,r=calibrate_costs(mixed,h,days)
        self.assertAlmostEqual(r[0]['bias'],(100+13*2)/14)
        self.assertGreater(r[0]['day_mean_se'],0)
        d,s,_=calibrate_costs(e[:24*3],h[:24*3],days[:24*3])
        np.testing.assert_array_equal(d,np.zeros(24));np.testing.assert_array_equal(s,np.zeros(24))

    def test_normalization_shuffle_and_gradient(self):
        h=np.repeat([0,1],4);c=np.array([.1,.2,.5,.9,.2,.3,.6,.8])
        d=np.ones(24,dtype=int);strength=np.ones(24)*.3;r=np.full(8,-2.)
        fixed=cost_coefficients(r,c,h,d,strength,'fixed')
        for kind in CANDIDATES:
            b=cost_coefficients(r,c,h,d,strength,kind)
            self.assertTrue(((b>=1)&(b<=2)).all())
            if kind!='ordinary':
                for horizon in [0,1]:self.assertAlmostEqual(b[h==horizon].mean(),fixed[h==horizon].mean())
            eps=1e-4
            def loss(x):return .5*np.sum(cost_coefficients(x,c,h,d,strength,kind)*x*x)
            plus=r.copy();minus=r.copy();plus[0]+=eps;minus[0]-=eps
            self.assertAlmostEqual((loss(plus)-loss(minus))/(2*eps),b[0]*r[0],places=6)
            self.assertAlmostEqual((loss(plus)-2*loss(r)+loss(minus))/eps**2,b[0],places=5)
        for suffix in ['03','10']:
            a=cost_coefficients(r,c,h,d,strength,'cas_'+suffix)
            b=cost_coefficients(r,c,h,d,strength,'shuffle_'+suffix)
            for horizon in [0,1]:np.testing.assert_array_equal(np.sort(a[h==horizon]),np.sort(b[h==horizon]))


if __name__=='__main__':unittest.main()
