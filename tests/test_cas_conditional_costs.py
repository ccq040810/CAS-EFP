from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pilot_cas_conditional_costs import conditional_profile,assign_costs


class ConditionalCostsTests(unittest.TestCase):
    def fixture(self):
        h=np.tile(np.repeat(np.arange(24),3),14)
        c=np.tile([.1,.5,.9],24*14)
        days=pd.date_range('2024-01-01',periods=14,freq='D').repeat(72)
        e=np.tile([-4.,0.,4.],24*14)
        return h,c,days,e

    def test_confidence_bands_detect_opposite_bias_despite_zero_h_mean(self):
        h,c,days,e=self.fixture()
        cuts,under,over,rows=conditional_profile(e,c,h,days,c,h)
        self.assertEqual(len(rows),72)
        self.assertTrue((under[:,2]>1).all());self.assertTrue((over[:,0]>1).all())
        np.testing.assert_array_equal(under[:,0],np.ones(24));np.testing.assert_array_equal(over[:,2],np.ones(24))
        self.assertTrue(all(abs(r['horizon_bias'])<1e-12 for r in rows))
        self.assertTrue(((under>=1)&(under<=1.5)).all());self.assertTrue(((over>=1)&(over<=1.5)).all())
        self.assertTrue(all(r['calibration_days']==14 for r in rows))

    def test_mean_and_shuffle_match_both_sides_within_h(self):
        h,c,days,e=self.fixture();cuts,under,over,_=conditional_profile(e,c,h,days,c,h)
        a,b=assign_costs(c,h,cuts,under,over,'conditional_c')
        ma,mb=assign_costs(c,h,cuts,under,over,'conditional_mean')
        sa,sb=assign_costs(c,h,cuts,under,over,'conditional_shuffle')
        for step in range(24):
            mask=h==step
            self.assertAlmostEqual(a[mask].mean(),ma[mask].mean());self.assertAlmostEqual(b[mask].mean(),mb[mask].mean())
            np.testing.assert_array_equal(np.sort(a[mask]),np.sort(sa[mask]));np.testing.assert_array_equal(np.sort(b[mask]),np.sort(sb[mask]))
        self.assertFalse(np.array_equal(a,sa))

    def test_piecewise_two_sided_cost_derivatives(self):
        h,c,days,e=self.fixture();cuts,under,over,_=conditional_profile(e,c,h,days,c,h)
        a,b=assign_costs(c,h,cuts,under,over,'conditional_c');r=np.where(np.arange(len(c))%2,-2.,3.)
        coef=np.where(r<0,a,b);eps=1e-4
        def loss(x):return .5*np.sum(np.where(x<0,a,b)*x*x)
        for i in [0,1,2]:
            plus=r.copy();minus=r.copy();plus[i]+=eps;minus[i]-=eps
            self.assertAlmostEqual((loss(plus)-loss(minus))/(2*eps),coef[i]*r[i],places=6)
            self.assertAlmostEqual((loss(plus)-2*loss(r)+loss(minus))/eps**2,coef[i],places=3)


if __name__=='__main__':unittest.main()
