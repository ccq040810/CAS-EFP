from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pilot_cas_robust_loss import robust_derivatives,robust_value,horizon_scales


class RobustLossTests(unittest.TestCase):
    def test_mixed_robust_gradient_hessian_and_positive_curvature(self):
        r=np.array([-50.,-2.,3.,70.]);delta=np.array([5.,10.,7.,4.]);coef=np.array([1.2,1.,1.5,1.1])
        for gamma in [.25,.5,1.]:
            g,h=robust_derivatives(r,delta,gamma)
            self.assertTrue((h>=gamma).all());self.assertTrue((h<=1).all())
            def loss(x):return float(np.sum(coef*robust_value(x,delta,gamma)))
            eps=1e-3
            for i in range(len(r)):
                plus=r.copy();minus=r.copy();plus[i]+=eps;minus[i]-=eps
                self.assertAlmostEqual((loss(plus)-loss(minus))/(2*eps),coef[i]*g[i],places=5)
                self.assertAlmostEqual((loss(plus)-2*loss(r)+loss(minus))/eps**2,coef[i]*h[i],places=4)
        g,h=robust_derivatives(r,delta,1.)
        np.testing.assert_array_equal(g,r);np.testing.assert_array_equal(h,np.ones(4))
        with self.assertRaises(ValueError):robust_derivatives(r,delta,0.)

    def test_outlier_influence_reduced_without_zero_hessian(self):
        g,h=robust_derivatives(np.array([1.,1000.]),np.array([10.,10.]),.25)
        self.assertLess(g[1],300.)
        self.assertGreaterEqual(h[1],.25)
        self.assertGreater(g[0],.99)

    def test_per_h_scales_handle_degenerate_and_invalid_calibration(self):
        h=np.tile(np.arange(24),5);r=np.ones(len(h))
        delta=horizon_scales(r,h)
        np.testing.assert_array_equal(delta,np.full(24,.001))
        changed=r.copy();changed[0]=float('nan')
        with self.assertRaises(ValueError):horizon_scales(changed,h)
        with self.assertRaises(ValueError):horizon_scales(r[h!=23],h[h!=23])


if __name__=='__main__':unittest.main()
