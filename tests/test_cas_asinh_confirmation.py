from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.confirm_cas_asinh_heads import frozen_costs
from scripts.pilot_cas_horizon_aligned import cost_coefficients


class FrozenHeadConfirmationTests(unittest.TestCase):
    def test_frozen_calibration_is_not_reestimated_or_mutated_by_costs(self):
        recipe=dict(horizons=[dict(h=3,direction=1,strength=.123)])
        d,s=frozen_costs(recipe);before_d=d.copy();before_s=s.copy()
        c=np.array([.1,.4,.8]);h=np.array([2,2,2]);r=np.array([-3.,2.,-1.])
        for method in ['ordinary','fixed','cas_10','shuffle_10']:
            b=cost_coefficients(r,c,h,d,s,method);self.assertTrue((b>=1).all())
        np.testing.assert_array_equal(d,before_d);np.testing.assert_array_equal(s,before_s)
        self.assertEqual(s[2],.123);self.assertEqual(np.count_nonzero(d),1)
        for invalid in [dict(horizons=[]),dict(horizons=[dict(h=3,direction=1,strength=.6)]),
                        dict(horizons=[dict(h=3,direction=1,strength=.1),dict(h=3,direction=-1,strength=.1)])]:
            with self.assertRaises(ValueError):frozen_costs(invalid)


if __name__=='__main__':unittest.main()
