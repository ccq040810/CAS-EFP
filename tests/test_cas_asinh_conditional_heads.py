from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pilot_cas_asinh_conditional_heads import calibrate_head
from scripts.pilot_cas_conditional_costs import assign_costs


class ConditionalHeadTests(unittest.TestCase):
    def test_opposite_group_bias_and_matched_two_sided_controls(self):
        c=np.repeat([.1,.5,.9],12);h=np.full(36,7);error=np.repeat([-3.,0.,3.],12)
        days=pd.DatetimeIndex(np.tile(pd.date_range('2020-01-01',periods=12).values,3))
        cuts,under,over,rows=calibrate_head(error,c,h,days,c)
        self.assertEqual(len(rows),3);self.assertGreater(over[7,0],under[7,0]);self.assertGreater(under[7,2],over[7,2])
        a,b=assign_costs(c,h,cuts,under,over,'conditional_c')
        for kind in ['conditional_mean','conditional_shuffle']:
            aa,bb=assign_costs(c,h,cuts,under,over,kind)
            self.assertAlmostEqual(aa.mean(),a.mean());self.assertAlmostEqual(bb.mean(),b.mean())
            self.assertTrue(np.all((aa>=1)&(aa<=1.5)));self.assertTrue(np.all((bb>=1)&(bb<=1.5)))

    def test_no_day_support_is_neutral(self):
        c=np.linspace(0,1,30);h=np.zeros(30,dtype=int);days=pd.DatetimeIndex(['2020-01-01']*30)
        _,a,b,_=calibrate_head(np.ones(30),c,h,days,c)
        np.testing.assert_array_equal(a,np.ones((24,3)));np.testing.assert_array_equal(b,np.ones((24,3)))


if __name__=='__main__':unittest.main()
