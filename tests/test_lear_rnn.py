from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_lear_rnn import build_inputs,fit_scaler,pack_targets


class ExternalInputsTests(unittest.TestCase):
    def test_filtered_horizon_rows_preserve_exact_targets(self):
        a=pd.Timestamp('2024-01-01',tz='Asia/Shanghai')
        steps=np.array([0,2,0,1,2]);values=np.array([1.,3.,4.,5.,6.])
        times=pd.DatetimeIndex([a,a+pd.Timedelta(hours=2),a+pd.Timedelta(hours=1),a+pd.Timedelta(hours=2),a+pd.Timedelta(hours=3)])
        origins,row_ids,Y=pack_targets(times,steps,values)
        self.assertEqual(len(origins),2)
        self.assertTrue(np.isnan(Y[0,1]))
        np.testing.assert_array_equal(Y[row_ids,steps],values)
    def test_no_post_origin_target_and_equal_time_units(self):
        index=pd.date_range('2024-01-01',periods=400,freq='h',tz='Asia/Shanghai')
        df=pd.DataFrame({'y':np.arange(400.),'known':np.arange(400.)+1000},index=index)
        origin=index[200]
        before=build_inputs(df,[origin],'y',['known'])
        self.assertEqual(before[0][0,-1,0],199)
        self.assertEqual(before[1].shape,(1,96+72+2+7))
        altered=df.copy();altered.loc[origin:,'y']=-99999
        for a,b in zip(before,build_inputs(altered,[origin],'y',['known'])):np.testing.assert_array_equal(a,b)
        other=df.copy();other.index=other.index.as_unit('us')
        for a,b in zip(before,build_inputs(other,[origin],'y',['known'])):np.testing.assert_array_equal(a,b)
        # First lag feature is price at origin-24h, followed by -48/-72/-168.
        np.testing.assert_array_equal(before[1][0,:4],[176,152,128,32])

    def test_missing_history_fills_only_from_past(self):
        index=pd.date_range('2024-01-01',periods=250,freq='h',tz='Asia/Shanghai')
        df=pd.DataFrame({'y':np.arange(250.),'known':np.arange(250.)},index=index)
        df.loc[index[199],'y']=np.nan
        seq,_,_=build_inputs(df,[index[200]],'y',['known'])
        self.assertEqual(seq[0,-1,0],198)
        df.loc[:index[50],'y']=np.nan
        with self.assertRaises(ValueError):build_inputs(df,[index[200]],'y',['known'])

    def test_invariant_scaler_handles_constant_features(self):
        X=np.array([[1.,2.],[3.,2.],[5.,2.]])
        center,scale=fit_scaler(X)
        np.testing.assert_allclose(np.sinh(np.arcsinh((X-center)/scale))*scale+center,X)
        self.assertTrue((scale>0).all())


if __name__=='__main__':unittest.main()
