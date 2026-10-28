from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_cas_legacy_diagnostic import LegacyIndex,LegacyFrame,METHODS,PROTOCOL,train_args
from scripts.run_mlp_transfer import job_config,suite_methods,training_weights


class DiagnosticTests(unittest.TestCase):
    def test_literal_legacy_lookup_keeps_environment_unit_difference(self):
        index=pd.date_range('2024-01-01',periods=10,freq='h',tz='Asia/Shanghai')
        anchor=index[4]
        self.assertEqual(LegacyIndex(index.as_unit('ns')).searchsorted(anchor),4)
        self.assertEqual(LegacyIndex(index.as_unit('us')).searchsorted(anchor),10)
        df=pd.DataFrame({'y':np.arange(10.)},index=index.as_unit('us'))
        proxy=LegacyFrame(df)
        pd.testing.assert_frame_equal(proxy.loc[:,['y']],df)
        self.assertEqual(proxy.index.searchsorted(anchor),len(df))

    def test_plan_and_pure_regressor_exclude_all_upstream_inputs(self):
        self.assertEqual(len(METHODS)*3*2*2*3*3,972)
        self.assertEqual(len(suite_methods('mechanism')),7)
        a=SimpleNamespace(output_root=Path('isolated_diagnostic'),source='cas_chronos2',backend='lightgbm',
            market='guangdong',window=1,method='plain_regressor',seed=42,threads=4)
        args=train_args(a)
        manifest=dict(feature_names=['h','pred_y','pred_y_conf'],confidence_index=2,point_index=1,
            direction=1,arrays_sha256='test',upstream=a.source)
        self.assertEqual(job_config(args,manifest,'plain_regressor',42)['feature_names'],['h'])
        self.assertIsNone(training_weights([.2,.8],'plain_regressor',1,42))
        self.assertEqual(args.selection,'legacy')
        self.assertIn('DIAGNOSTIC_ONLY',PROTOCOL)


if __name__=='__main__':unittest.main()
