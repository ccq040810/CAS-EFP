from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_cas_legacy_us_diagnostic import MicrosecondLegacyIndex,reusable_audit
from scripts.run_cas_legacy_diagnostic import LegacyIndex


class MicrosecondDiagnosticTests(unittest.TestCase):
    def test_both_environments_replay_same_old_lookup_without_changing_dates(self):
        index=pd.date_range('2024-01-01',periods=10,freq='h',tz='Asia/Shanghai')
        anchor=index[4]
        self.assertEqual(LegacyIndex(index.as_unit('ns')).searchsorted(anchor),4)
        for unit in ['ns','us']:
            lookup=MicrosecondLegacyIndex(index.as_unit(unit))
            self.assertEqual(lookup.searchsorted(anchor),10)
            self.assertEqual(lookup.index[4],anchor)

    def test_reuse_rejects_causal_or_invalid_history(self):
        r=dict(index_dtype='datetime64[us, Asia/Shanghai]',future_history=True,
               valid_history=True,legacy_position=100,correct_position=4)
        self.assertTrue(reusable_audit(dict(origins=[r])))
        self.assertFalse(reusable_audit(dict(origins=[])))
        for key,value in [('index_dtype','datetime64[ns, Asia/Shanghai]'),
                          ('future_history',False),('valid_history',False)]:
            self.assertFalse(reusable_audit(dict(origins=[dict(r,**{key:value})])))


if __name__=='__main__':unittest.main()
