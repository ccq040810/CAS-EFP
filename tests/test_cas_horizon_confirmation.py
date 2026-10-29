from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.confirm_cas_horizon_aligned import standalone_features,METHODS


class ConfirmationTests(unittest.TestCase):
    def test_standalone_tree_cannot_receive_upstream_derived_inputs(self):
        h=np.array([0,23,5,7]);ids=np.array([0,0,1,1]);n=2
        context=np.arange(2*(96+72*n+9),dtype=np.float32).reshape(2,-1)
        X=np.column_stack([h+1,np.arange(4),np.full(4,50.),np.linspace(.1,.9,4),np.full(4,60.)]).astype(np.float32)
        meta=dict(feature_names=['h','local','pred_point','pred_conf','da_price'],point_index=2,confidence_index=3)
        original=standalone_features(X,context,ids,h,meta,['load','da_price'])
        other=X.copy();other[:,2]=-999999;other[:,3]=99999
        np.testing.assert_array_equal(original,standalone_features(other,context,ids,h,meta,['load','da_price']))
        np.testing.assert_array_equal(original[:,3:7],context[ids[:,None],4*h[:,None]+np.arange(4)])
        self.assertEqual(len(METHODS)*3*2*3,108)
        self.assertNotIn('cas_03',METHODS)


if __name__=='__main__':unittest.main()
