from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pilot_cas_linear_leaves import fit_transformer,transform


class LinearLeafTests(unittest.TestCase):
    def test_preprocessing_fits_training_only_and_preserves_missing_flags(self):
        train=np.array([[1.,np.nan,5.],[3.,4.,5.],[5.,6.,5.]])
        recipe=fit_transformer(train);before=dict(recipe)
        result=transform(np.array([[1000.,np.nan,5.]]),recipe)
        self.assertTrue(np.isfinite(result).all());self.assertEqual(result.shape,(1,6))
        self.assertEqual(result[0,4],1.);self.assertEqual(recipe,before)
        np.testing.assert_allclose(transform(train,recipe)[:,:3].mean(axis=0),0,atol=1e-7)

    def test_native_custom_agree_and_linear_coefficients_survive_serialization(self):
        import lightgbm as lgb
        rng=np.random.default_rng(42);raw=rng.normal(size=(600,3));X=transform(raw,fit_transformer(raw))
        y=3*raw[:,0]+raw[:,1]+rng.normal(0,.2,len(raw));y-=y.mean()
        params=dict(objective='regression',metric='None',linear_tree=True,linear_lambda=1.,num_leaves=7,
            min_data_in_leaf=20,learning_rate=.1,verbosity=-1,num_threads=2,boost_from_average=False,seed=42)
        native=lgb.train(params,lgb.Dataset(X,label=y),num_boost_round=15)
        def neutral(pred,data):return pred-data.get_label(),np.ones(len(pred))
        custom=lgb.train(dict(params,objective=neutral),lgb.Dataset(X,label=y),num_boost_round=15)
        np.testing.assert_allclose(native.predict(X,num_threads=2),custom.predict(X,num_threads=2),atol=1e-8)
        leaves=[]
        def walk(node):
            if 'leaf_index' in node:leaves.append(node)
            else:walk(node['left_child']);walk(node['right_child'])
        walk(custom.dump_model()['tree_info'][-1]['tree_structure'])
        self.assertTrue(any(len(leaf.get('leaf_coeff',[]))>0 for leaf in leaves))
        restored=lgb.Booster(model_str=custom.model_to_string())
        np.testing.assert_allclose(custom.predict(X,num_threads=2),restored.predict(X,num_threads=2),atol=1e-10)
        confidence=np.linspace(.1,.9,len(y))
        def weighted(pred,data):
            r=pred-data.get_label();b=np.where(r<0,2-confidence,1.)
            return b*r,b
        weighted_model=lgb.train(dict(params,objective=weighted),lgb.Dataset(X,label=y),num_boost_round=15)
        self.assertGreater(float(np.max(np.abs(custom.predict(X,num_threads=2)-weighted_model.predict(X,num_threads=2)))),1e-5)


if __name__=='__main__':unittest.main()
