"""Frozen asinh-head Guangdong scoring; retain prior DE and external checkpoints.

Previously observed original tests: development comparison, not a fresh holdout.
No model, calibration, seed, date or sample selection is changed during scoring.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha
from scripts.run_cas_all import run_lock
from scripts.run_cas_directional_study import local_times
from scripts.pilot_cas_asinh_heads import full_features,forward,inverse
from scripts.run_cas_final_period import rank_acceptance

METHODS=['ordinary','fixed','cas_10','shuffle_10','tree_only']


def main():
    import lightgbm as lgb
    source=ROOT/'results/cas_asinh_heads_confirmation_20261029'
    old=ROOT/'results/cas_original_windows_saved_score_20261029'
    out=ROOT/'results/cas_asinh_original_windows_score_20261029';out.mkdir(parents=True,exist_ok=True)
    cp=read(source/'plan.json');pilot=Path(cp['pilot_root']);pp=read(pilot/'plan.json')
    assert read(source/'completion.json')['complete'] and read(source/'independent_audit.json')['complete']
    assert read(pilot/'independent_audit.json')['complete'] and read(old/'completion.json')['complete']
    assert cp['cas_profile']==cp['standalone_profile']=='asinh_per_h'
    for name,digest in cp['source_artifact_sha256'].items():assert sha(Path(name))==digest
    hashes={str(source/'plan.json'):sha(source/'plan.json'),str(pilot/'plan.json'):sha(pilot/'plan.json')}
    for w in [1,2,3]:
        for h in range(1,25):
            base=source/'guangdong'/f'window_{w}'/'asinh_per_h'/f'head_{h:02d}'
            recipe=pilot/'guangdong'/f'window_{w}'/'asinh_per_h'/f'head_{h:02d}'/'training_recipe.json'
            hashes[str(recipe)]=sha(recipe)
            for method in METHODS:
                for seed in [42,43,44]:
                    p=base/method/f'seed_{seed}'/'validation.json';r=read(p)
                    hashes[str(p)]=sha(p);hashes[r['model_source']]=sha(Path(r['model_source']))
    old_runs=pd.read_csv(old/'ALL_RUNS.csv')
    reuse=old_runs[(old_runs.market=='reale_delu')|old_runs.method.isin(['chronos_only','lear_adapted','rnn','fusion_absolute'])].copy()
    for r in reuse.itertuples():
        p=old/r.market/f'window_{r.window}'/f'predictions_{r.method}_seed_{r.seed}.csv';hashes[str(p)]=sha(p)
    plan=dict(protocol='frozen_asinh_GD_original_windows_development_score_v1',script_sha256=sha(__file__),
        confirmation_plan_sha256=sha(source/'plan.json'),source_artifact_sha256=hashes,
        methods_gd=METHODS+['chronos_only','lear_adapted','rnn','fusion_absolute'],seeds=[42,43,44],windows=[1,2,3],
        decision='Validation-only selection and three-seed confirmation complete before original test scoring',
        interpretation='Previously observed original tests; development evaluation, not an independent holdout',
        unchanged_de_recipe=True,retrain=False,recalibration=False,expected_records=162)
    with run_lock(out):
        if (out/'plan.json').exists() and read(out/'plan.json')!=plan:raise ValueError('Frozen scoring plan changed')
        write(out/'plan.json',plan)
        def case(w):
            ref=Path(pp['roots']['reference_root'])/'guangdong'/f'window_{w}'
            ext=Path(pp['roots']['external_root'])/'prepared/guangdong'/f'window_{w}'
            meta=read(ref/'manifest.json');em=read(ext/'manifest.json');identity=cp['source_identities'][str(w)]
            assert sha(ref/'features.npz')==identity['reference_arrays_sha256'] and sha(ext/'inputs.npz')==identity['external_arrays_sha256']
            assert sha(ref/'manifest.json')==identity['reference_manifest_sha256'] and sha(ext/'manifest.json')==identity['external_manifest_sha256']
            with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
                X=z['X_test'];y=z['y_test'];h=e['steps_test'];ids=e['row_ids_test']
                np.testing.assert_array_equal(y,e['yrows_test'])
                assert local_times(meta['test_time']).equals(local_times(em['test_time']))
                A,B=full_features(X,e['X_test'],ids,h,meta,em['covariates'])
            rows=[];dest=out/'guangdong'/f'window_{w}';dest.mkdir(parents=True,exist_ok=True)
            for method in METHODS:
                for seed in [42,43,44]:
                    pred=np.full(len(y),np.nan)
                    for step in range(24):
                        vi=np.flatnonzero(h==step)
                        record=read(source/'guangdong'/f'window_{w}'/'asinh_per_h'/f'head_{step+1:02d}'/method/f'seed_{seed}'/'validation.json')
                        model=lgb.Booster(model_file=record['model_source']);features=B if method=='tree_only' else A
                        assert model.num_feature()==features.shape[1]
                        base=0 if method=='tree_only' else forward(X[vi,meta['point_index']],record['target'])
                        pred[vi]=inverse(base+record['center']+model.predict(features[vi],num_iteration=record['best_iteration'],num_threads=2),record['target'])
                    assert np.isfinite(pred).all();r=pred-y
                    row=dict(market='guangdong',window=w,method=method,seed=seed,mae=float(np.abs(r).mean()),
                        rmse=float(np.sqrt(np.mean(r*r))),n_test=len(y),retrained=False,calibration_reestimated=False,source_kind='frozen_asinh_heads')
                    pd.DataFrame(dict(time=meta['test_time'],h=h+1,y_true=y,prediction=pred)).to_csv(dest/f'predictions_{method}_seed_{seed}.csv',index=False)
                    rows.append(row)
            print('SCORED frozen GD',w,flush=True);return rows
        rows=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for result in pool.map(case,[1,2,3]):rows.extend(result)
        # External and DE predictions are copied, not retrained or recalibrated.
        for r in reuse.itertuples():
            filename=f'predictions_{r.method}_seed_{r.seed}.csv'
            pred=pd.read_csv(old/r.market/f'window_{r.window}'/filename)
            ref=Path(pp['roots']['reference_root'])/r.market/f'window_{r.window}';meta=read(ref/'manifest.json')
            with np.load(ref/'features.npz') as z:y=z['y_test']
            np.testing.assert_array_equal(pred.y_true.to_numpy().astype(y.dtype),y)
            assert local_times(pred.time).equals(local_times(meta['test_time']))
            error=pred.prediction.to_numpy(dtype=float)-y
            dest=out/r.market/f'window_{r.window}';dest.mkdir(parents=True,exist_ok=True)
            pred.to_csv(dest/filename,index=False)
            rows.append(dict(market=r.market,window=r.window,method=r.method,seed=r.seed,mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.mean(error*error))),n_test=len(y),retrained=False,calibration_reestimated=False,source_kind='unchanged_original_saved_prediction'))
        assert all(sha(Path(name))==digest for name,digest in hashes.items())
        d=pd.DataFrame(rows);assert len(d)==162 and not d.duplicated(['market','window','method','seed']).any()
        d.to_csv(out/'ALL_RUNS.csv',index=False)
        per=d.groupby(['market','method','seed'],as_index=False).agg(mae=('mae','mean'),rmse=('rmse','mean'));per.to_csv(out/'PER_SEED.csv',index=False)
        summary=per.groupby(['market','method'],as_index=False).agg(mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'),rmse_std=('rmse','std'))
        summary.to_csv(out/'SUMMARY.csv',index=False)
        d.groupby(['market','window','method'],as_index=False).agg(mae=('mae','mean'),rmse=('rmse','mean')).to_csv(out/'PER_WINDOW.csv',index=False)
        ranks,passed=rank_acceptance(summary);pd.DataFrame(ranks).to_csv(out/'MAIN_COMPARISON_RANKS.csv',index=False)
        write(out/'completion.json',dict(complete=True,completed=162,retrained=False,recalibration=False,source_artifacts_unchanged=True,
            qualification_passed=passed,ranked_cells=ranks,interpretation=plan['interpretation']))
        print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
