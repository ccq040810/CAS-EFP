"""Score original W1/W2/W3 using saved models/calibration only; never refit.

Retains the later-period results separately. Original tests have been inspected
in prior development and are not relabelled a new independent holdout.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha,ROLLOUT_PROTOCOL
from scripts.run_cas_all import run_lock
from scripts.run_cas_directional_study import local_times
from scripts.pilot_cas_horizon_aligned import aligned_features
from scripts.confirm_cas_horizon_aligned import standalone_features,METHODS
from scripts.run_cas_final_period import rank_acceptance

PROTOCOL='original_W1_W2_W3_saved_models_test_scoring_only_v1'
EXTERNAL=['chronos_only','lear_adapted','rnn']


def score_case(args,market,window):
    import lightgbm as lgb
    ref=args.reference_root/market/f'window_{window}';meta=read(ref/'manifest.json')
    ext=args.external_root/'prepared'/market/f'window_{window}';em=read(ext/'manifest.json')
    if meta['rollout_protocol']!=ROLLOUT_PROTOCOL or sha(ref/'features.npz')!=meta['arrays_sha256']:raise ValueError('Reference changed')
    if sha(ext/'inputs.npz')!=em['arrays_sha256'] or em['identity']['reference_sha256']!=meta['arrays_sha256']:raise ValueError('External source changed')
    with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
        X=z['X_test'];y=z['y_test'];h=e['steps_test'];ids=e['row_ids_test']
        np.testing.assert_array_equal(y,e['yrows_test'])
        np.testing.assert_array_equal(X[:,meta['feature_names'].index('h')].astype(int),h+1)
        if not local_times(meta['test_time']).equals(local_times(em['test_time'])):raise ValueError('Time mismatch')
        A=aligned_features(X,e['X_test'],ids,h,meta,em['covariates'])
        B=standalone_features(X,e['X_test'],ids,h,meta,em['covariates'])
    dest=args.output_root/market/f'window_{window}';dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    def score(method,seed,pred,source):
        pred=np.asarray(pred,dtype=float)
        if pred.shape!=y.shape or not np.isfinite(pred).all():raise ValueError('Invalid prediction')
        error=pred-y
        row=dict(market=market,window=window,method=method,seed=seed,mae=float(np.abs(error).mean()),
            rmse=float(np.sqrt((error*error).mean())),bias=float(error.mean()),n_test=len(y),
            reference_sha256=meta['arrays_sha256'],external_sha256=em['arrays_sha256'],
            model_source=str(source),retrained=False,calibration_reestimated=False)
        pd.DataFrame(dict(time=meta['test_time'],h=h+1,y_true=y,prediction=pred)).to_csv(dest/f'predictions_{method}_seed_{seed}.csv',index=False)
        rows.append(row)
    for seed in [42,43,44]:
        for method in METHODS:
            source=args.model_root/market/f'window_{window}'/method/f'seed_{seed}'
            record=read(source/'validation.json')
            if record['prepared_sha256']!=meta['arrays_sha256'] or record['external_sha256']!=em['arrays_sha256']:raise ValueError('Saved model source mismatch')
            model=lgb.Booster(model_file=str(source/'model.txt'))
            features=B if method=='tree_only' else A
            if model.num_feature()!=features.shape[1]:raise ValueError('Model feature schema differs')
            base=np.zeros_like(y) if method in ['tree_only','fusion_absolute'] else X[:,meta['point_index']].copy()
            pred=base+record['center']+model.predict(features,num_iteration=record['best_iteration'],num_threads=4)
            score(method,seed,pred,source/'model.txt')
        score('chronos_only',seed,X[:,meta['point_index']],ref/'features.npz')
        for method in ['lear_adapted','rnn']:
            source=args.external_root/'training'/method/market/f'window_{window}'/f'seed_{seed}'
            record=read(source/'metrics.json');old=pd.read_csv(source/'predictions_test.csv')
            if record['prepared_sha256']!=em['arrays_sha256']:raise ValueError('External baseline source differs')
            np.testing.assert_allclose(old.y_true,y,rtol=1e-7,atol=1e-6)
            if not local_times(old.time).equals(local_times(meta['test_time'])):raise ValueError('External test keys differ')
            score(method,seed,old.prediction.to_numpy(),source/'predictions_test.csv')
    frame=pd.DataFrame(rows);frame.to_csv(dest/'ALL_RUNS.csv',index=False)
    write(dest/'completion.json',dict(complete=len(rows)==27,completed=len(rows),expected=27,retrained=False,calibration_reestimated=False))
    print('SCORED',market,window,len(rows),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--model-root',type=Path,default=ROOT/'results/cas_horizon_aligned_confirmation_20261029')
    p.add_argument('--external-root',type=Path,default=ROOT/'results/lear_rnn_external_20261029_v2')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_original_windows_saved_score_20261029')
    args=p.parse_args()
    for k in ['reference_root','model_root','external_root','output_root']:setattr(args,k,getattr(args,k).resolve())
    args.output_root.mkdir(parents=True,exist_ok=True)
    with run_lock(args.output_root):
        done=read(args.model_root/'completion.json')
        if not done['complete'] or done['completed']!=108:raise ValueError('Saved three-window models incomplete')
        source_plan=read(args.model_root/'plan.json')
        if sha(ROOT/'scripts/pilot_cas_horizon_aligned.py')!=source_plan['helper_sha256']:raise ValueError('Original feature helper changed')
        if sha(ROOT/'scripts/confirm_cas_horizon_aligned.py')!=source_plan['script_sha256']:raise ValueError('Original model training script changed')
        sources={}
        for m,w in source_plan['cases']:
            for seed in [42,43,44]:
                for method in METHODS:
                    folder=args.model_root/m/f'window_{w}'/method/f'seed_{seed}'
                    for name in ['model.txt','validation.json']:sources[str(folder/name)]=sha(folder/name)
            calibration=args.model_root/m/f'window_{w}'/'frozen_calibration.json'
            sources[str(calibration)]=sha(calibration)
        plan=dict(protocol=PROTOCOL,script_sha256=sha(__file__),source_confirmation_plan_sha256=sha(args.model_root/'plan.json'),
            roots={k:str(getattr(args,k)) for k in ['reference_root','model_root','external_root']},
            cases=source_plan['cases'],seeds=[42,43,44],methods=METHODS+EXTERNAL,
            existing_models_and_calibration_sha256=sources,retrain=False,reestimate_calibration=False,
            selection='Exact saved checkpoints selected on original validation RMSE; no new selection',
            interpretation='Original-period development comparison; fourth-window results retained separately; no new independent holdout claim')
        if (args.output_root/'plan.json').exists() and read(args.output_root/'plan.json')!=plan:raise ValueError('Scoring plan changed')
        write(args.output_root/'plan.json',plan)
        rows=[];failures=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(score_case,args,m,w):(m,w) for m,w in source_plan['cases']}
            for future in as_completed(futures):
                try:rows.extend(future.result())
                except Exception as exc:failures.append(dict(case=futures[future],error=str(exc)))
                write(args.output_root/'progress.json',dict(completed=len(rows),expected=162,failures=failures))
        if failures:
            write(args.output_root/'completion.json',dict(complete=False,completed=len(rows),expected=162,failures=failures))
            raise RuntimeError('Scoring failed; see completion.json')
        # Verify that no source model, checkpoint or calibration was modified.
        if any(sha(Path(path))!=value for path,value in sources.items()):raise ValueError('A source artifact changed during scoring')
        frame=pd.DataFrame(rows)
        if len(frame)!=162 or frame.duplicated(['market','window','method','seed']).any():raise ValueError('Incomplete/duplicate scoring matrix')
        frame.to_csv(args.output_root/'ALL_RUNS.csv',index=False)
        per=frame.groupby(['market','method','seed'],as_index=False).agg(mae=('mae','mean'),rmse=('rmse','mean'))
        per.to_csv(args.output_root/'PER_SEED.csv',index=False)
        summary=per.groupby(['market','method'],as_index=False).agg(mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'),rmse_std=('rmse','std'))
        summary.to_csv(args.output_root/'SUMMARY.csv',index=False)
        frame.groupby(['market','window','method'],as_index=False).agg(mae=('mae','mean'),rmse=('rmse','mean')).to_csv(args.output_root/'PER_WINDOW.csv',index=False)
        ranks,passed=rank_acceptance(summary);pd.DataFrame(ranks).to_csv(args.output_root/'MAIN_COMPARISON_RANKS.csv',index=False)
        write(args.output_root/'completion.json',dict(complete=True,completed=162,expected=162,retrained=False,
            calibration_reestimated=False,source_artifacts_unchanged=True,qualification_passed=passed,ranked_cells=ranks))
        print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
