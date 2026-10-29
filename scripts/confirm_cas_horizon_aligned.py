"""Frozen 3-seed validation confirmation of lambda=1; no test arrays accessed."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,sys,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha,ROLLOUT_PROTOCOL
from scripts.run_cas_all import run_lock
from scripts.run_cas_directional_study import local_times
from scripts.pilot_cas_horizon_aligned import aligned_features,cost_coefficients

METHODS=['ordinary','fixed','cas_10','shuffle_10','tree_only','fusion_absolute']
PROTOCOL='frozen_horizon_aligned_lambda1_three_seed_validation_v1'


def standalone_features(X,context,ids,h,meta,covariates):
    """Same causal target-h history, with no upstream point/c or point-derived spreads."""
    ids=np.asarray(ids,dtype=int);h=np.asarray(h,dtype=int);n=len(covariates)
    if not np.isin(h,np.arange(24)).all():raise ValueError('Invalid horizons')
    if context.shape[1]!=96+72*n+9:raise ValueError('Invalid curve schema')
    keep=[i for i in range(X.shape[1]) if i not in [meta['point_index'],meta['confidence_index']]]
    prices=context[ids[:,None],4*h[:,None]+np.arange(4)]
    parts=[X[:,keep],prices,prices[:,[0]]-prices[:,[1,3]]]
    for d in [0,1,2]:parts.append(context[ids[:,None],96+(3*h[:,None]+d)*n+np.arange(n)])
    return np.column_stack(parts).astype(np.float32)


def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--external-root',type=Path,default=ROOT/'results/lear_rnn_external_20261029_v2')
    p.add_argument('--pilot-root',type=Path,default=ROOT/'results/cas_horizon_aligned_validation_20261029')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_horizon_aligned_confirmation_20261029')
    args=p.parse_args()
    for k in ['reference_root','external_root','pilot_root','output_root']:setattr(args,k,getattr(args,k).resolve())
    args.output_root.mkdir(parents=True,exist_ok=True)
    pilot_plan=read(args.pilot_root/'plan.json');done=read(args.pilot_root/'completion.json')
    if not done['complete'] or done['completed']!=36 or done['test_evaluated']:raise ValueError('Complete validation-only pilot required')
    plan=dict(protocol=PROTOCOL,script_sha256=sha(__file__),helper_sha256=sha(ROOT/'scripts/pilot_cas_horizon_aligned.py'),
        pilot_plan_sha256=sha(args.pilot_root/'plan.json'),pilot_validation_sha256=sha(args.pilot_root/'VALIDATION_ALL.csv'),
        roots={k:str(getattr(args,k)) for k in ['reference_root','external_root','pilot_root']},
        methods=METHODS,seeds=[42,43,44],cases=pilot_plan['cases'],tree=pilot_plan['tree'],
        calibration='frozen seed42 train-only direction/strength, unchanged across confirmation seeds',
        selection='ordinary_validation_rmse_early_stopping',test_arrays_read=False,weight_lambda=1.)
    with run_lock(args.output_root):
        if (args.output_root/'plan.json').exists() and read(args.output_root/'plan.json')!=plan:raise ValueError('Confirmation plan changed')
        write(args.output_root/'plan.json',plan)
        def study(case):
            market,window=case;folder=args.output_root/market/f'window_{window}';folder.mkdir(parents=True,exist_ok=True)
            ref=args.reference_root/market/f'window_{window}';meta=read(ref/'manifest.json')
            ext=args.external_root/'prepared'/market/f'window_{window}';em=read(ext/'manifest.json')
            if meta['rollout_protocol']!=ROLLOUT_PROTOCOL or sha(ref/'features.npz')!=meta['arrays_sha256']:raise ValueError('Reference changed')
            if em['identity']['reference_sha256']!=meta['arrays_sha256'] or sha(ext/'inputs.npz')!=em['arrays_sha256']:raise ValueError('External source changed')
            pilot=args.pilot_root/market/f'window_{window}';cal=read(pilot/'training_calibration.json')
            rows=cal['horizons'];direction=np.array([r['direction'] for r in rows]);strength=np.array([r['strength'] for r in rows])
            if [r['h'] for r in rows]!=list(range(1,25)):raise ValueError('Invalid calibration')
            write(folder/'frozen_calibration.json',dict(source_sha256=sha(pilot/'training_calibration.json'),**cal))
            data={}
            with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
                for s in ['train','valid']:
                    X=z['X_'+s];y=z['y_'+s];ids=e['row_ids_'+s];h=e['steps_'+s]
                    np.testing.assert_array_equal(y,e['yrows_'+s])
                    if not local_times(meta[s+'_time']).equals(local_times(em[s+'_time'])):raise ValueError('Time mismatch')
                    A=aligned_features(X,e['X_'+s],ids,h,meta,em['covariates'])
                    B=standalone_features(X,e['X_'+s],ids,h,meta,em['covariates'])
                    data[s]=(A,B,y,X[:,meta['point_index']].copy(),X[:,meta['confidence_index']].astype(float),h)
            A,B,y,point,c,h=data['train'];V,W,yv,pv,_,_=data['valid'];results=[]
            for seed in plan['seeds']:
                params=dict(plan['tree'],objective='regression',metric='None',seed=seed,num_threads=4,
                    deterministic=True,force_col_wise=True,verbosity=-1,boost_from_average=False)
                for method in METHODS:
                    dest=folder/method/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
                    if (dest/'validation.json').exists():results.append(read(dest/'validation.json'));continue
                    absolute=method in ['tree_only','fusion_absolute'];base=np.zeros_like(y) if absolute else point
                    bv=np.zeros_like(yv) if absolute else pv
                    center=float(np.mean(y-base,dtype=np.float64));xx=B if method=='tree_only' else A;vv=W if method=='tree_only' else V
                    kind='ordinary' if absolute else method
                    def objective(pred,dataset):
                        error=pred-dataset.get_label();b=cost_coefficients(error,c,h,direction,strength,kind)
                        return b*error,b
                    def rmse(pred,dataset):return 'rmse',float(np.sqrt(np.mean((pred-dataset.get_label())**2))),False
                    reused=seed==42 and not absolute
                    if reused:
                        old=read(pilot/method/'validation.json')
                        if old['prepared_sha256']!=meta['arrays_sha256'] or old['external_sha256']!=em['arrays_sha256'] or old['center']!=center:
                            raise ValueError('Cannot reuse mismatched pilot')
                        shutil.copy2(pilot/method/'model.txt',dest/'model.txt');model=lgb.Booster(model_file=str(dest/'model.txt'))
                        best=old['best_iteration']
                    else:
                        model=lgb.train(dict(params,objective='regression' if kind=='ordinary' else objective),
                            lgb.Dataset(xx,label=y-base-center),num_boost_round=2000,
                            valid_sets=[lgb.Dataset(vv,label=yv-bv-center)],valid_names=['valid'],feval=rmse,
                            callbacks=[lgb.early_stopping(100,verbose=False)])
                        model.save_model(str(dest/'model.txt'));best=model.best_iteration
                    pred=bv+center+model.predict(vv,num_threads=4);error=pred-yv
                    row=dict(market=market,window=window,seed=seed,method=method,valid_mae=float(np.abs(error).mean()),
                        valid_rmse=float(np.sqrt((error*error).mean())),valid_bias=float(error.mean()),best_iteration=best,
                        prepared_sha256=meta['arrays_sha256'],external_sha256=em['arrays_sha256'],center=center,
                        feature_count=xx.shape[1],test_evaluated=False,reused_seed42_pilot=reused)
                    write(dest/'validation.json',row)
                    pd.DataFrame(dict(time=meta['valid_time'],h=data['valid'][5]+1,y_true=yv,prediction=pred)).to_csv(dest/'predictions_valid.csv',index=False)
                    if reused:
                        np.testing.assert_allclose([row['valid_mae'],row['valid_rmse']],[old['valid_mae'],old['valid_rmse']],atol=1e-9)
                    results.append(row);print('VALID',row,flush=True)
            return results
        rows=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in as_completed([pool.submit(study,case) for case in plan['cases']]):
                rows.extend(f.result());write(args.output_root/'progress.json',dict(expected=108,completed=len(rows),test_evaluated=False))
        table=pd.DataFrame(rows);table.to_csv(args.output_root/'VALIDATION_ALL.csv',index=False)
        per=table.groupby(['market','method','seed'],as_index=False).agg(valid_mae=('valid_mae','mean'),valid_rmse=('valid_rmse','mean'))
        per.to_csv(args.output_root/'PER_SEED.csv',index=False)
        per.groupby(['market','method'],as_index=False).agg(mae_mean=('valid_mae','mean'),mae_std=('valid_mae','std'),
            rmse_mean=('valid_rmse','mean'),rmse_std=('valid_rmse','std')).to_csv(args.output_root/'VALIDATION_SUMMARY.csv',index=False)
        write(args.output_root/'completion.json',dict(complete=len(rows)==108,expected=108,completed=len(rows),test_evaluated=False))


if __name__=='__main__':main()
