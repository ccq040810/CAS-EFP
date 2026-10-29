"""Validation-only follow-up: horizon-aligned history and soft training-only bias costs.

Fixed 36 development fits, no test-array access and no automatic test-driven search.
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
from scripts.pilot_cas_optimization import chronological_calibration

CANDIDATES=['ordinary','fixed','cas_03','cas_10','shuffle_03','shuffle_10']
PROTOCOL='validation_only_horizon_aligned_residual_soft_mean_bias_v1'


def aligned_features(X,context,row_ids,steps,meta,covariates):
    """Select d-1/d-2/d-3/d-7 prices at each target h from causal origin curves."""
    row_ids=np.asarray(row_ids,dtype=int);steps=np.asarray(steps,dtype=int)
    if not np.isin(steps,np.arange(24)).all():raise ValueError('Invalid horizon')
    if not np.array_equal(X[:,meta['feature_names'].index('h')].astype(int)-1,steps):
        raise ValueError('Misaligned horizon keys')
    n=len(covariates)
    if context.shape[1]!=96+72*n+9:raise ValueError('Unrecognized external curve layout')
    keep=[i for i in range(X.shape[1]) if i!=meta['confidence_index']]
    point=X[:,meta['point_index']]
    prices=context[row_ids[:,None],4*steps[:,None]+np.arange(4)]
    # Spread features let pooled trees learn corrections without first splitting h
    # to select one of 24 separate curve columns.
    parts=[X[:,keep],prices,prices-point[:,None],prices[:,[0]]-prices[:,[1,3]]]
    for lag_group in [0,1,2]:
        indices=96+(3*steps[:,None]+lag_group)*n+np.arange(n)
        parts.append(context[row_ids[:,None],indices])
    for i,name in enumerate(meta['feature_names']):
        if 'price' in name.lower():parts.append((X[:,i]-point)[:,None])
    return np.column_stack(parts).astype(np.float32)


def calibrate_costs(errors,horizons,origin_days):
    errors=np.asarray(errors,dtype=float);horizons=np.asarray(horizons,dtype=int)
    days=pd.DatetimeIndex(origin_days).normalize()
    direction=np.zeros(24,dtype=int);strength=np.zeros(24);rows=[]
    for h in range(24):
        mask=horizons==h
        if not mask.any():raise ValueError('Missing calibration horizon')
        e=errors[mask];bias=float(e.mean());sd=float(e.std())
        daily=pd.Series(e,index=days[mask]).groupby(level=0).mean()
        # Day blocks account for repeated, overlapping hourly forecast origins.
        # This SE is a shrinkage heuristic, not a formal confidence guarantee.
        se=float(daily.std(ddof=1)/np.sqrt(len(daily))) if len(daily)>1 else float('inf')
        if len(daily)>=7 and np.isfinite(se):
            direction[h]=1 if bias>0 else (-1 if bias<0 else 0)
            strength[h]=min(.5,abs(bias)/(sd+1e-6)*abs(bias)/(abs(bias)+2*se+1e-6))
        rows.append(dict(h=h+1,bias=bias,residual_sd=sd,day_mean_se=se if np.isfinite(se) else None,
            calibration_days=len(daily),calibration_samples=len(e),direction=int(direction[h]),strength=float(strength[h])))
    return direction,strength,rows


def cost_coefficients(error,c,h,direction,strength,kind):
    error=np.asarray(error);h=np.asarray(h,dtype=int);c=np.asarray(c,dtype=float)
    if not np.isfinite(c).all() or ((c<0)|(c>1)).any():raise ValueError('Invalid c')
    if kind=='ordinary':return np.ones_like(error,dtype=float)
    if kind not in CANDIDATES:raise ValueError(kind)
    w=np.ones_like(c)
    if kind!='fixed':
        weight_lambda=.3 if kind.endswith('03') else 1.
        w=1+weight_lambda*(1-c)
        rng=np.random.default_rng(42)
        for step in range(24):
            mask=h==step
            if mask.any():
                w[mask]/=w[mask].mean()
                if kind.startswith('shuffle'):w[mask]=rng.permutation(w[mask])
    focused=-error*direction[h]>0
    return np.where(focused,1+strength[h]*w,1.)


def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--external-root',type=Path,default=ROOT/'results/lear_rnn_external_20261029_v2')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_horizon_aligned_validation_20261029')
    args=p.parse_args()
    for k in ['reference_root','external_root','output_root']:setattr(args,k,getattr(args,k).resolve())
    args.output_root.mkdir(parents=True,exist_ok=True)
    plan=dict(protocol=PROTOCOL,script_sha256=sha(__file__),methods=CANDIDATES,seed=42,
        source_roots={k:str(getattr(args,k)) for k in ['reference_root','external_root']},
        helpers={name:sha(ROOT/'scripts'/name) for name in ['run_mlp_transfer.py','run_cas_directional_study.py','pilot_cas_optimization.py']},
        cases=[(m,w) for m in ['guangdong','reale_delu'] for w in [1,2,3]],test_arrays_read=False,
        selection='ordinary_validation_rmse',probe_rounds=300,probe_fraction=.8,
        strength='min(0.5, abs(mean_bias)/residual_sd * abs(mean_bias)/(abs(mean_bias)+2*day_block_se))',
        weight='1+strength_h * normalized_within_h(1+lambda*(1-c)) on focused side; other side 1',
        tree=dict(learning_rate=.05,num_leaves=63,min_data_in_leaf=40,feature_fraction=.9,bagging_fraction=.8,bagging_freq=5,min_gain_to_split=.08))
    with run_lock(args.output_root):
        if (args.output_root/'plan.json').exists() and read(args.output_root/'plan.json')!=plan:raise ValueError('Plan changed')
        write(args.output_root/'plan.json',plan)
        def study(case):
            market,window=case;folder=args.output_root/market/f'window_{window}';folder.mkdir(parents=True,exist_ok=True)
            ref=args.reference_root/market/f'window_{window}';meta=read(ref/'manifest.json')
            ext=args.external_root/'prepared'/market/f'window_{window}';em=read(ext/'manifest.json')
            if meta['rollout_protocol']!=ROLLOUT_PROTOCOL or sha(ref/'features.npz')!=meta['arrays_sha256']:raise ValueError('Wrong causal reference')
            if em['identity']['reference_sha256']!=meta['arrays_sha256'] or sha(ext/'inputs.npz')!=em['arrays_sha256']:raise ValueError('External source changed')
            arrays={}
            with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
                for s in ['train','valid']:
                    X=z['X_'+s];y=z['y_'+s];h=e['steps_'+s];ids=e['row_ids_'+s]
                    np.testing.assert_array_equal(y,e['yrows_'+s])
                    if not local_times(meta[s+'_time']).equals(local_times(em[s+'_time'])):raise ValueError('Times differ')
                    A=aligned_features(X,e['X_'+s],ids,h,meta,em['covariates'])
                    arrays[s]=(A,y,X[:,meta['point_index']].copy(),X[:,meta['confidence_index']].astype(float),h)
            A,y,point,c,h=arrays['train'];V,yv,pv,_,_=arrays['valid']
            params=dict(plan['tree'],objective='regression',metric='None',seed=42,num_threads=4,deterministic=True,
                force_col_wise=True,verbosity=-1,boost_from_average=False)
            fit,cal=chronological_calibration(pd.to_datetime(meta['train_time']),h+1)
            probe_center=float(np.mean(y[fit]-point[fit],dtype=np.float64))
            probe=lgb.train(params,lgb.Dataset(A[fit],label=y[fit]-point[fit]-probe_center),num_boost_round=300)
            error=y[cal]-point[cal]-probe_center-probe.predict(A[cal],num_threads=4)
            origins=local_times(meta['train_time'])-pd.to_timedelta(h,unit='h')
            direction,strength,diagnostic=calibrate_costs(error,h[cal],origins[cal])
            write(folder/'training_calibration.json',dict(horizons=diagnostic,fit_center=probe_center,fit_samples=int(fit.sum()),
                calibration_samples=int(cal.sum()),fit_max_target_time=str(local_times(meta['train_time'])[fit].max()),
                first_calibration_origin=str(origins[cal].min()),only_training_labels_used=True))
            center=float(np.mean(y-point,dtype=np.float64));rows=[]
            for kind in CANDIDATES:
                dest=folder/kind;dest.mkdir(parents=True,exist_ok=True)
                if (dest/'validation.json').exists():rows.append(read(dest/'validation.json'));continue
                def objective(pred,dataset):
                    r=pred-dataset.get_label();b=cost_coefficients(r,c,h,direction,strength,kind)
                    return b*r,b
                def rmse(pred,dataset):return 'rmse',float(np.sqrt(np.mean((pred-dataset.get_label())**2))),False
                model=lgb.train(dict(params,objective='regression' if kind=='ordinary' else objective),
                    lgb.Dataset(A,label=y-point-center),num_boost_round=2000,
                    valid_sets=[lgb.Dataset(V,label=yv-pv-center)],valid_names=['valid'],feval=rmse,
                    callbacks=[lgb.early_stopping(100,verbose=False)])
                pred=pv+center+model.predict(V,num_threads=4);r=pred-yv
                row=dict(market=market,window=window,method=kind,seed=42,valid_mae=float(np.abs(r).mean()),
                    valid_rmse=float(np.sqrt((r*r).mean())),valid_bias=float(r.mean()),best_iteration=model.best_iteration,
                    center=center,prepared_sha256=meta['arrays_sha256'],external_sha256=em['arrays_sha256'],test_evaluated=False,
                    feature_count=A.shape[1],active_horizons=int((strength>0).sum()))
                write(dest/'validation.json',row);model.save_model(str(dest/'model.txt'))
                pd.DataFrame(dict(time=meta['valid_time'],h=arrays['valid'][4]+1,y_true=yv,prediction=pred)).to_csv(dest/'predictions_valid.csv',index=False)
                rows.append(row);print('VALID',row,flush=True)
            return rows
        rows=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in as_completed([pool.submit(study,case) for case in plan['cases']]):
                rows.extend(f.result());write(args.output_root/'progress.json',dict(completed=len(rows),expected=36,test_evaluated=False))
        table=pd.DataFrame(rows);table.to_csv(args.output_root/'VALIDATION_ALL.csv',index=False)
        table.groupby(['market','method'],as_index=False).agg(valid_mae=('valid_mae','mean'),valid_rmse=('valid_rmse','mean')).to_csv(args.output_root/'VALIDATION_SUMMARY.csv',index=False)
        write(args.output_root/'completion.json',dict(complete=len(rows)==36,completed=len(rows),expected=36,test_evaluated=False))


if __name__=='__main__':main()
