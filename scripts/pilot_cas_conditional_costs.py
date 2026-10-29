"""Fixed 3-bin confidence-conditioned bias costs, training/validation only.

108 development records: reuse 54 prior fits; train 54 matched new fits.
No test arrays indexed, no test-driven parameter or subset selection.
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
from scripts.pilot_cas_horizon_aligned import aligned_features

METHODS=['ordinary','fixed','cas_10','conditional_c','conditional_mean','conditional_shuffle']
PROTOCOL='train_only_confidence_tercile_conditional_bias_cost_three_seed_v1'


def error_statistics(errors,days):
    errors=np.asarray(errors,dtype=float)
    daily=pd.Series(errors,index=pd.DatetimeIndex(days).normalize()).groupby(level=0).mean()
    se=float(daily.std(ddof=1)/np.sqrt(len(daily))) if len(daily)>1 else None
    return dict(bias=float(errors.mean()),sd=float(errors.std()),se=se,days=len(daily),n=len(errors))


def conditional_profile(errors,c,h,days,fit_c,fit_h):
    errors=np.asarray(errors);c=np.asarray(c);h=np.asarray(h,dtype=int)
    fit_c=np.asarray(fit_c);fit_h=np.asarray(fit_h,dtype=int);days=pd.DatetimeIndex(days)
    cuts=np.zeros((24,2));under=np.ones((24,3));over=np.ones((24,3));rows=[]
    for step in range(24):
        mask=h==step
        if not mask.any() or not (fit_h==step).any():raise ValueError('Missing horizon')
        cuts[step]=np.quantile(fit_c[fit_h==step],[1/3,2/3])
        groups=np.searchsorted(cuts[step],c[mask],side='right')
        global_stats=error_statistics(errors[mask],days[mask])
        for band in range(3):
            selected=groups==band
            stats=error_statistics(errors[mask][selected],days[mask][selected]) if selected.any() else global_stats
            n=int(selected.sum());rho=n/(n+100.)
            bias=rho*stats['bias']+(1-rho)*global_stats['bias']
            strength=0.
            if n>=10 and stats['days']>=7 and stats['se'] is not None:
                strength=min(.5,abs(bias)/(stats['sd']+1e-6)*abs(bias)/(abs(bias)+2*stats['se']+1e-6))
            if bias>0:under[step,band]+=strength
            elif bias<0:over[step,band]+=strength
            rows.append(dict(h=step+1,confidence_band=band,cut_low=float(cuts[step,0]),cut_high=float(cuts[step,1]),
                samples=n,calibration_days=stats['days'],bias=stats['bias'],horizon_bias=global_stats['bias'],
                shrunken_bias=bias,pseudo_sample_shrinkage=rho,residual_sd=stats['sd'],day_mean_se=stats['se'],
                under_coefficient=float(under[step,band]),over_coefficient=float(over[step,band])))
    return cuts,under,over,rows


def assign_costs(c,h,cuts,under,over,kind):
    c=np.asarray(c,dtype=float);h=np.asarray(h,dtype=int)
    if not np.isfinite(c).all() or ((c<0)|(c>1)).any():raise ValueError('Invalid confidence')
    positive=np.ones(len(c));negative=np.ones(len(c));rng=np.random.default_rng(42)
    for step in range(24):
        m=h==step
        if not m.any():continue
        band=np.searchsorted(cuts[step],c[m],side='right')
        a=under[step,band];b=over[step,band]
        if kind=='conditional_mean':a=np.full_like(a,a.mean());b=np.full_like(b,b.mean())
        elif kind=='conditional_shuffle':
            order=rng.permutation(len(a));a=a[order];b=b[order]
        elif kind!='conditional_c':raise ValueError(kind)
        positive[m]=a;negative[m]=b
    return positive,negative


def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--external-root',type=Path,default=ROOT/'results/lear_rnn_external_20261029_v2')
    p.add_argument('--confirmation-root',type=Path,default=ROOT/'results/cas_horizon_aligned_confirmation_20261029')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_conditional_costs_validation_20261029')
    args=p.parse_args()
    for k in ['reference_root','external_root','confirmation_root','output_root']:setattr(args,k,getattr(args,k).resolve())
    args.output_root.mkdir(parents=True,exist_ok=True)
    previous=read(args.confirmation_root/'completion.json')
    if not previous['complete'] or previous['completed']!=108 or previous['test_evaluated']:raise ValueError('Prior confirmation incomplete')
    plan=dict(protocol=PROTOCOL,script_sha256=sha(__file__),feature_helper_sha256=sha(ROOT/'scripts/pilot_cas_horizon_aligned.py'),
        previous_plan_sha256=sha(args.confirmation_root/'plan.json'),previous_validation_sha256=sha(args.confirmation_root/'VALIDATION_ALL.csv'),
        roots={k:str(getattr(args,k)) for k in ['reference_root','external_root','confirmation_root']},
        methods=METHODS,seeds=[42,43,44],markets=['guangdong','reale_delu'],windows=[1,2,3],
        bins=3,bin_cutoffs='early_training_confidence_terciles_within_h',pseudo_samples=100,
        max_excess_cost=.5,min_calibration_samples=10,min_calibration_days=7,
        probe_rounds=300,test_arrays_read=False,calibration='training_80/20_purged_target_time',
        tree=read(args.confirmation_root/'plan.json')['tree'])
    with run_lock(args.output_root):
        if (args.output_root/'plan.json').exists() and read(args.output_root/'plan.json')!=plan:raise ValueError('Plan changed')
        write(args.output_root/'plan.json',plan)
        def study(market,window):
            folder=args.output_root/market/f'window_{window}';folder.mkdir(parents=True,exist_ok=True)
            ref=args.reference_root/market/f'window_{window}';meta=read(ref/'manifest.json')
            ext=args.external_root/'prepared'/market/f'window_{window}';em=read(ext/'manifest.json')
            if meta['rollout_protocol']!=ROLLOUT_PROTOCOL or sha(ref/'features.npz')!=meta['arrays_sha256']:raise ValueError('Reference changed')
            if em['identity']['reference_sha256']!=meta['arrays_sha256'] or sha(ext/'inputs.npz')!=em['arrays_sha256']:raise ValueError('External source changed')
            data={}
            with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
                for s in ['train','valid']:
                    X=z['X_'+s];y=z['y_'+s];h=e['steps_'+s];ids=e['row_ids_'+s]
                    np.testing.assert_array_equal(y,e['yrows_'+s])
                    if not local_times(meta[s+'_time']).equals(local_times(em[s+'_time'])):raise ValueError('Times differ')
                    A=aligned_features(X,e['X_'+s],ids,h,meta,em['covariates'])
                    data[s]=(A,y,X[:,meta['point_index']].copy(),X[:,meta['confidence_index']].astype(float),h)
            A,y,point,c,h=data['train'];V,yv,pv,_,_=data['valid']
            params=dict(plan['tree'],objective='regression',metric='None',seed=42,num_threads=4,deterministic=True,
                force_col_wise=True,verbosity=-1,boost_from_average=False)
            fit,cal=chronological_calibration(pd.to_datetime(meta['train_time']),h+1)
            probe_center=float(np.mean(y[fit]-point[fit],dtype=np.float64))
            probe=lgb.train(params,lgb.Dataset(A[fit],label=y[fit]-point[fit]-probe_center),num_boost_round=300)
            error=y[cal]-point[cal]-probe_center-probe.predict(A[cal],num_threads=4)
            origins=local_times(meta['train_time'])-pd.to_timedelta(h,unit='h')
            cuts,under,over,profiles=conditional_profile(error,c[cal],h[cal],origins[cal],c[fit],h[fit])
            write(folder/'confidence_profiles.json',dict(profiles=profiles,fit_max_target_time=str(local_times(meta['train_time'])[fit].max()),
                  first_calibration_origin=str(origins[cal].min()),only_training_labels_used=True))
            pd.DataFrame(profiles).to_csv(folder/'confidence_profiles.csv',index=False)
            # Preserve causal training calibration observations for independent diagnostics.
            np.savez_compressed(folder/'training_calibration.npz',residual=error,confidence=c[cal],h=h[cal],
                origin_ns=origins[cal].as_unit('ns').asi8,cuts=cuts,under=under,over=over)
            center=float(np.mean(y-point,dtype=np.float64));rows=[]
            for seed in plan['seeds']:
                for method in METHODS:
                    dest=folder/method/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
                    if (dest/'validation.json').exists():rows.append(read(dest/'validation.json'));continue
                    reused=method in ['ordinary','fixed','cas_10']
                    if reused:
                        source=args.confirmation_root/market/f'window_{window}'/method/f'seed_{seed}'
                        row=read(source/'validation.json')
                        if row['prepared_sha256']!=meta['arrays_sha256'] or row['external_sha256']!=em['arrays_sha256']:raise ValueError('Reuse source differs')
                        row=dict(row,reused_prior_confirmation=True,reused_from=str(source))
                        write(dest/'validation.json',row);rows.append(row);continue
                    a,b=assign_costs(c,h,cuts,under,over,method)
                    def objective(pred,dataset):
                        r=pred-dataset.get_label();coeff=np.where(r<0,a,b)
                        return coeff*r,coeff
                    def rmse(pred,dataset):return 'rmse',float(np.sqrt(np.mean((pred-dataset.get_label())**2))),False
                    model=lgb.train(dict(params,seed=seed,objective=objective),lgb.Dataset(A,label=y-point-center),num_boost_round=2000,
                        valid_sets=[lgb.Dataset(V,label=yv-pv-center)],valid_names=['valid'],feval=rmse,
                        callbacks=[lgb.early_stopping(100,verbose=False)])
                    pred=pv+center+model.predict(V,num_threads=4);r=pred-yv
                    row=dict(market=market,window=window,method=method,seed=seed,valid_mae=float(np.abs(r).mean()),
                        valid_rmse=float(np.sqrt((r*r).mean())),valid_bias=float(r.mean()),best_iteration=model.best_iteration,
                        prepared_sha256=meta['arrays_sha256'],external_sha256=em['arrays_sha256'],center=center,
                        test_evaluated=False,reused_prior_confirmation=False)
                    write(dest/'validation.json',row);model.save_model(str(dest/'model.txt'))
                    pd.DataFrame(dict(time=meta['valid_time'],h=data['valid'][4]+1,y_true=yv,prediction=pred)).to_csv(dest/'predictions_valid.csv',index=False)
                    rows.append(row);print('VALID',row,flush=True)
            return rows
        rows=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in as_completed([pool.submit(study,m,w) for m in plan['markets'] for w in plan['windows']]):
                rows.extend(f.result());write(args.output_root/'progress.json',dict(completed=len(rows),expected=108,test_evaluated=False))
        table=pd.DataFrame(rows);table.to_csv(args.output_root/'VALIDATION_ALL.csv',index=False)
        per=table.groupby(['market','method','seed'],as_index=False).agg(valid_mae=('valid_mae','mean'),valid_rmse=('valid_rmse','mean'))
        per.to_csv(args.output_root/'PER_SEED.csv',index=False)
        per.groupby(['market','method'],as_index=False).agg(mae_mean=('valid_mae','mean'),mae_std=('valid_mae','std'),
            rmse_mean=('valid_rmse','mean'),rmse_std=('valid_rmse','std')).to_csv(args.output_root/'VALIDATION_SUMMARY.csv',index=False)
        write(args.output_root/'completion.json',dict(complete=len(rows)==108,completed=len(rows),expected=108,test_evaluated=False))


if __name__=='__main__':main()
