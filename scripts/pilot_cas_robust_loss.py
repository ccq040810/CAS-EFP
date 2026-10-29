"""Dual MAE/RMSE validation development: bounded robust asymmetric CAS losses.

Preserves Chronos + aligned LightGBM residual + c only in loss. No test access.
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
from scripts.pilot_cas_horizon_aligned import aligned_features,calibrate_costs,cost_coefficients

KINDS=['ordinary','fixed','cas_10','shuffle_10']
GAMMAS={'g1':1.,'g050':.5,'g025':.25}
PROTOCOL='dual_metric_validation_mixed_pseudohuber_asymmetric_CAS_v1'


def robust_derivatives(error,delta,gamma):
    if not 0<gamma<=1:raise ValueError('Positive quadratic fraction required')
    error=np.asarray(error,dtype=float);delta=np.asarray(delta,dtype=float)
    if not np.isfinite(delta).all() or (delta<=0).any():raise ValueError('Invalid training-only scale')
    q=np.hypot(1.,error/delta)
    return gamma*error+(1-gamma)*error/q,gamma+(1-gamma)/(q*q*q)


def robust_value(error,delta,gamma):
    error=np.asarray(error,dtype=float);q=np.hypot(1.,error/np.asarray(delta))
    # Stable form of delta^2*(sqrt(1+(r/delta)^2)-1).
    return .5*gamma*error*error+(1-gamma)*error*error/(q+1)


def horizon_scales(error,h):
    error=np.asarray(error,dtype=float);h=np.asarray(h,dtype=int);delta=np.zeros(24)
    if not np.isfinite(error).all():raise ValueError('Nonfinite calibration residuals')
    for step in range(24):
        e=error[h==step]
        if not len(e):raise ValueError('Missing training calibration h')
        delta[step]=max(1.4826*float(np.median(np.abs(e-np.median(e)))),.05*float(np.std(e)),1e-3)
    return delta


def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--external-root',type=Path,default=ROOT/'results/lear_rnn_external_20261029_v2')
    p.add_argument('--confirmation-root',type=Path,default=ROOT/'results/cas_horizon_aligned_confirmation_20261029')
    p.add_argument('--calibration-root',type=Path,default=ROOT/'results/cas_conditional_costs_validation_20261029')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_robust_loss_validation_20261029')
    p.add_argument('--markets',nargs='+',choices=['guangdong','reale_delu'],default=['guangdong'])
    args=p.parse_args()
    if len(set(args.markets))!=len(args.markets):raise ValueError('Duplicate markets')
    for k in ['reference_root','external_root','confirmation_root','calibration_root','output_root']:setattr(args,k,getattr(args,k).resolve())
    args.output_root.mkdir(parents=True,exist_ok=True)
    for source in [args.confirmation_root,args.calibration_root]:
        done=read(source/'completion.json')
        if not done['complete'] or done['test_evaluated']:raise ValueError('Complete validation-only prerequisite required')
    plan=dict(protocol=PROTOCOL,script_sha256=sha(__file__),feature_helper_sha256=sha(ROOT/'scripts/pilot_cas_horizon_aligned.py'),
        roots={k:str(getattr(args,k)) for k in ['reference_root','external_root','confirmation_root','calibration_root']},
        prior_confirmation_sha256=sha(args.confirmation_root/'plan.json'),calibration_plan_sha256=sha(args.calibration_root/'plan.json'),
        gammas=GAMMAS,kinds=KINDS,seeds=[42,43,44],markets=args.markets,windows=[1,2,3],
        scale='per-h MAD*1.4826 of train-only held-out residual; floor=max(.05 residual_sd, .001)',
        calibration='per-h mean robust score direction and day-block-SE soft excess cost<=.5',
        early_stopping='ordinary_validation_RMSE',test_arrays_read=False,
        success='Seek a substantial GD validation MAE improvement while retaining RMSE; report all candidates. DE recipe unchanged. No test-based selection.',
        tree=read(args.confirmation_root/'plan.json')['tree'])
    expected=len(plan['markets'])*len(plan['windows'])*len(plan['seeds'])*len(GAMMAS)*len(KINDS)
    with run_lock(args.output_root):
        if (args.output_root/'plan.json').exists() and read(args.output_root/'plan.json')!=plan:raise ValueError('Plan changed')
        write(args.output_root/'plan.json',plan)
        def study(market,window):
            folder=args.output_root/market/f'window_{window}';folder.mkdir(parents=True,exist_ok=True)
            ref=args.reference_root/market/f'window_{window}';meta=read(ref/'manifest.json')
            ext=args.external_root/'prepared'/market/f'window_{window}';em=read(ext/'manifest.json')
            if meta['rollout_protocol']!=ROLLOUT_PROTOCOL or sha(ref/'features.npz')!=meta['arrays_sha256']:raise ValueError('Reference changed')
            if em['identity']['reference_sha256']!=meta['arrays_sha256'] or sha(ext/'inputs.npz')!=em['arrays_sha256']:raise ValueError('External source changed')
            calibration=args.calibration_root/market/f'window_{window}'
            profile=read(calibration/'confidence_profiles.json')
            if pd.Timestamp(profile['fit_max_target_time'])>=pd.Timestamp(profile['first_calibration_origin']):raise ValueError('Unpurged training probe')
            with np.load(calibration/'training_calibration.npz') as z:
                error=z['residual'];ch=z['h'];days=pd.to_datetime(z['origin_ns'],unit='ns',utc=True).tz_convert('Asia/Shanghai')
            delta=horizon_scales(error,ch)
            data={}
            with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
                for s in ['train','valid']:
                    X=z['X_'+s];y=z['y_'+s];h=e['steps_'+s];ids=e['row_ids_'+s]
                    np.testing.assert_array_equal(y,e['yrows_'+s])
                    if not local_times(meta[s+'_time']).equals(local_times(em[s+'_time'])):raise ValueError('Time mismatch')
                    A=aligned_features(X,e['X_'+s],ids,h,meta,em['covariates'])
                    data[s]=(A,y,X[:,meta['point_index']].copy(),X[:,meta['confidence_index']].astype(float),h)
            A,y,point,c,h=data['train'];V,yv,pv,_,_=data['valid'];center=float(np.mean(y-point,dtype=np.float64));rows=[]
            for label,gamma in GAMMAS.items():
                score,_=robust_derivatives(error,delta[ch],gamma)
                direction,strength,diagnostics=calibrate_costs(score,ch,days)
                write(folder/(label+'_training_calibration.json'),dict(gamma=gamma,delta_by_h=delta.tolist(),horizons=diagnostics,
                    training_calibration_sha256=sha(calibration/'training_calibration.npz'),test_labels_used=False))
                for seed in plan['seeds']:
                    for kind in KINDS:
                        method=label+'_'+kind;dest=folder/method/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
                        if (dest/'validation.json').exists():rows.append(read(dest/'validation.json'));continue
                        if gamma==1:
                            source=args.confirmation_root/market/f'window_{window}'/kind/f'seed_{seed}'
                            old=read(source/'validation.json')
                            if old['prepared_sha256']!=meta['arrays_sha256'] or old['external_sha256']!=em['arrays_sha256']:raise ValueError('Reuse source changed')
                            row=dict(old,method=method,kind=kind,gamma=gamma,reused_prior_confirmation=True,reused_from=str(source))
                            write(dest/'validation.json',row);rows.append(row);continue
                        params=dict(plan['tree'],objective='regression',metric='None',seed=seed,num_threads=4,
                            deterministic=True,force_col_wise=True,verbosity=-1,boost_from_average=False)
                        def objective(pred,dataset):
                            residual=pred-dataset.get_label();g,hh=robust_derivatives(residual,delta[h],gamma)
                            b=cost_coefficients(residual,c,h,direction,strength,kind)
                            return b*g,b*hh
                        def rmse(pred,dataset):return 'rmse',float(np.sqrt(np.mean((pred-dataset.get_label())**2))),False
                        model=lgb.train(dict(params,objective=objective),lgb.Dataset(A,label=y-point-center),num_boost_round=2000,
                            valid_sets=[lgb.Dataset(V,label=yv-pv-center)],valid_names=['valid'],feval=rmse,
                            callbacks=[lgb.early_stopping(100,verbose=False)])
                        pred=pv+center+model.predict(V,num_threads=4);r=pred-yv
                        row=dict(market=market,window=window,seed=seed,method=method,kind=kind,gamma=gamma,
                            valid_mae=float(np.abs(r).mean()),valid_rmse=float(np.sqrt((r*r).mean())),valid_bias=float(r.mean()),
                            best_iteration=model.best_iteration,center=center,prepared_sha256=meta['arrays_sha256'],external_sha256=em['arrays_sha256'],
                            test_evaluated=False,reused_prior_confirmation=False)
                        write(dest/'validation.json',row);model.save_model(str(dest/'model.txt'))
                        pd.DataFrame(dict(time=meta['valid_time'],h=data['valid'][4]+1,y_true=yv,prediction=pred)).to_csv(dest/'predictions_valid.csv',index=False)
                        rows.append(row);print('VALID',row,flush=True)
            return rows
        rows=[]
        with ThreadPoolExecutor(max_workers=min(3,len(plan['markets'])*3)) as pool:
            for f in as_completed([pool.submit(study,m,w) for m in plan['markets'] for w in plan['windows']]):
                rows.extend(f.result());write(args.output_root/'progress.json',dict(completed=len(rows),expected=expected,test_evaluated=False))
        table=pd.DataFrame(rows);table.to_csv(args.output_root/'VALIDATION_ALL.csv',index=False)
        per=table.groupby(['market','method','seed'],as_index=False).agg(valid_mae=('valid_mae','mean'),valid_rmse=('valid_rmse','mean'))
        per.to_csv(args.output_root/'PER_SEED.csv',index=False)
        per.groupby(['market','method'],as_index=False).agg(mae_mean=('valid_mae','mean'),mae_std=('valid_mae','std'),
            rmse_mean=('valid_rmse','mean'),rmse_std=('valid_rmse','std')).to_csv(args.output_root/'VALIDATION_SUMMARY.csv',index=False)
        write(args.output_root/'completion.json',dict(complete=len(rows)==expected,completed=len(rows),expected=expected,test_evaluated=False))


if __name__=='__main__':main()
