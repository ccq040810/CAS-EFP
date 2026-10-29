"""Guangdong validation pilot: LightGBM linear leaves with matched CAS controls.

Three fixed configurations, seed42 screening only. All inputs fit on training
data, no test arrays accessed. Official linear-tree/custom-Newton support is
verified by unit tests before running. No test-driven parameter search.
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
from scripts.pilot_cas_horizon_aligned import aligned_features,calibrate_costs,cost_coefficients
from scripts.confirm_cas_horizon_aligned import standalone_features
from scripts.pilot_cas_robust_loss import robust_derivatives,horizon_scales

CONFIGS={'linear_l2_r1':dict(linear_lambda=1.,gamma=1.),
         'linear_l2_r100':dict(linear_lambda=100.,gamma=1.),
         'linear_mix_r1':dict(linear_lambda=1.,gamma=.5)}
METHODS=['ordinary','fixed','cas_10','shuffle_10','tree_only']
PROTOCOL='validation_only_linear_leaf_LightGBM_CAS_fixed_three_config_v1'


def fit_transformer(X):
    X=np.asarray(X,dtype=float);finite=np.isfinite(X)
    safe=np.where(finite,X,np.nan)
    med=np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0. for column in safe.T])
    filled=np.where(finite,X,med);mean=filled.mean(axis=0);scale=filled.std(axis=0)
    scale=np.where(scale>1e-8,scale,1.)
    return dict(median=med.tolist(),mean=mean.tolist(),scale=scale.tolist(),append_missing_indicators=True)


def transform(X,recipe):
    X=np.asarray(X,dtype=float);missing=~np.isfinite(X)
    filled=np.where(missing,np.asarray(recipe['median']),X)
    normalized=(filled-np.asarray(recipe['mean']))/np.asarray(recipe['scale'])
    return np.column_stack([normalized,missing.astype(float)]).astype(np.float32)


def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--external-root',type=Path,default=ROOT/'results/lear_rnn_external_20261029_v2')
    p.add_argument('--confirmation-root',type=Path,default=ROOT/'results/cas_horizon_aligned_confirmation_20261029')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_linear_leaf_validation_20261029')
    args=p.parse_args()
    for k in ['reference_root','external_root','confirmation_root','output_root']:setattr(args,k,getattr(args,k).resolve())
    args.output_root.mkdir(parents=True,exist_ok=True)
    tree=dict(read(args.confirmation_root/'plan.json')['tree'],linear_tree=True,tree_learner='serial',device_type='cpu')
    plan=dict(protocol=PROTOCOL,script_sha256=sha(__file__),lightgbm_version=lgb.__version__,configs=CONFIGS,methods=METHODS,
        roots={k:str(getattr(args,k)) for k in ['reference_root','external_root','confirmation_root']},
        helper_sha256={p:sha(ROOT/'scripts'/p) for p in ['pilot_cas_horizon_aligned.py','confirm_cas_horizon_aligned.py','pilot_cas_robust_loss.py']},
        market='guangdong',windows=[1,2,3],seeds=[42],tree=tree,expected=45,test_arrays_read=False,
        preprocessing='Training-only finite median imputation, mean/std normalization and explicit missing indicators; standalone excludes upstream-derived columns',
        probe='300-round linear L2, fit80% training only; independent probe preprocessing; purged target overlap',
        selection='CAS and standalone configs each selected by minimum max(validation_MAE/54.60651697764482,validation_RMSE/94.56828737273588), ties lower RMSE then config name',
        early_stopping='ordinary_validation_RMSE',confirmation='Only frozen selected configurations get seeds43/44 if useful validation improvement; no test selection')
    with run_lock(args.output_root):
        if (args.output_root/'plan.json').exists() and read(args.output_root/'plan.json')!=plan:raise ValueError('Pilot plan changed')
        write(args.output_root/'plan.json',plan)
        def study(w):
            ref=args.reference_root/'guangdong'/f'window_{w}';meta=read(ref/'manifest.json')
            ext=args.external_root/'prepared/guangdong'/f'window_{w}';em=read(ext/'manifest.json')
            if meta['rollout_protocol']!=ROLLOUT_PROTOCOL or sha(ref/'features.npz')!=meta['arrays_sha256']:raise ValueError('Wrong reference')
            if em['identity']['reference_sha256']!=meta['arrays_sha256'] or sha(ext/'inputs.npz')!=em['arrays_sha256']:raise ValueError('External source changed')
            data={}
            with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
                for s in ['train','valid']:
                    X=z['X_'+s];y=z['y_'+s];h=e['steps_'+s];ids=e['row_ids_'+s]
                    np.testing.assert_array_equal(y,e['yrows_'+s])
                    if not local_times(meta[s+'_time']).equals(local_times(em[s+'_time'])):raise ValueError('Time mismatch')
                    data[s]=(aligned_features(X,e['X_'+s],ids,h,meta,em['covariates']),
                        standalone_features(X,e['X_'+s],ids,h,meta,em['covariates']),y,X[:,meta['point_index']].copy(),X[:,meta['confidence_index']].astype(float),h)
            raw,plain,y,point,c,h=data['train'];raw_v,plain_v,yv,pv,_,_=data['valid']
            recipe=fit_transformer(raw);plain_recipe=fit_transformer(plain)
            A=transform(raw,recipe);V=transform(raw_v,recipe);B=transform(plain,plain_recipe);W=transform(plain_v,plain_recipe)
            folder=args.output_root/'guangdong'/f'window_{w}';folder.mkdir(parents=True,exist_ok=True)
            write(folder/'preprocessing.json',dict(cas=recipe,standalone=plain_recipe))
            fit,cal=chronological_calibration(pd.to_datetime(meta['train_time']),h+1)
            origins=local_times(meta['train_time'])-pd.to_timedelta(h,unit='h')
            probe_recipe=fit_transformer(raw[fit]);probe_A=transform(raw[fit],probe_recipe);probe_V=transform(raw[cal],probe_recipe)
            probes={};rows=[]
            for name,config in CONFIGS.items():
                gamma=config['gamma'];ridge=config['linear_lambda'];dest=folder/name;dest.mkdir(parents=True,exist_ok=True)
                params=dict(tree,linear_lambda=ridge,objective='regression',metric='None',seed=42,num_threads=4,
                    deterministic=True,force_col_wise=True,verbosity=-1,boost_from_average=False)
                if ridge not in probes:
                    center_probe=float(np.mean(y[fit]-point[fit],dtype=np.float64))
                    probe=lgb.train(params,lgb.Dataset(probe_A,label=y[fit]-point[fit]-center_probe),num_boost_round=300)
                    residual=y[cal]-point[cal]-center_probe-probe.predict(probe_V,num_threads=4)
                    probes[ridge]=(residual,horizon_scales(residual,h[cal]))
                residual,delta=probes[ridge]
                score,_=robust_derivatives(residual,delta[h[cal]],gamma)
                direction,strength,diagnostics=calibrate_costs(score,h[cal],origins[cal])
                write(dest/'training_calibration.json',dict(config=config,delta_by_h=delta.tolist(),horizons=diagnostics,
                    probe_preprocessing=probe_recipe,fit_max_target_time=str(local_times(meta['train_time'])[fit].max()),first_calibration_origin=str(origins[cal].min())))
                for method in METHODS:
                    job=dest/method;job.mkdir(parents=True,exist_ok=True)
                    if (job/'validation.json').exists():rows.append(read(job/'validation.json'));continue
                    absolute=method=='tree_only';xx=B if absolute else A;vv=W if absolute else V
                    base=np.zeros_like(y) if absolute else point;bv=np.zeros_like(yv) if absolute else pv
                    center=float(np.mean(y-base,dtype=np.float64));kind='ordinary' if absolute else method
                    current_delta=delta
                    if absolute and gamma<1:
                        bp_recipe=fit_transformer(plain[fit]);bp_center=float(np.mean(y[fit],dtype=np.float64))
                        bp=lgb.train(params,lgb.Dataset(transform(plain[fit],bp_recipe),label=y[fit]-bp_center),num_boost_round=300)
                        bp_error=y[cal]-bp_center-bp.predict(transform(plain[cal],bp_recipe),num_threads=4)
                        current_delta=horizon_scales(bp_error,h[cal])
                        write(job/'training_scale.json',dict(delta_by_h=current_delta.tolist(),source='standalone linear L2 train-only probe'))
                    def objective(pred,dataset):
                        r=pred-dataset.get_label();g,hh=robust_derivatives(r,current_delta[h],gamma)
                        b=cost_coefficients(r,c,h,direction,strength,kind)
                        return b*g,b*hh
                    native=kind=='ordinary' and gamma==1
                    history={}
                    def rmse(pred,dataset):return 'rmse',float(np.sqrt(np.mean((pred-dataset.get_label())**2))),False
                    model=lgb.train(dict(params,objective='regression' if native else objective),lgb.Dataset(xx,label=y-base-center),num_boost_round=2000,
                        valid_sets=[lgb.Dataset(vv,label=yv-bv-center)],valid_names=['valid'],feval=rmse,
                        callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(100),lgb.record_evaluation(history)])
                    pred=bv+center+model.predict(vv,num_threads=4);error=pred-yv
                    if not np.isfinite(pred).all():raise ValueError('Nonfinite linear-leaf predictions')
                    row=dict(market='guangdong',window=w,seed=42,configuration=name,method=method,gamma=gamma,linear_lambda=ridge,
                        valid_mae=float(np.abs(error).mean()),valid_rmse=float(np.sqrt((error*error).mean())),valid_bias=float(error.mean()),
                        best_iteration=model.best_iteration,center=center,feature_count=xx.shape[1],prepared_sha256=meta['arrays_sha256'],external_sha256=em['arrays_sha256'],test_evaluated=False)
                    model.save_model(str(job/'model.txt'));write(job/'validation.json',row)
                    pd.DataFrame(history['valid']).to_csv(job/'history.csv',index=False)
                    pd.DataFrame(dict(time=meta['valid_time'],h=data['valid'][5]+1,y_true=yv,prediction=pred)).to_csv(job/'predictions_valid.csv',index=False)
                    rows.append(row);print('VALID',row,flush=True)
            return rows
        rows=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in as_completed([pool.submit(study,w) for w in plan['windows']]):
                rows.extend(f.result());write(args.output_root/'progress.json',dict(completed=len(rows),expected=45,test_evaluated=False))
        d=pd.DataFrame(rows);d.to_csv(args.output_root/'VALIDATION_ALL.csv',index=False)
        summary=d.groupby(['configuration','method'],as_index=False).agg(valid_mae=('valid_mae','mean'),valid_rmse=('valid_rmse','mean'))
        summary.to_csv(args.output_root/'VALIDATION_SUMMARY.csv',index=False)
        selections={}
        for role,method in [('CAS','cas_10'),('standalone','tree_only')]:
            candidates=summary[summary.method==method].copy()
            candidates['selection_score']=np.maximum(candidates.valid_mae/54.60651697764482,candidates.valid_rmse/94.56828737273588)
            selections[role]=candidates.sort_values(['selection_score','valid_rmse','configuration']).iloc[0].to_dict()
        write(args.output_root/'selection.json',dict(rule=plan['selection'],selections=selections,test_used=False))
        write(args.output_root/'completion.json',dict(complete=len(rows)==45,completed=len(rows),expected=45,test_evaluated=False))


if __name__=='__main__':main()
