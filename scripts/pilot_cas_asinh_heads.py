"""Fixed Guangdong train/validation study: asinh target, pooled vs per-h heads.

Chronos median anchoring and confidence-asymmetric training retained. Full
causal LEAR curves are shared with standalone controls. No test arrays read.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,os,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha,ROLLOUT_PROTOCOL
from scripts.run_cas_all import run_lock
from scripts.run_cas_directional_study import local_times
from scripts.pilot_cas_optimization import chronological_calibration
from scripts.pilot_cas_horizon_aligned import aligned_features,cost_coefficients
from scripts.confirm_cas_horizon_aligned import standalone_features

METHODS=['ordinary','fixed','cas_10','shuffle_10','tree_only']
PROFILES=['asinh_shared','asinh_per_h']
PROTOCOL='validation_only_asinh_anchored_full_causal_curves_shared_vs_h_heads_v1'


def fit_target(y):
    y=np.asarray(y,dtype=float)
    if not np.isfinite(y).all() or not len(y):raise ValueError('Invalid training target')
    median=float(np.median(y));scale=float(np.median(np.abs(y-median))/.6744897501960817)
    return dict(median=median,scale=scale if scale>1e-8 else 1.)


def forward(y,recipe):return np.arcsinh((np.asarray(y,dtype=float)-recipe['median'])/recipe['scale'])


def inverse(z,recipe):
    with np.errstate(over='ignore',invalid='ignore'):y=np.sinh(np.asarray(z,dtype=float))*recipe['scale']+recipe['median']
    if not np.isfinite(y).all():raise ValueError('Nonfinite inverse target prediction; no clipping permitted')
    return y


def full_features(X,context,ids,h,meta,covariates):
    curves=context[ids]
    A=np.column_stack([aligned_features(X,context,ids,h,meta,covariates),curves]).astype(np.float32)
    B=np.column_stack([standalone_features(X,context,ids,h,meta,covariates),curves]).astype(np.float32)
    return A,B


def calibration(error,h,days):
    direction=np.zeros(24,dtype=int);strength=np.zeros(24);rows=[]
    for step in np.unique(h):
        mask=h==step;e=error[mask];bias=float(e.mean());sd=float(e.std())
        daily=pd.Series(e,index=pd.DatetimeIndex(days[mask]).normalize()).groupby(level=0).mean()
        se=float(daily.std(ddof=1)/np.sqrt(len(daily))) if len(daily)>1 else None
        if len(daily)>=7 and se is not None:
            direction[step]=1 if bias>0 else (-1 if bias<0 else 0)
            strength[step]=min(.5,abs(bias)/(sd+1e-6)*abs(bias)/(abs(bias)+2*se+1e-6))
        rows.append(dict(h=int(step)+1,bias=bias,sd=sd,day_mean_se=se,days=len(daily),direction=int(direction[step]),strength=float(strength[step])))
    return direction,strength,rows


def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--external-root',type=Path,default=ROOT/'results/lear_rnn_external_20261029_v2')
    p.add_argument('--confirmation-root',type=Path,default=ROOT/'results/cas_horizon_aligned_confirmation_20261029')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_asinh_heads_validation_20261029')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    for k in ['reference_root','external_root','confirmation_root','output_root']:setattr(args,k,getattr(args,k).resolve())
    if args.workers<1:raise ValueError('Positive workers required')
    args.output_root.mkdir(parents=True,exist_ok=True)
    tree=dict(read(args.confirmation_root/'plan.json')['tree'],num_leaves=31,min_data_in_leaf=80)
    sources={str(w):dict(reference_manifest_sha256=sha(args.reference_root/'guangdong'/f'window_{w}/manifest.json'),
        reference_arrays_sha256=read(args.reference_root/'guangdong'/f'window_{w}/manifest.json')['arrays_sha256'],
        external_manifest_sha256=sha(args.external_root/'prepared/guangdong'/f'window_{w}/manifest.json'),
        external_arrays_sha256=read(args.external_root/'prepared/guangdong'/f'window_{w}/manifest.json')['arrays_sha256']) for w in [1,2,3]}
    plan=dict(protocol=PROTOCOL,script_sha256=sha(__file__),roots={k:str(getattr(args,k)) for k in ['reference_root','external_root','confirmation_root']},
        source_identities=sources,
        helper_sha256={p:sha(ROOT/'scripts'/p) for p in ['pilot_cas_horizon_aligned.py','confirm_cas_horizon_aligned.py']},
        market='guangdong',windows=[1,2,3],profiles=PROFILES,methods=METHODS,seeds=[42],tree=tree,
        features='h-aligned local/lag/spread features + all existing causal LEAR origin curves/calendar; standalone excludes upstream-derived inputs',
        target='T=asinh((y-train_median)/train_MAD_scaled); forecast=inverse(T(Chronos_P50)+tree_residual)',
        probe='300 fixed rounds on purged training80%; probe target scale fitted only there; latter20% training calibrates costs',
        gain_threshold='0.08 / training_target_scale**2 to convert original central price units to target coordinates',
        evaluation='Raw-unit validation RMSE for early stopping; raw-unit MAE/RMSE for comparison',
        selection='CAS and standalone separately minimize max(MAE/54.60651697764482,RMSE/94.56828737273588) across all three windows; ties RMSE then profile',
        expected_aggregate=30,expected_child_models=375,expected_probes=75,test_arrays_read=False)
    with run_lock(args.output_root):
        if (args.output_root/'plan.json').exists() and read(args.output_root/'plan.json')!=plan:raise ValueError('Plan changed')
        write(args.output_root/'plan.json',plan)
        prepared={}
        for w in [1,2,3]:
            ref=args.reference_root/'guangdong'/f'window_{w}';meta=read(ref/'manifest.json')
            ext=args.external_root/'prepared/guangdong'/f'window_{w}';em=read(ext/'manifest.json')
            if meta['rollout_protocol']!=ROLLOUT_PROTOCOL or sha(ref/'features.npz')!=meta['arrays_sha256']:raise ValueError('Reference changed')
            if em['identity']['reference_sha256']!=meta['arrays_sha256'] or sha(ext/'inputs.npz')!=em['arrays_sha256']:raise ValueError('External curves changed')
            data={}
            with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
                for s in ['train','valid']:
                    X=z['X_'+s];y=z['y_'+s];h=e['steps_'+s];ids=e['row_ids_'+s]
                    np.testing.assert_array_equal(y,e['yrows_'+s])
                    if not local_times(meta[s+'_time']).equals(local_times(em[s+'_time'])):raise ValueError('Times changed')
                    A,B=full_features(X,e['X_'+s],ids,h,meta,em['covariates'])
                    data[s]=(A,B,y,X[:,meta['point_index']].copy(),X[:,meta['confidence_index']].astype(float),h)
            fit,cal=chronological_calibration(pd.to_datetime(meta['train_time']),data['train'][5]+1)
            origins=local_times(meta['train_time'])-pd.to_timedelta(data['train'][5],unit='h')
            prepared[w]=dict(data=data,meta=meta,em=em,fit=fit,cal=cal,origins=origins)
        def job(w,profile,head):
            source=prepared[w];data=source['data'];meta=source['meta'];em=source['em']
            tr=data['train'];va=data['valid'];ti=np.arange(len(tr[2])) if head is None else np.flatnonzero(tr[5]==head)
            vi=np.arange(len(va[2])) if head is None else np.flatnonzero(va[5]==head)
            A,B,y,point,c,h=(v[ti] for v in tr);V,W,yv,pv,_,hv=(v[vi] for v in va)
            fit=source['fit'][ti];cal=source['cal'][ti]
            if not fit.any() or not cal.any() or not len(vi):raise ValueError('Missing head fit/calibration/validation rows')
            folder=args.output_root/'guangdong'/f'window_{w}'/profile/('head_all' if head is None else f'head_{head+1:02d}')
            folder.mkdir(parents=True,exist_ok=True)
            recipe=fit_target(y);probe_recipe=fit_target(y[fit])
            z=forward(y,recipe);bp=forward(point,recipe);zv=forward(yv,recipe);bvp=forward(pv,recipe)
            probe_base=forward(point[fit],probe_recipe);probe_y=forward(y[fit],probe_recipe)
            center_probe=float(np.mean(probe_y-probe_base))
            params=dict(tree,objective='regression',metric='None',seed=42,num_threads=2,deterministic=True,
                force_col_wise=True,verbosity=-1,boost_from_average=False,min_gain_to_split=.08/(recipe['scale']**2))
            probe_params=dict(params,min_gain_to_split=.08/(probe_recipe['scale']**2))
            probe=lgb.train(probe_params,lgb.Dataset(A[fit],label=probe_y-probe_base-center_probe),num_boost_round=300)
            pz=forward(point[cal],probe_recipe)+center_probe+probe.predict(A[cal],num_threads=2)
            error=forward(y[cal],probe_recipe)-pz
            direction,strength,diagnostics=calibration(error,h[cal],source['origins'][ti][cal])
            write(folder/'training_recipe.json',dict(target=recipe,probe_target=probe_recipe,probe_center=center_probe,horizons=diagnostics,
                fit_max_target_time=str(local_times(meta['train_time'])[ti][fit].max()),first_calibration_origin=str(source['origins'][ti][cal].min()),
                original_h=head+1 if head is not None else None,feature_count=A.shape[1],standalone_feature_count=B.shape[1]))
            results={}
            for method in METHODS:
                dest=folder/method;dest.mkdir(parents=True,exist_ok=True)
                if (dest/'validation.json').exists():
                    record=read(dest/'validation.json');pred=pd.read_csv(dest/'predictions_valid.csv').prediction.to_numpy()
                    results[method]=(record,pred);continue
                absolute=method=='tree_only';base=np.zeros_like(z) if absolute else bp;bv=np.zeros_like(zv) if absolute else bvp
                xx=B if absolute else A;vv=W if absolute else V;center=float(np.mean(z-base));kind='ordinary' if absolute else method
                def objective(pred,dataset):
                    r=pred-dataset.get_label();b=cost_coefficients(r,c,h,direction,strength,kind)
                    return b*r,b
                def raw_rmse(pred,dataset):
                    raw=inverse(bv+center+pred,recipe)
                    return 'raw_rmse',float(np.sqrt(np.mean((raw-yv)**2))),False
                history={}
                model=lgb.train(dict(params,objective='regression' if kind=='ordinary' else objective),
                    lgb.Dataset(xx,label=z-base-center),num_boost_round=2000,
                    valid_sets=[lgb.Dataset(vv,label=zv-bv-center)],valid_names=['valid'],feval=raw_rmse,
                    callbacks=[lgb.early_stopping(100,verbose=False),lgb.record_evaluation(history)])
                pred=inverse(bv+center+model.predict(vv,num_threads=2),recipe);r=pred-yv
                record=dict(market='guangdong',window=w,profile=profile,head=head+1 if head is not None else None,seed=42,method=method,
                    center=center,target=recipe,valid_mae=float(np.abs(r).mean()),valid_rmse=float(np.sqrt((r*r).mean())),
                    best_iteration=model.best_iteration,n_train=len(y),n_valid=len(yv),prepared_sha256=meta['arrays_sha256'],external_sha256=em['arrays_sha256'],test_evaluated=False)
                write(dest/'validation.json',record);model.save_model(str(dest/'model.txt'))
                pd.DataFrame(history['valid']).to_csv(dest/'history.csv',index=False)
                pd.DataFrame(dict(time=np.asarray(meta['valid_time'])[vi],h=hv+1,y_true=yv,prediction=pred)).to_csv(dest/'predictions_valid.csv',index=False)
                results[method]=(record,pred)
            print('HEAD COMPLETE',w,profile,head,flush=True)
            return w,profile,head,vi,results
        jobs=[(w,'asinh_shared',None) for w in [1,2,3]]+[(w,'asinh_per_h',h) for w in [1,2,3] for h in range(24)]
        predictions={};children=[];completed=0
        for w in [1,2,3]:
            for profile in PROFILES:
                for method in METHODS:predictions[w,profile,method]=np.full(len(prepared[w]['data']['valid'][2]),np.nan)
        with ThreadPoolExecutor(max_workers=min(args.workers,max(1,(os.cpu_count() or 2)//2))) as pool:
            for future in as_completed([pool.submit(job,*j) for j in jobs]):
                w,profile,head,vi,results=future.result()
                for method,(record,pred) in results.items():predictions[w,profile,method][vi]=pred;children.append(record)
                completed+=1;write(args.output_root/'progress.json',dict(completed_heads=completed,expected_heads=75,completed_models=len(children),expected_models=375,test_evaluated=False))
        rows=[]
        for (w,profile,method),pred in predictions.items():
            if not np.isfinite(pred).all():raise ValueError('Missing head predictions')
            meta=prepared[w]['meta'];yv=prepared[w]['data']['valid'][2];h=prepared[w]['data']['valid'][5];r=pred-yv
            folder=args.output_root/'aggregate'/f'window_{w}'/profile/method;folder.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(dict(time=meta['valid_time'],h=h+1,y_true=yv,prediction=pred)).to_csv(folder/'predictions_valid.csv',index=False)
            row=dict(market='guangdong',window=w,profile=profile,method=method,seed=42,valid_mae=float(np.abs(r).mean()),
                valid_rmse=float(np.sqrt((r*r).mean())),n_valid=len(yv),test_evaluated=False)
            write(folder/'validation.json',row);rows.append(row)
        pd.DataFrame(children).to_csv(args.output_root/'HEAD_RUNS.csv',index=False)
        d=pd.DataFrame(rows);d.to_csv(args.output_root/'VALIDATION_ALL.csv',index=False)
        summary=d.groupby(['profile','method'],as_index=False).agg(valid_mae=('valid_mae','mean'),valid_rmse=('valid_rmse','mean'))
        summary.to_csv(args.output_root/'VALIDATION_SUMMARY.csv',index=False)
        selections={}
        for role,method in [('CAS','cas_10'),('standalone','tree_only')]:
            s=summary[summary.method==method].copy();s['selection_score']=np.maximum(s.valid_mae/54.60651697764482,s.valid_rmse/94.56828737273588)
            selections[role]=s.sort_values(['selection_score','valid_rmse','profile']).iloc[0].to_dict()
        write(args.output_root/'selection.json',dict(rule=plan['selection'],selections=selections,test_used=False))
        write(args.output_root/'completion.json',dict(complete=len(rows)==30 and len(children)==375,completed=30,child_models=len(children),test_evaluated=False))


if __name__=='__main__':main()
