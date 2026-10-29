"""Freeze selected asinh head profiles/calibration; confirm seeds43/44 on validation.

Seed42 checkpoints reused exactly. No probe refit, no test arrays accessed.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,os,sys,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha
from scripts.run_cas_all import run_lock
from scripts.run_cas_directional_study import local_times
from scripts.pilot_cas_horizon_aligned import cost_coefficients
from scripts.pilot_cas_asinh_heads import full_features,forward,inverse
from scripts.pilot_cas_linear_leaves import transform

CAS_METHODS=['ordinary','fixed','cas_10','shuffle_10']
METHODS=CAS_METHODS+['tree_only']


def frozen_costs(recipe):
    direction=np.zeros(24,dtype=int);strength=np.zeros(24);seen=set()
    for row in recipe['horizons']:
        h=row['h']-1
        if h in seen or h not in range(24) or row['direction'] not in [-1,0,1] or not 0<=row['strength']<=.5:
            raise ValueError('Invalid frozen calibration')
        seen.add(h);direction[h]=row['direction'];strength[h]=row['strength']
    if not seen:raise ValueError('Empty frozen calibration')
    return direction,strength


def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-root',type=Path,default=ROOT/'results/cas_asinh_linear_heads_validation_20261029')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_asinh_linear_heads_confirmation_20261029')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();args.pilot_root=args.pilot_root.resolve();args.output_root=args.output_root.resolve()
    args.output_root.mkdir(parents=True,exist_ok=True)
    pilot=read(args.pilot_root/'plan.json');selection=read(args.pilot_root/'selection.json');done=read(args.pilot_root/'completion.json')
    if not done['complete'] or done['child_models']!=360 or done['test_evaluated']:raise ValueError('Complete validation pilot required')
    if sha(ROOT/'scripts/pilot_cas_asinh_linear_heads.py')!=pilot['script_sha256']:raise ValueError('Pilot implementation changed')
    assert read(args.pilot_root/'independent_audit.json')['validation_advance_rule_passed']
    score=selection['selections']['CAS']
    assert score['valid_mae']<53.180123970647344 and score['valid_rmse']<92.69189747620028
    cas_profile=selection['selections']['CAS']['profile'];plain_profile=selection['selections']['standalone']['profile']
    profiles=sorted({cas_profile,plain_profile})
    methods_by_profile={profile:([*CAS_METHODS] if profile==cas_profile else [])+(['tree_only'] if profile==plain_profile else []) for profile in profiles}
    source_hashes={};jobs=[]
    for w in [1,2,3]:
        for profile in profiles:
            for head in ([None] if profile=='asinh_shared' else list(range(24))):
                folder=args.pilot_root/'guangdong'/f'window_{w}'/profile/('head_all' if head is None else f'head_{head+1:02d}')
                source_hashes[str(folder/'training_recipe.json')]=sha(folder/'training_recipe.json')
                source_hashes[str(folder/'preprocessing.json')]=sha(folder/'preprocessing.json')
                for method in methods_by_profile[profile]:
                    for name in ['validation.json','model.txt','predictions_valid.csv']:
                        path=folder/method/name;source_hashes[str(path)]=sha(path)
                for seed in [42,43,44]:jobs.append((w,profile,head,seed))
    expected=sum(len(methods_by_profile[j[1]]) for j in jobs)
    plan=dict(protocol='frozen_asinh_linear_head_validation_three_seeds_v1',script_sha256=sha(__file__),
        pilot_plan_sha256=sha(args.pilot_root/'plan.json'),pilot_selection_sha256=sha(args.pilot_root/'selection.json'),
        pilot_root=str(args.pilot_root),cas_profile=cas_profile,standalone_profile=plain_profile,methods_by_profile=methods_by_profile,
        source_artifact_sha256=source_hashes,source_identities=pilot['source_identities'],seeds=[42,43,44],windows=[1,2,3],
        expected_aggregate=45,expected_child_records=expected,
        confirmation='Same training preprocessing, target scales, centres, calibrated costs and hyperparameters as seed42; only boosting seed varies',test_arrays_read=False)
    with run_lock(args.output_root):
        if (args.output_root/'plan.json').exists() and read(args.output_root/'plan.json')!=plan:raise ValueError('Confirmation plan changed')
        write(args.output_root/'plan.json',plan)
        prepared={}
        for w in [1,2,3]:
            ref=Path(pilot['roots']['reference_root'])/'guangdong'/f'window_{w}';meta=read(ref/'manifest.json')
            ext=Path(pilot['roots']['external_root'])/'prepared/guangdong'/f'window_{w}';em=read(ext/'manifest.json')
            identity=plan['source_identities'][str(w)]
            if sha(ref/'manifest.json')!=identity['reference_manifest_sha256'] or sha(ref/'features.npz')!=identity['reference_arrays_sha256']:raise ValueError('Reference changed')
            if sha(ext/'manifest.json')!=identity['external_manifest_sha256'] or sha(ext/'inputs.npz')!=identity['external_arrays_sha256']:raise ValueError('External source changed')
            data={}
            with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
                for s in ['train','valid']:
                    X=z['X_'+s];y=z['y_'+s];h=e['steps_'+s];ids=e['row_ids_'+s]
                    np.testing.assert_array_equal(y,e['yrows_'+s])
                    if not local_times(meta[s+'_time']).equals(local_times(em[s+'_time'])):raise ValueError('Times differ')
                    A,B=full_features(X,e['X_'+s],ids,h,meta,em['covariates'])
                    data[s]=(A,B,y,X[:,meta['point_index']].copy(),X[:,meta['confidence_index']].astype(float),h)
            prepared[w]=dict(data=data,meta=meta,em=em)
        def job(w,profile,head,seed):
            source=prepared[w];meta=source['meta'];tr=source['data']['train'];va=source['data']['valid']
            vi=np.arange(len(va[2])) if head is None else np.flatnonzero(va[5]==head)
            src=args.pilot_root/'guangdong'/f'window_{w}'/profile/('head_all' if head is None else f'head_{head+1:02d}')
            folder=args.output_root/'guangdong'/f'window_{w}'/profile/('head_all' if head is None else f'head_{head+1:02d}')
            folder.mkdir(parents=True,exist_ok=True);recipe=read(src/'training_recipe.json');direction,strength=frozen_costs(recipe)
            methods=methods_by_profile[profile];results={}
            if seed==42:
                for method in methods:
                    dest=folder/method/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
                    old=read(src/method/'validation.json');pred=pd.read_csv(src/method/'predictions_valid.csv')
                    np.testing.assert_allclose(pred.y_true,va[2][vi],atol=1e-6)
                    record=dict(old,model_source=str(src/method/'model.txt'),reused_seed42_pilot=True)
                    write(dest/'validation.json',record);shutil.copy2(src/method/'predictions_valid.csv',dest/'predictions_valid.csv')
                    results[method]=(record,pred.prediction.to_numpy())
                return w,profile,head,seed,vi,results
            ti=np.arange(len(tr[2])) if head is None else np.flatnonzero(tr[5]==head)
            A,B,y,point,c,h=(v[ti] for v in tr);V,W,yv,pv,_,hv=(v[vi] for v in va)
            prep=read(src/'preprocessing.json')
            A=transform(A,prep['cas']);B=transform(B,prep['standalone']);V=transform(V,prep['cas']);W=transform(W,prep['standalone'])
            target=recipe['target'];z=forward(y,target);bp=forward(point,target);zv=forward(yv,target);bvp=forward(pv,target)
            params=dict(pilot['tree'],objective='regression',metric='None',seed=seed,num_threads=2,deterministic=True,
                force_col_wise=True,verbosity=-1,boost_from_average=False,min_gain_to_split=.08/(target['scale']**2))
            for method in methods:
                dest=folder/method/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
                if (dest/'validation.json').exists():
                    record=read(dest/'validation.json');pred=pd.read_csv(dest/'predictions_valid.csv').prediction.to_numpy()
                    results[method]=(record,pred);continue
                original=read(src/method/'validation.json');center=original['center']
                absolute=method=='tree_only';base=np.zeros_like(z) if absolute else bp;bv=np.zeros_like(zv) if absolute else bvp
                np.testing.assert_allclose(float(np.mean(z-base)),center,atol=1e-12)
                xx=B if absolute else A;vv=W if absolute else V;kind='ordinary' if absolute else method
                def objective(pred,dataset):
                    r=pred-dataset.get_label();b=cost_coefficients(r,c,h,direction,strength,kind)
                    return b*r,b
                def raw_rmse(pred,dataset):return 'raw_rmse',float(np.sqrt(np.mean((inverse(bv+center+pred,target)-yv)**2))),False
                model=lgb.train(dict(params,objective='regression' if kind=='ordinary' else objective),lgb.Dataset(xx,label=z-base-center),
                    num_boost_round=2000,valid_sets=[lgb.Dataset(vv,label=zv-bv-center)],valid_names=['valid'],feval=raw_rmse,
                    callbacks=[lgb.early_stopping(100,verbose=False)])
                pred=inverse(bv+center+model.predict(vv,num_threads=2),target);r=pred-yv
                record=dict(original,seed=seed,valid_mae=float(np.abs(r).mean()),valid_rmse=float(np.sqrt((r*r).mean())),
                    best_iteration=model.best_iteration,model_source=str(dest/'model.txt'),reused_seed42_pilot=False)
                write(dest/'validation.json',record);model.save_model(str(dest/'model.txt'))
                pd.DataFrame(dict(time=np.asarray(meta['valid_time'])[vi],h=hv+1,y_true=yv,prediction=pred)).to_csv(dest/'predictions_valid.csv',index=False)
                results[method]=(record,pred)
            print('CONFIRMED HEAD',w,profile,head,seed,flush=True)
            return w,profile,head,seed,vi,results
        arrays={(w,m,s):np.full(len(prepared[w]['data']['valid'][2]),np.nan) for w in [1,2,3] for m in METHODS for s in [42,43,44]}
        children=[];completed=0
        with ThreadPoolExecutor(max_workers=min(args.workers,max(1,(os.cpu_count() or 2)//2))) as pool:
            for f in as_completed([pool.submit(job,*j) for j in jobs]):
                w,profile,head,seed,vi,results=f.result()
                for method,(record,pred) in results.items():arrays[w,method,seed][vi]=pred;children.append(record)
                completed+=1;write(args.output_root/'progress.json',dict(completed_head_jobs=completed,expected_head_jobs=len(jobs),completed_child_records=len(children),expected_child_records=expected,test_evaluated=False))
        if any(sha(Path(p))!=v for p,v in source_hashes.items()):raise ValueError('Frozen pilot artifacts changed')
        rows=[]
        for (w,method,seed),pred in arrays.items():
            if not np.isfinite(pred).all():raise ValueError('Incomplete original-row reconstruction')
            source=prepared[w];yv=source['data']['valid'][2];h=source['data']['valid'][5];meta=source['meta'];r=pred-yv
            profile=plain_profile if method=='tree_only' else cas_profile
            folder=args.output_root/'aggregate'/f'window_{w}'/method/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(dict(time=meta['valid_time'],h=h+1,y_true=yv,prediction=pred)).to_csv(folder/'predictions_valid.csv',index=False)
            row=dict(market='guangdong',window=w,method=method,seed=seed,profile=profile,valid_mae=float(np.abs(r).mean()),valid_rmse=float(np.sqrt((r*r).mean())),test_evaluated=False)
            write(folder/'validation.json',row);rows.append(row)
        pd.DataFrame(children).to_csv(args.output_root/'HEAD_RUNS.csv',index=False)
        d=pd.DataFrame(rows);d.to_csv(args.output_root/'VALIDATION_ALL.csv',index=False)
        per=d.groupby(['method','seed'],as_index=False).agg(valid_mae=('valid_mae','mean'),valid_rmse=('valid_rmse','mean'))
        per.to_csv(args.output_root/'PER_SEED.csv',index=False)
        per.groupby('method',as_index=False).agg(mae_mean=('valid_mae','mean'),mae_std=('valid_mae','std'),rmse_mean=('valid_rmse','mean'),rmse_std=('valid_rmse','std')).to_csv(args.output_root/'VALIDATION_SUMMARY.csv',index=False)
        write(args.output_root/'completion.json',dict(complete=len(rows)==45 and len(children)==expected,completed_aggregate=len(rows),child_records=len(children),
            frozen_pilot_artifacts_unchanged=True,recalibration=False,test_evaluated=False))


if __name__=='__main__':main()
