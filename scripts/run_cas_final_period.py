"""Frozen CAS and baselines on a later chronological period, evaluated once.

Qualification: of 2 markets x MAE/RMSE, >=3 first places and no rank worse
than second, compared with LEAR/RNN/Chronos/standalone LightGBM only.
All ablations and seeds remain visible. No test-based retry/tuning.
"""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,os,queue,subprocess,sys,uuid
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha,ROLLOUT_PROTOCOL
from scripts.run_cas_all import run_lock
from scripts.run_cas_directional_study import local_times
from scripts.pilot_cas_optimization import chronological_calibration
from scripts.pilot_cas_horizon_aligned import aligned_features,calibrate_costs,cost_coefficients
from scripts.confirm_cas_horizon_aligned import standalone_features
from scripts.run_rolling_adaptive_experiment import DATASETS

METHODS=['ordinary','fixed','cas_10','shuffle_10','tree_only','fusion_absolute']
EXTERNAL=['chronos_only','lear_adapted','rnn']
PROTOCOL='frozen_cas_lambda1_later_period_four_cells_rank_v1'


def fresh_split(times,previous_test_times):
    raw=pd.DatetimeIndex(pd.to_datetime(times))
    if raw.hasnans or raw.has_duplicates:raise ValueError('Invalid raw time index')
    t=local_times(raw).sort_values();old=local_times(previous_test_times).max()
    candidate=t[int(len(t)*.9)]
    remaining=t[(t>=candidate)&(t>old)]
    if len(remaining)<192:raise ValueError('No sufficiently long unevaluated later period')
    start=remaining[0];train_end=t[int(len(t)*.8)]
    if train_end>=start:raise ValueError('Insufficient validation period')
    def text(value):return str(value.tz_localize(None) if raw.tz is None else value.tz_convert(raw.tz))
    return dict(train_start=text(t[0]),train_end=text(train_end),valid_start=text(train_end),
        valid_end=text(start),test_start=text(start),test_end=text(t[-1]+pd.Timedelta(hours=1))),dict(
        previous_scored_max=str(old),fresh_start=str(start),new_observed_hours=len(remaining),
        later_than_all_previous_test_targets=True,time_only_boundary_design=True)


def rank_acceptance(table):
    records=[]
    for market,group in table.groupby('market'):
        target=group[group.method=='cas_10'].iloc[0]
        controls=group[group.method.isin(['lear_adapted','rnn','chronos_only','tree_only'])]
        if len(controls)!=4:raise ValueError('Four external comparators required')
        for metric in ['mae_mean','rmse_mean']:
            rank=1+int((controls[metric]<target[metric]-1e-9).sum())
            records.append(dict(market=market,metric=metric,value=float(target[metric]),rank=rank))
    passed=len(records)==4 and sum(r['rank']==1 for r in records)>=3 and max(r['rank'] for r in records)<=2
    return records,passed


def source_plan(args):
    confirmation=read(args.confirmation_root/'plan.json')
    if not read(args.confirmation_root/'completion.json')['complete']:raise ValueError('Three-seed validation confirmation required')
    splits={};audits={};source_hashes={}
    for market in ['guangdong','reale_delu']:
        data,timecol,*_=DATASETS[market]
        times=pd.read_csv(ROOT/data,usecols=[timecol])[timecol]
        prior=[]
        for w in [1,2,3]:prior.extend(read(args.previous_reference_root/market/f'window_{w}/manifest.json')['test_time'])
        split,audit=fresh_split(times,prior);split.update(train=.8,valid=.1,test=.1,time_col=timecol)
        splits[market]=split;audits[market]=audit;source_hashes[market]=sha(ROOT/data)
    return dict(protocol=PROTOCOL,script_sha256=sha(__file__),markets=['guangdong','reale_delu'],window=4,
        model_path=str(args.model_path),model_config_sha256=sha(args.model_path/'config.json'),
        previous_reference_root=str(args.previous_reference_root),confirmation_root=str(args.confirmation_root),
        methods=METHODS,external=EXTERNAL,seeds=[42,43,44],weight_lambda=1.,
        selected_configuration='square_loss; target-h-aligned Chronos residual; train-only soft mean-bias cost; c only in loss',
        confirmation_plan_sha256=sha(args.confirmation_root/'plan.json'),confirmation_validation_sha256=sha(args.confirmation_root/'VALIDATION_ALL.csv'),
        helper_sha256={p:sha(ROOT/'scripts'/p) for p in ['run_mlp_transfer.py','run_lear_rnn.py','pilot_cas_horizon_aligned.py','confirm_cas_horizon_aligned.py']},
        sources=source_hashes,splits=splits,boundary_audit=audits,tree=confirmation['tree'],
        success='Across 2 markets x MAE/RMSE, >=3 rank1 and all ranks<=2 versus LEAR/RNN/Chronos/tree.',
        qualification_scope='Main external comparison only. Ablations fully disclosed; c-specific gain is a separate question.',
        no_test_driven_retry=True,old_test_periods_not_reused=True,
        caveat='Later targets were not scored in prior valid comparisons; this is not a claim the raw source file was never accessed.')


def prepare_market(args,market):
    from scripts import run_mlp_transfer as transfer
    from scripts import run_lear_rnn as external
    prepared=args.output_root/'prepared';ref=prepared/market/'window_4'
    if not (ref/'manifest.json').exists():
        stage=args.output_root/('.prepared-'+market+'-'+uuid.uuid4().hex)
        transfer.prepare(SimpleNamespace(markets=[market],windows=[4],upstream='cas_chronos2',prepared_root=stage,
            split_root=args.output_root/'splits',cache_root=args.output_root/'cache',model_path=str(args.model_path),tsfm_batch_size=16,device='cuda:0'))
        ref.parent.mkdir(parents=True,exist_ok=True)
        (stage/market/'window_4').rename(ref)
    meta=read(ref/'manifest.json');plan=read(args.output_root/'plan.json')
    if meta['rollout_protocol']!=ROLLOUT_PROTOCOL or sha(ref/'features.npz')!=meta['arrays_sha256']:raise ValueError('Reference changed')
    first=local_times(meta['test_time']).min();old=pd.Timestamp(plan['boundary_audit'][market]['previous_scored_max'])
    if first<=old:raise ValueError('New test overlaps prior evaluated targets')
    external.prepare(SimpleNamespace(output_root=args.output_root/'external',reference_root=prepared),market,4)
    write(args.output_root/(market+'_prepared.json'),dict(reference_sha256=meta['arrays_sha256'],first_new_target=str(first),previous_scored_max=str(old)))


def train_market(args,market):
    import lightgbm as lgb
    plan=read(args.output_root/'plan.json');ref=args.output_root/'prepared'/market/'window_4';meta=read(ref/'manifest.json')
    ext=args.output_root/'external/prepared'/market/'window_4';em=read(ext/'manifest.json')
    if sha(ref/'features.npz')!=meta['arrays_sha256'] or sha(ext/'inputs.npz')!=em['arrays_sha256'] or em['identity']['reference_sha256']!=meta['arrays_sha256']:
        raise ValueError('Prepared checksum mismatch')
    data={}
    with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
        for s in ['train','valid']:
            X=z['X_'+s];y=z['y_'+s];h=e['steps_'+s];ids=e['row_ids_'+s]
            np.testing.assert_array_equal(y,e['yrows_'+s])
            if not local_times(meta[s+'_time']).equals(local_times(em[s+'_time'])):raise ValueError('Time mismatch')
            data[s]=(aligned_features(X,e['X_'+s],ids,h,meta,em['covariates']),
                     standalone_features(X,e['X_'+s],ids,h,meta,em['covariates']),y,
                     X[:,meta['point_index']].copy(),X[:,meta['confidence_index']].astype(float),h)
    A,B,y,point,c,h=data['train'];V,W,yv,pv,_,_=data['valid']
    params=dict(plan['tree'],objective='regression',metric='None',seed=42,num_threads=4,deterministic=True,
        force_col_wise=True,verbosity=-1,boost_from_average=False)
    fit,cal=chronological_calibration(pd.to_datetime(meta['train_time']),h+1)
    center_probe=float(np.mean(y[fit]-point[fit],dtype=np.float64))
    probe=lgb.train(params,lgb.Dataset(A[fit],label=y[fit]-point[fit]-center_probe),num_boost_round=300)
    error=y[cal]-point[cal]-center_probe-probe.predict(A[cal],num_threads=4)
    origins=local_times(meta['train_time'])-pd.to_timedelta(h,unit='h')
    direction,strength,diagnostics=calibrate_costs(error,h[cal],origins[cal])
    folder=args.output_root/'training'/market;folder.mkdir(parents=True,exist_ok=True)
    write(folder/'training_calibration.json',dict(horizons=diagnostics,fit_center=center_probe,
        fit_max_target_time=str(local_times(meta['train_time'])[fit].max()),first_calibration_origin=str(origins[cal].min())))
    results=[]
    for seed in plan['seeds']:
        for method in METHODS:
            dest=folder/method/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'validation.json').exists():results.append(read(dest/'validation.json'));continue
            absolute=method in ['tree_only','fusion_absolute'];base=np.zeros_like(y) if absolute else point
            bv=np.zeros_like(yv) if absolute else pv;center=float(np.mean(y-base,dtype=np.float64))
            xx=B if method=='tree_only' else A;vv=W if method=='tree_only' else V;kind='ordinary' if absolute else method
            def objective(pred,dataset):
                r=pred-dataset.get_label();b=cost_coefficients(r,c,h,direction,strength,kind)
                return b*r,b
            def rmse(pred,dataset):return 'rmse',float(np.sqrt(np.mean((pred-dataset.get_label())**2))),False
            model=lgb.train(dict(params,seed=seed,objective='regression' if kind=='ordinary' else objective),
                lgb.Dataset(xx,label=y-base-center),num_boost_round=2000,
                valid_sets=[lgb.Dataset(vv,label=yv-bv-center)],valid_names=['valid'],feval=rmse,
                callbacks=[lgb.early_stopping(100,verbose=False)])
            pred=bv+center+model.predict(vv,num_threads=4);r=pred-yv
            row=dict(market=market,method=method,seed=seed,center=center,valid_mae=float(np.abs(r).mean()),valid_rmse=float(np.sqrt((r*r).mean())),
                best_iteration=model.best_iteration,reference_sha256=meta['arrays_sha256'],test_evaluated=False)
            write(dest/'validation.json',row);model.save_model(str(dest/'model.txt'));results.append(row)
            print('TRAINED',row,flush=True)
    write(folder/'training_frozen.json',dict(plan_sha256=sha(args.output_root/'plan.json'),reference_sha256=meta['arrays_sha256'],models=results))


def score_market(args,market):
    import lightgbm as lgb
    if not (args.output_root/'TRAINING_FROZEN.json').exists():raise ValueError('Freeze all downstream training before scoring')
    plan=read(args.output_root/'plan.json');ref=args.output_root/'prepared'/market/'window_4';meta=read(ref/'manifest.json')
    ext=args.output_root/'external/prepared'/market/'window_4';em=read(ext/'manifest.json')
    with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
        X=z['X_test'];y=z['y_test'];h=e['steps_test'];ids=e['row_ids_test']
        np.testing.assert_array_equal(y,e['yrows_test'])
        A=aligned_features(X,e['X_test'],ids,h,meta,em['covariates']);B=standalone_features(X,e['X_test'],ids,h,meta,em['covariates'])
    rows=[];predroot=args.output_root/'predictions'/market;predroot.mkdir(parents=True,exist_ok=True)
    def score(method,seed,pred,extra=None):
        error=np.asarray(pred,dtype=float)-y
        row=dict(market=market,method=method,seed=seed,mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error*error).mean())),
            n_test=len(y),reference_sha256=meta['arrays_sha256'])
        if extra:row.update(extra)
        pd.DataFrame(dict(time=meta['test_time'],h=h+1,y_true=y,prediction=pred)).to_csv(predroot/f'{method}_seed_{seed}.csv',index=False)
        rows.append(row)
    for seed in plan['seeds']:
        for method in METHODS:
            dest=args.output_root/'training'/market/method/f'seed_{seed}';r=read(dest/'validation.json')
            model=lgb.Booster(model_file=str(dest/'model.txt'));xx=B if method=='tree_only' else A
            base=np.zeros_like(y) if method in ['tree_only','fusion_absolute'] else X[:,meta['point_index']].copy()
            score(method,seed,base+r['center']+model.predict(xx,num_threads=4),dict(best_iteration=r['best_iteration']))
        score('chronos_only',seed,X[:,meta['point_index']],dict(deterministic_reuse=True))
        for method in ['lear_adapted','rnn']:
            actual_seed=42 if method=='lear_adapted' else seed
            dest=args.output_root/'external/training'/method/market/'window_4'/f'seed_{actual_seed}'
            r=read(dest/'metrics.json');pred=pd.read_csv(dest/'predictions_test.csv')
            if r['prepared_sha256']!=em['arrays_sha256']:raise ValueError('External source mismatch')
            np.testing.assert_allclose(pred.y_true,y,rtol=1e-7,atol=1e-6)
            if not local_times(pred.time).equals(local_times(meta['test_time'])):raise ValueError('External sample time mismatch')
            score(method,seed,pred.prediction.to_numpy(),dict(deterministic_reuse=method=='lear_adapted',actual_training_seed=actual_seed))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['all','check','prepare','train'],default='all');p.add_argument('--market',choices=['guangdong','reale_delu'])
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_final_later_period_20261029')
    p.add_argument('--previous-reference-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--confirmation-root',type=Path,default=ROOT/'results/cas_horizon_aligned_confirmation_20261029')
    p.add_argument('--model-path',type=Path,default=ROOT.parent/'CAS-EFP/models/chronos-2')
    p.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    args=p.parse_args()
    for k in ['output_root','previous_reference_root','confirmation_root','model_path']:setattr(args,k,getattr(args,k).resolve())
    args.output_root.mkdir(parents=True,exist_ok=True)
    if args.phase in ['prepare','train']:
        if not args.market:raise ValueError('Worker requires market')
        (prepare_market if args.phase=='prepare' else train_market)(args,args.market);return
    with run_lock(args.output_root):
        plan=source_plan(args)
        if (args.output_root/'plan.json').exists() and read(args.output_root/'plan.json')!=plan:raise ValueError('Frozen evaluation plan changed')
        write(args.output_root/'plan.json',plan)
        for market in plan['markets']:
            split=args.output_root/'splits'/market/'window_4';split.mkdir(parents=True,exist_ok=True)
            write(split/'data_split.json',[dict(data_split=plan['splits'][market])])
        print('BOUNDARIES',plan['boundary_audit'],flush=True)
        if args.phase=='check':return
        logs=args.output_root/'logs';logs.mkdir(exist_ok=True)
        def internal(phase,market,gpu):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            cmd=[sys.executable,__file__,'--phase',phase,'--market',market,'--output-root',str(args.output_root),'--model-path',str(args.model_path)]
            with (logs/f'{phase}_{market}.log').open('a') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for phase in ['prepare','train']:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(internal,phase,m,args.gpus[i%len(args.gpus)] if phase=='prepare' else '') for i,m in enumerate(plan['markets'])]
                for f in as_completed(futures):f.result()
            write(args.output_root/'progress.json',dict(phase=phase,complete=True))
        write(args.output_root/'TRAINING_FROZEN.json',dict(plan_sha256=sha(args.output_root/'plan.json'),
            markets={m:sha(args.output_root/'training'/m/'training_frozen.json') for m in plan['markets']},
            external_settings='Unchanged LEAR AIC adaptation/RNN64; seeded configs fixed before scoring'))
        cards=queue.Queue()
        for gpu in args.gpus:cards.put(gpu)
        jobs=[(m,model,seed) for m in plan['markets'] for model in ['lear_adapted','rnn'] for seed in ([42] if model=='lear_adapted' else plan['seeds'])]
        def external(job):
            m,model,seed=job;gpu=cards.get() if model=='rnn' else ''
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            try:
                with (logs/f'{model}_{m}_s{seed}.log').open('a') as log:
                    subprocess.run([sys.executable,str(ROOT/'scripts/run_lear_rnn.py'),'--action','worker','--output-root',str(args.output_root/'external'),
                        '--model',model,'--market',m,'--window','4','--seed',str(seed)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            finally:
                if model=='rnn':cards.put(gpu)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for f in as_completed([pool.submit(external,j) for j in jobs]):f.result()
        rows=[]
        for market in plan['markets']:rows.extend(score_market(args,market))
        table=pd.DataFrame(rows);table.to_csv(args.output_root/'ALL_RUNS.csv',index=False)
        summary=table.groupby(['market','method'],as_index=False).agg(mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'),rmse_std=('rmse','std'))
        summary.to_csv(args.output_root/'SUMMARY.csv',index=False)
        ranks,passed=rank_acceptance(summary);pd.DataFrame(ranks).to_csv(args.output_root/'MAIN_COMPARISON_RANKS.csv',index=False)
        write(args.output_root/'completion.json',dict(complete=len(rows)==54,completed=len(rows),expected=54,qualification_passed=passed,
             acceptance_scope=plan['qualification_scope'],ranked_cells=ranks,test_based_retry=False))
        print('FINAL',read(args.output_root/'completion.json'),flush=True)


if __name__=='__main__':main()
