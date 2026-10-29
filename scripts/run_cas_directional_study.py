"""Bounded real-data optimization and explicitly semi-synthetic directional stress study.

No test-driven retry/search. All signed trends, strengths, windows and methods are
registered before execution. Existing real baselines are read-only. Seed 42 is a
development study, not final independent confirmation.
"""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,json,os,queue,shutil,subprocess,sys,time,uuid
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha,ROLLOUT_PROTOCOL
from scripts.run_cas_all import run_lock
from scripts.pilot_cas_optimization import chronological_calibration

PROTOCOL='causal_real_and_signed_target_ramp_development_v1'
VARIANTS=['ordinary','fixed','cas','mean','shuffle']
METHODS=['tree_only_rich','chronos_tree_absolute']+['original_residual_'+v for v in VARIANTS]+['rich_residual_'+v for v in VARIANTS+['gated_fixed','gated_cas','gated_shuffle']]
CAS_METHODS=['original_residual_cas','rich_residual_cas','rich_residual_gated_cas']


def local_times(values):
    """Prepared naive wall times use the existing pipeline's Shanghai convention."""
    index=pd.DatetimeIndex(pd.to_datetime(values))
    return index.tz_localize('Asia/Shanghai') if index.tz is None else index.tz_convert('Asia/Shanghai')


def cases():
    result=[dict(id=f'real_{m}_w{w}',market=m,window=w,kind='real',sign=0,strength=0.)
            for m in ['guangdong','reale_delu'] for w in [1,2,3]]
    result += [dict(id=f'synthetic_gd_{label}_{strength}_w{w}',market='guangdong',window=w,
                    kind='semi_synthetic_signed_ramp',sign=sign,strength=strength)
               for label,sign in [('negative',-1),('positive',1)] for strength in [.5,1.] for w in [1,2,3]]
    return result


def signed_ramp(times,y,train_start,train_end,sign,strength):
    """Target-only intervention. Scale uses original training values only."""
    times=pd.DatetimeIndex(times);y=np.asarray(y,dtype=float)
    start=pd.Timestamp(train_start);end=pd.Timestamp(train_end)
    mask=np.asarray((times>=start)&(times<end))
    scale=float(np.nanstd(y[mask]))
    if sign not in [-1,1] or strength not in [.5,1.] or not np.isfinite(scale) or scale<=0:
        raise ValueError('Invalid preregistered ramp or empty training segment')
    elapsed=np.maximum(np.asarray((times-start)/pd.Timedelta(days=90),dtype=float),0.)
    delta=sign*strength*scale*elapsed
    return y+delta,dict(training_std=scale,change_per_90_days=sign*strength*scale,
                        anchor=str(start),target_only=True,synthetic=True)


def weights(error,c,variant,direction):
    error=np.asarray(error);c=np.asarray(c,dtype=float)
    if not np.isfinite(c).all() or ((c<0)|(c>1)).any():raise ValueError('c outside [0,1]')
    if variant=='ordinary':return np.ones_like(error,dtype=float)
    focused=-error*np.asarray(direction)>0
    if variant.startswith('gated_'):
        w=1+.3*(1-c);w=w/w.mean()
        if variant=='gated_fixed':w=np.ones_like(w)
        elif variant=='gated_shuffle':w=np.random.default_rng(42).permutation(w)
        elif variant!='gated_cas':raise ValueError(variant)
        return np.where(focused,1+.25*w,1.)
    w=2-c
    if variant=='fixed':w=np.ones_like(w)
    elif variant=='mean':w=np.full_like(w,w.mean())
    elif variant=='shuffle':w=np.random.default_rng(42).permutation(w)
    elif variant!='cas':raise ValueError(variant)
    return np.where(focused,2*w,1.)


def reference(args,case,folder):
    from scripts import run_mlp_transfer as transfer
    from scripts import run_lear_rnn as external
    from scripts.run_rolling_adaptive_experiment import DATASETS
    market=case['market'];w=case['window']
    original=args.reference_root/market/f'window_{w}'
    if case['kind']=='real':
        return original,args.external_root
    rel,timecol,reg,tsfm,std=DATASETS[market]
    generated=folder/'synthetic_source.csv'
    source=ROOT/rel
    if not generated.exists() or not (folder/'intervention.json').exists():
        raw=pd.read_csv(source);dates=pd.to_datetime(raw[timecol])
        target=read(ROOT/reg)[0]['target_columns'][-1]
        split=read(original/'data_split.json')[0]['data_split']
        raw[target],details=signed_ramp(dates,raw[target],split['train_start'],split['train_end'],case['sign'],case['strength'])
        temporary=folder/'synthetic_source.tmp.csv';raw.to_csv(temporary,index=False);temporary.replace(generated)
        write(folder/'intervention.json',dict(case=case,source_sha256=sha(source),generated_sha256=sha(generated),
                target=target,unchanged_covariates=True,**details))
    intervention=read(folder/'intervention.json')
    if sha(generated)!=intervention['generated_sha256'] or sha(source)!=intervention['source_sha256']:
        raise ValueError('Synthetic data provenance changed')
    # Override process-local dataset mapping only. No original data/config is edited.
    DATASETS[market]=(str(generated),timecol,reg,tsfm,std)
    prepared=folder/'reference';ref=prepared/market/f'window_{w}'
    if not (ref/'manifest.json').exists():
        stage=folder/('.reference-'+uuid.uuid4().hex)
        a=SimpleNamespace(markets=[market],windows=[w],upstream='cas_chronos2',prepared_root=stage,
            split_root=args.reference_root,cache_root=folder/'cache',model_path=str(args.model_path),
            tsfm_batch_size=16,device='cuda:0')
        transfer.prepare(a)
        if prepared.exists():raise ValueError('Incomplete published reference')
        stage.rename(prepared)
    out=folder/'external'
    external.prepare(SimpleNamespace(output_root=out,reference_root=prepared),market,w)
    return ref,out


def fit_case(args,case):
    import lightgbm as lgb
    folder=args.output_root/'cases'/case['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists() and read(folder/'completion.json')['complete']:return
    ref,extroot=reference(args,case,folder)
    meta=read(ref/'manifest.json');market=case['market'];window=case['window']
    if meta['rollout_protocol']!=ROLLOUT_PROTOCOL or sha(ref/'features.npz')!=meta['arrays_sha256']:
        raise ValueError('Wrong causal reference or checksum')
    ext=extroot/'prepared'/market/f'window_{window}'
    em=read(ext/'manifest.json')
    if em['identity']['reference_sha256']!=meta['arrays_sha256'] or sha(ext/'inputs.npz')!=em['arrays_sha256']:
        raise ValueError('External input checksum/reference mismatch')
    pi=meta['point_index'];ci=meta['confidence_index'];hi=meta['feature_names'].index('h')
    # Development stage reads train and valid only. Test loading is below, after selection is frozen.
    data={};extra={}
    with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
        for s in ['train','valid']:
            X=z['X_'+s];y=z['y_'+s];ids=e['row_ids_'+s];steps=e['steps_'+s]
            np.testing.assert_allclose(y,e['yrows_'+s],rtol=0,atol=0)
            np.testing.assert_array_equal(X[:,hi].astype(int)-1,steps)
            if not local_times(meta[s+'_time']).equals(local_times(em[s+'_time'])):
                raise ValueError('Time alignment changed')
            data[s]=(X,y)
            extra[s]=e['X_'+s][ids].astype(np.float32)
    X,y=data['train'];V,yv=data['valid'];c=X[:,ci].astype(float);point=X[:,pi].copy()
    keep=[i for i in range(X.shape[1]) if i!=ci];plain=[i for i in keep if i!=pi]
    original=X[:,keep];ov=V[:,keep];rich=np.column_stack([original,extra['train']]);rv=np.column_stack([ov,extra['valid']])
    params=dict(objective='regression',metric='None',seed=42,num_threads=4,deterministic=True,
        force_col_wise=True,verbosity=-1,boost_from_average=False,learning_rate=.05,num_leaves=63,
        min_data_in_leaf=40,feature_fraction=.9,bagging_fraction=.8,bagging_freq=5,min_gain_to_split=.08)
    fitmask,cal=chronological_calibration(pd.to_datetime(meta['train_time']),X[:,hi])
    # Probe centering uses fit segment only, including no held-out calibration labels.
    center_probe=float(np.mean(y[fitmask]-point[fitmask],dtype=np.float64))
    probe=lgb.train(params,lgb.Dataset(rich[fitmask],label=y[fitmask]-point[fitmask]-center_probe),num_boost_round=300)
    err=y[cal]-point[cal]-center_probe-probe.predict(rich[cal],num_threads=4)
    threshold=.05*float(np.std(y[fitmask]));gates=np.zeros(24,dtype=int);details=[]
    horizon=X[:,hi].astype(int)-1
    for h in range(24):
        m=horizon[cal]==h;bias=float(np.median(err[m])) if m.any() else 0.
        gates[h]=0 if abs(bias)<=threshold else (1 if bias>0 else -1)
        details.append(dict(h=h+1,bias=bias,threshold=threshold,direction=int(gates[h]),samples=int(m.sum())))
    write(folder/'training_calibration.json',dict(gates=details,probe_fit_center=center_probe,
          target_time_purged=True,fit_samples=int(fitmask.sum()),calibration_samples=int(cal.sum())))
    direction=1 if np.median(y-point)>=0 else -1
    results=[]
    for method in METHODS:
        dest=folder/'models'/method;dest.mkdir(parents=True,exist_ok=True)
        identity=dict(protocol=PROTOCOL,method=method,seed=42,prepared_sha256=meta['arrays_sha256'],external_sha256=em['arrays_sha256'])
        if (dest/'identity.json').exists() and read(dest/'identity.json')!=identity:raise ValueError('Different fitted model')
        write(dest/'identity.json',identity)
        if (dest/'validation.json').exists():results.append(read(dest/'validation.json'));continue
        is_plain=method=='tree_only_rich';absolute=method=='chronos_tree_absolute'
        variant='ordinary' if is_plain or absolute else method.split('_residual_')[1]
        xx=rich if method.startswith('rich_') else original
        vv=rv if method.startswith('rich_') else ov
        if is_plain:xx=np.column_stack([X[:,plain],extra['train']]);vv=np.column_stack([V[:,plain],extra['valid']])
        base=np.zeros_like(y) if is_plain or absolute else point
        bv=np.zeros_like(yv) if is_plain or absolute else V[:,pi].copy()
        center=float(np.mean(y-base,dtype=np.float64));d=gates[horizon] if variant.startswith('gated_') else direction
        def objective(pred,dataset):
            error=pred-dataset.get_label();b=weights(error,c,variant,d)
            return b*error,b
        def rmse(pred,dataset):return 'rmse',float(np.sqrt(np.mean((pred-dataset.get_label())**2))),False
        reused=None
        if case['kind']=='real' and not method.startswith('rich_') and not is_plain:
            previous=args.reference_root.parents[1]
            if absolute:
                reused=previous/'cas_validation_pilot_centered_20261029'/market/f'window_{window}'/'absolute/ordinary'
            else:
                name={'ordinary':'ordinary','fixed':'fixed_alpha2','cas':'original_cas','mean':'mean_alpha2','shuffle':'shuffle_alpha2'}[variant]
                reused=previous/'cas_residual_strength_controls_20261029'/market/f'window_{window}'/'residual'/name
            prior=read(reused/'validation_metrics.json')
            if prior['prepared_sha256']!=meta['arrays_sha256'] or abs(prior['target_center']-center)>1e-10:raise ValueError('Incompatible reused fit')
            shutil.copy2(reused/'model.txt',dest/'model.txt');model=lgb.Booster(model_file=str(dest/'model.txt'))
        else:
            model=lgb.train(dict(params,objective='regression' if variant=='ordinary' else objective),
                lgb.Dataset(xx,label=y-base-center),num_boost_round=2000,
                valid_sets=[lgb.Dataset(vv,label=yv-bv-center)],valid_names=['valid'],feval=rmse,
                callbacks=[lgb.early_stopping(100,verbose=False)])
            model.save_model(str(dest/'model.txt'))
        pred=bv+center+model.predict(vv,num_threads=4)
        error=pred-yv
        row=dict(method=method,valid_mae=float(np.mean(abs(error))),valid_rmse=float(np.sqrt(np.mean(error**2))),
            best_iteration=model.best_iteration,center=center,reused_from=str(reused) if reused else None)
        write(dest/'validation.json',row);results.append(row);print('VALID',case['id'],json.dumps(row),flush=True)
    # Freeze selection using validation only, before this process accesses test arrays or external test scores.
    selected=min((r for r in results if r['method'] in CAS_METHODS),key=lambda r:(r['valid_rmse'],r['method']))['method']
    freeze=dict(protocol=PROTOCOL,selection='validation_rmse_only_among_3_CAS_variants',selected_cas=selected,
                reference_sha256=meta['arrays_sha256'],validation=results)
    if (folder/'selection.json').exists() and read(folder/'selection.json')!=freeze:raise ValueError('Frozen selection changed')
    write(folder/'selection.json',freeze)
    with np.load(ref/'features.npz') as z,np.load(ext/'inputs.npz') as e:
        T=z['X_test'];yt=z['y_test'];te=e['X_test'][e['row_ids_test']].astype(np.float32)
        np.testing.assert_array_equal(yt,e['yrows_test'])
        np.testing.assert_array_equal(T[:,hi].astype(int)-1,e['steps_test'])
        if not local_times(meta['test_time']).equals(local_times(em['test_time'])):
            raise ValueError('Test time mismatch')
    rows=[]
    def score(method,pred,validation=None):
        error=pred-yt
        row=dict(case_id=case['id'],kind=case['kind'],market=market,window=window,sign=case['sign'],strength=case['strength'],
            method=method,seed=42,selected_cas=method==selected,test_mae=float(np.mean(abs(error))),
            test_rmse=float(np.sqrt(np.mean(error**2))),test_bias=float(error.mean()),n_test=len(yt),
            reference_sha256=meta['arrays_sha256'])
        if validation:row.update({k:v for k,v in validation.items() if k.startswith('valid_')})
        rows.append(row)
        pd.DataFrame(dict(time=meta['test_time'],h=T[:,hi].astype(int),y_true=yt,prediction=pred)).to_csv(folder/('predictions_'+method+'.csv'),index=False)
    for r in results:
        method=r['method'];model=lgb.Booster(model_file=str(folder/'models'/method/'model.txt'))
        xx=T[:,keep];base=T[:,pi].copy()
        if method.startswith('rich_'):xx=np.column_stack([xx,te])
        if method=='tree_only_rich':xx=np.column_stack([T[:,plain],te]);base=np.zeros_like(yt)
        if method=='chronos_tree_absolute':base=np.zeros_like(yt)
        score(method,base+r['center']+model.predict(xx,num_threads=4),r)
    ve=V[:,pi]-yv
    score('chronos_only',T[:,pi],dict(valid_mae=float(np.mean(abs(ve))),valid_rmse=float(np.sqrt(np.mean(ve**2)))))
    # External baselines are unchanged for real data; rerun on changed source for synthetic cases.
    for model in ['lear_adapted','rnn']:
        dest=extroot/'training'/model/market/f'window_{window}'/'seed_42'
        if case['kind']!='real':
            subprocess.run([sys.executable,str(ROOT/'scripts/run_lear_rnn.py'),'--action','worker','--output-root',str(extroot),
                '--model',model,'--market',market,'--window',str(window),'--seed','42'],cwd=ROOT,check=True)
        m=read(dest/'metrics.json');pred=pd.read_csv(dest/'predictions_test.csv')
        if m['prepared_sha256']!=em['arrays_sha256']:raise ValueError('External score source mismatch')
        np.testing.assert_allclose(pred.y_true,yt,rtol=1e-7,atol=1e-6)
        if not local_times(pred.time).equals(local_times(meta['test_time'])):
            raise ValueError('External test time mismatch')
        score(model,pred.prediction.to_numpy(),m)
    pd.DataFrame(rows).to_csv(folder/'ALL_RUNS.csv',index=False)
    write(folder/'completion.json',dict(complete=len(rows)==len(METHODS)+3,expected=len(METHODS)+3,completed=len(rows)))
    print('CASE COMPLETE',case['id'],flush=True)


def summarize(args):
    found=[]
    for case in cases():
        p=args.output_root/'cases'/case['id']/'ALL_RUNS.csv'
        if p.exists():found.append(pd.read_csv(p))
    if found:
        allruns=pd.concat(found,ignore_index=True)
        if allruns.duplicated(['case_id','method','seed']).any():raise ValueError('Duplicate metrics')
        allruns.to_csv(args.output_root/'ALL_RUNS.csv',index=False)
        # Synthetic strengths and signs remain separate; never pooled into real-data headline scores.
        allruns.groupby(['kind','market','sign','strength','method'],as_index=False).agg(
            mae=('test_mae','mean'),rmse=('test_rmse','mean'),windows=('window','nunique')).to_csv(args.output_root/'SUMMARY.csv',index=False)
    return len(found)


def preflight(args):
    import lightgbm,torch,chronos.chronos2
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for fresh Chronos and RNN runs')
    if not (args.model_path/'config.json').is_file():raise FileNotFoundError(args.model_path/'config.json')
    for case in [c for c in cases() if c['kind']=='real']:
        m=case['market'];w=case['window']
        ref=args.reference_root/m/f'window_{w}';meta=read(ref/'manifest.json')
        if meta['rollout_protocol']!=ROLLOUT_PROTOCOL or sha(ref/'features.npz')!=meta['arrays_sha256']:
            raise ValueError('Reference not ready: '+str(ref))
        ext=args.external_root/'prepared'/m/f'window_{w}';em=read(ext/'manifest.json')
        if em['identity']['reference_sha256']!=meta['arrays_sha256'] or sha(ext/'inputs.npz')!=em['arrays_sha256']:
            raise ValueError('External features not ready: '+str(ext))
        for model in ['lear_adapted','rnn']:
            folder=args.external_root/'training'/model/m/f'window_{w}'/'seed_42'
            info=read(folder/'metrics.json')
            if info['prepared_sha256']!=em['arrays_sha256'] or not (folder/'predictions_test.csv').is_file():
                raise ValueError('Existing baseline not ready: '+str(folder))
        previous=args.reference_root.parents[1]
        for kind in ['ordinary','fixed_alpha2','original_cas','mean_alpha2','shuffle_alpha2']:
            folder=previous/'cas_residual_strength_controls_20261029'/m/f'window_{w}'/'residual'/kind
            if read(folder/'validation_metrics.json')['prepared_sha256']!=meta['arrays_sha256'] or not (folder/'model.txt').is_file():
                raise ValueError('Reused fit not ready: '+str(folder))
    print('PREFLIGHT READY',dict(cases=18,evaluations=324,torch=torch.__version__,lightgbm=lightgbm.__version__,visible_gpus=torch.cuda.device_count()),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['all','case','summarize','check'],default='all')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_directional_study_20261029')
    p.add_argument('--reference-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--external-root',type=Path,default=ROOT/'results/lear_rnn_external_20261029_v2')
    p.add_argument('--model-path',type=Path,default=ROOT.parent/'CAS-EFP/models/chronos-2')
    p.add_argument('--case-id');p.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    args=p.parse_args()
    for key in ['output_root','reference_root','external_root','model_path']:setattr(args,key,getattr(args,key).resolve())
    args.output_root.mkdir(parents=True,exist_ok=True)
    plan=dict(protocol=PROTOCOL,script_sha256=sha(__file__),cases=cases(),methods=METHODS,
              roots={k:str(getattr(args,k)) for k in ['reference_root','external_root','model_path']},
              helper_sha256={name:sha(ROOT/'scripts'/name) for name in ['run_mlp_transfer.py','run_lear_rnn.py','pilot_cas_optimization.py']},
              external=['chronos_only','lear_adapted','rnn'],seed=42,test_driven_search=False,
              stress='target_only_signed_linear_ramp_after_training_start; +/-0.5/1.0 original train std per 90 days',
              note='Semi-synthetic cases are mechanism stress tests, not real Guangdong performance claims. Previously inspected real test periods are development results.')
    if (args.output_root/'plan.json').exists():
        if read(args.output_root/'plan.json')!=plan:raise ValueError('Preregistered plan changed')
    else:write(args.output_root/'plan.json',plan)
    if args.action=='case':
        case=next(c for c in cases() if c['id']==args.case_id)
        folder=args.output_root/'cases'/case['id'];folder.mkdir(parents=True,exist_ok=True)
        # Implementation-only recovery can reuse an inference still running in an
        # earlier process. Wait for its OS lock, then resume verified artifacts.
        import fcntl
        with (folder/'launcher.lock').open('a+') as handle:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        with run_lock(folder):fit_case(args,case)
        return
    if args.action=='summarize':summarize(args);return
    preflight(args)
    if args.action=='check':return
    with run_lock(args.output_root):
        cards=queue.Queue()
        for gpu in args.gpus:cards.put(gpu)
        logs=args.output_root/'logs';logs.mkdir(exist_ok=True)
        failures=[]
        def run(case):
            gpu=cards.get();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            try:
                print('START',case['id'],'GPU',gpu,flush=True)
                with (logs/(case['id']+'.log')).open('a') as log:
                    subprocess.run([sys.executable,__file__,'--action','case','--case-id',case['id'],
                        '--output-root',str(args.output_root),'--reference-root',str(args.reference_root),
                        '--external-root',str(args.external_root),'--model-path',str(args.model_path)],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                print('DONE',case['id'],flush=True)
            finally:cards.put(gpu)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures={pool.submit(run,c):c for c in cases()}
            for f in as_completed(futures):
                try:f.result()
                except Exception as e:
                    failures.append(dict(case=futures[f],error=str(e)));print('FAILED',futures[f]['id'],str(e),flush=True)
                write(args.output_root/'progress.json',dict(completed_cases=summarize(args),expected_cases=len(cases()),failures=failures))
        n=summarize(args)
        write(args.output_root/'completion.json',dict(complete=n==len(cases()) and not failures,completed_cases=n,expected_cases=len(cases()),failures=failures))
        if failures:raise RuntimeError('See completion.json and individual logs')


if __name__=='__main__':main()
