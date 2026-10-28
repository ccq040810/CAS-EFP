"""DIAGNOSTIC ONLY: replay legacy inputs and parameters, never a valid forecast benchmark.

Chronos uses the exact archived native-quantile features. Replacement upstreams
use the literal legacy integer history lookup in their respective environments.
No imputation or timestamp-unit fix is applied. Failures remain visible.
"""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_mlp_transfer as transfer
from scripts.run_cas_all import run_lock

PROTOCOL='legacy_integer_lookup_native_environment_DIAGNOSTIC_ONLY_v1'
METHODS=[*transfer.MECHANISM,'tsfm_point_confidence_asym_noC','plain_regressor']
SOURCES=['cas_chronos2','cas_moirai2','cas_sundial']
MARKETS=['guangdong','reale_delu']
TREE_PARAMS=dict(feature_fraction=.9,bagging_fraction=.8,bagging_freq=5,min_gain_to_split=.08)


class LegacyIndex:
    def __init__(self,index):self.index=index
    def searchsorted(self,anchor,side='left'):
        import numpy as np
        return int(np.searchsorted(self.index.view('int64'),int(anchor.value),side=side))


class LegacyFrame:
    def __init__(self,frame):self.frame=frame;self.index=LegacyIndex(frame.index)
    def __getattr__(self,name):return getattr(self.frame,name)


def install_training_protocol():
    from exp.pipeline import cas_lgbm_regressor as backend
    base=backend.CASLGBMRegressor
    class OriginalParameters(base):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.params.update(TREE_PARAMS)
            self.params.pop('deterministic',None)
            self.params.pop('force_col_wise',None)
    backend.CASLGBMRegressor=OriginalParameters
    original=transfer.job_config
    def settings(args,manifest,method,seed):
        config=original(args,manifest,method,seed)
        config.update(diagnostic_only=True,history_protocol=PROTOCOL,
                      lgbm_extra_params=TREE_PARAMS if args.regressor=='lightgbm' else None)
        return config
    transfer.job_config=settings


def train_args(args):
    return SimpleNamespace(prepared_root=args.output_root/'prepared'/args.source,
        output_root=args.output_root/'training'/args.source/args.backend,
        regressor=args.backend,market=args.market,window=args.window,method=args.method,
        seed=args.seed,alpha=2.,weight_lambda=1.,hidden=[128,64],lr=.001,batch_size=8192,
        epochs=300,patience=30,selection='legacy',device='cuda:0' if args.backend=='mlp' else 'cpu',
        threads=args.threads,lgbm_rounds=2000,lgbm_patience=100)


def prepare_child(args):
    import numpy as np
    from exp.pipeline import tsfm_infer
    final=args.output_root/'prepared'/args.source/args.market/f'window_{args.window}'
    if (final/'manifest.json').exists():
        manifest=transfer.read(final/'manifest.json')
        if manifest.get('diagnostic_protocol')!=PROTOCOL or transfer.sha(final/'features.npz')!=manifest['arrays_sha256']:
            raise ValueError('Unverified prepared features')
        for path,digest in manifest['source_sha256'].items():
            if transfer.sha(path)!=digest:raise ValueError(f'Prepared source changed: {path}')
        return
    staging=args.output_root/'staging'/uuid.uuid4().hex
    staging.mkdir(parents=True)
    if args.source=='cas_chronos2':
        old=ROOT/'results/reproduction_audit_20261028/old_cache_prepared'/args.market/f'window_{args.window}'
        manifest=transfer.read(old/'manifest.json')
        if transfer.sha(old/'features.npz')!=manifest['arrays_sha256']:
            raise ValueError('Archived original feature checksum mismatch')
        ready=staging/args.market/f'window_{args.window}'
        shutil.copytree(old,ready)
        audit=dict(mode='exact_archived_native_chronos_features',source=str(old),
                   causal_validity='known future-history contamination; diagnostic only')
    else:
        original=tsfm_infer._predict_multivar
        audit=dict(mode='literal_legacy_integer_lookup',origins=[])
        def legacy(**kwargs):
            df=kwargs['df'];index=LegacyIndex(df.index)
            for anchor in kwargs['anchors']:
                pos=index.searchsorted(anchor)
                correct=int(df.index.searchsorted(anchor))
                history=df.iloc[pos-kwargs['seq_len']:pos][list(kwargs['mv_cols'])].to_numpy(np.float32)
                audit['origins'].append(dict(anchor=str(anchor),index_dtype=str(df.index.dtype),
                    legacy_position=pos,correct_position=correct,
                    history_end=str(df.index[pos-1]) if pos>0 else None,
                    valid_history=bool(history.shape==(kwargs['seq_len'],len(kwargs['mv_cols'])) and np.isfinite(history).all()),
                    future_history=bool(pos>correct)))
            kwargs['df']=LegacyFrame(df)
            kwargs['causal_context_fill']=False
            return original(**kwargs)
        tsfm_infer._predict_multivar=legacy
        transfer.ROLLOUT_PROTOCOL=PROTOCOL
        paths=dict(cas_moirai2=args.bundle_root/'models/moirai-2.0-R-small',
                   cas_sundial=args.bundle_root/'models/sundial-base-128m')
        try:
            transfer.prepare(SimpleNamespace(upstream=args.source,markets=[args.market],windows=[args.window],
                prepared_root=staging,split_root=args.old_root/'results/rolling_adaptive_v3',
                cache_root=args.output_root/'cache',model_path=str(paths[args.source]),device='cuda:0',tsfm_batch_size=16))
        finally:
            transfer.write(staging/'HISTORY_AUDIT.json',audit)
            audit_path=args.output_root/'history_audits'/args.source/args.market
            audit_path.mkdir(parents=True,exist_ok=True)
            transfer.write(audit_path/f'window_{args.window}.json',audit)
        ready=staging/args.market/f'window_{args.window}'
        manifest=transfer.read(ready/'manifest.json')
    manifest.update(diagnostic_only=True,diagnostic_protocol=PROTOCOL,
                    upstream=args.source,source_distribution_protocol=manifest['distribution_protocol'])
    transfer.write(ready/'manifest.json',manifest)
    transfer.write(ready/'HISTORY_AUDIT.json',audit)
    final.parent.mkdir(parents=True,exist_ok=True)
    ready.rename(final)


def summarize(args,jobs,failures):
    import numpy as np
    import pandas as pd
    rows=[];missing=[];upstream=[]
    for source,backend,market,window,method,seed in jobs:
        folder=args.output_root/'training'/source/backend/market/f'window_{window}'/method/f'seed_{seed}'
        if not (folder/'metrics.json').exists():missing.append([source,backend,market,window,method,seed]);continue
        row=transfer.read(folder/'metrics.json');row['diagnostic_only']=True
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_root/'DIAGNOSTIC_ALL_RUNS.csv',index=False)
    if rows:
        frame=pd.DataFrame(rows)
        per=frame.groupby(['upstream','regressor','market','method','seed'],as_index=False).agg(
            mae=('test_mae','mean'),rmse=('test_rmse','mean'),windows=('window','nunique'))
        per.to_csv(args.output_root/'DIAGNOSTIC_PER_SEED.csv',index=False)
        complete=per[per.windows==3]
        report=complete.groupby(['upstream','regressor','market','method'],as_index=False).agg(
            mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),seeds=('seed','nunique'))
        report.to_csv(args.output_root/'DIAGNOSTIC_SUMMARY.csv',index=False)
    else:report=pd.DataFrame()
    for source in SOURCES:
        for market in MARKETS:
            for window in [1,2,3]:
                folder=args.output_root/'prepared'/source/market/f'window_{window}'
                if not (folder/'manifest.json').exists():continue
                manifest=transfer.read(folder/'manifest.json')
                with np.load(folder/'features.npz') as z:
                    e=z['X_test'][:,manifest['point_index']]-z['y_test']
                    upstream.append(dict(upstream=source,market=market,window=window,
                        mae=float(np.mean(np.abs(e))),rmse=float(np.sqrt(np.mean(e**2)))))
    pd.DataFrame(upstream).to_csv(args.output_root/'DIAGNOSTIC_UPSTREAM_ONLY.csv',index=False)
    transfer.write(args.output_root/'completion.json',dict(diagnostic_only=True,expected=len(jobs),
        completed=len(rows),missing=missing,complete=not missing,failures=failures))
    lines=['# 旧协议诊断结果','',
        '**仅用于协议诊断，包含已知或待审计的未来历史输入，不是有效预测性能结果。**','',
        f'完成 {len(rows)}/{len(jobs)} 次训练。失败与缺失详见completion.json。',
        'Chronos保留原缓存原生分位网格；替换模型沿用已有共同Q10—Q90适配器。不同环境原始日期单位可能不同，逐起报点记录于HISTORY_AUDIT.json；不进行填充或时间修复。',
        '树参数与早停恢复旧版，MLP沿用既定配置。plain_regressor排除上游点和c；按上游重复训练的纯小模型不是独立模型家族。','',
        '|上游|下游|市场|配置|MAE|RMSE|种子数|','|---|---|---|---|---:|---:|---:|']
    for r in report.itertuples():lines.append(f'|{r.upstream}|{r.regressor}|{r.market}|{r.method}|{r.mae_mean:.6f}|{r.rmse_mean:.6f}|{r.seeds}|')
    (args.output_root/'DIAGNOSTIC_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['all','prepare','worker'],default='all')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/legacy_protocol_diagnostic_20261028')
    p.add_argument('--old-root',type=Path,default=ROOT.parent/'CAS-EFP')
    p.add_argument('--bundle-root',type=Path,default=ROOT.parent/'cas-offline-bundle-20261027')
    p.add_argument('--env-root',type=Path,default=ROOT.parent/'cas-envs')
    p.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--source',choices=SOURCES);p.add_argument('--backend',choices=['lightgbm','mlp'])
    p.add_argument('--market',choices=MARKETS);p.add_argument('--window',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--method',choices=METHODS)
    args=p.parse_args()
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or args.threads<1:p.error('Unique GPU IDs and positive threads required')
    args.output_root=args.output_root.resolve();args.output_root.mkdir(parents=True,exist_ok=True)
    if 'diagnostic' not in args.output_root.name.lower():p.error('Output root must explicitly contain diagnostic')
    if args.action=='prepare':prepare_child(args);return
    install_training_protocol()
    if args.action=='worker':
        transfer.worker(train_args(args))
        dest=args.output_root/'training'/args.source/args.backend/args.market/f'window_{args.window}'/args.method/f'seed_{args.seed}'
        metrics=transfer.read(dest/'metrics.json');metrics.update(diagnostic_only=True,diagnostic_protocol=PROTOCOL)
        transfer.write(dest/'metrics.json',metrics)
        return
    jobs=[(s,b,m,w,t,seed) for s in SOURCES for b in ['lightgbm','mlp'] for m in MARKETS
          for w in [1,2,3] for t in METHODS for seed in [42,43,44]]
    logs=args.output_root/'logs';logs.mkdir(exist_ok=True)
    cards=queue.Queue()
    for card in args.gpus:cards.put(card)
    python=dict(cas_chronos2=sys.executable,cas_moirai2=str(args.env_root/'moirai-py312/bin/python'),
                cas_sundial=str(args.env_root/'sundial-py312/bin/python'))
    failures=[]
    def execute(command,log,gpu):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if gpu=='cpu' else gpu,
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
            CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with log.open('w') as stream:subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    with run_lock(args.output_root):
        config=dict(diagnostic_only=True,protocol=PROTOCOL,methods=METHODS,tree_params=TREE_PARAMS,
                    selection='legacy',seeds=[42,43,44],expected=len(jobs),script_sha256=transfer.sha(Path(__file__)))
        path=args.output_root/'experiment.json'
        if path.exists() and transfer.read(path)!=config:raise ValueError('Different diagnostic configuration')
        transfer.write(path,config)
        def prepare(job):
            source,market,window=job;gpu=cards.get()
            try:
                print('PREPARE',source,market,window,flush=True)
                execute([python[source],__file__,'--action','prepare','--source',source,'--market',market,'--window',window,
                         '--output-root',args.output_root,'--old-root',args.old_root,'--bundle-root',args.bundle_root],logs/f'prepare_{source}_{market}_w{window}.log',gpu)
            finally:cards.put(gpu)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            pending={pool.submit(prepare,(s,m,w)):(s,m,w) for s in SOURCES for m in MARKETS for w in [1,2,3]}
            for future in as_completed(pending):
                try:future.result()
                except Exception as e:failures.append(dict(stage='prepare',job=pending[future],error=str(e)))
        transfer.write(args.output_root/'preparation_failures.json',failures)
        def train(job):
            source,backend,market,window,method,seed=job
            folder=args.output_root/'prepared'/source/market/f'window_{window}'
            if not (folder/'manifest.json').exists():return
            dest=args.output_root/'training'/source/backend/market/f'window_{window}'/method/f'seed_{seed}'
            a=SimpleNamespace(**vars(args));a.source=source;a.backend=backend;a.market=market;a.window=window;a.method=method;a.seed=seed
            manifest=transfer.read(folder/'manifest.json')
            if transfer.sha(folder/'features.npz')!=manifest['arrays_sha256']:raise ValueError('Prepared checksum changed')
            required=transfer.job_config(train_args(a),manifest,method,seed)
            if (dest/'run_config.json').exists() and transfer.read(dest/'run_config.json')!=required:raise ValueError('Different training settings')
            if (dest/'metrics.json').exists() and (dest/'run_config.json').exists():return
            gpu=cards.get() if backend=='mlp' else 'cpu'
            try:
                print('TRAIN',job,flush=True)
                execute([sys.executable,__file__,'--action','worker','--source',source,'--backend',backend,'--market',market,
                         '--window',window,'--seed',seed,'--method',method,'--threads',args.threads,'--output-root',args.output_root],
                        logs/f'{source}_{backend}_{market}_w{window}_{method}_s{seed}.log',gpu)
            finally:
                if backend=='mlp':cards.put(gpu)
        def train_pool(backend,workers):
            errors=[]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending={pool.submit(train,j):j for j in jobs if j[1]==backend}
                for future in as_completed(pending):
                    try:future.result()
                    except Exception as e:errors.append(dict(stage='train',job=pending[future],error=str(e)))
            return errors
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(train_pool,b,n) for b,n in [('mlp',len(args.gpus)),('lightgbm',4)]]
            for future in futures:failures.extend(future.result())
        summarize(args,jobs,failures)
        print('DIAGNOSTIC FINISHED',args.output_root/'completion.json',flush=True)


if __name__=='__main__':main()
