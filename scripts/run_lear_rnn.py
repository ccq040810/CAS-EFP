"""Train only external LEAR-adapted and vanilla RNN baselines; existing results are read-only."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,os,sys,subprocess,queue
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha
from scripts.run_cas_all import run_lock
from scripts.run_rolling_adaptive_experiment import DATASETS
PROTOCOL='external_hourly_origin_24h_causal_LEAR_adapted_RNN_v1'


def build_inputs(df,origins,target,covariates):
    origins=pd.DatetimeIndex(origins)
    # Origins are timezone-aware; object arithmetic retains exact instants.
    def take(offsets,columns,filled=False):
        offsets=list(offsets)
        times=origins.repeat(len(offsets))+pd.to_timedelta(np.tile(offsets,len(origins)),unit='h')
        source=df[columns].ffill() if filled else df[columns]
        return source.reindex(times).to_numpy(dtype=float).reshape(len(origins),len(offsets),len(columns))
    history=take(range(-168,0),[target],True)
    if not np.isfinite(history).all():raise ValueError('Unfillable target history; refusing to change test rows')
    prices=np.stack([history[:,168-24*d:168-24*d+24,0] for d in [1,2,3,7]],axis=-1).reshape(len(origins),96)
    cov=[]
    for h in range(24):
        for d in [1,7,0]:
            cov.append(take([h-24*d],covariates)[:,0,:])
    exog=np.concatenate(cov,axis=1) if cov else np.empty((len(origins),0))
    dummy=np.eye(7)[origins.dayofweek]
    hour=np.column_stack([np.sin(origins.hour*2*np.pi/24),np.cos(origins.hour*2*np.pi/24)])
    return history.astype(np.float32),np.concatenate([prices,exog,hour,dummy],axis=1),np.concatenate([exog,hour,dummy],axis=1)


def prepare(args,market,window):
    from exp.pipeline.tsfm_infer import _read_table,_prepare_time_index,_to_shanghai
    folder=args.output_root/'prepared'/market/f'window_{window}'
    reference=args.reference_root/market/f'window_{window}'
    manifest=read(reference/'manifest.json')
    identity=dict(protocol=PROTOCOL,reference_sha256=manifest['arrays_sha256'],data_sha256=sha(ROOT/DATASETS[market][0]))
    if (folder/'manifest.json').exists():
        old=read(folder/'manifest.json')
        if old['identity']!=identity or sha(folder/'inputs.npz')!=old['arrays_sha256']:raise ValueError('Prepared inputs changed')
        return
    folder.mkdir(parents=True,exist_ok=True)
    data,timecol,config,_,_=DATASETS[market]
    columns=read(ROOT/config)[0]['target_columns'];target=columns[-1]
    df=_prepare_time_index(_read_table(str(ROOT/data)),timecol)
    if not df.index.is_unique:raise ValueError('Duplicate source timestamps')
    cov=[c for c in columns[:-1] if c in df.columns and c not in ['hour_of_day','day_of_week','is_holiday']]
    arrays={};metadata={}
    with np.load(reference/'features.npz') as ref:
        if sha(reference/'features.npz')!=manifest['arrays_sha256']:raise ValueError('Reference checksum mismatch')
        for split,key in [('train','train'),('valid','valid'),('test','test')]:
            times=pd.DatetimeIndex([_to_shanghai(pd.Timestamp(t)) for t in manifest[split+'_time']])
            Y=ref['y_'+split].reshape(-1,24)
            matrix=np.asarray(times).reshape(-1,24);origins=pd.DatetimeIndex(matrix[:,0])
            for h in range(24):
                if not (pd.DatetimeIndex(matrix[:,h])==origins+pd.Timedelta(hours=h)).all():raise ValueError('Noncontiguous target horizon')
            observed=df[target].reindex(times).to_numpy().reshape(-1,24)
            np.testing.assert_allclose(observed,Y,rtol=1e-6,atol=1e-6)
            seq,X,Z=build_inputs(df,origins,target,cov)
            arrays.update({f'seq_{split}':seq,f'X_{split}':X,f'Z_{split}':Z,f'y_{split}':Y})
            metadata[split+'_time']=[str(t) for t in times]
    np.savez(folder/'inputs.npz',**arrays)
    write(folder/'manifest.json',dict(identity=identity,arrays_sha256=sha(folder/'inputs.npz'),covariates=cov,**metadata))
    print('PREPARED',market,window,flush=True)


def fit_scaler(X):
    med=np.nanmedian(np.where(np.isfinite(X),X,np.nan),axis=0);med=np.where(np.isfinite(med),med,0)
    clean=np.where(np.isfinite(X),X,med)
    scale=np.median(np.abs(clean-med),axis=0)/.6744897501960817
    return med,np.where(scale>1e-8,scale,1.)


def worker(args):
    import torch
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    folder=args.output_root/'prepared'/args.market/f'window_{args.window}'
    meta=read(folder/'manifest.json')
    if sha(folder/'inputs.npz')!=meta['arrays_sha256']:raise ValueError('Inputs changed')
    with np.load(folder/'inputs.npz') as z:arrays={k:z[k] for k in z.files}
    dest=args.output_root/'training'/args.model/args.market/f'window_{args.window}'/f'seed_{args.seed}'
    dest.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol=PROTOCOL,model=args.model,seed=args.seed,prepared_sha256=meta['arrays_sha256'])
    if (dest/'run_config.json').exists() and read(dest/'run_config.json')!=identity:raise ValueError('Different job')
    if (dest/'metrics.json').exists():return
    write(dest/'run_config.json',identity)
    y=arrays['y_train'];ym,ys=fit_scaler(y)
    predictions={};history=[]
    if args.model=='lear_adapted':
        from sklearn.linear_model import LassoLarsIC,Lasso
        import joblib
        X=arrays['X_train'];xm,xs=fit_scaler(X[:,:-7])
        def transform(v):return np.column_stack([np.arcsinh((np.where(np.isfinite(v[:,:-7]),v[:,:-7],xm)-xm)/xs),v[:,-7:]])
        Xt=transform(X);Yt=np.arcsinh((y-ym)/ys);models=[]
        predictions={s:np.zeros_like(arrays['y_'+s],dtype=float) for s in ['valid','test']}
        for h in range(24):
            alpha=LassoLarsIC(criterion='aic',max_iter=2500).fit(Xt,Yt[:,h]).alpha_
            model=Lasso(alpha=alpha,max_iter=2500).fit(Xt,Yt[:,h]);models.append(model)
            for s in predictions:predictions[s][:,h]=np.sinh(model.predict(transform(arrays['X_'+s])))*ys[h]+ym[h]
            print('LEAR_HOUR',h,'alpha',alpha,flush=True)
        joblib.dump(dict(models=models,xm=xm,xs=xs,ym=ym,ys=ys),dest/'model.joblib')
    else:
        from torch import nn
        torch.use_deterministic_algorithms(True)
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        m=float(y.mean());sd=max(float(y.std()),1e-6)
        zm,zs=fit_scaler(arrays['Z_train'])
        def tensors(s):
            seq=torch.tensor((arrays['seq_'+s]-m)/sd,dtype=torch.float32)
            v=arrays['Z_'+s];v=np.where(np.isfinite(v),v,zm)
            return seq,torch.tensor((v-zm)/zs,dtype=torch.float32),torch.tensor((arrays['y_'+s]-m)/sd,dtype=torch.float32)
        tensors_by_split={s:tensors(s) for s in ['train','valid','test']}
        class RNN(nn.Module):
            def __init__(self):
                super().__init__();self.rnn=nn.RNN(1,64,batch_first=True,nonlinearity='tanh')
                self.head=nn.Sequential(nn.Linear(64+len(zm),64),nn.ReLU(),nn.Linear(64,24))
            def forward(self,a,b):
                _,state=self.rnn(a);return self.head(torch.cat([state[-1],b],dim=1))
        model=RNN().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        def predict(s):
            model.eval();a,b,_=tensors_by_split[s]
            with torch.no_grad():return np.concatenate([model(a[i:i+512].to(device),b[i:i+512].to(device)).cpu().numpy() for i in range(0,len(a),512)])*sd+m
        best=float('inf');stale=0
        for epoch in range(1,151):
            model.train();a,b,t=tensors_by_split['train'];order=torch.randperm(len(a))
            for idx in order.split(256):
                optimizer.zero_grad();loss=((model(a[idx].to(device),b[idx].to(device))-t[idx].to(device))**2).mean();loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            pred=predict('valid');score=float(np.sqrt(np.mean((pred-arrays['y_valid'])**2)))
            history.append(dict(epoch=epoch,valid_rmse=score));print('EPOCH',epoch,score,flush=True)
            if score<best:
                best=score;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if stale>=20:break
        model.load_state_dict(state)
        predictions={s:predict(s) for s in ['valid','test']}
        torch.save(dict(state=state,price_mean=m,price_std=sd,exog_median=zm,exog_scale=zs),dest/'model.pt')
    metrics=dict(**identity,market=args.market,window=args.window)
    for split,pred in predictions.items():
        if not np.isfinite(pred).all():raise ValueError('Nonfinite forecast')
        error=pred-arrays['y_'+split]
        metrics[split+'_mae']=float(np.abs(error).mean());metrics[split+'_rmse']=float(np.sqrt(np.mean(error**2)))
        pd.DataFrame(dict(time=meta[split+'_time'],y_true=arrays['y_'+split].ravel(),prediction=pred.ravel())).to_csv(dest/f'predictions_{split}.csv',index=False)
    pd.DataFrame(history).to_csv(dest/'history.csv',index=False);write(dest/'metrics.json',metrics)
    print(json.dumps(metrics),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['all','worker'],default='all')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/lear_rnn_external_20261029')
    p.add_argument('--reference-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--model',choices=['lear_adapted','rnn']);p.add_argument('--market');p.add_argument('--window',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--gpus',nargs='+',default=['0','1','2','3','4','5','6','7'])
    args=p.parse_args();args.output_root=args.output_root.resolve();args.output_root.mkdir(parents=True,exist_ok=True)
    if args.action=='worker':worker(args);return
    with run_lock(args.output_root):
        config=dict(protocol=PROTOCOL,reference_root=str(args.reference_root.resolve()),script_sha256=sha(__file__),models=['lear_adapted','rnn'],seeds=[42,43,44])
        if (args.output_root/'experiment.json').exists() and read(args.output_root/'experiment.json')!=config:raise ValueError('Different experiment')
        write(args.output_root/'experiment.json',config)
        for m in ['guangdong','reale_delu']:
            for w in [1,2,3]:prepare(args,m,w)
        cards=queue.Queue()
        for card in args.gpus:cards.put(card)
        logs=args.output_root/'logs';logs.mkdir(exist_ok=True)
        jobs=[(model,m,w,s) for model in ['lear_adapted','rnn'] for m in ['guangdong','reale_delu'] for w in [1,2,3] for s in [42,43,44]]
        failures=[]
        def run(j):
            model,m,w,s=j;gpu=cards.get() if model=='rnn' else ''
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            try:
                print('TRAIN',j,flush=True)
                with (logs/f'{model}_{m}_w{w}_s{s}.log').open('w') as log:
                    subprocess.run([sys.executable,__file__,'--action','worker','--output-root',str(args.output_root),'--model',model,'--market',m,'--window',str(w),'--seed',str(s)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            finally:
                if model=='rnn':cards.put(gpu)
        def pool(model,n):
            errors=[]
            with ThreadPoolExecutor(max_workers=n) as executor:
                futures={executor.submit(run,j):j for j in jobs if j[0]==model}
                for f in as_completed(futures):
                    try:f.result()
                    except Exception as e:errors.append(dict(job=futures[f],error=str(e)))
            return errors
        with ThreadPoolExecutor(max_workers=2) as executor:
            for future in [executor.submit(pool,'lear_adapted',2),executor.submit(pool,'rnn',len(args.gpus))]:failures.extend(future.result())
        rows=[read(x) for x in (args.output_root/'training').glob('*/*/window_*/seed_*/metrics.json')]
        pd.DataFrame(rows).to_csv(args.output_root/'ALL_RUNS.csv',index=False)
        write(args.output_root/'completion.json',dict(expected=36,completed=len(rows),complete=len(rows)==36,failures=failures))
        if len(rows)==36:
            frame=pd.DataFrame(rows);per=frame.groupby(['model','market','seed'],as_index=False).agg(mae=('test_mae','mean'),rmse=('test_rmse','mean'))
            per.groupby(['model','market'],as_index=False).agg(mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'),rmse_std=('rmse','std')).to_csv(args.output_root/'SUMMARY.csv',index=False)


if __name__=='__main__':main()
