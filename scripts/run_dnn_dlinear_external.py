"""Only add DNN-adapted and native univariate DLinear; prior results read-only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,os,sys,subprocess,queue
import numpy as np
import pandas as pd
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha
from scripts.run_cas_all import run_lock
from scripts.run_lear_rnn import fit_scaler


class DLinear(nn.Module):
    """Official DLinear decomposition/linear mapping, single price channel, K25."""
    def __init__(self):
        super().__init__();self.avg=nn.AvgPool1d(25,stride=1)
        self.seasonal=nn.Linear(168,24);self.trend=nn.Linear(168,24)
    def forward(self,x):
        trend=self.avg(torch.cat([x[:,:1].repeat(1,12),x,x[:,-1:].repeat(1,12)],dim=1).unsqueeze(1)).squeeze(1)
        return self.seasonal(x-trend)+self.trend(trend)


def masked_mse(pred,target,mask):
    return ((pred-target).square()*mask).sum()/mask.sum()


def worker(args):
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);torch.use_deterministic_algorithms(True)
    src=args.prepared_root/args.market/f'window_{args.window}';meta=read(src/'manifest.json')
    assert sha(src/'inputs.npz')==meta['arrays_sha256']
    out=args.output_root/'training'/args.model/args.market/f'window_{args.window}'/f'seed_{args.seed}';out.mkdir(parents=True,exist_ok=True)
    identity=dict(model=args.model,market=args.market,window=args.window,seed=args.seed,prepared_sha256=meta['arrays_sha256'],script_sha256=sha(__file__))
    if (out/'run_config.json').exists():assert read(out/'run_config.json')==identity
    write(out/'run_config.json',identity)
    if (out/'metrics.json').exists():return
    with np.load(src/'inputs.npz') as z:
        data={s:{k:z[k+'_'+s] for k in ['X','seq','y','row_ids','steps','yrows']} for s in ['train','valid']}
    y=data['train']['y'];is_dnn=args.model=='dnn_adapted'
    if is_dnn:
        xm,xs=fit_scaler(data['train']['X']);ym,ys=fit_scaler(y)
        model=nn.Sequential(nn.Linear(len(xm),256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,24))
        def inputs(d):return np.arcsinh((np.where(np.isfinite(d['X']),d['X'],xm)-xm)/xs)
        def targets(v):return np.arcsinh((v-ym)/ys)
        def inverse(v):return np.sinh(v.astype(float))*ys+ym
        scaler=dict(xm=xm,xs=xs,ym=ym,ys=ys)
    else:
        mean=float(np.nanmean(y));sd=max(float(np.nanstd(y)),1e-6);model=DLinear()
        def inputs(d):return (d['seq'][:,:,0]-mean)/sd
        def targets(v):return (v-mean)/sd
        def inverse(v):return v.astype(float)*sd+mean
        scaler=dict(mean=mean,sd=sd)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');model=model.to(device)
    tensors={}
    for s,d in data.items():
        mask=np.isfinite(d['y']);target=np.where(mask,targets(d['y']),0)
        tensors[s]=(torch.tensor(inputs(d),dtype=torch.float32),torch.tensor(target,dtype=torch.float32),torch.tensor(mask,dtype=torch.float32))
    def predict(d):
        model.eval();x=torch.tensor(inputs(d),dtype=torch.float32)
        with torch.no_grad():pred=np.concatenate([model(x[i:i+512].to(device)).cpu().numpy() for i in range(0,len(x),512)])
        result=inverse(pred)
        if not np.isfinite(result).all():raise ValueError('Nonfinite inverse prediction; no clipping')
        return result[d['row_ids'],d['steps']]
    opt=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');stale=0;history=[];best_epoch=0
    for epoch in range(1,(151 if is_dnn else 201)):
        model.train();x,t,mask=tensors['train']
        for idx in torch.randperm(len(x)).split(256):
            opt.zero_grad();loss=masked_mse(model(x[idx].to(device)),t[idx].to(device),mask[idx].to(device));loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        p=predict(data['valid']);score=float(np.sqrt(np.mean((p-data['valid']['yrows'])**2)))
        history.append(dict(epoch=epoch,valid_rmse=score));print('EPOCH',epoch,score,flush=True)
        if score<best:
            best=score;stale=0;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=20:break
    model.load_state_dict(state);torch.save(dict(state=state,scaler=scaler),out/'model.pt')
    write(out/'TRAINING_FROZEN.json',dict(**identity,best_epoch=best_epoch,model_sha256=sha(out/'model.pt'),test_arrays_read=False))
    # Test arrays are accessed only after the selected checkpoint is saved.
    with np.load(src/'inputs.npz') as z:data['test']={k:z[k+'_test'] for k in ['X','seq','y','row_ids','steps','yrows']}
    metrics=dict(**identity,best_epoch=best_epoch,device=str(device))
    for s in ['valid','test']:
        d=data[s];pred=predict(d);error=pred-d['yrows'].astype(float)
        metrics[s+'_mae']=float(np.abs(error).mean());metrics[s+'_rmse']=float(np.sqrt(np.mean(error*error)))
        pd.DataFrame(dict(time=meta[s+'_time'],h=d['steps']+1,y_true=d['yrows'],prediction=pred)).to_csv(out/f'predictions_{s}.csv',index=False)
    pd.DataFrame(history).to_csv(out/'history.csv',index=False);write(out/'metrics.json',metrics);print(metrics,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--prepared-root',type=Path,default=ROOT/'results/lear_rnn_external_20261029_v2/prepared')
    p.add_argument('--prior-root',type=Path,default=ROOT/'results/cas_original_windows_saved_score_20261029')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/dnn_dlinear_external_20261029')
    p.add_argument('--model',choices=['dnn_adapted','dlinear']);p.add_argument('--market');p.add_argument('--window',type=int);p.add_argument('--seed',type=int)
    args=p.parse_args()
    for name in ['prepared_root','prior_root','output_root']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args);return
    args.output_root.mkdir(parents=True,exist_ok=True)
    with run_lock(args.output_root):
        sources={}
        for m in ['guangdong','reale_delu']:
            for w in [1,2,3]:
                src=args.prepared_root/m/f'window_{w}';meta=read(src/'manifest.json')
                assert sha(src/'inputs.npz')==meta['arrays_sha256']
                sources[str(src/'inputs.npz')]=meta['arrays_sha256'];sources[str(src/'manifest.json')]=sha(src/'manifest.json')
        sources[str(args.prior_root/'ALL_RUNS.csv')]=sha(args.prior_root/'ALL_RUNS.csv')
        plan=dict(script_sha256=sha(__file__),models=['dnn_adapted','dlinear'],seeds=[42,43,44],windows=[1,2,3],source_sha256=sources,
            prior_root=str(args.prior_root),prepared_root=str(args.prepared_root),expected=36,selection='raw validation RMSE only; fixed hyperparameters',
            dnn='hourly-origin adapted MLP 256/128 ReLU; full LEAR curves/exog; train-only asinh feature/target scaling; not exact epftoolbox reproduction',
            dlinear='univariate price168 ->24; moving average25 with endpoint replication; separate trend/seasonal linear maps; no exogenous inputs',
            optimizer='Adam .001 batch256 clip5; patience20; DNN max150/DLinear max200; checkpoints frozen before test access',
            evaluation='Identical original flat target/h/time rows; missing target masks; float64 raw-unit metrics; no prior models retrained')
        path=args.output_root/'plan.json'
        if path.exists():assert read(path)==plan
        write(path,plan);cards=queue.Queue()
        for card in range(8):cards.put(str(card))
        logs=args.output_root/'logs';logs.mkdir(exist_ok=True)
        jobs=[(model,m,w,s) for model in plan['models'] for m in ['guangdong','reale_delu'] for w in [1,2,3] for s in plan['seeds']]
        def run(job):
            model,m,w,s=job;card=cards.get();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            try:
                with (logs/f'{model}_{m}_w{w}_s{s}.log').open('w') as log:
                    subprocess.run([sys.executable,__file__,'--worker','--prepared-root',str(args.prepared_root),'--output-root',str(args.output_root),
                        '--model',model,'--market',m,'--window',str(w),'--seed',str(s)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            finally:cards.put(card)
        failures=[]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for f in as_completed([pool.submit(run,j) for j in jobs]):
                try:f.result()
                except Exception as exc:failures.append(str(exc))
                rows=[read(x) for x in (args.output_root/'training').glob('*/*/window_*/seed_*/metrics.json')]
                write(args.output_root/'progress.json',dict(completed=len(rows),expected=36,failures=failures))
        assert all(sha(Path(k))==v for k,v in sources.items())
        d=pd.DataFrame(rows);d.to_csv(args.output_root/'ALL_RUNS.csv',index=False)
        complete=len(d)==36 and not failures
        write(args.output_root/'completion.json',dict(complete=complete,completed=len(d),expected=36,failures=failures,source_artifacts_unchanged=True))
        if not complete:raise RuntimeError('Incomplete jobs; see logs')
        per=d.groupby(['model','market','seed'],as_index=False).agg(mae=('test_mae','mean'),rmse=('test_rmse','mean'));per.to_csv(args.output_root/'PER_SEED.csv',index=False)
        summary=per.groupby(['model','market'],as_index=False).agg(mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'),rmse_std=('rmse','std'))
        summary.to_csv(args.output_root/'SUMMARY.csv',index=False)
        old=pd.read_csv(args.prior_root/'SUMMARY.csv')
        pd.concat([old.rename(columns={'method':'model'}),summary],ignore_index=True).to_csv(args.output_root/'COMPARISON_WITH_PRE_REVISION.csv',index=False)


if __name__=='__main__':main()
