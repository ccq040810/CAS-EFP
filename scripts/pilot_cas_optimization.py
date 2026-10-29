"""Bounded validation-only CAS pilot. Never reads test arrays or changes datasets."""
from pathlib import Path
import argparse,json,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha,ROLLOUT_PROTOCOL
from scripts.run_cas_all import run_lock

CANDIDATES=['ordinary','fixed_mild','cas_mild','cas_medium','matched_mean','shuffled','reliability_symmetric','original_cas']
MODES=['absolute','residual']
ORIGINAL_DIRECTION={'original_cas'}


def coefficient(error,c,kind,direction,alpha,seed=42):
    if kind=='ordinary':return np.ones_like(error,dtype=float)
    if kind=='reliability_symmetric':return 1+.3*(c-c.mean())
    focused=(-error*direction)>0
    if kind=='original_cas':return np.where(focused,2*(2-c),1.)
    strength=.1 if kind=='cas_mild' else .3
    w=(1+strength*(1-c));w=w/w.mean()
    if kind in ['fixed_mild','matched_mean']:w=np.ones_like(w)
    if kind=='shuffled':w=np.random.default_rng(seed).permutation(w)
    return np.where(focused,1+(alpha-1)*w,1.)


def chronological_calibration(times,horizon):
    times=pd.DatetimeIndex(times)
    origins=times-pd.to_timedelta(horizon-1,unit='h')
    unique=origins.unique().sort_values()
    cutoff=unique[int(len(unique)*.8)]
    # All training labels precede the first held-out forecast origin.
    return np.asarray(times<cutoff),np.asarray(origins>=cutoff)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-root',type=Path,default=ROOT/'results/cas_transfer_prepared_v2/cas_chronos2')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/cas_validation_pilot_centered_20261029')
    args=p.parse_args();args.output_root.mkdir(parents=True,exist_ok=True)
    config=dict(protocol='causal_validation_only_equal_initialization_centered_weights_v2',
        script_sha256=sha(__file__),candidates=CANDIDATES,modes=MODES,
        train_only_calibration_fraction=.2,calibration_probe_rounds=300,
        selection='ordinary_valid_rmse',seed=42,test_arrays_read=False,
        initialization='train_target_mean_centered_plus_boost_from_average_false',
        tree=dict(learning_rate=.05,num_leaves=63,min_data_in_leaf=40,feature_fraction=.9,
                  bagging_fraction=.8,bagging_freq=5,min_gain_to_split=.08))
    with run_lock(args.output_root):
        path=args.output_root/'experiment.json'
        if path.exists() and read(path)!=config:raise ValueError('Different pilot in this output root')
        write(path,config)
        rows=[];diagnostics=[]
        def study(job):
            market,window,mode=job
            folder=args.prepared_root/market/f'window_{window}'
            meta=read(folder/'manifest.json')
            if meta['rollout_protocol']!=ROLLOUT_PROTOCOL:raise ValueError('Only corrected causal protocol allowed')
            if sha(folder/'features.npz')!=meta['arrays_sha256']:raise ValueError('Prepared checksum changed')
            # Test arrays are deliberately never indexed or materialized.
            with np.load(folder/'features.npz') as z:
                X=z['X_train'];y=z['y_train'];V=z['X_valid'];yv=z['y_valid']
            ci=meta['confidence_index'];pi=meta['point_index'];hi=meta['feature_names'].index('h')
            c=X[:,ci].astype(float);point=X[:,pi].copy();pv=V[:,pi].copy()
            train,cal=chronological_calibration(pd.to_datetime(meta['train_time']),X[:,hi])
            keep=[i for i in range(X.shape[1]) if i!=ci]
            X=X[:,keep];V=V[:,keep]
            base=point if mode=='residual' else np.zeros_like(point)
            bv=pv if mode=='residual' else np.zeros_like(pv)
            center=float(np.mean(y-base,dtype=np.float64))
            target=y-base-center;valid_target=yv-bv-center
            params=dict(config['tree'],objective='regression',metric='None',seed=42,num_threads=4,
                        deterministic=True,force_col_wise=True,verbosity=-1,boost_from_average=False)
            probe=lgb.train(params,lgb.Dataset(X[train],label=target[train]),num_boost_round=300)
            error=y[cal]-(base[cal]+center+probe.predict(X[cal]))
            bias=float(np.median(error));threshold=.02*float(np.std(y[train]))
            direction=0 if abs(bias)<=threshold else (1 if bias>0 else -1)
            corr=float(spearmanr(c[cal],np.abs(error)).statistic)
            diagnostic=dict(market=market,window=window,mode=mode,median_downstream_calibration_residual=bias,
                neutrality_threshold=threshold,train_only_direction=direction,
                confidence_vs_abs_error_spearman=corr if np.isfinite(corr) else None,
                calibration_samples=int(cal.sum()),probe_training_samples=int(train.sum()))
            items=[]
            for kind in CANDIDATES:
                dest=args.output_root/market/f'window_{window}'/mode/kind
                dest.mkdir(parents=True,exist_ok=True)
                if (dest/'validation_metrics.json').exists():
                    old=read(dest/'validation_metrics.json')
                    if old['prepared_sha256']!=meta['arrays_sha256']:raise ValueError('Changed pilot source')
                    items.append(old);continue
                d=(1 if np.median(y-point)>=0 else -1) if kind in ORIGINAL_DIRECTION else direction
                alpha=1.25 if direction else 1.
                # c is frozen; coefficients are piecewise constant in prediction.
                def objective(pred,data):
                    e=pred-data.get_label();b=coefficient(e,c,kind,d,alpha)
                    return b*e,b
                current=dict(params)
                if kind!='ordinary':current['objective']=objective
                history={}
                def rmse(pred,data):return 'rmse',float(np.sqrt(np.mean((pred-data.get_label())**2))),False
                model=lgb.train(current,lgb.Dataset(X,label=target),num_boost_round=2000,
                    valid_sets=[lgb.Dataset(V,label=valid_target)],valid_names=['valid'],feval=rmse,
                    callbacks=[lgb.early_stopping(100,verbose=False),lgb.record_evaluation(history)])
                pred=bv+center+model.predict(V,num_iteration=model.best_iteration)
                r=pred-yv
                row=dict(market=market,window=window,mode=mode,candidate=kind,
                    valid_mae=float(np.mean(np.abs(r))),valid_rmse=float(np.sqrt(np.mean(r*r))),
                    valid_bias=float(r.mean()),best_iteration=model.best_iteration,
                    direction=d,alpha=2. if kind in ORIGINAL_DIRECTION else alpha,target_center=center,
                    prepared_sha256=meta['arrays_sha256'],test_evaluated=False)
                write(dest/'validation_metrics.json',row)
                pd.DataFrame(history['valid']).to_csv(dest/'history.csv',index=False)
                model.save_model(str(dest/'model.txt'))
                items.append(row);print('DONE',json.dumps(row),flush=True)
            return items,diagnostic
        jobs=[(m,w,mode) for m in ['guangdong','reale_delu'] for w in [1,2,3] for mode in MODES]
        expected=len(jobs)*len(CANDIDATES)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(study,j) for j in jobs]):
                result,d=future.result();rows.extend(result);diagnostics.append(d)
                write(args.output_root/'progress.json',dict(completed=len(rows),expected=expected,test_evaluated=False))
        table=pd.DataFrame(rows);table.to_csv(args.output_root/'VALIDATION_ALL.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(args.output_root/'TRAIN_ONLY_CONFIDENCE_DIAGNOSTICS.csv',index=False)
        report=table.groupby(['market','mode','candidate'],as_index=False).agg(valid_mae=('valid_mae','mean'),valid_rmse=('valid_rmse','mean'))
        report.to_csv(args.output_root/'VALIDATION_SUMMARY.csv',index=False)
        write(args.output_root/'completion.json',dict(expected=expected,completed=len(rows),complete=len(rows)==expected,test_evaluated=False))
        print(report.to_string(index=False),flush=True)


if __name__=='__main__':main()
