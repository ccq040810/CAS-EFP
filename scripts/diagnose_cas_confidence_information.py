"""Finite confidence-information audit on training/validation only; no fitting.

Thresholds come from training confidence within each horizon. Ordinary-model
validation residuals avoid using the c objective to construct its own evidence.
Day-block intervals are descriptive and do not establish causal benefit.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import read,write,sha
from scripts.run_cas_directional_study import local_times
from scripts.pilot_cas_asinh_heads import forward


def block_difference(error,group,days,repetitions=400):
    frame=pd.DataFrame(dict(e=error,g=group,day=days))
    sums=frame.groupby(['day','g']).e.sum().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    counts=frame.groupby(['day','g']).e.size().unstack(fill_value=0).reindex(index=sums.index,columns=[0,1,2],fill_value=0)
    rng=np.random.default_rng(42);samples=[]
    for _ in range(repetitions):
        idx=rng.integers(0,len(sums),len(sums));total=sums.to_numpy()[idx].sum(axis=0);n=counts.to_numpy()[idx].sum(axis=0)
        if n[0]>0 and n[2]>0:samples.append(total[0]/n[0]-total[2]/n[2])
    return np.quantile(samples,[.025,.975]).tolist() if samples else [None,None]


def main():
    source=ROOT/'results/cas_asinh_linear_heads_confirmation_20261029';cp=read(source/'plan.json');pilot=Path(cp['pilot_root']);pp=read(pilot/'plan.json')
    assert read(source/'independent_audit.json')['complete']
    out=ROOT/'results/cas_confidence_information_20261029';out.mkdir(parents=True,exist_ok=True)
    plan=dict(script_sha256=sha(__file__),confirmation_plan_sha256=sha(source/'plan.json'),windows=[1,2,3],seeds=[42,43,44],
        groups='Within-h training c terciles; duplicates are retained as degenerate bins, never forced apart',
        residuals='Raw and transformed truth minus ordinary-model prediction; Chronos reference once per head',
        uncertainty='400 fixed-seed origin-day block bootstrap low-c minus high-c transformed mean errors; descriptive only',
        no_training=True,test_evaluated=False)
    if (out/'plan.json').exists():assert read(out/'plan.json')==plan
    write(out/'plan.json',plan);rows=[];bins=[];training=[]
    for w in [1,2,3]:
        ref=Path(pp['roots']['reference_root'])/'guangdong'/f'window_{w}';meta=read(ref/'manifest.json');ident=cp['source_identities'][str(w)]
        assert sha(ref/'features.npz')==ident['reference_arrays_sha256'] and sha(ref/'manifest.json')==ident['reference_manifest_sha256']
        with np.load(ref/'features.npz') as z:
            ct=z['X_train'][:,meta['confidence_index']].astype(float);ht=z['X_train'][:,meta['feature_names'].index('h')].astype(int)
            cv=z['X_valid'][:,meta['confidence_index']].astype(float);hv=z['X_valid'][:,meta['feature_names'].index('h')].astype(int)
            y=z['y_valid'];point=z['X_valid'][:,meta['point_index']]
        times=local_times(meta['valid_time'])
        for h in range(1,25):
            ti=ht==h;vi=hv==h;c=ct[ti];v=cv[vi];yv=y[vi];q=np.quantile(c,[1/3,2/3]);groups=np.searchsorted(q,v,side='right')
            recipe=read(pilot/'guangdong'/f'window_{w}'/'asinh_linear_per_h'/f'head_{h:02d}'/'training_recipe.json');target=recipe['target']
            costs=next(row for row in recipe['horizons'] if row['h']==h);s=costs['strength'];weight=(2-c)/np.mean(2-c)
            multiplier=(1+s*weight)/(1+s)
            training.append(dict(window=w,h=h,c_min=c.min(),c_max=c.max(),c_mean=c.mean(),c_std=c.std(),c_q33=q[0],c_q67=q[1],
                c_distinct=len(np.unique(c)),linear_weight_cv=weight.std()/weight.mean(),focused_vs_fixed_cv=multiplier.std()/multiplier.mean(),
                training_direction=costs['direction'],training_strength=s))
            origin_days=(times[vi]-pd.to_timedelta(h-1,unit='h')).normalize()
            predictions=[('chronos_only',42,point[vi])]
            for seed in [42,43,44]:
                path=source/'guangdong'/f'window_{w}'/'asinh_linear_per_h'/f'head_{h:02d}'/'ordinary'/f'seed_{seed}'/'predictions_valid.csv'
                p=pd.read_csv(path);np.testing.assert_array_equal(p.y_true.to_numpy().astype(y.dtype),yv)
                assert local_times(p.time).equals(times[vi]);predictions.append(('ordinary',seed,p.prediction.to_numpy()))
            for model,seed,pred in predictions:
                error=yv-pred;ez=forward(yv,target)-forward(pred,target);low=groups==0;high=groups==2
                rho_abs=pd.Series(1-v).corr(pd.Series(np.abs(ez)),method='spearman')
                rho_signed=pd.Series(1-v).corr(pd.Series(ez),method='spearman')
                lo,hi=block_difference(ez,groups,origin_days)
                diff=float(ez[low].mean()-ez[high].mean()) if low.any() and high.any() else None
                rows.append(dict(window=w,h=h,method=model,seed=seed,n=len(v),origin_days=len(np.unique(origin_days)),
                    spearman_uncertainty_abs_transformed_error=rho_abs,spearman_uncertainty_signed_transformed_error=rho_signed,
                    low_minus_high_transformed_bias=diff,day_block_lower=lo,day_block_upper=hi,
                    training_direction=costs['direction'],training_strength=s,mean_raw_error=float(error.mean())))
                for group in range(3):
                    mask=groups==group
                    bins.append(dict(window=w,h=h,method=model,seed=seed,c_group=group,n=int(mask.sum()),
                        c_mean=float(v[mask].mean()) if mask.any() else None,raw_bias=float(error[mask].mean()) if mask.any() else None,
                        transformed_bias=float(ez[mask].mean()) if mask.any() else None,transformed_mae=float(np.abs(ez[mask]).mean()) if mask.any() else None))
        print('DIAGNOSED validation window',w,flush=True)
    d=pd.DataFrame(rows);g=pd.DataFrame(bins);t=pd.DataFrame(training)
    assert len(d)==288 and len(g)==864 and len(t)==72
    d.to_csv(out/'BY_HORIZON.csv',index=False);g.to_csv(out/'CONFIDENCE_GROUPS.csv',index=False);t.to_csv(out/'TRAINING_WEIGHT_SPREAD.csv',index=False)
    d.groupby(['method','window','seed'],as_index=False).agg(median_abs_error_rho=('spearman_uncertainty_abs_transformed_error','median'),
        median_signed_error_rho=('spearman_uncertainty_signed_transformed_error','median')).to_csv(out/'SUMMARY.csv',index=False)
    write(out/'completion.json',dict(complete=True,horizon_records=len(d),groups=len(g),training_profiles=len(t),test_evaluated=False,no_training=True))
    print(pd.read_csv(out/'SUMMARY.csv').to_string(index=False),flush=True)


if __name__=='__main__':main()
