"""Validation-only residual CAS strength controls. Reuse 12 pilot fits; train 18 controls."""
from pathlib import Path
import shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import pilot_cas_optimization as base


def controlled_coefficient(error,c,kind,direction,alpha,seed=42):
    if kind=='ordinary':return np.ones_like(error,dtype=float)
    mask=(-error*direction)>0
    w=2-c
    if kind=='fixed_alpha2':w=np.ones_like(c)
    elif kind=='mean_alpha2':w=np.full_like(c,w.mean())
    elif kind=='shuffle_alpha2':w=np.random.default_rng(seed).permutation(w)
    elif kind!='original_cas':raise ValueError(kind)
    return np.where(mask,2*w,1.)


def main():
    if any(a.split('=')[0]=='--output-root' for a in sys.argv[1:]):
        raise ValueError('This follow-up uses a fixed isolated output root for provenance')
    out=ROOT/'results/cas_residual_strength_controls_20261029'
    previous=ROOT/'results/cas_validation_pilot_centered_20261029'
    if not base.read(previous/'completion.json')['complete']:raise ValueError('Complete initial pilot first')
    for m in ['guangdong','reale_delu']:
        for w in [1,2,3]:
            for kind in ['ordinary','original_cas']:
                src=previous/m/f'window_{w}/residual'/kind
                dst=out/m/f'window_{w}/residual'/kind
                if not dst.exists():
                    shutil.copytree(src,dst)
                    row=base.read(dst/'validation_metrics.json');row['reused_from']=str(src)
                    base.write(dst/'validation_metrics.json',row)
    sys.argv.extend(['--output-root',str(out)])
    base.CANDIDATES=['ordinary','fixed_alpha2','mean_alpha2','shuffle_alpha2','original_cas']
    base.MODES=['residual'];base.ORIGINAL_DIRECTION=set(base.CANDIDATES)-{'ordinary'}
    base.coefficient=controlled_coefficient
    base.__file__=str(Path(__file__).resolve())
    base.main()


if __name__=='__main__':main()
