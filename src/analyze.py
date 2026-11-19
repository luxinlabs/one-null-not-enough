"""Analyze results/grid: per-cell metrics with 95% bootstrap CIs and seed-paired policy contrasts.

Writes results/analysis/metrics.csv and results/analysis/contrasts.csv. Test seeds only, except the sens block (validation).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CFG=['block','sigma','budget','fault_rate','n_mc','truth','prior_fault','plan_rep','voi_tol','support_eps','credal_lo','credal_hi','scenario']
# (policy, reference): the preregistered comparisons. H3 is the first; the rest are H1/H2 and E4 ablations.
CONTRASTS=[('robust_bayes','identifiability'),('robust_bayes','joint_risk_voi'),('identifiability','joint_risk_voi'),('identifiability_stop','joint_risk_voi'),('identifiability','identifiability_stop'),('identifiability_eig','identifiability'),('oracle_fault','joint_risk_voi'),
           ('joint_risk_voi','joint_risk'),('joint_risk_voi','replicate'),('joint_risk_voi','scientific_only'),('identifiability','scientific_only')]
DIFF_METRICS=['correct','false_rejection','abstain','spent','brier']

def load(root):
    cols=None;frames=[]
    for f in sorted(Path(root).glob('*/*.csv.gz')):
        if cols is None:cols=[c for c in pd.read_csv(f,nrows=0).columns if c!='history']
        frames.append(pd.read_csv(f,usecols=cols))
    d=pd.concat(frames,ignore_index=True)
    d['prior_fault']=d['prior_fault'].fillna(d['fault_rate']) # matched prior
    p=d['posterior_h1'].clip(1e-12,1-1e-12)
    d['brier']=(d['posterior_h1']-d['true_h'])**2
    d['log_loss']=-(d['true_h']*np.log(p)+(1-d['true_h'])*np.log(1-p))
    d['forced']=(d['posterior_h1']>.5).astype(int) # E4 "remove abstention" ablation
    return d

def metric_arrays(g,idx):
    """Metrics for each bootstrap resample; idx is (B, n) row indices into g."""
    a=lambda c:g[c].to_numpy(float)[idx]
    h,f,pred,ab,correct,forced=a('true_h'),a('fault'),a('predicted_h'),a('abstain'),a('correct'),a('forced')
    div=lambda num,den:np.where(den>0,num/np.maximum(den,1),np.nan)
    out={'coverage':1-ab.mean(1),'accuracy':correct.mean(1),
         'selective_accuracy':div((correct*(1-ab)).sum(1),(1-ab).sum(1)),
         'false_rejection':div(((pred==0)&(h==1)).sum(1),(h==1).sum(1)),
         'false_rejection_faulty':div(((pred==0)&(h==1)&(f==1)).sum(1),((h==1)&(f==1)).sum(1)),
         'mean_cost':a('spent').mean(1),'brier':a('brier').mean(1),'log_loss':a('log_loss').mean(1),
         'forced_accuracy':(forced==h).mean(1),'forced_false_rejection':div(((forced==0)&(h==1)).sum(1),(h==1).sum(1))}
    # Area under the risk-coverage curve of the forced decision, ranked by confidence.
    conf=np.abs(a('posterior_h1')-.5);err=(forced!=h).astype(float)
    order=np.argsort(-conf,axis=1,kind='stable');err=np.take_along_axis(err,order,1)
    out['aurc']=(np.cumsum(err,1)/np.arange(1,err.shape[1]+1)).mean(1)
    return out

def summarize(d,rng,n_boot):
    rows=[]
    for key,g in d.groupby(CFG+['policy'],dropna=False,sort=True):
        g=g.sort_values('seed').reset_index(drop=True);n=len(g)
        point=metric_arrays(g,np.arange(n)[None])
        boot=metric_arrays(g,rng.integers(0,n,(n_boot,n)))
        row=dict(zip(CFG+['policy'],key));row['n']=n
        for m,v in point.items():
            row[m]=v[0];row[m+'_lo'],row[m+'_hi']=np.nanpercentile(boot[m],[2.5,97.5]) if np.isfinite(boot[m]).any() else (np.nan,np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def contrasts(d,rng,n_boot):
    rows=[]
    for key,g in d.groupby(CFG,dropna=False,sort=True):
        by=dict(tuple(g.groupby('policy')))
        for pol,ref in CONTRASTS:
            if pol not in by or ref not in by:continue
            m=by[pol].merge(by[ref],on='seed',suffixes=('','_ref'))
            idx=rng.integers(0,len(m),(n_boot,len(m)))
            for metric in DIFF_METRICS:
                diff=(m[metric]-m[metric+'_ref']).to_numpy(float)
                lo,hi=np.percentile(diff[idx].mean(1),[2.5,97.5])
                rows.append({**dict(zip(CFG,key)),'policy':pol,'reference':ref,'metric':metric,'n_pairs':len(m),'mean_diff':diff.mean(),'lo':lo,'hi':hi,
                             'frac_better':float((diff>0).mean()),'frac_worse':float((diff<0).mean())})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/grid');ap.add_argument('--out',default='results/analysis')
    ap.add_argument('--n-boot',type=int,default=2000);ap.add_argument('--seed',type=int,default=0);args=ap.parse_args()
    rng=np.random.default_rng(args.seed);d=load(args.root)
    d=d[(d['split']=='test')|(d['block']=='sens')]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    summarize(d,rng,args.n_boot).to_csv(out/'metrics.csv',index=False)
    contrasts(d,rng,args.n_boot).to_csv(out/'contrasts.csv',index=False)
    print(f'{len(d)} rows -> {out}/metrics.csv, {out}/contrasts.csv')
if __name__=='__main__':main()
