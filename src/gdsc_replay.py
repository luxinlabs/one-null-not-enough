"""Real-data replay on GDSC (docs/PREREG_GDSC.md, amendment 6). Thresholds are fitted on the exploration split only.

Writes results/gdsc/thresholds.json, metrics.csv, units_<split>.csv.gz.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

RAW='data/raw/gdsc'
RULES=('single_plate','control_gated','confirm_or_reject','confirm_or_abstain')
Z_STAR=0.452 # 10th percentile of exploration-plate Z', fixed in amendment 6
COST={'plate':1.,'controls':.6,'independent':2.5}

def labelled_units(split):
    u=pd.read_csv('data/processed/gdsc_units.csv.gz',dtype={'BARCODE':str})
    g1=pd.read_csv(f'{RAW}/GDSC1_fitted_dose_response_24Jul22.csv');g2=pd.read_csv(f'{RAW}/GDSC2_fitted_dose_response_24Jul22.csv')
    name=g1.drop_duplicates('DRUG_ID').set_index('DRUG_ID').DRUG_NAME
    gt=g2.groupby(['COSMIC_ID','DRUG_NAME'],as_index=False).AUC.mean()
    d=u[u.split==split].copy();d['DRUG_NAME']=d.DRUG_ID.map(name)
    d=d.merge(gt,on=['COSMIC_ID','DRUG_NAME'])
    d['H']=np.where(d.AUC<=.8,1,np.where(d.AUC>=.95,0,-1));d=d[d.H>=0].copy()
    # Independent plate: the same (cell line, drug) on a different plate of the same split; take the lowest other barcode.
    s=u[u.split==split][['COSMIC_ID','DRUG_ID','BARCODE','effect']].sort_values('BARCODE')
    first=s.drop_duplicates(['COSMIC_ID','DRUG_ID']).rename(columns={'BARCODE':'b1','effect':'e1'})
    second=s[~s.index.isin(s.drop_duplicates(['COSMIC_ID','DRUG_ID']).index)].drop_duplicates(['COSMIC_ID','DRUG_ID']).rename(columns={'BARCODE':'b2','effect':'e2'})
    d=d.merge(first,on=['COSMIC_ID','DRUG_ID'],how='left').merge(second,on=['COSMIC_ID','DRUG_ID'],how='left')
    d['alt_effect']=np.where(d.BARCODE==d.b1,d.e2,d.e1)
    return d.drop(columns=['b1','e1','b2','e2']).reset_index(drop=True)

def fit_thresholds(d,target=.9):
    grid=np.round(np.arange(0,1.0001,.05),2)
    tau=max(t for t in grid if (d.effect<t).any() and 1-d.H[d.effect<t].mean()>=target)
    tau_p=min(t for t in grid if t>=tau and (d.effect>t).any() and d.H[d.effect>t].mean()>=target)
    return float(tau),float(tau_p)

def decide(d,rule,tau,tau_p):
    pred=np.where(d.effect<tau,0,np.where(d.effect>tau_p,1,-1));cost=np.full(len(d),COST['plate'])
    if rule=='control_gated':
        cost+=COST['controls'];pred=np.where((pred==0)&(d.zprime<Z_STAR),-1,pred)
    if rule in ('confirm_or_reject','confirm_or_abstain'):
        want=pred==0;has=d.alt_effect.notna().to_numpy()
        cost+=np.where(want&has,COST['independent'],0.)
        confirmed=d.alt_effect.to_numpy()<tau
        pred=np.where(want&has,np.where(confirmed,0,-1),pred)
        if rule=='confirm_or_abstain':pred=np.where(want&~has,-1,pred)
    return pred,cost

def metrics(h,pred,cost):
    dec=pred>=0
    return {'coverage':dec.mean(),'selective_accuracy':(pred[dec]==h[dec]).mean() if dec.any() else np.nan,
            'false_rejection':((pred==0)&(h==1)).sum()/max((h==1).sum(),1),'true_rejection':((pred==0)&(h==0)).sum()/max((h==0).sum(),1),
            'mean_cost':cost.mean()}

def evaluate(d,tau,tau_p,rng,n_boot):
    cells=d.COSMIC_ID.to_numpy();uc=np.unique(cells);idx={c:np.flatnonzero(cells==c) for c in uc};rows=[]
    preds={r:decide(d,r,tau,tau_p) for r in RULES};h=d.H.to_numpy()
    boots=[np.concatenate([idx[c] for c in rng.choice(uc,len(uc))]) for _ in range(n_boot)]
    for strat,mask in (('all',np.ones(len(d),bool)),('independent plate available',d.alt_effect.notna().to_numpy()),('no independent plate',d.alt_effect.isna().to_numpy())):
        for r,(pred,cost) in preds.items():
            point=metrics(h[mask],pred[mask],cost[mask]);bm=[]
            for b in boots:
                b=b[mask[b]];bm.append(metrics(h[b],pred[b],cost[b]))
            row={'stratum':strat,'rule':r,'n_units':int(mask.sum()),'n_H1':int(h[mask].sum())}
            for k,v in point.items():
                vals=np.array([x[k] for x in bm],float);row[k]=v;row[k+'_lo'],row[k+'_hi']=np.nanpercentile(vals,[2.5,97.5])
            rows.append(row)
    return pd.DataFrame(rows),preds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=('explore','test'),required=True);ap.add_argument('--n-boot',type=int,default=2000)
    ap.add_argument('--out',default='results/gdsc');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);th=out/'thresholds.json'
    if args.split=='explore':
        tau,tau_p=fit_thresholds(labelled_units('explore'))
        th.write_text(json.dumps({'tau':tau,'tau_prime':tau_p,'z_star':Z_STAR,'fitted_on':'explore'},indent=1))
    t=json.loads(th.read_text());d=labelled_units(args.split)
    m,preds=evaluate(d,t['tau'],t['tau_prime'],np.random.default_rng(0),args.n_boot)
    m.insert(0,'split',args.split);m.to_csv(out/f'metrics_{args.split}.csv',index=False)
    for r,(pred,cost) in preds.items():d[f'pred_{r}']=pred
    d.to_csv(out/f'units_{args.split}.csv.gz',index=False)
    print(json.dumps(t),f'{len(d)} labelled {args.split} units')
    print(m[['stratum','rule','n_units','coverage','selective_accuracy','false_rejection','true_rejection','mean_cost']].round(3).to_string(index=False))
if __name__=='__main__':main()
