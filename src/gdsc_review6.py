"""Review round 8 (post hoc; disclosed). Writes results/gdsc/review6.json.

R9: do discordant rejections track measurable plate anomalies once dose, assay, cell line, and compound are controlled for? Logistic
    regression on effective test units (outcome: single-plate rejection) with cell-line, compound, and drug-set (plate layout) fixed effects, dose matching, and assay;
    plate features standardized; odds ratios per SD with a cell-line bootstrap.
R10: sensitivity of retest prioritization to which nulls have a replicate. (a) Inverse-probability weights that make held-out-cell-line test
    nulls with a replicate resemble all such nulls on the retest features and the GDSC2 label; (b) tilts that make the replicate overturn
    effective nulls more or less often than observed (lambda x the weight of recoverable effective nulls).
"""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,'src')
from retest_priority import half,FEATURES,TAU
from gdsc_review2 import with_dose

PLATE=['zprime','nc_cv','log_signal_window','plate_null_share']

def plate_features(u):
    allu=pd.read_csv('data/processed/gdsc_units.csv.gz',dtype={'BARCODE':str});allu=allu[allu.eligible]
    pl=allu.assign(null=(allu.effect<TAU).astype(float)).groupby('BARCODE').null.agg(['sum','size'])
    p=pd.read_csv('data/processed/gdsc_plates.csv.gz',dtype={'BARCODE':str}).set_index('BARCODE')
    u=u.join(pl,on='BARCODE');null=(u.effect<TAU).astype(float)
    u['plate_null_share']=(u['sum']-null)/(u['size']-1).replace(0,np.nan)  # other compounds on the plate, the unit itself left out
    u['nc_cv']=(p.nc_sd/p.nc_mean).reindex(u.BARCODE).values;u['log_signal_window']=np.log10((p.nc_mean/p.b_mean).reindex(u.BARCODE).values)
    u['assay_a']=(p.ASSAY.reindex(u.BARCODE).values=='a').astype(float)
    return u.drop(columns=['sum','size'])

def design(d,sc):
    Z=sc.transform(d[PLATE].to_numpy(float));base=np.column_stack([Z,d.dose_matched.astype(float),d.assay_a])
    fe=pd.get_dummies(d[['COSMIC_ID','DRUG_ID','DRUGSET_ID']].astype(str),drop_first=True,sparse=True).sparse.to_coo().tocsr()
    return sparse.hstack([sparse.csr_matrix(base),fe]).tocsr()

def r9():
    u=with_dose(pd.read_csv('results/gdsc/units_test.csv.gz',dtype={'BARCODE':str}));u=plate_features(u)
    d=u[(u.H==1)].dropna(subset=PLATE+['dose_matched']).reset_index(drop=True);y=(d.effect<TAU).to_numpy()
    sc=StandardScaler().fit(d[PLATE].to_numpy(float))
    fit=lambda dd,yy:LogisticRegression(C=10.,max_iter=3000,solver='lbfgs').fit(design(dd,sc),yy).coef_[0][:len(PLATE)+2]
    est=fit(d,y);rng=np.random.default_rng(0);cells=d.COSMIC_ID.to_numpy();uc=np.unique(cells);idx={c:np.flatnonzero(cells==c) for c in uc};bs=[]
    for _ in range(200):
        i=np.concatenate([idx[c] for c in rng.choice(uc,len(uc))]);bs.append(fit(d.iloc[i].reset_index(drop=True),y[i]))
    bs=np.array(bs);names=PLATE+['dose_matched','assay_a']
    out={'n_effective_units':int(len(d)),'FR':float(y.mean()),
         'odds_ratio_per_sd':{n:{'or':float(np.exp(est[k])),'ci':[float(x) for x in np.exp(np.percentile(bs[:,k],[2.5,97.5]))]} for k,n in enumerate(names)}}
    # Same model on dose-matched units only.
    m=d.dose_matched.to_numpy();dm=d[m].reset_index(drop=True)
    e2=LogisticRegression(C=10.,max_iter=3000).fit(design(dm,sc),y[m]).coef_[0][:len(PLATE)]
    out['dose_matched_only_or_per_sd']={n:float(np.exp(e2[k])) for k,n in enumerate(PLATE)}
    return out

def wcurve(te,score,w,b=.1):
    h1=(te.H==1).to_numpy();conf=(te.alt_effect<TAU).to_numpy();k=int(round(b*len(te)));r=np.zeros(len(te),bool);r[np.argsort(-score,kind='stable')[:k]]=True
    return float((w*(r&~conf&h1)).sum()/(w*h1).sum()),float((w*(~conf&h1)).sum()/(w*h1).sum())

def r10():
    tr_all=pd.read_csv('results/gdsc/null_features_explore.csv.gz',dtype={'BARCODE':str});te_all=pd.read_csv('results/gdsc/null_features_test.csv.gz',dtype={'BARCODE':str})
    tr=tr_all[half(tr_all.COSMIC_ID)].reset_index(drop=True);pop=te_all[~half(te_all.COSMIC_ID)].reset_index(drop=True)
    gbm=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.,random_state=0).fit(tr[FEATURES].to_numpy(float),tr.H)
    has=pop.alt_effect.notna().to_numpy()
    Xp=np.column_stack([np.nan_to_num(pop[FEATURES].to_numpy(float)),pop.H]);Xp=StandardScaler().fit_transform(Xp)
    ps=LogisticRegression(max_iter=2000).fit(Xp,has).predict_proba(Xp)[:,1]
    te=pop[has].reset_index(drop=True);ipw=np.clip(1/ps[has],None,np.percentile(1/ps[has],99))
    rng=np.random.default_rng(0)
    scores={'random':rng.random(len(te)),'lowest_zprime':-te.zprime.to_numpy(),'compound_history':te.drug_resp.fillna(0).to_numpy(),
            'learned_gbm':gbm.predict_proba(te[FEATURES].to_numpy(float))[:,1]}
    rec=((te.H==1)&(te.alt_effect>=TAU)).to_numpy();out={'n_population_nulls':int(len(pop)),'n_with_replicate':int(has.sum()),
         'propensity_range':[float(ps[has].min()),float(ps[has].max())]}
    for name,w in [('unweighted',np.ones(len(te))),('ipw',ipw)]+[(f'tilt_{l}',np.where(rec,l,1.)) for l in (.5,.75,1.5,2.)]:
        r={k:wcurve(te,v,w) for k,v in scores.items()};allr=r['random'][1]
        out[name]={'retest_all':allr,**{k:v[0] for k,v in r.items()},'compound_history_share_of_all':r['compound_history'][0]/allr}
    return out

def main():
    out={'R9_plate_anomalies':r9(),'R10_replicate_availability':r10()}
    Path('results/gdsc/review6.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
if __name__=='__main__':main()
