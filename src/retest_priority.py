"""Which nulls should an autonomous lab retest? (review round 4; post hoc). Writes results/gdsc/retest.json, results/gdsc/retest_curve.csv,
and results/gdsc/null_features_{explore,test}.csv.gz.

Pool: units whose single plate would reject (effect < tau). A retest is the replicate plate; a retested null that the replicate does not
confirm (replicate effect >= tau) is withheld from rejection. Scores use only information a lab has before retesting: the plate reading,
Z', the GDSC1 maximum concentration, the assay, the compound's and cell line's response rates on exploration plates (leaving the pair out),
and how many other compounds on the same plate also look null. The learned score is fitted on exploration-split nulls (label: GDSC2) and
evaluated on test-split nulls that have a replicate, at fixed retest budgets. Because GDSC2 labels are per (cell line, compound) pair, training
and evaluation use disjoint cell lines (primary) or disjoint compounds, so no evaluated pair's label is seen in training.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW='data/raw/gdsc'
TAU=.3
FEATURES=['effect','zprime','log_max_conc','assay_a','drug_resp','cell_resp','plate_null_share','n_slots']
BUDGETS=(.05,.1,.2,.3,.5,1.)

def features(split):
    allu=pd.read_csv('data/processed/gdsc_units.csv.gz',dtype={'BARCODE':str});allu=allu[allu.eligible]
    d=pd.read_csv(f'results/gdsc/units_{split}.csv.gz',dtype={'BARCODE':str});d=d[d.effect<TAU].reset_index(drop=True)
    mc=pd.read_csv(f'{RAW}/GDSC1_fitted_dose_response_24Jul22.csv').groupby(['COSMIC_ID','DRUG_ID']).MAX_CONC.max().rename('max_conc')
    d=d.join(mc,on=['COSMIC_ID','DRUG_ID']);d['log_max_conc']=np.log10(d.max_conc)
    plates=pd.read_csv('data/processed/gdsc_plates.csv.gz',dtype={'BARCODE':str}).set_index('BARCODE').ASSAY;d['assay_a']=(plates.loc[d.BARCODE].values=='a').astype(float)
    # Response history on exploration plates, leaving the (cell line, compound) pair itself out.
    ex=allu[allu.split=='explore'].assign(resp=lambda x:(x.effect>=.7).astype(float))
    pair=ex.groupby(['COSMIC_ID','DRUG_ID']).resp.agg(['sum','size']).rename(columns={'sum':'ps','size':'pn'})
    drug=ex.groupby('DRUG_ID').resp.agg(['sum','size']).rename(columns={'sum':'ds','size':'dn'});cell=ex.groupby('COSMIC_ID').resp.agg(['sum','size']).rename(columns={'sum':'cs','size':'cn'})
    d=d.join(pair,on=['COSMIC_ID','DRUG_ID']).join(drug,on='DRUG_ID').join(cell,on='COSMIC_ID');d[['ps','pn']]=d[['ps','pn']].fillna(0)
    d['drug_resp']=(d.ds-d.ps)/(d.dn-d.pn).replace(0,np.nan);d['cell_resp']=(d.cs-d.ps)/(d.cn-d.pn).replace(0,np.nan)
    # Other compounds on the same plate that also look null.
    pl=allu.assign(null=(allu.effect<TAU).astype(float)).groupby('BARCODE').null.agg(['sum','size'])
    d=d.join(pl,on='BARCODE');d['n_slots']=d['size'];d['plate_null_share']=(d['sum']-1)/(d['size']-1).replace(0,np.nan)
    return d.drop(columns=['ps','pn','ds','dn','cs','cn','sum','size'])

def curve(d,score,budgets=BUDGETS):
    """Retest the top-scoring fraction b of nulls; FR and CR after withholding unconfirmed retests."""
    h1=(d.H==1).to_numpy();conf=(d.alt_effect<TAU).to_numpy();order=np.argsort(-score,kind='stable');out=[]
    for b in budgets:
        k=int(round(b*len(d)));r=np.zeros(len(d),bool);r[order[:k]]=True;rej=~r|conf
        out.append({'budget':b,'FR':(rej&h1).sum()/h1.sum(),'CR':(rej&~h1).sum()/(~h1).sum(),'recovered_share':((r&~conf)&h1).sum()/h1.sum(),
                    'effective_per_100_retests':100*(r&h1&~conf).sum()/max(k,1),'retest_precision':(r&h1).sum()/max(k,1),
                    'correct_rejections_withheld_share':(r&~conf&~h1).sum()/(~h1).sum()})
    return out

def half(x):return np.array([int(hashlib.sha256(str(v).encode()).hexdigest(),16)%2==0 for v in x])

def run(tr_all,te_all,group):
    # group 'both': train on cell lines AND compounds of one hash-half, evaluate on pairs whose cell line AND compound are both in the other half.
    cols=['COSMIC_ID','DRUG_ID'] if group=='both' else [group]
    trm=np.logical_and.reduce([half(tr_all[c]) for c in cols]);tem=np.logical_and.reduce([~half(te_all[c]) for c in cols])
    tr=tr_all[trm].reset_index(drop=True);te=te_all[tem&te_all.alt_effect.notna()].reset_index(drop=True)
    X=lambda d:d[FEATURES].to_numpy(float)
    gbm=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.,random_state=0).fit(X(tr),tr.H)
    lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(np.nan_to_num(X(tr),nan=0.),tr.H)
    rng=np.random.default_rng(0)
    scores={'random':rng.random(len(te)),'lowest_zprime':-te.zprime.to_numpy(),'weakest_null':te.effect.to_numpy(),
            'compound_history':te.drug_resp.fillna(0).to_numpy(),'learned_logistic':lr.predict_proba(np.nan_to_num(X(te),nan=0.))[:,1],
            'learned_gbm':gbm.predict_proba(X(te))[:,1]}
    h1=(te.H==1).to_numpy();rec=h1&(te.alt_effect>=TAU).to_numpy()
    out={'n_train_nulls':int(len(tr)),'n_test_nulls_with_replicate':int(len(te)),'test_effective_share':float(h1.mean()),
         'auc_effective':{k:float(roc_auc_score(h1,v)) for k,v in scores.items()},'auc_recoverable':{k:float(roc_auc_score(rec,v)) for k,v in scores.items()}}
    rows=[]
    for k,v in scores.items():
        for r in curve(te,v):rows.append({'policy':k,**r})
    c=pd.DataFrame(rows);out['curve']=c.round(4).to_dict('records')
    # Cell-line bootstrap of the gain in recovered effective nulls, learned (GBM) minus each baseline, at 10% and 20% retest budgets.
    cells=te.COSMIC_ID.to_numpy();uc=np.unique(cells);idx={x:np.flatnonzero(cells==x) for x in uc};gain={}
    for b in (.1,.2):
        diffs={k:[] for k in scores if k!='learned_gbm'}
        for _ in range(1000):
            i=np.concatenate([idx[x] for x in rng.choice(uc,len(uc))]);s=te.iloc[i].reset_index(drop=True)
            g=curve(s,scores['learned_gbm'][i],(b,))[0]['recovered_share']
            for k in diffs:diffs[k].append(g-curve(s,scores[k][i],(b,))[0]['recovered_share'])
        gain[str(b)]={k:[float(x) for x in np.percentile(v,[2.5,50,97.5])] for k,v in diffs.items()}
    out['bootstrap_gain_recovered_share']=gain;c['group']=group
    return out,c

def main():
    tr,te=features('explore'),features('test')
    tr.to_csv('results/gdsc/null_features_explore.csv.gz',index=False);te.to_csv('results/gdsc/null_features_test.csv.gz',index=False)
    res={};curves=[]
    for g in ('COSMIC_ID','DRUG_ID','both'):
        res[g],c=run(tr,te,g);curves.append(c)
    pd.concat(curves).to_csv('results/gdsc/retest_curve.csv',index=False)
    Path('results/gdsc/retest.json').write_text(json.dumps(res,indent=1))
    pd.set_option('display.width',200)
    for g,out in res.items():
        print(g,json.dumps({k:out[k] for k in ('n_train_nulls','n_test_nulls_with_replicate','test_effective_share','auc_effective','auc_recoverable')},indent=1))
        c=pd.DataFrame(out['curve']);print(c[c.budget.isin([.05,.1,.2,.3])].round(3).to_string(index=False));print(json.dumps(out['bootstrap_gain_recovered_share'],indent=1))
if __name__=='__main__':main()
