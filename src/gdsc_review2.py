"""Second-round review analyses (post hoc; disclosed). Writes results/gdsc/review2.json.

M5: replicate the post-hoc dose-matched analysis on the exploration split (not used to design it).
M2: replicate-plate batch structure (same cell expansion / same seeding date) and co-rejection by batch.
Assay: single-plate false rejection by GDSC1 assay type.
M3: expected points of each rule under the 10:1 and 50:1 losses shown to the agents.
"""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm,multivariate_normal,rankdata
import gdsc_replay as G

RAW='data/raw/gdsc'

def with_dose(u):
    g1=pd.read_csv(f'{RAW}/GDSC1_fitted_dose_response_24Jul22.csv');g2=pd.read_csv(f'{RAW}/GDSC2_fitted_dose_response_24Jul22.csv');k=['COSMIC_ID','DRUG_NAME']
    mc=g1.groupby(k).MAX_CONC.max().rename('max1').reset_index().merge(g2.groupby(k).MAX_CONC.max().rename('max2').reset_index(),on=k)
    u=u.merge(mc,on=k,how='left');u['dose_matched']=u.max1>=u.max2*0.99;return u

def fr(d,rule,tau=.3,tau_p=.7,mask=None):
    p,_=G.decide(d,rule,tau,tau_p);h=d.H.to_numpy();m=np.ones(len(d),bool) if mask is None else mask
    return float(((p==0)&(h==1)&m).sum()/((h==1)&m).sum())

def copula(e):
    """Gaussian-copula noise null: share of first-plate rejections of effective units that the replicate also rejects."""
    a=e[(e.H==1)&e.alt_effect.notna()];e1,e2=a.effect.to_numpy(),a.alt_effect.to_numpy();n=len(e1)
    z1,z2=norm.ppf(rankdata(e1)/(n+1)),norm.ppf(rankdata(e2)/(n+1));rho=np.corrcoef(z1,z2)[0,1]
    p1,p2=(e1<.3).mean(),(e2<.3).mean();pb=multivariate_normal([0,0],[[1,rho],[rho,1]]).cdf([norm.ppf(p1),norm.ppf(p2)])
    return {'copula_null_P_alt_rej_given_rej':float(pb/p1),'observed_P_alt_rej_given_rej':float((e2[e1<.3]<.3).mean())}

def main():
    out={}
    # M5: exploration-split replication
    e=with_dose(pd.read_csv('results/gdsc/units_explore.csv.gz',dtype={'BARCODE':str})).reset_index(drop=True)
    h=e[e.H==1];alt=e.alt_effect.notna().to_numpy();dm=e.dose_matched.to_numpy()
    out['M5_explore']={'n_units':int(len(e)),'FR_single_all':fr(e,'single_plate'),'FR_single_dose_matched':fr(e,'single_plate',mask=dm),
        'share_FR_lower_dose':float((~h[h.effect<.3].dose_matched).mean()),
        'with_replicate_single':fr(e,'single_plate',mask=alt),'with_replicate_confirm':fr(e,'confirm_or_reject',mask=alt),
        'dose_matched_with_replicate_single':fr(e,'single_plate',mask=alt&dm),'dose_matched_with_replicate_confirm':fr(e,'confirm_or_reject',mask=alt&dm)}
    out['M5_explore'].update(copula(e))
    # M2 + assay on the test split
    u=pd.read_csv('results/gdsc/units_test.csv.gz',dtype={'BARCODE':str})
    out['copula_test']=copula(u)
    raw=pd.read_csv(f'{RAW}/GDSC1_public_raw_data_24Jul22.csv',usecols=['BARCODE','CELL_ID','DATE_CREATED','ASSAY'],dtype=str).drop_duplicates('BARCODE').set_index('BARCODE')
    s=u[['COSMIC_ID','DRUG_ID','BARCODE']].sort_values('BARCODE');first=s.drop_duplicates(['COSMIC_ID','DRUG_ID']);second=s[~s.index.isin(first.index)].drop_duplicates(['COSMIC_ID','DRUG_ID'])
    pr=first.merge(second,on=['COSMIC_ID','DRUG_ID'],suffixes=('1','2'))
    pr['same_cell_stock']=raw.loc[pr.BARCODE1,'CELL_ID'].values==raw.loc[pr.BARCODE2,'CELL_ID'].values
    pr['same_date']=raw.loc[pr.BARCODE1,'DATE_CREATED'].values==raw.loc[pr.BARCODE2,'DATE_CREATED'].values
    h=u[(u.H==1)&u.alt_effect.notna()].merge(pr[['COSMIC_ID','DRUG_ID','same_cell_stock','same_date']],on=['COSMIC_ID','DRUG_ID'])
    h['rej']=h.effect<.3;h['arej']=h.alt_effect<.3
    out['M2_batch']={'pairs':int(len(pr)),'share_same_cell_stock':float(pr.same_cell_stock.mean()),'share_same_date':float(pr.same_date.mean()),
        'co_rejection_same_date':float(h[h.same_date&h.rej].arej.mean()),'co_rejection_different_date':float(h[~h.same_date&h.rej].arej.mean()),
        'n_rej_same_date':int((h.same_date&h.rej).sum()),'n_rej_different_date':int((~h.same_date&h.rej).sum())}
    u['ASSAY']=raw.loc[u.BARCODE,'ASSAY'].values
    out['assay']={str(k):{'n_H1':int((g.H==1).sum()),'FR_single':float(((g.effect<.3)&(g.H==1)).sum()/(g.H==1).sum())} for k,g in u.groupby('ASSAY')}
    # M3: expected points of the rules on the 400 agent episodes
    ep=pd.read_csv('results/llm/episodes.csv');w=ep.weight
    def pts(pred,req,d,c_fr):
        return (np.where(pred==0,c_fr*(d.H==1),np.where(pred==1,10*(d.H==0),1))+0.1*(1+2.5*(req&d.alt))).astype(float)
    out['M3']={'P_H1_given_single_null':float((w*ep.H).sum()/w.sum())}
    for c in (10,50):
        r={}
        for rule in G.RULES:
            p,_=G.decide(ep,rule,.3,.7);req=np.zeros(len(ep),bool) if rule in ('single_plate','control_gated') else np.ones(len(ep),bool)
            r[rule]=float((w*pts(p,req,ep,c)).sum()/w.sum())
        r['always_abstain']=1.1
        out['M3'][f'rules_expected_points_{c}to1']=r
    Path('results/gdsc/review2.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))

if __name__=='__main__':main()
