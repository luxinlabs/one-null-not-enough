"""Post-hoc robustness analyses requested in review (W1, W2, W8, W9). All on the GDSC test split; disclosed as post hoc.

W1: dose-matched stratum (GDSC1 maximum concentration >= GDSC2 maximum concentration for the pair).
W2: null baselines for the plate component: same cell line on other plates, and a Gaussian measurement-noise null for confirmation.
W8: test-split versions of the exploration-split statistics quoted in the paper.
W9: sensitivity to the GDSC2 AUC thresholds that define ground truth.
Writes results/gdsc/robustness.json.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal,norm
import gdsc_replay as G

RAW='data/raw/gdsc'

def load():
    u=pd.read_csv('results/gdsc/units_test.csv.gz',dtype={'BARCODE':str})
    g1=pd.read_csv(f'{RAW}/GDSC1_fitted_dose_response_24Jul22.csv');g2=pd.read_csv(f'{RAW}/GDSC2_fitted_dose_response_24Jul22.csv')
    k=['COSMIC_ID','DRUG_NAME']
    mc=g1.groupby(k).MAX_CONC.max().rename('max1').reset_index().merge(g2.groupby(k).MAX_CONC.max().rename('max2').reset_index(),on=k)
    u=u.merge(mc,on=k,how='left');u['dose_matched']=u.max1>=u.max2*0.99
    return u

def rule_table(d,tau,tau_p,rng,n_boot):
    m,_=G.evaluate(d.reset_index(drop=True),tau,tau_p,rng,n_boot)
    return m[['stratum','rule','n_units','n_H1','false_rejection','false_rejection_lo','false_rejection_hi','true_rejection','coverage']].round(4).to_dict('records')

def main():
    t=json.loads(Path('results/gdsc/thresholds.json').read_text());tau,tau_p=t['tau'],t['tau_prime']
    u=load();rng=np.random.default_rng(0);out={}
    h=u[u.H==1].copy();h['rej']=h.effect<tau
    # W1
    out['W1']={'share_H1_dose_matched':float(h.dose_matched.mean()),
               'single_plate_FR_dose_matched':float(h[h.dose_matched].rej.mean()),'single_plate_FR_lower_GDSC1_dose':float(h[~h.dose_matched].rej.mean()),
               'share_of_false_rejections_with_lower_GDSC1_dose':float(h[h.rej].dose_matched.eq(False).mean()),
               'rules_dose_matched':rule_table(u[u.dose_matched],tau,tau_p,rng,1000)}
    # W2a: same plate vs same cell line on other plates
    h['n_plate']=h.groupby('BARCODE').rej.transform('size');s_plate=h.groupby('BARCODE').rej.transform('sum')
    h['others_plate']=(s_plate-h.rej)/(h.n_plate-1).replace(0,np.nan)
    cl=h.groupby('COSMIC_ID').rej.agg(cl_sum='sum',cl_n='size');h=h.join(cl,on='COSMIC_ID')
    h['others_cell']=(h.cl_sum-s_plate)/(h.cl_n-h.n_plate).replace(0,np.nan)
    out['W2_plate_vs_cell_line']={k:float(v) for k,v in {
        'same_plate_given_rej':h[h.rej].others_plate.mean(),'same_plate_given_acc':h[~h.rej].others_plate.mean(),
        'other_plates_same_cell_line_given_rej':h[h.rej].others_cell.mean(),'other_plates_same_cell_line_given_acc':h[~h.rej].others_cell.mean()}.items()}
    # W2b: Gaussian measurement-noise null for confirmation among H=1 units with an independent plate
    a=h[h.alt_effect.notna()]
    e1,e2=a.effect.to_numpy(),a.alt_effect.to_numpy()
    mu=np.array([e1.mean(),e2.mean()]);C=np.cov(np.vstack([e1,e2]))
    p1=norm.cdf(tau,mu[0],np.sqrt(C[0,0]));p12=multivariate_normal(mu,C).cdf([tau,tau])
    out['W2_noise_null']={'n_pairs':int(len(a)),'corr':float(np.corrcoef(e1,e2)[0,1]),
        'observed_P_alt_rej_given_rej':float((e2[e1<tau]<tau).mean()),'gaussian_null_P_alt_rej_given_rej':float(p12/p1),
        'observed_FR_single':float((e1<tau).mean()),'observed_FR_both':float(((e1<tau)&(e2<tau)).mean()),'gaussian_null_FR_both':float(p12)}
    # W8: test-split versions of exploration-split statistics
    terc=pd.qcut(u.zprime,3,labels=['low','mid','high'])
    out['W8_test']={'FR_by_zprime_tertile':{str(k):float(v) for k,v in h.assign(t=terc[h.index]).groupby('t',observed=True).rej.mean().items()},
                    'second_plate_rej_given_first_rej':out['W2_noise_null']['observed_P_alt_rej_given_rej'],
                    'second_plate_rej_given_first_acc':float((e2[e1>=tau]<tau).mean())}
    # W9: ground-truth threshold sensitivity
    sens={}
    for lo,hi in ((0.7,0.9),(0.8,0.95),(0.85,0.95),(0.75,0.95)):
        d=u.copy();d['H']=np.where(d.AUC<=lo,1,np.where(d.AUC>=hi,0,-1));d=d[d.H>=0]
        r={}
        for rule in ('single_plate','control_gated','confirm_or_reject'):
            pred,_=G.decide(d,rule,tau,tau_p);hh=d.H.to_numpy()
            r[rule]={'FR':float(((pred==0)&(hh==1)).sum()/(hh==1).sum()),'CR':float(((pred==0)&(hh==0)).sum()/(hh==0).sum())}
        alt=d.alt_effect.notna().to_numpy();pred,_=G.decide(d,'confirm_or_reject',tau,tau_p);ps,_=G.decide(d,'single_plate',tau,tau_p);hh=d.H.to_numpy()
        r['with_independent_plate']={'single_FR':float(((ps==0)&(hh==1)&alt).sum()/((hh==1)&alt).sum()),'confirm_FR':float(((pred==0)&(hh==1)&alt).sum()/((hh==1)&alt).sum())}
        sens[f'{lo}/{hi}']=r
    out['W9_threshold_sensitivity']=sens
    Path('results/gdsc/robustness.json').write_text(json.dumps(out,indent=1))
    print(json.dumps({k:v for k,v in out.items() if k!='W1'},indent=1));print(json.dumps({k:v for k,v in out['W1'].items() if k!='rules_dose_matched'},indent=1))
    for r in out['W1']['rules_dose_matched']:print(r)
if __name__=='__main__':main()
