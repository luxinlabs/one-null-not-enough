"""Fifth-round review analysis (post hoc; disclosed). Writes results/gdsc/review5.json.

R9: curve-level dose comparability. Using GDSC2's fitted IC50, a pair is dose-reachable if GDSC2's IC50 lies within GDSC1's tested range
(IC50_2 <= GDSC1 maximum concentration), and well within range if IC50_2 <= max/4. Single-plate FR and confirmation are reported per stratum,
with cell-line bootstrap intervals.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from gdsc_review2 import with_dose

RAW='data/raw/gdsc';TAU=.3

def ci(d,f,n=1000,seed=0):
    rng=np.random.default_rng(seed);c=d.COSMIC_ID.to_numpy();uc=np.unique(c);idx={x:np.flatnonzero(c==x) for x in uc}
    return [float(x) for x in np.percentile([f(d.iloc[np.concatenate([idx[x] for x in rng.choice(uc,len(uc))])]) for _ in range(n)],[2.5,97.5])]

def main():
    u=with_dose(pd.read_csv('results/gdsc/units_test.csv.gz',dtype={'BARCODE':str}))
    g2=pd.read_csv(f'{RAW}/GDSC2_fitted_dose_response_24Jul22.csv').groupby(['COSMIC_ID','DRUG_NAME']).agg(ic50_2=('LN_IC50',lambda x:float(np.exp(x.mean()))),min2=('MIN_CONC','min'))
    u=u.join(g2,on=['COSMIC_ID','DRUG_NAME']);A=pd.read_csv('data/processed/gdsc_plates.csv.gz',dtype={'BARCODE':str}).set_index('BARCODE').ASSAY;u['ASSAY']=A.loc[u.BARCODE].values
    h=u[u.H==1].copy();h['rej']=h.effect<TAU
    strata={'all':h.index==h.index,'max_dose_matched':h.dose_matched,'reachable':h.ic50_2<=h.max1,'well_within':h.ic50_2<=h.max1/4,
            'well_within_assay_a':(h.ic50_2<=h.max1/4)&(h.ASSAY=='a'),'well_within_assay_s':(h.ic50_2<=h.max1/4)&(h.ASSAY=='s')}
    fr=lambda d:float(d.rej.mean());out={}
    for k,m in strata.items():
        g=h[m];r=g[g.alt_effect.notna()]
        out[k]={'n_H1':int(len(g)),'share_of_H1':float(m.mean()),'FR_single':fr(g),'FR_single_ci':ci(g,fr),'n_with_replicate':int(len(r)),
                'FR_single_with_replicate':fr(r),'FR_confirm_with_replicate':float((r.rej&(r.alt_effect<TAU)).mean())}
    out['share_of_FR_not_reachable']=float((h[h.rej].ic50_2>h[h.rej].max1).mean())
    Path('results/gdsc/review5.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
if __name__=='__main__':main()
