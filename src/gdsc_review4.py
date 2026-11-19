"""Fourth-round review analyses (post hoc; disclosed). Writes results/gdsc/review4.json.

R7: paired bootstrap of the matched-correct-rejection gap, FR(stricter single-plate threshold) - FR(confirm), with the single-plate
    threshold re-matched inside every resample; cell-line and crossed (cell line x compound) resampling.
R8: replicate-availability bias: units with and without a replicate plate compared on label, Z', dose matching, and single-plate FR.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from gdsc_review2 import with_dose

TAU=.3;TS=np.round(np.arange(.05,.61,.005),3)

def gap(e1,e2,h1,w):
    conf=(e1<TAU)&(e2<TAU);cr=(w*(conf&~h1)).sum()/(w*~h1).sum();fr=(w*(conf&h1)).sum()/(w*h1).sum()
    FR=[(w*((e1<t)&h1)).sum()/(w*h1).sum() for t in TS];CR=[(w*((e1<t)&~h1)).sum()/(w*~h1).sum() for t in TS]
    return float(np.interp(cr,CR,FR)-fr)

def main(n=1000):
    u=with_dose(pd.read_csv('results/gdsc/units_test.csv.gz',dtype={'BARCODE':str}));d=u[u.alt_effect.notna()].reset_index(drop=True)
    e1,e2,h1=d.effect.to_numpy(),d.alt_effect.to_numpy(),(d.H==1).to_numpy()
    ci,cl=np.unique(d.COSMIC_ID,return_inverse=True);di,dr=np.unique(d.DRUG_ID,return_inverse=True);rng=np.random.default_rng(0)
    cnt=lambda k:np.bincount(rng.integers(0,k,k),minlength=k)
    out={'R7_matched_gap':{'point':gap(e1,e2,h1,np.ones(len(d)))}}
    for name,wf in (('cell_line',lambda:cnt(len(ci))[cl].astype(float)),('crossed',lambda:(cnt(len(ci))[cl]*cnt(len(di))[dr]).astype(float))):
        b=np.array([gap(e1,e2,h1,wf()) for _ in range(n)]);out['R7_matched_gap'][name]={'ci95':[float(x) for x in np.percentile(b,[2.5,97.5])],'share_le_0':float((b<=0).mean())}
    u['has']=u.alt_effect.notna();h=u[u.H==1]
    out['R8_availability']={str(k):{'n':int(len(g)),'share_effective':float(g.H.mean()),'median_zprime':float(g.zprime.median()),'share_dose_matched':float(g.dose_matched.mean()),
        'n_cell_lines':int(g.COSMIC_ID.nunique()),'n_compounds':int(g.DRUG_ID.nunique()),'FR_single':float((h[h.has==k].effect<TAU).mean()),
        'FR_single_dose_matched':float((h[(h.has==k)&h.dose_matched].effect<TAU).mean())} for k,g in u.groupby('has')}
    Path('results/gdsc/review4.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
if __name__=='__main__':main()
