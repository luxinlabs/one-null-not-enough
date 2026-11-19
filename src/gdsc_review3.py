"""Third-round review analyses (post hoc; disclosed). Writes results/gdsc/review3.json and results/gdsc/frontier.csv.

R1: layered decomposition of single-plate false rejections (all / dose-matched / dose-matched by assay / replicate outcome).
R2: matched risk-coverage-cost comparison on real plates: threshold sweeps for single-plate, control-gated, confirm, and two-plate-mean rules.
R3: grouped splits (by cell line, by compound) with thresholds refitted on the grouped exploration part.
R4: crossed (cell line x compound) bootstrap for the headline rates.
R5: dose-matching rule sensitivity.
R6: exploration-split conditional rates given to the structured LLM prompt, and a Bayes decision-support baseline on the agent episodes.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import gdsc_replay as G
from gdsc_review2 import with_dose

RAW='data/raw/gdsc'
TAU,TAU_P=.3,.7

def assay():
    return pd.read_csv('data/processed/gdsc_plates.csv.gz',dtype={'BARCODE':str}).set_index('BARCODE').ASSAY

def fr(rej,h1,w=None):
    w=np.ones(len(rej)) if w is None else w
    return float((w*(rej&h1)).sum()/(w*h1).sum())

def decomposition(u):
    h=u[u.H==1].copy();h['rej']=h.effect<TAU;h['has']=h.alt_effect.notna();h['arej']=h.alt_effect<TAU
    strata={'all':h.index==h.index,'dose_lower':~h.dose_matched,'dose_matched':h.dose_matched,
            'dose_matched_assay_a':h.dose_matched&(h.ASSAY=='a'),'dose_matched_assay_s':h.dose_matched&(h.ASSAY=='s')}
    out={}
    for k,m in strata.items():
        g=h[m];r=g[g.has]
        out[k]={'n_H1':int(len(g)),'FR_single':float(g.rej.mean()),'n_H1_with_replicate':int(len(r)),
                'FR_single_with_replicate':float(r.rej.mean()),'FR_confirm_with_replicate':float((r.rej&r.arej).mean()),
                'share_of_FR_replicate_persistent':float(r[r.rej].arej.mean())}
    # Three-way attribution of false rejections that have a replicate.
    r=h[h.has&h.rej]
    out['attribution_of_FR_with_replicate']={'n':int(len(r)),'replicate_corrected':float((~r.arej).mean()),
        'persistent_dose_lower':float((r.arej&~r.dose_matched).mean()),'persistent_dose_matched':float((r.arej&r.dose_matched).mean())}
    return out

def frontier(u):
    """FR among effective, CR among ineffective, extra plates per unit, on test units with a replicate plate."""
    d=u[u.alt_effect.notna()].reset_index(drop=True);h1=(d.H==1).to_numpy();h0=~h1;e1=d.effect.to_numpy();e2=d.alt_effect.to_numpy();z=d.zprime.to_numpy()
    rows=[]
    def add(rule,param,rej,extra):rows.append({'rule':rule,'param':param,'FR':fr(rej,h1),'CR':fr(rej,h0),'extra_plates':float(extra)})
    for t in np.round(np.arange(.05,.61,.025),3):
        add('single_plate',t,e1<t,0);add('confirm',t,(e1<t)&(e2<t),(e1<t).mean());add('two_plate_mean',t,(e1+e2)/2<t,1.)
    for zs in np.round(np.quantile(z,np.arange(0,.61,.05)),3):add('control_gated',zs,(e1<TAU)&(z>=zs),0)
    f=pd.DataFrame(rows);f.to_csv('results/gdsc/frontier.csv',index=False)
    # Matched comparison: single-plate FR interpolated at the confirm rule's CR (tau=0.3).
    c=f[(f.rule=='confirm')&(f.param==TAU)].iloc[0];s=f[f.rule=='single_plate'].sort_values('CR')
    m=f[(f.rule=='two_plate_mean')].sort_values('CR')
    return {'confirm_tau03':c[['FR','CR','extra_plates']].to_dict(),
            'single_plate_FR_at_matched_CR':float(np.interp(c.CR,s.CR,s.FR)),'single_plate_tau_at_matched_CR':float(np.interp(c.CR,s.CR,s.param)),
            'two_plate_mean_FR_at_matched_CR':float(np.interp(c.CR,m.CR,m.FR)),
            'control_gated_FR_at_matched_CR':float(np.interp(c.CR,*f[f.rule=='control_gated'].sort_values('CR')[['CR','FR']].T.to_numpy())) if c.CR>=f[f.rule=='control_gated'].CR.min() else None}

def grouped(col):
    """Re-split all labelled plates by a hash of `col`, refit thresholds on the 30% part, evaluate on the rest."""
    u=pd.read_csv('data/processed/gdsc_units.csv.gz',dtype={'BARCODE':str})
    u['split']=np.where([int(hashlib.sha256(str(x).encode()).hexdigest(),16)%10<3 for x in u[col]],'explore','test')
    g1=pd.read_csv(f'{RAW}/GDSC1_fitted_dose_response_24Jul22.csv');g2=pd.read_csv(f'{RAW}/GDSC2_fitted_dose_response_24Jul22.csv')
    name=g1.drop_duplicates('DRUG_ID').set_index('DRUG_ID').DRUG_NAME;gt=g2.groupby(['COSMIC_ID','DRUG_NAME'],as_index=False).AUC.mean()
    res={}
    for split in ('explore','test'):
        s=u[u.split==split].copy();s['DRUG_NAME']=s.DRUG_ID.map(name);d=s.merge(gt,on=['COSMIC_ID','DRUG_NAME'])
        d['H']=np.where(d.AUC<=.8,1,np.where(d.AUC>=.95,0,-1));d=d[d.H>=0].copy()
        o=s[['COSMIC_ID','DRUG_ID','BARCODE','effect']].sort_values('BARCODE');first=o.drop_duplicates(['COSMIC_ID','DRUG_ID'])
        second=o[~o.index.isin(first.index)].drop_duplicates(['COSMIC_ID','DRUG_ID'])
        d=d.merge(first.rename(columns={'BARCODE':'b1','effect':'e1'}),on=['COSMIC_ID','DRUG_ID'],how='left').merge(second.rename(columns={'BARCODE':'b2','effect':'e2'}),on=['COSMIC_ID','DRUG_ID'],how='left')
        d['alt_effect']=np.where(d.BARCODE==d.b1,d.e2,d.e1);res[split]=with_dose(d.drop(columns=['b1','e1','b2','e2']).reset_index(drop=True))
    tau,tau_p=G.fit_thresholds(res['explore']);d=res['test'];h=d[d.H==1];r=h[h.alt_effect.notna()]
    return {'tau':tau,'tau_prime':tau_p,'n_test_units':int(len(d)),'n_test_H1':int(len(h)),'FR_single':float((h.effect<tau).mean()),
            'FR_single_dose_matched':float((h[h.dose_matched].effect<tau).mean()),
            'FR_single_with_replicate':float((r.effect<tau).mean()),'FR_confirm_with_replicate':float(((r.effect<tau)&(r.alt_effect<tau)).mean()),
            'CR_single':float((d[d.H==0].effect<tau).mean())}

def crossed_bootstrap(u,n=2000,seed=0):
    """Pigeonhole bootstrap: resample cell lines and compounds independently; unit weight = product of multiplicities."""
    rng=np.random.default_rng(seed);h=u[u.H==1].reset_index(drop=True)
    ci,cl=np.unique(h.COSMIC_ID,return_inverse=True);di,dr=np.unique(h.DRUG_ID,return_inverse=True)
    rej=(h.effect<TAU).to_numpy();dm=h.dose_matched.to_numpy();has=h.alt_effect.notna().to_numpy();conf=rej&(h.alt_effect<TAU).to_numpy()
    stats={'FR_single':[],'FR_single_dose_matched':[],'FR_single_with_replicate':[],'FR_confirm_with_replicate':[]}
    for _ in range(n):
        w=np.bincount(rng.integers(0,len(ci),len(ci)),minlength=len(ci))[cl]*np.bincount(rng.integers(0,len(di),len(di)),minlength=len(di))[dr]
        stats['FR_single'].append(fr(rej,np.ones(len(h),bool),w));stats['FR_single_dose_matched'].append(fr(rej,dm,w))
        stats['FR_single_with_replicate'].append(fr(rej,has,w));stats['FR_confirm_with_replicate'].append(fr(conf,has,w))
    return {k:[float(x) for x in np.percentile(v,[2.5,97.5])] for k,v in stats.items()}

def dose_rule_sensitivity(u):
    h=u[u.H==1];out={}
    for name,m in {'max1>=max2 (primary)':h.max1>=h.max2*.99,'max1>=0.5*max2':h.max1>=h.max2*.5,'max1>=2*max2':h.max1>=h.max2*2,
                   'max1 within 1% of max2':(h.max1/h.max2-1).abs()<=.01}.items():
        out[name]={'share_H1':float(m.mean()),'FR_single':float((h[m].effect<TAU).mean())}
    return out

def decision_support(ex,ep):
    """Exploration-split P(H=1) among would-be rejections, and given the replicate outcome; Bayes actions under 10:1 and 50:1 losses."""
    a=ex[ex.effect<TAU];r=a[a.alt_effect.notna()]
    rates={'P_H1_given_null':float(a.H.mean()),'P_H1_given_null_replicate_null':float(r[r.alt_effect<TAU].H.mean()),
           'P_H1_given_null_replicate_effect':float(r[r.alt_effect>=TAU].H.mean()),'P_replicate_null_given_null':float((r.alt_effect<TAU).mean()),
           'P_replicate_available_given_null':float(a.alt_effect.notna().mean()),'n_explore_null':int(len(a))}
    w=ep.weight.to_numpy();H=ep.H.to_numpy();alt=ep.alt.to_numpy();e2=ep.alt_effect.to_numpy();out={'rates':rates}
    def loss_final(p,c):return min(c*p,10*(1-p),1.)  # reject, accept, abstain
    def act(p,c):return int(np.argmin([c*p,10*(1-p),1.]))  # 0 reject, 1 accept, 2 abstain
    for c in (10,50):
        p0=rates['P_H1_given_null'];pn,pe=rates['P_H1_given_null_replicate_null'],rates['P_H1_given_null_replicate_effect'];q=rates['P_replicate_null_given_null']
        # Request if the expected loss after the replicate (when available) plus its cost beats deciding now.
        after=q*loss_final(pn,c)+(1-q)*loss_final(pe,c)+.25
        request=after<loss_final(p0,c)
        pts=[];pred=[]
        for i in range(len(ep)):
            pts_i=.1
            if request and alt[i]:
                p=pn if e2[i]<TAU else pe;pts_i+=.25
            else:p=p0
            k=act(p,c);pred.append([0,1,-1][k]);pts_i+=[c*H[i],10*(1-H[i]),1.][k];pts.append(pts_i)
        pred=np.array(pred);pts=np.array(pts)
        out[f'bayes_{c}to1']={'requests':bool(request),'expected_points':float((w*pts).sum()/w.sum()),
            'FR':float((w*((pred==0)&(H==1))).sum()/(w*(H==1)).sum()),'CR':float((w*((pred==0)&(H==0))).sum()/(w*(H==0)).sum())}
    return out

def main():
    A=assay();u=with_dose(pd.read_csv('results/gdsc/units_test.csv.gz',dtype={'BARCODE':str}));u['ASSAY']=A.loc[u.BARCODE].values
    ex=pd.read_csv('results/gdsc/units_explore.csv.gz',dtype={'BARCODE':str})
    ep=pd.read_csv('results/llm/episodes.csv')
    out={'R1_decomposition':decomposition(u),'R2_matched_frontier':frontier(u),
         'R3_grouped_split':{'cell_line':grouped('COSMIC_ID'),'compound':grouped('DRUG_ID')},
         'R4_crossed_bootstrap_CI':crossed_bootstrap(u),'R5_dose_rule':dose_rule_sensitivity(u),'R6_decision_support':decision_support(ex,ep)}
    Path('results/gdsc/review3.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))

if __name__=='__main__':main()
