"""Compare LLM agents with the GDSC decision rules on the same test episodes.

Writes results/llm/summary.csv (one row per rule or LLM condition, repeats pooled by averaging per-repeat metrics,
with the min-max across repeats) and results/llm/summary_by_rep.csv.
"""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
import gdsc_replay as G

OUT=Path('results/llm')
Z_LOW=G.Z_STAR

def table(e,pred,cost,requested,rng,n_boot=2000):
    """Population-reweighted metrics among would-be rejections, with 95% bootstrap CIs over episodes."""
    h=e.H.to_numpy();w=e.weight.to_numpy();alt=e.alt.to_numpy();low=(e.zprime<Z_LOW).to_numpy()
    def m(i):
        ww=w[i];hh=h[i];pp=pred[i];rq=requested[i]
        return {'false_rejection':(ww*((pp==0)&(hh==1))).sum()/(ww*(hh==1)).sum(),'true_rejection':(ww*((pp==0)&(hh==0))).sum()/(ww*(hh==0)).sum(),
                'abstain':(ww*(pp==-1)).sum()/ww.sum(),'request_rate':(ww*rq).sum()/ww.sum(),'extra_plates':(ww*(cost[i]-G.COST['plate'])/G.COST['independent']).sum()/ww.sum()}
    point=m(np.arange(len(e)));boot=[m(rng.integers(0,len(e),len(e))) for _ in range(n_boot)];row={}
    for k,v in point.items():row[k]=v;row[k+'_lo'],row[k+'_hi']=np.percentile([b[k] for b in boot],[2.5,97.5])
    row['request_low_z']=requested[low].mean() if low.any() else np.nan;row['request_normal_z']=requested[~low].mean()
    row['request_alt_available']=requested[alt].mean();row['n']=len(e)
    return row

def main():
    e_all=pd.read_csv(OUT/'episodes.csv',dtype={'BARCODE':str});t=json.loads(Path('results/gdsc/thresholds.json').read_text())
    rng=np.random.default_rng(0);rows=[]
    for r in G.RULES:
        pred,cost=G.decide(e_all,r,t['tau'],t['tau_prime'])
        if r=='control_gated':cost=cost-G.COST['controls'] # on-plate controls are not an extra plate
        req=np.zeros(len(e_all),bool) if r in ('single_plate','control_gated') else np.ones(len(e_all),bool)
        rows.append({'agent':r,'model':'rule','prompt':'','rep':0,'kind':'rule','invalid':0,'usd':0.,**table(e_all,pred,cost,req,rng)})
    for f in sorted(OUT.glob('*__*.jsonl')):
        m=re.match(r'(.+?)__(.+?)(?:__r(\d+))?$',f.stem);model,prompt,rep=m.group(1),m.group(2),int(m.group(3) or 0)
        res={x['episode']:x for x in map(json.loads,f.open())}
        e=e_all[e_all.episode.isin(res)].reset_index(drop=True) # gpt-5.5 neutral_z covers a 200-episode prefix
        x=[res[i] for i in e.episode]
        pred=np.array([r['pred'] for r in x]);cost=np.array([r['exp_cost'] for r in x]);req=np.array([r['requested'] for r in x])
        rows.append({'agent':f'{model}__{prompt}','model':model,'prompt':prompt,'rep':rep,'kind':'llm','invalid':sum(any(t['action']=='INVALID' for t in r['turns']) for r in x),
                     'usd':sum(r['usd'] for r in x),**table(e,pred,cost,req,rng)})
    by=pd.DataFrame(rows);by.to_csv(OUT/'summary_by_rep.csv',index=False)
    metrics=['false_rejection','true_rejection','abstain','request_rate','extra_plates','request_low_z','request_normal_z','request_alt_available']
    agg=[]
    for a,g in by.groupby('agent',sort=False):
        row=g.iloc[0].to_dict();row['n_reps']=len(g);row['usd']=g.usd.sum()
        for k in metrics:row[k]=g[k].mean();row[k+'_rep_min'],row[k+'_rep_max']=g[k].min(),g[k].max()
        if len(g)>1:
            for k in ('false_rejection','true_rejection'):row[k+'_lo'],row[k+'_hi']=np.nan,np.nan # CI replaced by across-repeat range
        agg.append(row)
    s=pd.DataFrame(agg);s.to_csv(OUT/'summary.csv',index=False)
    f=lambda r,c:f"{r[c]:.2f}"+(f" [{r[c+'_lo']:.2f},{r[c+'_hi']:.2f}]" if pd.notna(r.get(c+'_lo')) else f" (reps {r[c+'_rep_min']:.2f}-{r[c+'_rep_max']:.2f})")
    for _,r in s.iterrows():
        print(f"{r.agent:28s} n={r.n:3d} reps={r.n_reps} FR={f(r,'false_rejection')} CR={f(r,'true_rejection')} abst={r.abstain:.2f} req={r.request_rate:.2f} req|lowZ={r.request_low_z:.2f} req|normalZ={r.request_normal_z:.2f} req|alt={r.request_alt_available:.2f} extra={r.extra_plates:.2f} invalid={r.invalid} ${r.usd:.2f}")
if __name__=='__main__':main()
