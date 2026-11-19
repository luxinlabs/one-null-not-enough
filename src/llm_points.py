"""Expected points of LLM conditions and rules under the 50:1 loss, on the episodes each condition covers (review round 3). Writes results/llm/points.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gdsc_replay as G

def points(pred,req,d,c):
    return (np.where(pred==0,c*(d.H==1),np.where(pred==1,10*(d.H==0),1))+0.1*(1+2.5*(req&d.alt))).astype(float)

def main():
    ep=pd.read_csv('results/llm/episodes.csv');out={}
    runs={k:{x['episode']:x for x in map(json.loads,open(f'results/llm/{k}.jsonl'))} for k in ('gpt-5.4-mini-2026-03-17__loss50','gpt-5.4-mini-2026-03-17__baserate','gpt-5.4-mini-2026-03-17__baserate_explain','gpt-5.4-mini-2026-03-17__structured')}
    common=set.intersection(*(set(v) for v in runs.values()));d=ep[ep.episode.isin(common)].reset_index(drop=True);w=d.weight
    for k,res in runs.items():
        x=[res[i] for i in d.episode];pred=np.array([r['pred'] for r in x]);req=np.array([r['requested'] for r in x])
        out[k]={'expected_points_50to1':float((w*points(pred,req,d,50)).sum()/w.sum()),'FR':float((w*((pred==0)&(d.H==1))).sum()/(w*(d.H==1)).sum()),
                'CR':float((w*((pred==0)&(d.H==0))).sum()/(w*(d.H==0)).sum()),'abstain':float((w*(pred==-1)).sum()/w.sum()),'request':float((w*req).sum()/w.sum())}
    for rule in G.RULES:
        p,_=G.decide(d,rule,.3,.7);req=np.zeros(len(d),bool) if rule in ('single_plate','control_gated') else np.ones(len(d),bool)
        out[rule]={'expected_points_50to1':float((w*points(p,req,d,50)).sum()/w.sum())}
    out['always_abstain']={'expected_points_50to1':1.1};out['n_common']=len(d)
    Path('results/llm/points.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
if __name__=='__main__':main()
