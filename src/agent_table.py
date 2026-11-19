"""Table IV of the paper: agent conditions and rule baselines on the 30 campaigns shared by all agent conditions (post hoc). Writes results/agent/table4.csv."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path('results/agent');M=['effective_retested','false_rejections','abstentions','points']
RULES=['random','lowest_zprime','compound_history','retest_model','retest_model_bayes','oracle','voi_optimal']
AGENTS=['gpt-5.4-mini-2026-03-17__plain','gpt-5.4-mini-2026-03-17__score','gpt-5.4-mini-2026-03-17__calibrated','gpt-5.4-mini-2026-03-17__voi',
        'gpt-5.4-mini-2026-03-17__voi_forced','gpt-5.5__plain','gpt-5.5__score','gpt-5.5__calibrated']

def boot(x,rng,n=2000):
    x=np.asarray(x,float);return np.percentile(x[rng.integers(0,len(x),(n,len(x)))].mean(1),[2.5,97.5])

def main():
    runs={a:pd.DataFrame([json.loads(l) for l in open(OUT/f'{a}.jsonl')]) for a in AGENTS}
    shared=sorted(set.intersection(*(set(d.campaign) for d in runs.values())));rules=pd.read_csv(OUT/'rules.csv');rows=[]
    groups=[(p,rules[(rules.policy==p)&rules.campaign.isin(shared)]) for p in RULES]+[(a,d[d.campaign.isin(shared)]) for a,d in runs.items()]
    for name,g in groups:
        lo,hi=boot(g.sort_values('campaign').points,np.random.default_rng(0))
        rows.append({'policy':name,'n':len(g),**{k:g[k].mean() for k in M},'points_lo':lo,'points_hi':hi})
    t=pd.DataFrame(rows);t.to_csv(OUT/'table4.csv',index=False);print(t.round(2).to_string(index=False))
if __name__=='__main__':main()
