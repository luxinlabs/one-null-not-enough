"""Summarize agent campaigns against rule baselines on the same campaigns (results/agent/*.jsonl, rules.csv). Writes results/agent/summary.csv."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path('results/agent');M=['effective_retested','effective_recovered','false_rejections','correct_rejections','false_acceptances','abstentions','points']

def boot(x,rng,n=2000):
    x=np.asarray(x,float);b=x[rng.integers(0,len(x),(n,len(x)))].mean(1);return np.percentile(b,[2.5,97.5])

def main():
    rules=pd.read_csv(OUT/'rules.csv');rng=np.random.default_rng(0);rows=[]
    for f in sorted(OUT.glob('*__*.jsonl')):
        a=pd.DataFrame([json.loads(l) for l in f.open()]);model,cond=f.stem.split('__')
        served={t['served_model'] for t in a.turns.explode().dropna()}
        rows.append({'agent':f.stem,'kind':'llm','n_campaigns':len(a),'submitted':a.submitted.mean(),'mean_turns':a.turns.map(len).mean(),'usd':a.usd.sum(),'served':';'.join(sorted(served)),
                     **{k:a[k].mean() for k in M},**{k+'_ci':boot(a[k],rng).round(2).tolist() for k in ('effective_retested','points')},
                     'retest_zprime_corr':np.nan})
        for p,g in rules[rules.campaign.isin(a.campaign)].groupby('policy',sort=False):
            rows.append({'agent':f'{p} (on {f.stem} campaigns)','kind':'rule','n_campaigns':len(g),**{k:g[k].mean() for k in M},**{k+'_ci':boot(g[k],rng).round(2).tolist() for k in ('effective_retested','points')}})
    s=pd.DataFrame(rows);s.to_csv(OUT/'summary.csv',index=False);pd.set_option('display.width',250)
    print(s[['agent','n_campaigns','effective_retested','effective_retested_ci','effective_recovered','false_rejections','abstentions','points','points_ci']].round(2).to_string(index=False))
if __name__=='__main__':main()
