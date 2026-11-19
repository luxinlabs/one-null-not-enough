"""Direct test of belief updating vs. the terminal decision (camera-ready round; post hoc). Writes results/agent/update_test.json.

All runs use the model-optimal policy's 6 retests on the same 30 campaigns. voi_forced shows the prior and the replicate reading (the agent
must update); forced_lr also shows the fitted likelihood ratio of each replicate reading (the agent must apply it); forced_post also shows the posterior and the expected points of each action (the agent only has to pick the cheapest);
__high runs use high reasoning effort. Reported per run: campaign points (95% CI over campaigns), deviations from the expected-loss action
on retested pairs and their realized cost, and the share of null-replicate pairs shown p in [0.5, 0.8) that the agent rejected.
"""
import json
from pathlib import Path
import numpy as np
import agent_campaign as A
from agent_2x2 import bayes_decisions
from agent_decisions import decisions

RUNS=['gpt-5.4-mini-2026-03-17__voi_forced','gpt-5.4-mini-2026-03-17__voi_forced__high','gpt-5.4-mini-2026-03-17__forced_post',
      'gpt-5.4-mini-2026-03-17__forced_post__high','gpt-5.5__forced_post','gpt-5.4-mini-2026-03-17__forced_lr','gpt-5.5__forced_lr']

def main():
    cs={int(d.campaign.iloc[0]):d for d in A.campaigns(50)};rng=np.random.default_rng(0);out={}
    for run in RUNS:
        f=Path(f'results/agent/{run}.jsonl')
        if not f.exists():continue
        xs=[json.loads(l) for l in f.open()];pts=np.array([x['points'] for x in xs]);dev=[];exc=[];rej=[];unret=[0,0]
        for x in xs:
            d=cs[x['campaign']];dec=decisions(x);rt=set(x['retested_ids']);bd=bayes_decisions(d,rt);q=d.camp_p.to_numpy();k=0;e=0.
            for i,(p,e2,h) in enumerate(zip(d.pair_id,d.alt_effect,d.H)):
                a=dec.get(p,'ABSTAIN')
                if p not in rt:unret[0]+=a!=bd[p];unret[1]+=1;continue
                if a!=bd[p]:
                    cost={'REJECT':A.COST_FR*h,'ACCEPT':A.COST_FA*(1-h),'ABSTAIN':A.COST_ABSTAIN};k+=1;e+=cost[a]-cost[bd[p]]
                if e2<A.TAU and .5<=q[i]<.8:rej.append(a=='REJECT')
            dev.append(k);exc.append(e)
        bs=pts[rng.integers(0,len(pts),(2000,len(pts)))].mean(1)
        out[run]={'campaigns':len(xs),'points':float(pts.mean()),'points_ci':[float(v) for v in np.percentile(bs,[2.5,97.5])],
                  'unretested_deviation_share':unret[0]/unret[1],'retested_deviations_per_campaign':float(np.mean(dev)),
                  'retested_excess_points_per_campaign':float(np.mean(exc)),'null_replicate_prior_05_08':{'n':len(rej),'reject_share':float(np.mean(rej)) if rej else None},
                  'usd':float(sum(x['usd'] for x in xs)),'served_models':sorted({t['served_model'] for x in xs for t in x['turns']})}
    Path('results/agent/update_test.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
if __name__=='__main__':main()
