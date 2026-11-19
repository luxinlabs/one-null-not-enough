"""2x2 decomposition of the VOI-scaffolded agent's gap (review round 7; post hoc). Writes results/agent/decomposition_2x2.json.

Rows: who selects the 6 retests (agent with the VOI scaffold, or the model-optimal VOI policy); columns: who makes the final decisions
(the agent, or programmatic expected-loss decisions under the estimated decision model). Cells use the same 30 campaigns.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import agent_campaign as A

def bayes_decisions(d,retested):
    q=d.camp_p.to_numpy();post=lambda x,k:x*A.LIK[1][k]/(x*A.LIK[1][k]+(1-x)*A.LIK[0][k]);dec={}
    for i,(p,e2) in enumerate(zip(d.pair_id,d.alt_effect)):
        x=post(q[i],int(np.digitize(e2,[A.TAU,A.TAU_P]))) if p in retested else q[i]
        dec[p]=['REJECT','ACCEPT','ABSTAIN'][int(np.argmin([A.COST_FR*x,A.COST_FA*(1-x),A.COST_ABSTAIN]))]
    return dec

def ci(x,rng,n=2000):
    x=np.asarray(x,float);return [float(v) for v in np.percentile(x[rng.integers(0,len(x),(n,len(x)))].mean(1),[2.5,97.5])]

def main():
    cs={int(d.campaign.iloc[0]):d for d in A.campaigns(50)}
    ag={x['campaign']:x for x in map(json.loads,open('results/agent/gpt-5.4-mini-2026-03-17__voi.jsonl'))}
    fo={x['campaign']:x for x in map(json.loads,open('results/agent/gpt-5.4-mini-2026-03-17__voi_forced.jsonl'))}
    camp=sorted(set(ag)&set(fo));rows=[]
    for c in camp:
        d=cs[c];ln,vi=A.voi_columns(d);voi_top=list(d.pair_id.iloc[np.argsort(-vi,kind='stable')[:A.BUDGET]])
        rows.append({'campaign':c,'agent_retests_agent_decisions':ag[c]['points'],
                     'agent_retests_bayes_decisions':A.evaluate(d,ag[c]['retested_ids'],bayes_decisions(d,set(ag[c]['retested_ids'])))['points'],
                     'voi_retests_agent_decisions':fo[c]['points'],'voi_retests_bayes_decisions':A.evaluate(d,voi_top,bayes_decisions(d,set(voi_top)))['points'],
                     'retest_overlap':len(set(voi_top)&set(ag[c]['retested_ids']))})
    r=pd.DataFrame(rows);rng=np.random.default_rng(0);out={'n_campaigns':len(r),'mean_retest_overlap':float(r.retest_overlap.mean())}
    for k in r.columns[1:-1]:out[k]={'mean':float(r[k].mean()),'ci':ci(r[k],rng)}
    gap=r.agent_retests_agent_decisions-r.voi_retests_bayes_decisions
    out['total_gap']={'mean':float(gap.mean()),'ci':ci(gap,rng)}
    out['decision_effect_given_agent_retests']={'mean':float((r.agent_retests_agent_decisions-r.agent_retests_bayes_decisions).mean()),'ci':ci(r.agent_retests_agent_decisions-r.agent_retests_bayes_decisions,rng)}
    out['decision_effect_given_voi_retests']={'mean':float((r.voi_retests_agent_decisions-r.voi_retests_bayes_decisions).mean()),'ci':ci(r.voi_retests_agent_decisions-r.voi_retests_bayes_decisions,rng)}
    out['selection_effect_given_bayes_decisions']={'mean':float((r.agent_retests_bayes_decisions-r.voi_retests_bayes_decisions).mean()),'ci':ci(r.agent_retests_bayes_decisions-r.voi_retests_bayes_decisions,rng)}
    out['selection_effect_given_agent_decisions']={'mean':float((r.agent_retests_agent_decisions-r.voi_retests_agent_decisions).mean()),'ci':ci(r.agent_retests_agent_decisions-r.voi_retests_agent_decisions,rng)}
    # Symmetric (Shapley) attribution: average each component's effect over both replacement orders; the interaction is split equally.
    AA,AB,VA,VB=(r[k] for k in ('agent_retests_agent_decisions','agent_retests_bayes_decisions','voi_retests_agent_decisions','voi_retests_bayes_decisions'))
    for k,v in {'symmetric_selection':((AA-VA)+(AB-VB))/2,'symmetric_decision':((AA-AB)+(VA-VB))/2,'interaction':(AA-VA)-(AB-VB)}.items():
        out[k]={'mean':float(v.mean()),'ci':ci(v,rng)}
    out['served_models']=sorted({t['served_model'] for x in fo.values() for t in x['turns']})
    Path('results/agent/decomposition_2x2.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
if __name__=='__main__':main()
