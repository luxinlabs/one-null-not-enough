"""Terminal-decision errors when the agent is shown the probability (review round 8; post hoc). Writes results/agent/decision_errors.json.

On pairs the agent did not retest, the calibrated and VOI conditions show p_effective, the campaign probability the decision model uses, and
the prompt states the loss, so the expected-loss-minimizing action is determined by what the agent sees; the VOI condition also shows
loss_now, the expected points of that action. We count decisions that differ from it and the expected points they add, separately from
retest selection. On retested pairs the agent must itself combine p_effective with the replicate reading; there we compare with
expected-loss decisions under the posterior of the estimated decision model (agent_2x2.bayes_decisions) and count the realized points.
"""
import json
from pathlib import Path
import numpy as np
import agent_campaign as A
from agent_2x2 import bayes_decisions

RUNS=['gpt-5.4-mini-2026-03-17__calibrated','gpt-5.4-mini-2026-03-17__voi','gpt-5.4-mini-2026-03-17__voi_forced','gpt-5.5__calibrated']
ACT=['REJECT','ACCEPT','ABSTAIN']

def decisions(x):
    for t in x['turns']:
        for c in t['calls']:
            if c['name']=='submit_decisions':return {e['pair_id']:e['decision'] for e in json.loads(c['arguments'])['decisions']}
    return {}

def main():
    cs={int(d.campaign.iloc[0]):d for d in A.campaigns(50)};out={}
    shared=sorted({json.loads(l)['campaign'] for l in open('results/agent/gpt-5.5__calibrated.jsonl')})
    for run in RUNS:
        n=wrong=0;excess=0.;kinds={};per=[]
        for x in map(json.loads,open(f'results/agent/{run}.jsonl')):
            if x['campaign'] not in shared:continue
            d=cs[x['campaign']];dec=decisions(x);q=d.camp_p.round(3).to_numpy();k_wrong=0
            for i,p in enumerate(d.pair_id):
                if p in x['retested_ids']:continue
                loss=np.array([A.COST_FR*q[i],A.COST_FA*(1-q[i]),A.COST_ABSTAIN]);best=ACT[int(np.argmin(loss))];a=dec.get(p,'ABSTAIN');n+=1
                if a!=best:
                    wrong+=1;k_wrong+=1;excess+=loss[ACT.index(a)]-loss.min();kinds[f'{a} where {best} optimal']=kinds.get(f'{a} where {best} optimal',0)+1
            per.append(k_wrong)
        ret=[0,0.];rk={}
        for x in map(json.loads,open(f'results/agent/{run}.jsonl')):
            if x['campaign'] not in shared:continue
            d=cs[x['campaign']];dec=decisions(x);rt=set(x['retested_ids']);bd=bayes_decisions(d,rt)
            for p,h in zip(d.pair_id,d.H):
                if p not in rt or dec.get(p,'ABSTAIN')==bd[p]:continue
                pts={'REJECT':A.COST_FR*h,'ACCEPT':A.COST_FA*(1-h),'ABSTAIN':A.COST_ABSTAIN};a=dec.get(p,'ABSTAIN')
                ret[0]+=1;ret[1]+=pts[a]-pts[bd[p]];rk[f'{a} where {bd[p]} optimal']=rk.get(f'{a} where {bd[p]} optimal',0)+1
        out[run+' (retested pairs)']={'deviations_per_campaign':ret[0]/len(per),'realized_excess_points_per_campaign':ret[1]/len(per),
                                      'kinds':dict(sorted(rk.items(),key=lambda z:-z[1]))}
        out[run]={'campaigns':len(per),'unretested_pairs':n,'share_not_loss_minimizing':wrong/n,'per_campaign':float(np.mean(per)),
                  'expected_excess_points_per_campaign':excess/len(per),'kinds':dict(sorted(kinds.items(),key=lambda z:-z[1]))}
    Path('results/agent/decision_errors.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
if __name__=='__main__':main()
