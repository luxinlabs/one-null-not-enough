"""What the agents' decisions on retested pairs imply about their beliefs (review round 9; post hoc). Writes results/agent/beliefs.json.

The logs record actions, not beliefs. On unretested pairs the agents take the expected-loss action for the probability shown in nearly every
case (src/agent_decisions.py), so we read each decision after a retest as the loss-minimizing action for some belief: REJECT means
P(effective) < 0.2, ACCEPT means > 0.8, ABSTAIN lies in between. A REJECT after a replicate in bin k then bounds the likelihood ratio the agent
acts on, LR_k < odds(0.2)/odds(prior), which we compare with the fitted LR_k. We also tabulate how often decisions match two simple rules
that ignore the prior: 'reading rule' (replicate null -> REJECT, effect -> ACCEPT, intermediate -> ABSTAIN) and the Bayes decision.
"""
import json
from pathlib import Path
import numpy as np
import agent_campaign as A
from agent_2x2 import bayes_decisions
from agent_decisions import decisions

RUNS=['gpt-5.4-mini-2026-03-17__voi_forced','gpt-5.4-mini-2026-03-17__voi','gpt-5.4-mini-2026-03-17__calibrated','gpt-5.5__calibrated']
odds=lambda p:p/(1-p)

def main():
    cs={int(d.campaign.iloc[0]):d for d in A.campaigns(50)};shared=sorted({json.loads(l)['campaign'] for l in open('results/agent/gpt-5.5__calibrated.jsonl')})
    lr=A.LIK[1]/A.LIK[0];out={'fitted_likelihood_ratio_by_bin':[float(x) for x in lr]};rng=np.random.default_rng(0)
    for run in RUNS:
        rows=[]
        for x in map(json.loads,open(f'results/agent/{run}.jsonl')):
            if x['campaign'] not in shared:continue
            d=cs[x['campaign']];dec=decisions(x);rt=set(x['retested_ids']);bd=bayes_decisions(d,rt);q=d.camp_p.to_numpy()
            for i,(p,e2,h) in enumerate(zip(d.pair_id,d.alt_effect,d.H)):
                if p not in rt:continue
                k=int(np.digitize(e2,[A.TAU,A.TAU_P]));post=q[i]*lr[k]/(q[i]*lr[k]+1-q[i])
                rows.append({'c':x['campaign'],'bin':k,'prior':q[i],'post':post,'agent':dec.get(p,'ABSTAIN'),'bayes':bd[p],'H':int(h)})
        r={'retested_pairs':len(rows)};null=[z for z in rows if z['bin']==0]
        r['null_replicate']={'n':len(null),'agent_reject_share':float(np.mean([z['agent']=='REJECT' for z in null])),
                             'bayes_reject_share':float(np.mean([z['bayes']=='REJECT' for z in null])),'effective_share':float(np.mean([z['H'] for z in null]))}
        bands=[(0,.2),(.2,.5),(.5,.8),(.8,1.01)];r['null_replicate_by_prior']={}
        for lo,hi in bands:
            s=[z for z in null if lo<=z['prior']<hi]
            if s:r['null_replicate_by_prior'][f'{lo}-{min(hi,1)}']={'n':len(s),'agent_reject':float(np.mean([z['agent']=='REJECT' for z in s])),
                    'bayes_reject':float(np.mean([z['bayes']=='REJECT' for z in s])),'mean_bayes_posterior':float(np.mean([z['post'] for z in s])),'effective_share':float(np.mean([z['H'] for z in s]))}
        rej=[z for z in null if z['agent']=='REJECT' and z['bayes']!='REJECT']
        if rej:
            bound=np.array([odds(.2)/odds(z['prior']) for z in rej])
            r['rejected_against_bayes_after_null']={'n':len(rej),'median_prior':float(np.median([z['prior'] for z in rej])),
                'median_bayes_posterior':float(np.median([z['post'] for z in rej])),'effective_share':float(np.mean([z['H'] for z in rej])),
                'implied_LR_upper_bound_median':float(np.median(bound)),'fitted_LR':float(lr[0]),'implied_overweighting_median':float(lr[0]/np.median(bound))}
        reading={0:'REJECT',1:'ABSTAIN',2:'ACCEPT'}
        r['agreement_with_reading_rule']=float(np.mean([z['agent']==reading[z['bin']] for z in rows]))
        r['agreement_with_bayes']=float(np.mean([z['agent']==z['bayes'] for z in rows]))
        # Campaign-level bootstrap for the agent's reject share after a null replicate.
        camps=sorted({z['c'] for z in null});by={c:[z['agent']=='REJECT' for z in null if z['c']==c] for c in camps};bs=[]
        for _ in range(2000):
            s=sum((by[c] for c in rng.choice(camps,len(camps))),[]);bs.append(np.mean(s))
        r['null_replicate']['agent_reject_share_ci']=[float(v) for v in np.percentile(bs,[2.5,97.5])]
        out[run]=r
    Path('results/agent/beliefs.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
if __name__=='__main__':main()
