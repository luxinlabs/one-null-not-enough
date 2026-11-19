"""Loss sensitivity of the decision-stage gap without new agent runs (review round 9; post hoc). Writes results/agent/cost_sensitivity.json.

Agents were run under one loss (10/10/2), so their decisions cannot be rescored under another. We instead ask whether the mechanism they
exhibit costs points under other losses: for each loss, the model-optimal policy (VOI retests and expected-loss decisions, both recomputed
for that loss) is compared on all 50 campaigns with the same retests followed by a 'conclusive-replicate' rule that reads the replicate as
decisive (null -> REJECT, effect -> ACCEPT, intermediate -> ABSTAIN) and decides unretested pairs by expected loss.
"""
import json
from pathlib import Path
import numpy as np
import agent_campaign as A

def run(d,cfr,cfa,cab):
    q=d.camp_p.to_numpy();h=d.H.to_numpy();e2=d.alt_effect.to_numpy()
    L=lambda x:np.minimum.reduce([cfr*x,cfa*(1-x),np.full_like(x,cab)]);post=lambda x,k:x*A.LIK[1][k]/(x*A.LIK[1][k]+(1-x)*A.LIK[0][k])
    voi=L(q)-sum((q*A.LIK[1][k]+(1-q)*A.LIK[0][k])*L(post(q,k)) for k in range(3));top=np.argsort(-voi,kind='stable')[:A.BUDGET]
    k=np.digitize(e2,[A.TAU,A.TAU_P]);pts=lambda a:np.where(a==0,cfr*h,np.where(a==1,cfa*(1-h),cab)).sum()
    best=lambda x:np.argmin(np.stack([cfr*x,cfa*(1-x),np.full_like(x,cab)]),0)
    bay=best(q);bay[top]=best(post(q[top],k[top]));con=best(q);con[top]=np.array([0,2,1])[k[top]]
    return float(pts(bay)),float(pts(con))

def main():
    cs=A.campaigns(50);rng=np.random.default_rng(0);out={}
    for cfr,cab in [(5,2),(10,1),(10,2),(10,4),(20,2)]:
        r=np.array([run(d,cfr,10,cab) for d in cs]);g=r[:,1]-r[:,0];bs=g[rng.integers(0,len(g),(2000,len(g)))].mean(1)
        out[f'FR{cfr}_FA10_abstain{cab}']={'bayes':float(r[:,0].mean()),'conclusive_replicate':float(r[:,1].mean()),'gap':float(g.mean()),
                                           'gap_ci':[float(v) for v in np.percentile(bs,[2.5,97.5])]}
    Path('results/agent/cost_sensitivity.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
if __name__=='__main__':main()
