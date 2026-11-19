"""Risk-coverage-cost frontier in the simulator (review round 3; post hoc). Writes results/sim_frontier.csv and results/sim_frontier.json.

Families on the same seeds (sigma 0.5, budget 5, test seeds 100-299):
  bayes_t:   joint VOI rule, rejection threshold t swept post hoc on its final posterior (acceptance fixed at 0.9);
  voi_tol:   joint VOI rule with different stopping tolerances (spends more as the tolerance falls);
  guard_eps: identifiability guard with different support thresholds epsilon;
  credal_Q:  credal rule with nested prior sets Q.
Matched comparison: at each guard point, the Bayes-threshold false rejection interpolated at the same coverage.
"""
import itertools,json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import experiment as E

SEEDS=range(100,300)
CONDS={'fault0.5_matched':dict(fault_rate=.5,prior_fault=None,truth='specified'),'fault0.1_matched':dict(fault_rate=.1,prior_fault=None,truth='specified'),
       'fault0.5_prior0.1':dict(fault_rate=.5,prior_fault=.1,truth='specified'),'partial_fault0.5':dict(fault_rate=.5,prior_fault=None,truth='partial_fault')}
CLASSES=('resolvable','unidentifiable')
VOI_TOL=(1e-2,1e-3,1e-4)
EPS=(1e-4,1e-3,1e-2,5e-2)
Q=((.05,.95),(.1,.9),(.2,.8),(.3,.7),(.4,.6))
BAYES_T=(.01,.02,.05,.1,.15,.2,.3)

def job(args):
    cond,cls=args;c=CONDS[cond];rows=[]
    kw=dict(sigma=.5,budget=5.,n_mc=256,**c)
    for s in SEEDS:
        for tol in VOI_TOL:
            r=E.run(s,'joint_risk_voi',cls,voi_tol=tol,**kw)
            if tol==1e-3:
                for t in BAYES_T:
                    p=r['posterior_h1'];pred=1 if p>E.DECIDE else (0 if p<t else -1)
                    rows.append(dict(family='bayes_t',param=t,seed=s,h=r['true_h'],pred=pred,spent=r['spent']))
            rows.append(dict(family='voi_tol',param=tol,seed=s,h=r['true_h'],pred=r['predicted_h'],spent=r['spent']))
        for eps in EPS:
            r=E.run(s,'identifiability',cls,support_eps=eps,**kw);rows.append(dict(family='guard_eps',param=eps,seed=s,h=r['true_h'],pred=r['predicted_h'],spent=r['spent']))
        r=E.run(s,'identifiability_budget',cls,**kw);rows.append(dict(family='guard_budget',param=1e-3,seed=s,h=r['true_h'],pred=r['predicted_h'],spent=r['spent']))
        for q in Q:
            r=E.run(s,'robust_bayes',cls,credal=q,**kw);rows.append(dict(family='credal_Q',param=f'{q[0]}-{q[1]}',seed=s,h=r['true_h'],pred=r['predicted_h'],spent=r['spent']))
    d=pd.DataFrame(rows);d['condition']=cond;d['class']=cls;return d

def budget_job(args):
    """Structural vs budget-aware guard across budgets (review round 4)."""
    cond,cls,b=args;c=CONDS[cond];rows=[]
    for s in SEEDS:
        for pol in ('joint_risk_voi','identifiability','identifiability_budget'):
            r=E.run(s,pol,cls,sigma=.5,budget=b,n_mc=256,**c);rows.append(dict(family=pol,param=b,seed=s,h=r['true_h'],pred=r['predicted_h'],spent=r['spent']))
    d=pd.DataFrame(rows);d['condition']=cond;d['class']=cls;return d

def summarize(d):
    g=d.groupby(['condition','class','family','param'],sort=False)
    return g.apply(lambda x:pd.Series({'coverage':(x.pred>=0).mean(),'false_rejection':((x.pred==0)&(x.h==1)).sum()/(x.h==1).sum(),
        'false_acceptance':((x.pred==1)&(x.h==0)).sum()/(x.h==0).sum(),'mean_cost':x.spent.mean()}),include_groups=False).reset_index()

def matched(s):
    out=[]
    for (cond,cls),x in s.groupby(['condition','class'],sort=False):
        b=x[x.family=='bayes_t'].sort_values('coverage');cr=x[x.family=='credal_Q'].sort_values('coverage')
        for _,gp in x[x.family=='guard_eps'].iterrows():
            out.append({'condition':cond,'class':cls,'guard_eps':gp.param,'guard_coverage':gp.coverage,'guard_FR':gp.false_rejection,'guard_cost':gp.mean_cost,
                        'bayes_FR_at_matched_coverage':float(np.interp(gp.coverage,b.coverage,b.false_rejection)),
                        'credal_FR_at_matched_coverage':float(np.interp(gp.coverage,cr.coverage,cr.false_rejection)),
                        'bayes_cost':float(b.mean_cost.iloc[0]),'coverage_in_bayes_range':bool(b.coverage.min()<=gp.coverage<=b.coverage.max())})
    return out

def main():
    with Pool() as p:d=pd.concat(p.map(job,list(itertools.product(CONDS,CLASSES))))
    s=summarize(d);s.to_csv('results/sim_frontier.csv',index=False)
    Path('results/sim_frontier.json').write_text(json.dumps(matched(s),indent=1))
    with Pool() as p:b=pd.concat(p.map(budget_job,list(itertools.product(CONDS,CLASSES,(2.,5.,10.)))))
    sb=summarize(b);sb.to_csv('results/sim_budget.csv',index=False)
    pd.set_option('display.width',200);print(s.round(3).to_string(index=False));print(pd.DataFrame(matched(s)).round(3).to_string(index=False));print(sb.round(3).to_string(index=False))
if __name__=='__main__':main()
