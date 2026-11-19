"""Finite-state scientific/instrument ambiguity benchmark. No real laboratory data."""
import argparse,csv,hashlib,json,platform,sys
from pathlib import Path
import numpy as np

ACTIONS=('ordinary','replicate','calibrate','positive_control','alternate_sensor')
COST={'ordinary':1.,'replicate':1.,'calibrate':0.6,'positive_control':1.4,'alternate_sensor':2.5}
# Worlds: (scientific effect h, faulty instrument f). Worlds 0 and 1 have identical ordinary readings.
WORLDS=((0,0),(1,1),(1,0),(0,1))
H1=np.array([h for h,_ in WORLDS],dtype=bool)
FAULTY=np.array([f for _,f in WORLDS],dtype=bool)
POLICIES=('scientific_only','replicate','calibrate_first','joint_eig','joint_risk','joint_risk_voi','identifiability','robust_bayes')
# identifiability_stop: stopping rule without the rejection guard; identifiability_eig: entropy objective.
ABLATION_POLICIES=('identifiability_stop','identifiability_eig','oracle_fault')
IDENTIFIABILITY=('identifiability','identifiability_stop','identifiability_eig','identifiability_budget')
GUARDED=('identifiability','identifiability_eig')
# Budget-aware variant (review round 4): also abstains when a consequential pair is still supported but the budget ran out before it was separated.
BUDGET_GUARDED=('identifiability_budget',)
# robust_bayes: credal set of fault priors [lo, hi]; primary set fixed before its first run.
CREDAL=(0.05,0.95)
SCENARIOS=('resolvable','unidentifiable','budget_limited')
# E4 control restrictions: full action set minus one diagnostic.
ABLATION_SCENARIOS={'no_calibrate':'calibrate','no_positive_control':'positive_control','no_alternate':'alternate_sensor'}
# Data-generating processes. The planning model always assumes 'specified'.
TRUTHS=('specified','drift','intermittent','correlated_noise','imperfect_control','unknown_fault')
# Outside the main grid (review round 3): a continuous fault whose attenuation b ~ U(0.3, 1) is drawn per episode.
EXTRA_TRUTHS=('partial_fault',)
MAIN_SENSOR=('ordinary','replicate')
DRIFT=0.1 # main-sensor offset added per measurement step
INTERMITTENT=0.5 # probability a faulty instrument misbehaves on a given measurement
RHO=0.8 # AR(1) coefficient of correlated noise
CONTROL_FALSE_PASS=0.3 # imperfect positive control passes a faulty instrument with this probability
CALIBRATION_GAIN=0.5 # imperfect calibration reads this fraction of the fault signal
SENSITIVITY=0.4 # unknown fault: main-sensor gain, undetected by calibration
DECIDE=0.9 # accept H if P(H)>DECIDE, reject if P(H)<1-DECIDE, otherwise abstain

def mean(world,action):
    h,f=WORLDS[world]
    if action in MAIN_SENSOR: return float(h*(1-f))
    if action=='calibrate': return float(f)
    if action=='positive_control': return float(1-f)
    if action=='alternate_sensor': return float(h)
    raise ValueError(action)

MU={a:np.array([mean(w,a) for w in range(4)]) for a in ACTIONS}

def truth_mean(h,f,action,step,truth,u,b=1.):
    """Mean of the next reading under the data-generating process; u is a uniform draw for stochastic faults, b the partial-fault attenuation."""
    if truth=='partial_fault' and f:
        return {'ordinary':h*(1-b),'replicate':h*(1-b),'calibrate':b,'positive_control':1-b,'alternate_sensor':float(h)}[action]
    if truth=='unknown_fault' and f:
        # A fault type outside the planning model: attenuated main sensor that calibration does not detect.
        return {'ordinary':SENSITIVITY*h,'replicate':SENSITIVITY*h,'calibrate':0.,'positive_control':SENSITIVITY,'alternate_sensor':float(h)}[action]
    if truth=='intermittent' and f and u>=INTERMITTENT:f=0
    if truth=='imperfect_control' and f:
        if action=='calibrate':return CALIBRATION_GAIN
        if action=='positive_control':return 1. if u<CONTROL_FALSE_PASS else 0.
    m=mean(WORLDS.index((h,f)),action)
    if truth=='drift' and action in MAIN_SENSOR:m+=DRIFT*step
    return m

def posterior_update(p,action,y,sigma):
    logw=np.log(np.maximum(p,1e-300))-0.5*((y-MU[action])/sigma)**2
    z=np.exp(logw-logw.max());return z/z.sum()

def naive_update(p,y,sigma):
    """Fault-unaware model over h only: an ordinary reading is N(h, sigma), instrument assumed healthy."""
    logw=np.log(np.maximum(p,1e-300))-0.5*((y-np.array([0.,1.]))/sigma)**2
    z=np.exp(logw-logw.max());return z/z.sum()

def entropy(p):
    """Entropy of the last axis; works on a single posterior or a batch."""
    return -(p*np.log(np.where(p>0,p,1.))).sum(-1)

def risk(p):
    """Bayes 0-1 risk of the decision on H; works on a single posterior or a batch."""
    return np.minimum(p[...,~H1].sum(-1),p[...,H1].sum(-1))

def expected_gain(p,action,sigma,rng,n_mc=64,loss='entropy'):
    """Monte Carlo expected reduction in loss from one more observation of action (vectorized over draws)."""
    f=entropy if loss=='entropy' else risk
    mu=MU[action];w=rng.choice(4,size=n_mc,p=p);y=rng.normal(mu[w],sigma)
    logw=np.log(np.maximum(p,1e-300))-0.5*((y[:,None]-mu)/sigma)**2
    q=np.exp(logw-logw.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
    return float(f(p)-f(q).mean())

def consequential_pairs(p,support_eps):
    """Pairs of posterior-supported worlds that disagree on H."""
    live=[w for w in range(4) if p[w]>support_eps]
    return [(w,v) for w in live for v in live if not H1[w] and H1[v]]

def separating_actions(p,actions,support_eps,eq_tol):
    """Actions that separate at least one consequential pair. For Gaussian observations with equal variance, differing means imply distinguishability."""
    pairs=consequential_pairs(p,support_eps)
    return [a for a in actions if any(abs(mean(w,a)-mean(v,a))>eq_tol for w,v in pairs)]

def unseparable_pair(p,actions,support_eps,eq_tol):
    """True if some posterior-supported pair disagreeing on H is observationally equivalent under every action in the class."""
    return any(all(abs(mean(w,a)-mean(v,a))<=eq_tol for a in actions) for w,v in consequential_pairs(p,support_eps))

def first_available(prefs,actions):
    return next((a for a in prefs if a in actions),None)

def choose(policy,p,sigma,rng,actions,eq_tol=1e-9,n_mc=64,support_eps=1e-3,voi_tol=1e-3):
    """Return the next action, or None to stop."""
    if policy=='scientific_only':return first_available(['ordinary'],actions)
    if policy=='replicate':return first_available(['replicate','ordinary'],actions)
    if policy=='calibrate_first':
        # Calibrate until the fault state is resolved, then read H with the instrument judged reliable.
        p_fault=float(p[FAULTY].sum())
        if 1-DECIDE<p_fault<DECIDE:prefs=['calibrate','positive_control']
        elif p_fault<=1-DECIDE:prefs=['ordinary','alternate_sensor']
        else:prefs=['alternate_sensor']
        return first_available(prefs+['ordinary'],actions)
    objective='entropy' if policy in ('joint_eig','identifiability_eig') else 'risk'
    if policy in IDENTIFIABILITY:
        # Stop when no allowed action separates any posterior-supported pair that disagrees on H.
        actions=separating_actions(p,actions,support_eps,eq_tol)
        if not actions:return None
    gains={a:expected_gain(p,a,sigma,rng,n_mc,objective) for a in actions}
    if policy in ('joint_risk_voi','oracle_fault','robust_bayes') and max(gains.values())<=voi_tol:return None
    return max(actions,key=lambda a:gains[a]/COST[a])

def prior(fault_rate):
    return np.array([.5*(1-fault_rate) if not f else .5*fault_rate for _,f in WORLDS])

def credal_bounds(p,nominal_prior,credal):
    """Range of P(H|D) over fault priors in [lo, hi]. P(H|D) is a ratio of functions linear in the prior, so it is monotone and the endpoints suffice."""
    lik=p/np.maximum(nominal_prior,1e-300)
    vals=[]
    for q in credal:
        post=prior(q)*lik;vals.append(float(post[H1].sum()/post.sum()))
    return min(vals),max(vals)

def run(seed,policy,scenario='resolvable',sigma=0.5,budget=5.,n_mc=64,fault_rate=0.5,tight_budget=2.,eq_tol=1e-9,support_eps=1e-3,voi_tol=1e-3,truth='specified',prior_fault=None,plan_rep=0,credal=CREDAL):
    """One episode. prior_fault is the planning model's fault prior (defaults to the true fault_rate)."""
    # Independent streams so that every policy and scenario sees the same world, noise, and fault draws for a seed.
    world_rng,noise_rng,plan_rng,fault_rng=(np.random.default_rng([seed,k,plan_rep] if k==2 else [seed,k]) for k in range(4))
    h=int(world_rng.random()<0.5);f=int(world_rng.random()<fault_rate);b=.3+.7*world_rng.random()
    # Common random numbers: the k-th use of an action gets the same draws under every policy.
    max_uses=int(budget/min(COST.values()))+1
    eps={a:noise_rng.standard_normal(max_uses) for a in ACTIONS}
    unif={a:fault_rng.random(max_uses) for a in ACTIONS}
    actions=list(ACTIONS)
    if scenario=='unidentifiable':actions=['ordinary','replicate']
    elif scenario=='budget_limited':budget=min(budget,tight_budget)
    elif scenario in ABLATION_SCENARIOS:actions.remove(ABLATION_SCENARIOS[scenario])
    elif scenario!='resolvable':raise ValueError(scenario)
    if truth not in TRUTHS+EXTRA_TRUTHS:raise ValueError(truth)
    naive=policy=='scientific_only'
    p=np.array([.5,.5]) if naive else prior(fault_rate if prior_fault is None else prior_fault)
    p0=p.copy()
    if policy=='oracle_fault':p=np.where(FAULTY==bool(f),p,0.);p/=p.sum() # privileged upper bound: knows whether the instrument is faulty
    spent=0.;history=[];used=dict.fromkeys(ACTIONS,0);z=0.
    while True:
        feasible=[a for a in actions if spent+COST[a]<=budget+1e-9]
        if not feasible:break
        a=choose(policy,p,sigma,plan_rng,feasible,eq_tol,n_mc,support_eps,voi_tol)
        if a is None:break
        k=used[a];used[a]+=1
        z=RHO*z+np.sqrt(1-RHO**2)*eps[a][k] if truth=='correlated_noise' else eps[a][k]
        y=truth_mean(h,f,a,len(history),truth,unif[a][k],b)+sigma*float(z)
        p=naive_update(p,y,sigma) if naive else posterior_update(p,a,y,sigma);spent+=COST[a]
        history.append({'action':a,'measurement':y,'spent':spent})
    prob_true=float(p[1]) if naive else float(p[H1].sum())
    prediction=1 if prob_true>DECIDE else (0 if prob_true<1-DECIDE else -1)
    # Proposal step 3: never reject H while an explanation with H true is observationally equivalent, under the
    # scenario's action class, to one with H false. Acceptance is left to the posterior threshold.
    if policy in GUARDED and prediction==0 and unseparable_pair(p,actions,support_eps,eq_tol):prediction=-1
    if policy in BUDGET_GUARDED and prediction==0 and consequential_pairs(p,support_eps):prediction=-1
    if policy=='robust_bayes':
        # Plans like joint_risk_voi under the nominal prior; decides only if every prior in the credal set agrees.
        lo,hi=credal_bounds(p,p0,credal)
        prediction=1 if lo>DECIDE else (0 if hi<1-DECIDE else -1)
    return {'seed':seed,'policy':policy,'scenario':scenario,'world':WORLDS.index((h,f)),'true_h':h,'fault':f,'predicted_h':prediction,'correct':int(prediction==h),'false_rejection':int(h==1 and prediction==0),'abstain':int(prediction==-1),'spent':spent,'posterior_h1':prob_true,'history':history}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=30);ap.add_argument('--budget',type=float,default=5.);ap.add_argument('--sigma',type=float,default=.5)
    ap.add_argument('--n-mc',type=int,default=64);ap.add_argument('--fault-rate',type=float,default=.5);ap.add_argument('--tight-budget',type=float,default=2.)
    ap.add_argument('--eq-tol',type=float,default=1e-9);ap.add_argument('--support-eps',type=float,default=1e-3);ap.add_argument('--voi-tol',type=float,default=1e-3)
    ap.add_argument('--truth',choices=TRUTHS,default='specified');ap.add_argument('--prior-fault',type=float,default=None)
    ap.add_argument('--out',default='results/pilot.csv');args=ap.parse_args()
    rows=[]
    for scenario in SCENARIOS:
        for seed in range(args.seeds):
            for policy in POLICIES:
                r=run(seed,policy,scenario,args.sigma,args.budget,args.n_mc,args.fault_rate,args.tight_budget,args.eq_tol,args.support_eps,args.voi_tol,args.truth,args.prior_fault)
                r['history']=json.dumps(r['history']);rows.append(r)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    meta={'command':' '.join([Path(sys.argv[0]).name]+sys.argv[1:]),'args':vars(args),'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__}
    path.with_suffix('.meta.json').write_text(json.dumps(meta,indent=1))
    print(f'Wrote {len(rows)} rows to {path}')
if __name__=='__main__':main()
