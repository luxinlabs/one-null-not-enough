"""Run the preregistered experiment grid (E2, E3, E4, Monte Carlo precision, threshold sensitivity) in parallel.

Each task writes one gzipped CSV under results/grid/<block>/; existing files are skipped, so reruns resume.
Seeds 0-99 are validation (sensitivity block only); seeds 100-299 are test.
"""
import argparse,csv,gzip,hashlib,itertools,json,os,platform,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import experiment as E

VALIDATION=range(0,100)
TEST=range(100,300)
JOINT=('joint_eig','joint_risk','joint_risk_voi','identifiability','identifiability_stop','robust_bayes')
# budget_limited is omitted from grid blocks: the budget sweep subsumes it.
MAIN_SCENARIOS=('resolvable','unidentifiable')
BASE=dict(sigma=.5,budget=5.,fault_rate=.5,n_mc=256,truth='specified',prior_fault=None,plan_rep=0,voi_tol=1e-3,support_eps=1e-3,credal_lo=E.CREDAL[0],credal_hi=E.CREDAL[1])

def tasks():
    out=[]
    def add(block,seeds,scenarios,policies,**kw):
        cfg={**BASE,**kw}
        out.append({'block':block,'seeds':list(seeds),'scenarios':list(scenarios),'policies':list(policies),'cfg':cfg})
    for sg,b,fr in itertools.product((.25,.5,1.),(2.,5.,10.),(.1,.3,.5)):
        add('e2',TEST,MAIN_SCENARIOS,E.POLICIES+('identifiability_stop',),sigma=sg,budget=b,fault_rate=fr)
    for n_mc,rep in itertools.product((64,128,256,512),(0,1,2)):
        add('nmc',TEST,MAIN_SCENARIOS,JOINT,n_mc=n_mc,plan_rep=rep)
    for voi,eps in itertools.product((1e-4,1e-3,1e-2),(1e-4,1e-3,1e-2)):
        add('sens',VALIDATION,MAIN_SCENARIOS,('joint_risk_voi','identifiability','identifiability_stop','robust_bayes'),voi_tol=voi,support_eps=eps)
    # Credal-set sensitivity on test seeds: matched and shifted priors, specified truth.
    for (lo,hi),pf,fr in itertools.product(((.01,.99),(.2,.8)),(None,.1),(.5,.1)):
        add('credal',TEST,MAIN_SCENARIOS,('joint_risk_voi','identifiability','robust_bayes'),credal_lo=lo,credal_hi=hi,prior_fault=pf,fault_rate=fr)
    # E3: truth fault rate 0.5; planning prior matched (0.5) or shifted to an overconfident-healthy prior (0.1).
    for truth,pf,sg,b in itertools.product(E.TRUTHS,(None,.1),(.5,1.),(5.,10.)):
        add('e3',TEST,MAIN_SCENARIOS,E.POLICIES+('identifiability_stop',),truth=truth,prior_fault=pf,sigma=sg,budget=b)
    for truth in E.TRUTHS:
        add('e4',TEST,MAIN_SCENARIOS+tuple(E.ABLATION_SCENARIOS),('scientific_only','joint_risk','joint_risk_voi','identifiability','robust_bayes')+E.ABLATION_POLICIES,truth=truth)
    for t in out:
        key=json.dumps(t['cfg'],sort_keys=True)
        t['path']=f"results/grid/{t['block']}/{hashlib.sha1(key.encode()).hexdigest()[:12]}.csv.gz"
    return out

def execute(task):
    path=Path(task['path'])
    if path.exists():return task['path'],0,0.
    t0=time.perf_counter();cfg=task['cfg'];rows=[]
    for seed in task['seeds']:
        for scenario in task['scenarios']:
            for policy in task['policies']:
                r=E.run(seed,policy,scenario,sigma=cfg['sigma'],budget=cfg['budget'],n_mc=cfg['n_mc'],fault_rate=cfg['fault_rate'],support_eps=cfg['support_eps'],voi_tol=cfg['voi_tol'],truth=cfg['truth'],prior_fault=cfg['prior_fault'],plan_rep=cfg['plan_rep'],credal=(cfg['credal_lo'],cfg['credal_hi']))
                r['history']=json.dumps(r['history'])
                rows.append({'block':task['block'],'split':'validation' if seed in VALIDATION else 'test',**{k:('' if v is None else v) for k,v in cfg.items()},**r})
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with gzip.open(tmp,'wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    os.replace(tmp,path)
    return task['path'],len(rows),time.perf_counter()-t0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--blocks',default='e2,nmc,sens,credal,e3,e4');ap.add_argument('--workers',type=int,default=os.cpu_count())
    ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
    todo=[t for t in tasks() if t['block'] in args.blocks.split(',')]
    print(f'{len(todo)} tasks, {sum(len(t["seeds"])*len(t["scenarios"])*len(t["policies"]) for t in todo)} runs')
    if args.dry_run:return
    meta={'command':' '.join([Path(sys.argv[0]).name]+sys.argv[1:]),'experiment_sha256':hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest(),
          'run_grid_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'started':time.strftime('%Y-%m-%d %H:%M:%S')}
    Path('results/grid').mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter();done=0;n=0
    with Pool(args.workers) as pool:
        for path,rows,secs in pool.imap_unordered(execute,todo):
            done+=1;n+=rows
            print(f'[{done}/{len(todo)}] {path} rows={rows} {secs:.0f}s',flush=True)
    meta.update(finished=time.strftime('%Y-%m-%d %H:%M:%S'),wall_seconds=round(time.perf_counter()-t0,1),new_rows=n)
    Path('results/grid/meta.json').write_text(json.dumps(meta,indent=1))
if __name__=='__main__':main()
