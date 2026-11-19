"""Per-(scenario, policy) metrics with 95% bootstrap CIs and paired differences against a reference policy."""
import argparse,csv,collections,warnings
import numpy as np

def metrics(rows):
    a=lambda k:np.array([float(r[k]) for r in rows])
    correct,abstain,spent,h,fault,pred=a('correct'),a('abstain'),a('spent'),a('true_h'),a('fault'),a('predicted_h')
    decided=abstain==0
    nan=lambda m:float('nan')
    return {'coverage':1-abstain.mean(),
     'accuracy':correct.mean(), # abstentions count as errors
     'selective_accuracy':correct[decided].mean() if decided.any() else nan(0),
     'false_rejection':(pred[h==1]==0).mean() if (h==1).any() else nan(0), # denominator: true H=1
     'false_rejection_faulty':(pred[(h==1)&(fault==1)]==0).mean() if ((h==1)&(fault==1)).any() else nan(0),
     'mean_cost':spent.mean()}

def bootstrap(rows,stat,rng,n_boot):
    vals=[stat([rows[i] for i in rng.integers(0,len(rows),len(rows))]) for _ in range(n_boot)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning) # all-NaN when no resample has a decided/H=1 row
        return np.nanpercentile(vals,[2.5,97.5])

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/pilot.csv');p.add_argument('--reference',default='joint_risk')
    p.add_argument('--n-boot',type=int,default=1000);p.add_argument('--seed',type=int,default=0);args=p.parse_args()
    rng=np.random.default_rng(args.seed)
    with open(args.input,newline='') as f:rows=list(csv.DictReader(f))
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['scenario'],r['policy'])].append(r)
    names=list(metrics(next(iter(groups.values()))))
    print('scenario,policy,n,'+','.join(f'{m},{m}_lo,{m}_hi' for m in names))
    for (scenario,policy),g in sorted(groups.items()):
        point=metrics(g);out=[scenario,policy,str(len(g))]
        for m in names:
            lo,hi=bootstrap(g,lambda s:metrics(s)[m],rng,args.n_boot)
            out+=[f'{point[m]:.3f}',f'{lo:.3f}',f'{hi:.3f}']
        print(','.join(out))
    print(f'\npaired difference vs {args.reference} (policy minus reference, matched seeds)')
    print('scenario,policy,n_pairs,metric,mean_diff,lo,hi')
    for (scenario,policy),g in sorted(groups.items()):
        ref={r['seed']:r for r in groups.get((scenario,args.reference),[])}
        if policy==args.reference or not ref:continue
        pairs=[(r,ref[r['seed']]) for r in g if r['seed'] in ref]
        for m in ('correct','abstain','spent'):
            d=np.array([float(x[m])-float(y[m]) for x,y in pairs])
            boot=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(args.n_boot)]
            lo,hi=np.percentile(boot,[2.5,97.5])
            print(f'{scenario},{policy},{len(d)},{m},{d.mean():.3f},{lo:.3f},{hi:.3f}')
if __name__=='__main__':main()
