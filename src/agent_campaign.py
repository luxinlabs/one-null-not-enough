"""Multi-step agent campaign (review round 4): an LLM agent with tools decides which null-looking pairs to retest under a budget.

Each campaign has 24 test-split pairs that looked null on their first plate and have a replicate plate, from cell lines held out from the
retest model's training (src/retest_priority.py, disjoint by cell line); 6 are effective and 18 ineffective according to GDSC2. The agent
sees a table of lab-available information, may call request_replicate for up to 6 pairs (over as many calls as it likes), and then calls
submit_decisions. Conditions: 'plain' and 'score' (the table adds the held-out retest model's estimated probability of effectiveness).
Rule baselines on the same campaigns retest the top 6 by a score. Writes results/agent/<model>__<condition>.jsonl.
"""
import argparse,json,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold,cross_val_predict
from llm_agent import PRICE,Budget,load_key
from retest_priority import FEATURES,half

OUT=Path('results/agent');TAU,TAU_P=.3,.7;N_PAIRS,N_EFF,BUDGET=24,6,6
COST_FR,COST_FA,COST_ABSTAIN=10,10,2
FORCED=('voi_forced','forced_lr','forced_post')
SYSTEM=f"""You run the hit-confirmation stage of an autonomous drug-screening laboratory. Each of the {N_PAIRS} compound-cell line pairs in the table looked ineffective on its first viability plate (mean viability of at least 0.70 at the two highest doses; untreated controls = 1.0, cell-free blanks = 0.0). In campaigns like this one, about one in four such pairs is effective according to an independent screen.

You may retest up to {BUDGET} pairs on a replicate plate by calling request_replicate with one or more pair ids; you may call it several times, and each call returns the replicate readings. When you are done, call submit_decisions exactly once with a decision for every pair: REJECT (drop as ineffective), ACCEPT (effective), or ABSTAIN (flag for later follow-up).
Scoring: rejecting an effective pair costs {COST_FR} points, accepting an ineffective pair costs {COST_FA} points, abstaining costs {COST_ABSTAIN} points; retests within the budget are free. Minimise total points.

Columns: viability = first-plate mean viability at the two highest doses; Zprime = plate control quality (1 = perfect); max_conc_uM = highest concentration tested; compound_resp = share of other cell lines on which this compound reduced viability below 0.30 in this screen; cell_resp = share of other compounds to which this cell line responded that way; plate_null = share of the other compounds on the same plate that also look null."""
CAL_NOTE="\np_effective = calibrated probability that the pair is effective in this campaign, from a model trained on other cell lines of this screen and adjusted to this campaign's base rate."
VOI_NOTE=CAL_NOTE+"\nloss_now = expected points of the best decision for the pair without a retest; voi = expected reduction in points from retesting the pair (computed from p_effective and how replicate readings behave for effective and ineffective pairs)."
SCORE_NOTE="\nmodel_p = probability that the pair is effective, estimated by a model trained on other cell lines of this screen (it ranks pairs well but is not calibrated to this campaign)."
TOOLS=[{'type':'function','name':'request_replicate','description':'Retest pairs on a replicate plate. Returns each pair\'s replicate mean viability.',
        'parameters':{'type':'object','properties':{'pair_ids':{'type':'array','items':{'type':'string'}}},'required':['pair_ids'],'additionalProperties':False},'strict':True},
       {'type':'function','name':'submit_decisions','description':'Submit the final decision for every pair.',
        'parameters':{'type':'object','properties':{'decisions':{'type':'array','items':{'type':'object','properties':{'pair_id':{'type':'string'},'decision':{'type':'string','enum':['REJECT','ACCEPT','ABSTAIN']}},
        'required':['pair_id','decision'],'additionalProperties':False}}},'required':['decisions'],'additionalProperties':False},'strict':True}]

def campaigns(n,seed=0):
    tr=pd.read_csv('results/gdsc/null_features_explore.csv.gz',dtype={'BARCODE':str});te=pd.read_csv('results/gdsc/null_features_test.csv.gz',dtype={'BARCODE':str})
    tr=tr[half(tr.COSMIC_ID)];te=te[~half(te.COSMIC_ID)&te.alt_effect.notna()].reset_index(drop=True)
    gbm=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.,random_state=0).fit(tr[FEATURES].to_numpy(float),tr.H)
    te['model_p']=gbm.predict_proba(te[FEATURES].to_numpy(float))[:,1]
    # For the VOI baseline: isotonic calibration from cell-line-grouped cross-validated training predictions, and replicate likelihoods.
    cv=cross_val_predict(HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.,random_state=0),
                         tr[FEATURES].to_numpy(float),tr.H,groups=tr.COSMIC_ID,cv=GroupKFold(5),method='predict_proba')[:,1]
    te['cal_p']=IsotonicRegression(out_of_bounds='clip',y_min=1e-3,y_max=1-1e-3).fit(cv,tr.H).predict(te.model_p)
    r=tr[tr.alt_effect.notna()];b=np.digitize(r.alt_effect,[TAU,TAU_P])
    o=te.cal_p/(1-te.cal_p)*(N_EFF/N_PAIRS/(1-N_EFF/N_PAIRS))/(.091/.909);te['camp_p']=o/(1+o)
    global LIK;LIK={h:np.array([((b==k)&(r.H==h)).sum()+1 for k in range(3)])/((r.H==h).sum()+3) for h in (0,1)}
    rng=np.random.default_rng(seed);pos=list(rng.permutation(np.flatnonzero(te.H==1)));neg=list(rng.permutation(np.flatnonzero(te.H==0)));out=[]
    codes=rng.permutation(9000)+1000
    for c in range(n):
        idx=[pos.pop() for _ in range(N_EFF)]+[neg.pop() for _ in range(N_PAIRS-N_EFF)];d=te.iloc[rng.permutation(idx)].copy().reset_index(drop=True)
        d['pair_id']=[f'P{i+1:02d}' for i in range(N_PAIRS)]
        cmap={x:f'C-{codes[(c*50+i)%9000]}' for i,x in enumerate(d.DRUG_ID.unique())};lmap={x:f'CL-{codes[(c*50+25+i)%9000]}' for i,x in enumerate(d.COSMIC_ID.unique())}
        d['compound']=d.DRUG_ID.map(cmap);d['cell_line']=d.COSMIC_ID.map(lmap);d['campaign']=c;out.append(d)
    return out

def voi_columns(d):
    q=d.camp_p.to_numpy();L=lambda x:np.minimum.reduce([COST_FR*x,COST_FA*(1-x),np.full_like(x,COST_ABSTAIN)])
    post=lambda x,k:x*LIK[1][k]/(x*LIK[1][k]+(1-x)*LIK[0][k])
    return L(q),L(q)-sum((q*LIK[1][k]+(1-q)*LIK[0][k])*L(post(q,k)) for k in range(3))

def table(d,score,calibrated=False,voi=False):
    cols=['pair_id','compound','cell_line','viability','Zprime','max_conc_uM','compound_resp','cell_resp','plate_null']+(['p_effective','loss_now','voi'] if voi else (['p_effective'] if calibrated else (['model_p'] if score else [])))
    ln,vi=voi_columns(d)
    t=pd.DataFrame({'pair_id':d.pair_id,'compound':d.compound,'cell_line':d.cell_line,'viability':(1-d.effect).round(2),'Zprime':d.zprime.round(2),
                    'max_conc_uM':d.max_conc.map(lambda x:f'{x:.3g}' if pd.notna(x) else 'NA'),'compound_resp':d.drug_resp.map(lambda x:f'{x:.2f}' if pd.notna(x) else 'NA'),
                    'cell_resp':d.cell_resp.map(lambda x:f'{x:.2f}' if pd.notna(x) else 'NA'),'plate_null':d.plate_null_share.map(lambda x:f'{x:.2f}' if pd.notna(x) else 'NA'),
                    'model_p':d.model_p.round(3),'p_effective':d.camp_p.round(3),'loss_now':np.round(ln,2),'voi':np.round(np.maximum(vi,0),2)})
    return t[cols].to_csv(index=False)

def evaluate(d,retested,decisions):
    h=d.H.to_numpy();dec=np.array([decisions.get(p,'ABSTAIN') for p in d.pair_id]);r=d.pair_id.isin(retested).to_numpy();e2=d.alt_effect.to_numpy()
    pts=np.where(dec=='REJECT',COST_FR*h,np.where(dec=='ACCEPT',COST_FA*(1-h),COST_ABSTAIN)).sum()
    return {'retested':int(r.sum()),'effective_retested':int((r&(h==1)).sum()),'effective_recovered':int((r&(h==1)&(e2>=TAU)).sum()),
            'false_rejections':int(((dec=='REJECT')&(h==1)).sum()),'correct_rejections':int(((dec=='REJECT')&(h==0)).sum()),
            'false_acceptances':int(((dec=='ACCEPT')&(h==0)).sum()),'abstentions':int((dec=='ABSTAIN').sum()),'points':float(pts)}

def rule(d,score):
    """Retest the top BUDGET by score; reject if confirmed or not retested, accept if the replicate shows an effect, abstain otherwise."""
    top=set(d.pair_id.iloc[np.argsort(-score,kind='stable')[:BUDGET]]);dec={}
    for p,e2 in zip(d.pair_id,d.alt_effect):dec[p]='REJECT' if p not in top or e2<TAU else ('ACCEPT' if e2>TAU_P else 'ABSTAIN')
    return evaluate(d,top,dec)

def bayes_rule(d,p_train=.091,p_camp=N_EFF/N_PAIRS):
    """Retest the top BUDGET by model score; decide every pair by expected points, with the model probability shifted to the campaign base rate."""
    o=d.model_p/(1-d.model_p)*(p_camp/(1-p_camp))/(p_train/(1-p_train));q=(o/(1+o)).to_numpy()
    top=set(d.pair_id.iloc[np.argsort(-q,kind='stable')[:BUDGET]]);dec={}
    for p,qi,e2 in zip(d.pair_id,q,d.alt_effect):
        if p in top:dec[p]='REJECT' if e2<TAU else ('ACCEPT' if e2>TAU_P else 'ABSTAIN')
        else:dec[p]=['REJECT','ACCEPT','ABSTAIN'][int(np.argmin([COST_FR*qi,COST_FA*(1-qi),COST_ABSTAIN]))]
    return evaluate(d,top,dec)

def voi_rule(d,p_train=.091,p_camp=N_EFF/N_PAIRS):
    """Calibrated probability shifted to the campaign base rate; retest the BUDGET pairs with the largest value of information under the
    campaign loss (optimal here because pairs are independent and the budget is fixed); decide every pair by posterior expected loss."""
    o=d.cal_p/(1-d.cal_p)*(p_camp/(1-p_camp))/(p_train/(1-p_train));q=(o/(1+o)).to_numpy()
    L=lambda x:np.minimum.reduce([COST_FR*x,COST_FA*(1-x),np.full_like(x,COST_ABSTAIN)])
    post=lambda x,k:x*LIK[1][k]/(x*LIK[1][k]+(1-x)*LIK[0][k])
    after=sum((q*LIK[1][k]+(1-q)*LIK[0][k])*L(post(q,k)) for k in range(3));voi=L(q)-after
    top=set(d.pair_id.iloc[np.argsort(-voi,kind='stable')[:BUDGET]]);dec={}
    for p,qi,e2 in zip(d.pair_id,q,d.alt_effect):
        x=post(qi,int(np.digitize(e2,[TAU,TAU_P]))) if p in top else qi
        dec[p]=['REJECT','ACCEPT','ABSTAIN'][int(np.argmin([COST_FR*x,COST_FA*(1-x),COST_ABSTAIN]))]
    return evaluate(d,top,dec)

def run_campaign(client,model,cond,d,budget,effort,max_turns=10):
    sys=SYSTEM+({'score':SCORE_NOTE,'calibrated':CAL_NOTE,'voi':VOI_NOTE}.get(cond) or (VOI_NOTE if cond in FORCED else ''));user='Campaign table (CSV):\n'+table(d,cond=='score',cond=='calibrated',cond=='voi' or cond in FORCED)
    retested=[]
    if cond in FORCED:
        # 2x2 decomposition cell: the model-optimal policy's 6 retests are performed for the agent, which only makes the final decisions.
        # Round 9 (not yet run): forced_lr adds the fitted likelihood ratio of each replicate reading; forced_post adds the posterior and the
        # expected points of each action, so that belief updating (voi_forced), use of evidence strength (forced_lr), and the terminal
        # decision given the posterior (forced_post) are tested separately on the same retests.
        ln,vi=voi_columns(d);retested=list(d.pair_id.iloc[np.argsort(-vi,kind='stable')[:BUDGET]]);lines=[]
        for p in retested:
            i=int(np.flatnonzero(d.pair_id==p)[0]);e2=d.alt_effect[i];k=int(np.digitize(e2,[TAU,TAU_P]));lr=LIK[1][k]/LIK[0][k];q=d.camp_p[i];x=q*lr/(q*lr+1-q)
            line=f'{p}: replicate plate mean viability = {1-e2:.2f}'
            if cond=='forced_lr':line+=f'; likelihood ratio of this reading (P(reading | effective) / P(reading | ineffective)) = {lr:.2f}'
            if cond=='forced_post':line+=f'; p_effective after the replicate = {x:.3f}; expected points: REJECT {COST_FR*x:.2f}, ACCEPT {COST_FA*(1-x):.2f}, ABSTAIN {COST_ABSTAIN:.2f}'
            lines.append(line)
        user+='\nThe retest budget has already been used on these pairs:\n'+'\n'.join(lines)+'\nNo further retests are possible. Call submit_decisions.'
    resp=None;inp=[{'role':'system','content':sys},{'role':'user','content':user}];log=[];usd=0.;decisions=None
    for turn in range(max_turns):
        if not budget.ok():return None
        resp=client.responses.create(model=model,input=inp,tools=TOOLS,reasoning={'effort':effort},max_output_tokens=8000,store=True,
                                     **({'previous_response_id':resp.id} if resp is not None else {}))
        usd+=budget.add(model,resp.usage);calls=[o for o in resp.output if o.type=='function_call'];inp=[]
        log.append({'served_model':resp.model,'in':resp.usage.input_tokens,'out':resp.usage.output_tokens,'calls':[{'name':c.name,'arguments':c.arguments} for c in calls],'text':resp.output_text[:500]})
        if not calls:inp=[{'role':'user','content':'Please continue: use request_replicate or submit_decisions.'}];continue
        for c in calls:
            a=json.loads(c.arguments)
            if c.name=='request_replicate':
                lines=[]
                for p in a['pair_ids']:
                    row=d[d.pair_id==p]
                    if row.empty:lines.append(f'{p}: unknown pair id')
                    elif p in retested:lines.append(f'{p}: already retested, replicate viability = {1-row.alt_effect.iloc[0]:.2f}')
                    elif len(retested)>=BUDGET:lines.append(f'{p}: not retested, budget of {BUDGET} retests exhausted')
                    else:retested.append(p);lines.append(f'{p}: replicate plate mean viability = {1-row.alt_effect.iloc[0]:.2f}')
                out='\n'.join(lines)+f'\nRetests used: {len(retested)} of {BUDGET}.'
            else:
                decisions={x['pair_id']:x['decision'] for x in a['decisions']};out='Decisions recorded.'
            inp.append({'type':'function_call_output','call_id':c.call_id,'output':out})
        if decisions is not None:break
    res=evaluate(d,retested,decisions or {})
    return {'campaign':int(d.campaign.iloc[0]),'model':model,'condition':cond,'effort':effort,'submitted':decisions is not None,
            'missing_decisions':N_PAIRS-len(set(decisions or {})&set(d.pair_id)),'retested_ids':retested,'usd':usd,'turns':log,**res}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=list(PRICE),required=True);ap.add_argument('--condition',choices=('plain','score','calibrated','voi')+FORCED,default='plain')
    ap.add_argument('--n',type=int,default=25);ap.add_argument('--cap',type=float,default=17.5);ap.add_argument('--effort',default='low');ap.add_argument('--workers',type=int,default=5)
    ap.add_argument('--rules-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);cs=campaigns(args.n)
    if args.rules_only:
        rows=[]
        for d in cs:
            rng=np.random.default_rng(int(d.campaign.iloc[0]))
            for name,s in {'random':rng.random(len(d)),'lowest_zprime':-d.zprime.to_numpy(),'weakest_null':d.effect.to_numpy(),
                           'compound_history':d.drug_resp.fillna(0).to_numpy(),'retest_model':d.model_p.to_numpy(),'oracle':d.H.to_numpy()+1e-3*d.alt_effect.to_numpy()}.items():
                rows.append({'campaign':int(d.campaign.iloc[0]),'policy':name,**rule(d,s)})
            rows.append({'campaign':int(d.campaign.iloc[0]),'policy':'retest_model_bayes',**bayes_rule(d)})
            rows.append({'campaign':int(d.campaign.iloc[0]),'policy':'voi_optimal',**voi_rule(d)})
        r=pd.DataFrame(rows);r.to_csv(OUT/'rules.csv',index=False);print(r.groupby('policy',sort=False).mean(numeric_only=True).drop(columns='campaign').round(2));return
    log=OUT/(f'{args.model}__{args.condition}'+('' if args.effort=='low' else f'__{args.effort}')+'.jsonl');done={json.loads(l)['campaign'] for l in log.open()} if log.exists() else set()
    todo=[d for d in cs if int(d.campaign.iloc[0]) not in done];budget=Budget(args.cap);client=OpenAI(api_key=load_key());lock=threading.Lock()
    print(f'{len(todo)} campaigns to run; spent so far ${budget.spent:.4f} of ${args.cap}')
    def work(d):
        try:res=run_campaign(client,args.model,args.condition,d,budget,args.effort)
        except Exception as ex:print('error',int(d.campaign.iloc[0]),type(ex).__name__,str(ex)[:200]);return
        if res is None:return
        with lock,log.open('a') as f:f.write(json.dumps(res)+'\n')
    with ThreadPoolExecutor(args.workers) as pool:list(pool.map(work,todo))
    print(f'spent ${budget.spent:.4f} of ${args.cap}')
if __name__=='__main__':main()
