"""LLM-agent evaluation on real GDSC test episodes (Part 2). Hard budget cap on OpenAI spend.

Episodes: test-split units that a single plate would reject (effect < tau), stratified by H and by independent-plate
availability. The agent sees one plate, may buy an independent plate, and then accepts, rejects, or abstains.
Writes results/llm/episodes.csv, results/llm/<model>__<prompt>.jsonl, results/llm/spend.json.
"""
import argparse,json,re,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from openai import OpenAI

# USD per 1M tokens, standard tier, from https://developers.openai.com/api/docs/pricing (read 2026-11-02).
PRICE={'gpt-5.5':(5.00,30.00),'gpt-5.4-mini':(0.75,4.50),'gpt-5.4-mini-2026-03-17':(0.75,4.50)} # dated snapshot priced as its alias
OUT=Path('results/llm')
COST_PLATE,COST_INDEPENDENT=1.0,2.5
SYSTEM_BASE="""You are the decision module of an autonomous drug-screening laboratory. For one compound and one cancer cell line you must decide whether the compound is EFFECTIVE (it substantially reduces cell viability) or INEFFECTIVE.

Evidence comes from 384/1536-well viability plates read by a fluorescence plate reader. Viability is normalised so that untreated control wells = 1.0 and cell-free blank wells = 0.0. Z' is the plate's control-quality score (1 = perfect separation of untreated and blank controls; values above ~0.5 are conventionally considered good).

Each turn, reply with JSON only: {"action": one of "REQUEST_INDEPENDENT_PLATE", "ACCEPT_EFFECTIVE", "REJECT_INEFFECTIVE", "ABSTAIN", "rationale": "<= 30 words"}.
- REQUEST_INDEPENDENT_PLATE: retrieve the same compound and cell line measured on a different plate (experimental cost 2.5 units; the first plate cost 1 unit). It may be unavailable.
- ACCEPT_EFFECTIVE / REJECT_INEFFECTIVE: final decision.
- ABSTAIN: no decision; the pair is flagged for later follow-up.
A rejected compound is dropped from further study for this cell line."""
FAULT_HINT="""

Note: plate-level technical failures (e.g. dispensing errors, edge effects, cell-health problems) can make an effective compound look ineffective, and such failures are not always visible in the control wells."""
# Ablations (reviewer W3): neutral_z drops the conventional-threshold hint for Z'; loss adds an explicit scoring rule.
SYSTEM_NEUTRAL_Z=SYSTEM_BASE.replace(" (1 = perfect separation of untreated and blank controls; values above ~0.5 are conventionally considered good)","")
LOSS="""

Scoring: rejecting an effective compound costs 10 points, accepting an ineffective compound costs 10 points, abstaining costs 1 point, and each experimental cost unit costs 0.1 points. Minimise expected points."""
LOSS50=LOSS.replace('rejecting an effective compound costs 10 points','rejecting an effective compound costs 50 points') # reviewer M3: a loss under which confirming is clearly optimal
# Review round 3: 'context' adds per-episode dose and screening history to the first message; 'structured' adds, on top of that,
# decision support estimated on exploration plates (results/gdsc/review3.json, R6), explicit explanations, and the 50:1 loss.
# Split for the round-4 ablation ladder; SUPPORT is unchanged (BASE_RATES+EXPLAIN).
BASE_RATES="""

Decision support, estimated on separate calibration plates of this screen: among compound-cell line pairs whose first plate shows mean viability of at least 0.70 at the two highest doses, 9.1% are effective according to an independent screen. An independent plate is available for about 18% of such pairs. When it is available, it also shows viability of at least 0.70 in 87% of cases, and then 5.5% of pairs are effective; otherwise 32.5% are effective."""
EXPLAIN="""
A plate that looks null can mean: (a) the compound is ineffective on this cell line; (b) the tested dose range is too low; (c) a plate- or batch-level technical failure; (d) measurement noise. Before deciding, compute the expected points of each available action."""
SUPPORT=BASE_RATES+EXPLAIN
PROMPTS={'base':SYSTEM_BASE,'fault_aware':SYSTEM_BASE+FAULT_HINT,'neutral_z':SYSTEM_NEUTRAL_Z,'loss':SYSTEM_NEUTRAL_Z+LOSS,'loss50':SYSTEM_NEUTRAL_Z+LOSS50,
         'context':SYSTEM_BASE,'structured':SYSTEM_NEUTRAL_Z+LOSS50+SUPPORT,
         'baserate':SYSTEM_NEUTRAL_Z+LOSS50+BASE_RATES,'baserate_explain':SYSTEM_NEUTRAL_Z+LOSS50+BASE_RATES+EXPLAIN} # ladder: loss50 -> +base rates -> +explanations -> +context (structured)
CONTEXT_PROMPTS=('context','structured')
ACTIONS=('REQUEST_INDEPENDENT_PLATE','ACCEPT_EFFECTIVE','REJECT_INEFFECTIVE','ABSTAIN')

def load_key(path='.env'):
    for line in open(path):
        m=re.match(r'\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)\s*$',line)
        if m:return m.group(1).strip().strip('"').strip("'")
    raise SystemExit('OPENAI_API_KEY not found in .env')

def build_episodes(per_stratum,seed=0):
    d=pd.read_csv('results/gdsc/units_test.csv.gz',dtype={'BARCODE':str})
    tau=json.loads(Path('results/gdsc/thresholds.json').read_text())['tau']
    d=d[d.effect<tau].copy();d['alt']=d.alt_effect.notna()
    parts=[g.sample(min(per_stratum,len(g)),random_state=seed) for _,g in d.groupby(['H','alt'])]
    e=pd.concat(parts).reset_index(drop=True)
    e['episode']=[f'ep{i:04d}' for i in range(len(e))]
    # Anonymised identifiers so the model cannot recall published sensitivities.
    rng=np.random.default_rng(seed)
    e['compound']=[f'C-{x:04d}' for x in rng.permutation(10000)[:len(e)]];e['cell_line']=[f'CL-{x:04d}' for x in rng.permutation(10000)[:len(e)]]
    # Population share of each stratum among would-be rejections, for reweighting.
    share=d.groupby(['H','alt']).size()/len(d)
    e['weight']=[share[(h,a)]/(e[(e.H==h)&(e.alt==a)].shape[0]/len(e)) for h,a in zip(e.H,e.alt)]
    return e

def episode_context(e):
    """Dose and screening history for each episode, from the GDSC1 fitted file and exploration plates only (no GDSC2 information)."""
    g1=pd.read_csv('data/raw/gdsc/GDSC1_fitted_dose_response_24Jul22.csv').groupby(['COSMIC_ID','DRUG_ID']).MAX_CONC.max().rename('max_conc')
    ex=pd.read_csv('data/processed/gdsc_units.csv.gz',dtype={'BARCODE':str});ex=ex[ex.split=='explore'];ex['resp']=ex.effect>=.7
    drug=ex.groupby('DRUG_ID').agg(drug_resp=('resp','mean'),drug_n=('COSMIC_ID','nunique'));cell=ex.groupby('COSMIC_ID').agg(cell_resp=('resp','mean'),cell_n=('DRUG_ID','nunique'))
    two=ex.sort_values('BARCODE').groupby(['COSMIC_ID','DRUG_ID']).effect.agg(lambda x:abs(x.iloc[0]-x.iloc[1]) if len(x)>1 else np.nan).dropna()
    e=e.join(g1,on=['COSMIC_ID','DRUG_ID']).join(drug,on='DRUG_ID').join(cell,on='COSMIC_ID');e['plate_mad']=float(two.median())
    return e

def first_message(r,context=False):
    m=(f"Compound {r.compound}, cell line {r.cell_line}.\nPlate 1: mean viability at the two highest doses = {1-r.effect:.2f}. "
       f"Plate control quality Z' = {r.zprime:.2f}.\nExperimental cost so far: {COST_PLATE:g} unit.")
    if context:
        dose=f"{r.max_conc:.3g} uM" if pd.notna(r.max_conc) else "not recorded"
        drug=f"this compound reduced viability below 0.30 in {100*r.drug_resp:.0f}% of the {r.drug_n:.0f} cell lines tested" if pd.notna(r.drug_resp) else "this compound has no other measurements"
        cell=f"this cell line responded this way to {100*r.cell_resp:.0f}% of the {r.cell_n:.0f} compounds tested on it" if pd.notna(r.cell_resp) else "this cell line has no other measurements"
        m+=(f"\nContext: maximum concentration tested for this pair in this screen = {dose}. On other plates of this screen, {drug}, and {cell}. "
            f"For the same compound and cell line on two plates, readings differ by a median of {r.plate_mad:.2f}.")
    return m

def second_message(r):
    if pd.isna(r.alt_effect):return "No independent plate is available for this compound and cell line. Make a final decision (ACCEPT_EFFECTIVE, REJECT_INEFFECTIVE, or ABSTAIN)."
    return (f"Independent plate: mean viability at the two highest doses = {1-r.alt_effect:.2f}. Experimental cost so far: {COST_PLATE+COST_INDEPENDENT:g} units. "
            "Make a final decision (ACCEPT_EFFECTIVE, REJECT_INEFFECTIVE, or ABSTAIN).")

class Budget:
    def __init__(self,cap):
        self.cap=cap;self.lock=threading.Lock();self.path=OUT/'spend.json'
        self.spent=json.loads(self.path.read_text())['usd'] if self.path.exists() else 0.
    def ok(self):
        with self.lock:return self.spent<self.cap
    def add(self,model,usage):
        pin,pout=PRICE[model];usd=usage.input_tokens*pin/1e6+usage.output_tokens*pout/1e6
        with self.lock:
            self.spent+=usd;self.path.write_text(json.dumps({'usd':round(self.spent,6),'cap':self.cap}))
        return usd

def parse(text):
    m=re.search(r'\{.*\}',text,re.S)
    try:a=json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:a={}
    act=str(a.get('action','')).strip().upper()
    return (act if act in ACTIONS else 'INVALID'),str(a.get('rationale',''))[:300]

def run_episode(client,model,prompt,r,budget,effort):
    msgs=[{'role':'system','content':PROMPTS[prompt]},{'role':'user','content':first_message(r,prompt in CONTEXT_PROMPTS)}];turns=[];usd=0.
    for turn in range(2):
        if not budget.ok():return None
        resp=client.responses.create(model=model,input=msgs,reasoning={'effort':effort},max_output_tokens=4000)
        usd+=budget.add(model,resp.usage);act,why=parse(resp.output_text)
        turns.append({'action':act,'rationale':why,'in':resp.usage.input_tokens,'out':resp.usage.output_tokens,'served_model':resp.model})
        if act!='REQUEST_INDEPENDENT_PLATE' or turn==1:break
        msgs+=[{'role':'assistant','content':resp.output_text},{'role':'user','content':second_message(r)}]
    final=turns[-1]['action'];requested=turns[0]['action']=='REQUEST_INDEPENDENT_PLATE'
    if final=='REQUEST_INDEPENDENT_PLATE':final='ABSTAIN' # asked twice: treated as no decision
    pred={'ACCEPT_EFFECTIVE':1,'REJECT_INEFFECTIVE':0}.get(final,-1)
    cost=COST_PLATE+(COST_INDEPENDENT if requested and pd.notna(r.alt_effect) else 0.)
    return {'episode':r.episode,'model':model,'prompt':prompt,'effort':effort,'final':final,'pred':pred,'requested':requested,'exp_cost':cost,'usd':usd,'turns':turns}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=list(PRICE),required=True);ap.add_argument('--prompt',choices=list(PROMPTS),default='base')
    ap.add_argument('--n',type=int,default=10,help='episodes to run (first n of the frozen list)');ap.add_argument('--per-stratum',type=int,default=100)
    ap.add_argument('--cap',type=float,default=17.5);ap.add_argument('--rep',type=int,default=0,help='independent sample index; 0 keeps the original file name');ap.add_argument('--episodes-from',default=None,help='run only the episodes logged in this jsonl');ap.add_argument('--effort',default='low');ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);ep=OUT/'episodes.csv'
    if not ep.exists():build_episodes(args.per_stratum).sample(frac=1,random_state=1).to_csv(ep,index=False) # shuffled once, so any prefix is stratum-balanced in expectation
    e=pd.read_csv(ep,dtype={'BARCODE':str}).head(args.n)
    if args.episodes_from:e=e[e.episode.isin({json.loads(l)['episode'] for l in open(args.episodes_from)})]
    if args.prompt in CONTEXT_PROMPTS:e=episode_context(e)
    log=OUT/(f'{args.model}__{args.prompt}.jsonl' if args.rep==0 else f'{args.model}__{args.prompt}__r{args.rep}.jsonl');done={json.loads(l)['episode'] for l in log.open()} if log.exists() else set()
    todo=[r for r in e.itertuples() if r.episode not in done];budget=Budget(args.cap);client=OpenAI(api_key=load_key());lock=threading.Lock()
    print(f'{len(todo)} episodes to run; spent so far ${budget.spent:.4f} of ${args.cap}')
    def work(r):
        try:res=run_episode(client,args.model,args.prompt,r,budget,args.effort)
        except Exception as ex:print('error',r.episode,type(ex).__name__,str(ex)[:120]);return
        if res is None:return
        with lock,log.open('a') as f:f.write(json.dumps(res)+'\n')
    with ThreadPoolExecutor(args.workers) as pool:list(pool.map(work,todo))
    print(f'spent ${budget.spent:.4f} of ${args.cap}')
if __name__=='__main__':main()
