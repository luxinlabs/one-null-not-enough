"""F7: which nulls to retest. (a) recovered share of false rejections vs retest budget (held-out cell lines); (b) campaign points
(lower is better) for rules, the VOI policy, and LLM agents on the same campaigns."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figures import INK,MUTED,save

C={'voi_optimal':'#0b0b0b','random':'#9e9d99','lowest_zprime':'#eda100','weakest_null':'#1baf7a','compound_history':'#6250d6','learned_gbm':'#2a78d6','retest_model':'#2a78d6','oracle':'#52514e'}
L={'voi_optimal':'VOI policy','random':'random','lowest_zprime':r"lowest $Z^{\prime}$ first",'weakest_null':'weakest null first','compound_history':'compound history','learned_gbm':'retest model','retest_model':'retest model','oracle':'oracle'}
c=pd.read_csv('results/gdsc/retest_curve.csv');c=c[c.group=='COSMIC_ID']
fig,(a,b)=plt.subplots(1,2,figsize=(6.4,2.75),gridspec_kw={'width_ratios':[1.1,1]})
for k in ('random','lowest_zprime','weakest_null','compound_history','learned_gbm'):
    s=c[c.policy==k].sort_values('budget');a.plot(100*s.budget,100*s.recovered_share,'-o',ms=3.5,color=C[k],lw=2 if k=='learned_gbm' else 1.4,label=L[k])
a.axhline(100*c[c.budget==1].recovered_share.iloc[0],color=MUTED,lw=.7,ls='--');a.text(30,46.3,'retest every null',fontsize=7.6,color=MUTED)
a.set_xlim(0,55);a.set_ylim(0,50);a.set_xlabel('nulls retested (%)');a.set_ylabel('false rejections recovered (%)');a.legend(fontsize=7.6,loc='lower right')
a.set_title('(a) held-out cell lines',fontsize=9,loc='left')
r=pd.read_csv('results/agent/rules.csv');rows=[]
agents=sorted(f for f in Path('results/agent').glob('*__*.jsonl') if f.stem.count('__')==1 and f.stem.split('__')[1] in ('plain','score','calibrated','voi','voi_forced'))  # main-figure conditions
camp=set.intersection(*[{json.loads(l)['campaign'] for l in f.open()} for f in agents]) if agents else set(r.campaign)
for k in ('random','lowest_zprime','compound_history','retest_model','voi_optimal'):
    v=r[(r.policy==k)&r.campaign.isin(camp)].points;rows.append((L[k],v.mean(),v.sem()*1.96,C[k]))
AC={'plain':'#eb6834','score':'#e87ba4','calibrated':'#b4447a','voi':'#7a2150','voi_forced':'#c9a0b8'}
for f in agents:
    m,cond=f.stem.split('__');v=pd.DataFrame([json.loads(l) for l in f.open()]);v=v[v.campaign.isin(camp)].points
    rows.append((('gpt-5.5' if m=='gpt-5.5' else 'gpt-5.4-mini')+{'score':' + raw score','calibrated':' + calibrated','voi':' + VOI scaffold','voi_forced':', decides only'}.get(cond,''),v.mean(),v.sem()*1.96,AC[cond]))
ORDER=['random',r"lowest $Z^{\prime}$ first",'compound history','retest model','gpt-5.4-mini','gpt-5.4-mini + calibrated','gpt-5.5','gpt-5.5 + calibrated',
       'gpt-5.4-mini + VOI scaffold','gpt-5.4-mini, decides only','VOI policy']  # central contrast last, adjacent
rows=sorted([x for x in rows if x[0] in ORDER],key=lambda x:ORDER.index(x[0]))  # raw-score rows are in Table V only
y=np.arange(len(rows))[::-1]
bars=b.barh(y,[x[1] for x in rows],xerr=[x[2] for x in rows],color=[x[3] for x in rows],height=.7,error_kw={'elinewidth':.8,'ecolor':MUTED})
for bar,x in zip(bars,rows):  # neutral baselines, light other agents, strong central contrast
    bar.set_color('#0b0b0b' if x[0]=='VOI policy' else '#b4447a' if x[0]=='gpt-5.4-mini + VOI scaffold' else '#e6b8a8' if x[0].startswith('gpt') else '#c4c2bc')
b.set_yticks(y,[x[0] for x in rows],fontsize=8);b.set_xlim(0,62);b.set_xlabel('campaign points (lower is better)')
for yi,x in zip(y,rows):b.text(x[1]+x[2]+.8,yi,f'{x[1]:.1f}',va='center',fontsize=7.6,color=INK)
b.set_title(f'(b) agent campaigns (n={len(camp)})',fontsize=9,loc='left');b.grid(axis='y',visible=False)
fig.tight_layout();save(fig,Path('results/figures'),'F7_retest');print('ok',len(camp))
