"""F6: real-data risk-coverage comparison and false-rejection decomposition (results/gdsc/frontier.csv, review3.json)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figures import INK,MUTED,save

C={'single_plate':'#eb6834','confirm':'#2a78d6','two_plate_mean':'#6250d6','control_gated':'#eda100'}
L={'single_plate':'single plate, threshold swept','confirm':'confirm with replicate','two_plate_mean':'mean of two plates','control_gated':r"single plate, $Z^{\prime}$ gate swept"}
f=pd.read_csv('results/gdsc/frontier.csv');r=json.loads(Path('results/gdsc/review3.json').read_text())['R1_decomposition']
fig,(a,b)=plt.subplots(1,2,figsize=(6.4,2.55),gridspec_kw={'width_ratios':[1.15,1]})
for k in ('single_plate','control_gated','two_plate_mean','confirm'):
    s=f[f.rule==k].sort_values('FR');a.plot(s.FR,s.CR,color=C[k],lw=2 if k!='two_plate_mean' else 1.2,ls='-' if k!='two_plate_mean' else ':',label=L[k])
    if k!='control_gated':
        p=s[np.isclose(s.param,.3)];a.plot(p.FR,p.CR,'o',color=C[k],ms=7,mec='white',mew=1.5,zorder=4)
a.axhline(f[(f.rule=='confirm')&np.isclose(f.param,.3)].CR.iloc[0],color=MUTED,lw=.7,ls='--')
a.set_xlim(0,.4);a.set_ylim(.3,1);a.set_xlabel('false rejection of effective compounds');a.set_ylabel('correct rejection of\nineffective compounds')
a.set_title('(a) units with a replicate plate; dots: $\\tau=0.3$',fontsize=9,loc='left');a.legend(fontsize=7.6,loc='lower right')
strata=[('dose_lower','lower\nGDSC1 dose'),('dose_matched_assay_s','matched,\nassay s'),('dose_matched_assay_a','matched,\nassay a')]
x=np.arange(len(strata));w=.38
for i,(k,lab,c) in enumerate((('FR_single_with_replicate','single plate',C['single_plate']),('FR_confirm_with_replicate','confirm with replicate',C['confirm']))):
    v=[r[s][k] for s,_ in strata];a2=b.bar(x+(i-.5)*w,v,w*.92,color=c,label=lab)
    for xi,vi in zip(x+(i-.5)*w,v):b.text(xi,vi+.004,f'{100*vi:.0f}%',ha='center',fontsize=7.6,color=INK)
b.set_xticks(x,[s for _,s in strata],fontsize=7.6);b.set_ylabel('false rejection');b.set_ylim(0,.23);b.legend(fontsize=7.6,loc='upper right')
b.set_title('(b) by GDSC1 dose and assay',fontsize=9,loc='left')
fig.tight_layout();save(fig,Path('results/figures'),'F6_gdsc');print('ok')
