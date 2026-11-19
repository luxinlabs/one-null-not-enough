"""F5: false vs correct rejection among would-be rejections, rules and LLM agents (results/llm/summary.csv)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from figures import COLOR,INK,MUTED,save
from pathlib import Path

LAB={'single_plate':'single plate','control_gated':'Z′-gated','confirm_or_reject':'confirm if possible','confirm_or_abstain':'guard (abstain if no confirmation)',
     'gpt-5.4-mini__base':'gpt-5.4-mini','gpt-5.4-mini__fault_aware':'gpt-5.4-mini + fault hint','gpt-5.5__base':'gpt-5.5','gpt-5.5__fault_aware':'gpt-5.5 + fault hint',
     'gpt-5.4-mini__neutral_z':'mini, no $Z\'$ hint','gpt-5.4-mini__loss':'mini, explicit loss','gpt-5.5__neutral_z':'gpt-5.5, no $Z\'$ hint'}
OFF={'single_plate':(6,-12),'gpt-5.4-mini__base':(-30,8),'confirm_or_reject':(-95,6),'gpt-5.5__base':(6,4),'gpt-5.5__fault_aware':(-112,-3),
     'control_gated':(6,4),'gpt-5.4-mini__fault_aware':(6,-12),'confirm_or_abstain':(8,4),
     'gpt-5.4-mini__neutral_z':(-62,-14),'gpt-5.4-mini__loss':(6,6),'gpt-5.5__neutral_z':(6,-10)}
s=pd.read_csv('results/llm/summary.csv')
s=s[~s.agent.str.contains('neutral_z|loss|baserate|structured')] # ablations are reported in the table only
fig,ax=plt.subplots(figsize=(5.6,4.2))
ax.plot([0,1],[0,1],color=MUTED,lw=.8,ls='--',zorder=1);ax.text(.62,.55,'non-specific\n(equal cut)',color=MUTED,fontsize=7,rotation=33)
for _,r in s.iterrows():
    llm=r.kind=='llm';c=COLOR['joint_risk_voi'] if 'gpt-5.5' in r.agent else (COLOR['identifiability_eig'] if llm else COLOR['identifiability'])
    lo=lambda k:r[k+'_lo'] if pd.notna(r[k+'_lo']) else r[k+'_rep_min'];hi=lambda k:r[k+'_hi'] if pd.notna(r[k+'_hi']) else r[k+'_rep_max']
    fmt='s' if not llm else ('^' if 'neutral_z' in r.agent else ('v' if 'loss' in r.agent else 'o'))
    ax.errorbar(r.false_rejection,r.true_rejection,xerr=[[r.false_rejection-lo('false_rejection')],[hi('false_rejection')-r.false_rejection]],
                yerr=[[r.true_rejection-lo('true_rejection')],[hi('true_rejection')-r.true_rejection]],fmt=fmt,ms=6,color=c,
                mfc=c if 'fault' not in r.agent else 'white',mec=c,mew=1.4,elinewidth=.7,capsize=0,zorder=3)
    ax.annotate(LAB[r.agent],(r.false_rejection,r.true_rejection),textcoords='offset points',xytext=OFF[r.agent],fontsize=7,color=INK)
ax.set_xlabel('false rejection of effective compounds  (lower is better)');ax.set_ylabel('correct rejection of ineffective compounds');ax.set_xlim(0,1.08);ax.set_ylim(0,1.08)
save(fig,Path('results/figures'),'F5_llm_agents');print('ok')
