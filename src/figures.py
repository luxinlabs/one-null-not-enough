"""Publication figures F1-F4 from results/analysis/metrics.csv. Writes PNG and PDF to results/figures/."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import experiment as E

# Reference categorical palette, fixed per entity (never by rank).
COLOR={'identifiability':'#2a78d6','joint_risk_voi':'#eb6834','scientific_only':'#1baf7a','calibrate_first':'#eda100',
       'identifiability_eig':'#e87ba4','oracle_fault':'#6250d6','robust_bayes':'#008300'}
LABEL={'identifiability':'identifiability (guarded)','joint_risk_voi':'joint risk + VOI stop','scientific_only':'fault-unaware',
       'calibrate_first':'calibrate first','identifiability_eig':'identifiability, entropy','oracle_fault':'oracle fault state','robust_bayes':'credal rule, Q=[0.05,0.95]'}
INK,MUTED,GRID='#0b0b0b','#52514e','#e4e3df'
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9,'axes.edgecolor':MUTED,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':.6,
                     'axes.axisbelow':True,'legend.frameon':False,'figure.dpi':150,'savefig.bbox':'tight','font.family':'sans-serif'})

def err(d,m):
    return np.vstack([d[m]-d[m+'_lo'],d[m+'_hi']-d[m]])

def bars(ax,d,policies,groups,group_col,metric):
    w=.8/len(policies);x=np.arange(len(groups))
    for i,p in enumerate(policies):
        s=d[d.policy==p].set_index(group_col).reindex(groups)
        ax.bar(x+(i-(len(policies)-1)/2)*w,s[metric],w*.92,yerr=err(s,metric),color=COLOR[p],label=LABEL[p],
               error_kw={'elinewidth':.8,'ecolor':MUTED,'capsize':0})
    ax.set_xticks(x,[g.replace('_','\n') for g in groups])

def f1(out):
    worlds=[f'H={h}, fault={f}' for h,f in E.WORLDS];m=np.array([[E.mean(w,a) for a in E.ACTIONS] for w in range(4)])
    fig,ax=plt.subplots(figsize=(4.3,1.95));ax.grid(False)
    ax.pcolormesh(np.arange(6)-.5,np.arange(5)-.5,m,cmap=matplotlib.colors.ListedColormap(['#cde2fb','#2a78d6']),vmin=0,vmax=1,edgecolors='face',linewidth=.4);ax.set_xlim(-.5,4.5);ax.set_ylim(3.5,-.5)  # vector cells, not a bitmap
    for i in range(4):
        for j in range(5):ax.text(j,i,f'{m[i,j]:.0f}',ha='center',va='center',color='white' if m[i,j] else INK)
    ax.set_xticks(range(5),[a.replace('_','\n') for a in E.ACTIONS]);ax.set_yticks(range(4),worlds)
    ax.add_patch(plt.Rectangle((-.5,-.5),2,2,fill=False,ec='#eb6834',lw=2))
    save(fig,out,'F1_equivalence')

def f2(m,out):
    d=m.query("block=='e2' and sigma==0.5 and fault_rate==0.1")
    pols=['scientific_only','joint_risk_voi','identifiability','robust_bayes']
    fig,axes=plt.subplots(1,2,figsize=(7,2.6),sharey=True)
    for ax,sc in zip(axes,('resolvable','unidentifiable')):
        for p in pols:
            s=d[(d.policy==p)&(d.scenario==sc)].sort_values('budget')
            ax.errorbar(s.budget,s.false_rejection,yerr=err(s,'false_rejection'),color=COLOR[p],lw=2,marker='o',ms=5,capsize=0,elinewidth=.8,label=LABEL[p])
        ax.set_title(sc,loc='left',fontsize=9);ax.set_xlabel('experiment budget');ax.set_xticks([2,5,10])
    axes[0].set_ylabel('false rejection of true H')
    fig.suptitle('Rare faults (rate 0.1, matched prior, σ=0.5): false rejection vs budget, 95% CI',x=.02,y=1.04,ha='left',fontsize=9,color=MUTED)
    legend_below(fig,axes[0],4)
    save(fig,out,'F2_false_rejection_budget')

def f3(m,out):
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.1),gridspec_kw={'width_ratios':[1.6,1]})
    d=m.query("block=='e3' and sigma==0.5 and budget==5 and prior_fault==0.5 and scenario=='resolvable'")
    bars(axes[0],d,['scientific_only','calibrate_first','joint_risk_voi'],list(E.TRUTHS),'truth','false_rejection_faulty')
    axes[0].set_ylabel('false rejection | H true, faulty');axes[0].set_title('(a) masked-effect false rejection by fault type',loc='left',fontsize=9)
    axes[0].tick_params(axis='x',labelsize=7.5,labelrotation=25)
    # (b) coverage vs false rejection in the unidentifiable action class across prior conditions.
    conds=[("block=='e2' and sigma==0.5 and budget==5 and fault_rate==0.5",'fault 0.5'),("block=='e2' and sigma==0.5 and budget==5 and fault_rate==0.1",'fault 0.1'),
           ("block=='e3' and truth=='specified' and sigma==0.5 and budget==5 and prior_fault==0.1",'prior shift')]
    ax=axes[1]
    for q,lab in conds:
        s=m.query(q+" and scenario=='unidentifiable'");pts={}
        for p in ('joint_risk_voi','identifiability'):
            r=s[s.policy==p].iloc[0];pts[p]=(r.coverage,r.false_rejection)
            ax.errorbar(r.coverage,r.false_rejection,xerr=[[r.coverage-r.coverage_lo],[r.coverage_hi-r.coverage]],yerr=[[r.false_rejection-r.false_rejection_lo],[r.false_rejection_hi-r.false_rejection]],
                        color=COLOR[p],marker='o',ms=6,elinewidth=.8,capsize=0,mec='white',mew=1.2,zorder=3)
        (x0,y0),(x1,y1)=pts['joint_risk_voi'],pts['identifiability']
        r=s[s.policy=='robust_bayes'].iloc[0]
        ax.errorbar(r.coverage,r.false_rejection,xerr=[[r.coverage-r.coverage_lo],[r.coverage_hi-r.coverage]],color=COLOR['robust_bayes'],marker='D',ms=5,elinewidth=.8,capsize=0,mec='white',mew=1.2,zorder=3)
        if abs(x0-x1)+abs(y0-y1)>.02:
            ax.annotate('',(x1,y1),(x0,y0),arrowprops={'arrowstyle':'->','color':MUTED,'lw':.8,'shrinkA':5,'shrinkB':5})
            ax.annotate(lab,(x0,y0),textcoords='offset points',xytext=(-8,6),ha='right',fontsize=7,color=MUTED)
        else:ax.annotate(lab+' (identical)',(x0,y0),textcoords='offset points',xytext=(6,6),fontsize=7,color=MUTED)
    ax.set_xlabel('coverage (decided fraction)');ax.set_ylabel('false rejection of true H');ax.set_xlim(0,1)
    ax.set_title('(b) unidentifiable action class',loc='left',fontsize=9)
    ax.set_ylim(-.03,.65)
    leg=('scientific_only','calibrate_first','joint_risk_voi','identifiability','robust_bayes')
    h=[plt.Rectangle((0,0),1,1,color=COLOR[p]) for p in leg]
    fig.legend(h,[LABEL[p] for p in leg],loc='lower center',ncol=5,bbox_to_anchor=(.5,-.02),fontsize=7.5)
    fig.subplots_adjust(bottom=.3,wspace=.3)
    save(fig,out,'F3_misspecification')

def f4(m,out):
    d=m.query("block=='e4' and truth=='specified'");pols=['joint_risk_voi','identifiability','robust_bayes','identifiability_eig','oracle_fault']
    sc=['resolvable','no_calibrate','no_positive_control','no_alternate','unidentifiable']
    fig,axes=plt.subplots(1,2,figsize=(9,2.8))
    for ax,met,lab in zip(axes,('coverage','false_rejection'),('coverage','false rejection of true H')):
        bars(ax,d,pols,sc,'scenario',met);ax.set_ylabel(lab)
    for ax in axes:ax.tick_params(axis='x',labelsize=7.5)
    fig.suptitle('E4 ablations by action class (specified truth, σ=0.5, budget 5), 95% CI',x=.02,y=1.02,ha='left',fontsize=9,color=MUTED)
    legend_below(fig,axes[0],5)
    save(fig,out,'F4_ablations')

def legend_below(fig,ax,n):
    h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=n,bbox_to_anchor=(.5,-.02),fontsize=8)
    fig.subplots_adjust(bottom=.3)

def save(fig,out,name):
    for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--metrics',default='results/analysis/metrics.csv');ap.add_argument('--out',default='results/figures');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);m=pd.read_csv(args.metrics)
    f1(out);f2(m,out);f3(m,out);f4(m,out);print('figures ->',out)
if __name__=='__main__':main()
