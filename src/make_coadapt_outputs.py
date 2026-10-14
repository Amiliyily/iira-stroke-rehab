"""Figure 8 and booktabs Table 6 from an explicitly selected coadapt run."""
import argparse
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import make_patent_flow as mpf

ROOT=Path(__file__).resolve().parents[1]
MODELS=['pid','impedance','smc','mlp','lstm','iira','iira_noaan','iira_tuned']
DISPLAY=['PID','Impedance','SMC','MLP','LSTM','IIRA','IIRA w/o AAN','IIRA-tuned']
COLORS=['#777777','#8764ad','#2878b5','#299c53','#18a6b4','#d1495b','#d394a5','#ee8b2d']
FLOW_CFG=dict(name='fig8a_patient_flow',fonts=['DejaVu Sans'],figsize=(15,4.8),
              fs_title=13,fs_node=10,fs_branch=10,fs_caption=10,
              rect_w=24,rect_h=11,rect_w_no=28,dia_w=22,
              layout=dict(xlim=110,title_bar=(6,104),caption_x=55,
                          start=10,r1=32,r2=60,d=86,r3=86,r4=49,end=10,r5=86,x_fb=70),
              title='(a) Closed-loop adaptive patient model',start='Trial k',end='Trial k+1',
              r1='Error e, ed; previous\ndynamic assistance (tau_a - g)',
              r2='120 ms feedback delay\nFF + K_h e + K_hv ed\nSynergy projection, LPF, saturation',
              diamond='Trial\ncomplete?',
              r3='Active participation drives K_h\nupdate; FMA proxy increases',
              r4='Partial fatigue recovery\nCarry m, F, K_h, assist EMA',
              r5='Patient torque + robot -> plant\nUpdate m (slacking), F (fatigue)\nRe-evaluate error and assistance',
              yes='Yes',no='No',
              caption='Patient torque: (1 - lambda_w F) U S c_eff - spasticity + noise;  c_eff = m x dissolved synergy activation')

def plot_band(ax,groups,xcol,ycol,color,label,marker=None):
    agg=groups.groupby(xcol)[ycol].agg(['mean','std','count'])
    x=agg.index.to_numpy();y=agg['mean'].to_numpy()
    se=agg['std'].to_numpy()/np.sqrt(agg['count'].to_numpy())
    ax.plot(x,y,color=color,label=label,lw=1.9,marker=marker,ms=4)
    ax.fill_between(x,y-se,y+se,color=color,alpha=.09,lw=0)

def figure(long,summ,tests,sessions,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    mpf.draw_flow(FLOW_CFG)
    fig=plt.figure(figsize=(15,13))
    gs=fig.add_gridspec(3,2,height_ratios=[1.25,1,1],hspace=.55,wspace=.30,
                       left=.075,right=.97,bottom=.075,top=.99)
    a=fig.add_subplot(gs[0,:]);a.imshow(plt.imread(out/'fig8a_patient_flow.png'));a.axis('off')
    b=fig.add_subplot(gs[1,0])
    for m,c,label in zip(MODELS,COLORS,DISPLAY):
        plot_band(b,long[long.model==m],'trial_global','rho_dynamic',c,label)
    for x in (8.5,16.5):b.axvline(x,color='.7',ls='--',lw=.8)
    b.set(xlim=(1,24),xlabel='Trial (3 sessions x 8 directions)',
          ylabel=r'$\rho_{dynamic}$',title='(b) Patient dynamic contribution (mean +/- SEM)')
    b.grid(axis='y',alpha=.15)
    c=fig.add_subplot(gs[1,1]);S=summ.set_index('model')
    vals=S.loc[MODELS,'slacking_pct_mean'].to_numpy()
    err=1.96*S.loc[MODELS,'slacking_pct_std'].to_numpy()/np.sqrt(S.loc[MODELS,'n_subjects'])
    c.bar(np.arange(8),vals,yerr=err,color=COLORS,capsize=3)
    sig=tests[tests.family=='slacking_vs0'].set_index('model_a')
    span=max(np.max(vals+err)-np.min(vals-err),1.)
    for i,m in enumerate(MODELS):
        if sig.loc[m,'significant']:c.text(i,vals[i]+err.iloc[i]+span*.04,'*',ha='center',fontsize=13)
    c.axhline(0,color='.2',lw=.8)
    c.set_xticks(range(8),DISPLAY,rotation=30,ha='right',fontsize=9)
    c.set(ylabel='Volitional effort loss, session 1 to 3 (%)',
          title='(c) Slacking index (positive = less patient effort)')
    c.set_ylim(min(0,np.min(vals-err))-span*.1,np.max(vals+err)+span*.2)
    d=fig.add_subplot(gs[2,0])
    for m,col,label in zip(MODELS,COLORS,DISPLAY):
        g=sessions[sessions.model==m].copy()
        g['fma_change']=g.fma_proxy-g.fma_baseline
        plot_band(d,g,'session','fma_change',col,label,marker='o')
    d.set(xticks=[1,2,3],xlabel='Completed session',ylabel='FMA proxy change from baseline (points)',
          title='(d) Simulated learning (mean +/- SEM)')
    d.grid(axis='y',alpha=.15)
    e=fig.add_subplot(gs[2,1])
    # Separate labels at fixed axes positions with leader lines, robust to close points.
    label_pos=[(.88,.10),(.12,.92),(.08,.10),(.32,.92),(.75,.76),(.88,.45),(.40,.10),(.62,.10)]
    for m,col,label,pos in zip(MODELS,COLORS,DISPLAY,label_pos):
        r=S.loc[m];xy=(r.rmse_p_mm_mean,r.rho_dynamic_mean)
        e.scatter(*xy,s=55,color=col,zorder=4)
        e.annotate(label,xy,xytext=pos,textcoords='axes fraction',ha='center',va='center',
                   fontsize=9,color=col,bbox=dict(fc='white',ec='none',pad=1.5,alpha=.9),
                   arrowprops=dict(arrowstyle='-',color=col,lw=.75))
    e.margins(x=.13,y=.30)
    e.set(xlabel='End-effector RMSE (mm)',ylabel=r'Mean $\rho_{dynamic}$',
          title='(e) Accuracy-participation tradeoff')
    handles,labels=b.get_legend_handles_labels()
    fig.legend(handles,labels,ncol=8,loc='lower center',bbox_to_anchor=(.52,.014),
               frameon=False,fontsize=9,handlelength=1.9,columnspacing=1.1)
    fig.savefig(out/'fig8_coadapt.png',dpi=300,facecolor='white')
    fig.savefig(out/'fig8_coadapt.pdf',facecolor='white')
    plt.close(fig)

def table(summ,tests,out,validation_dir):
    S=summ.set_index('model').loc[MODELS]
    cols=['rmse_p_mm_mean','rmse_p_mm_std','rho_dynamic_mean','rho_dynamic_std',
          'slacking_pct_mean','slacking_pct_std','delta_fma_mean','delta_fma_std',
          'norm_jerk_mean','viol_pct_mean']
    frame=S[cols].reset_index()
    frame['controller']=DISPLAY
    frame.to_csv(out/'table6_coadapt.csv',index=False)
    sig=tests[tests.family=='slacking_vs0'].set_index('model_a')
    def star(m):return r'$^{*}$' if sig.loc[m,'significant'] else ''
    lines=[r'\begin{tabular}{lcccccc}',r'\toprule',
           r'Controller & RMSE (mm) & $\rho_{dyn}$ & Effort loss (\%) & $\Delta$FMA & Norm. jerk & Viol. (\%) \\',r'\midrule']
    expected=[]
    for m,label in zip(MODELS,DISPLAY):
        r=S.loc[m]
        nums=[r.rmse_p_mm_mean,r.rmse_p_mm_std,r.rho_dynamic_mean,r.rho_dynamic_std,
              r.slacking_pct_mean,r.slacking_pct_std,r.delta_fma_mean,r.delta_fma_std,
              r.norm_jerk_mean,r.viol_pct_mean]
        lines.append(f'{label} & {nums[0]:.1f}$\\pm${nums[1]:.1f} & '
                     f'{nums[2]:.3f}$\\pm${nums[3]:.3f} & '
                     f'{nums[4]:.2f}$\\pm${nums[5]:.2f}{star(m)} & '
                     f'{nums[6]:.2f}$\\pm${nums[7]:.2f} & {nums[8]:.2e} & {nums[9]:.2f} \\\\')
        expected.append([float(f'{v:.{p}f}') for v,p in zip(nums[:8],[1,1,3,3,2,2,2,2])]
                        +[float(f'{nums[8]:.2e}'),float(f'{nums[9]:.2f}')])
    lines += [r'\bottomrule',r'\end{tabular}',
              '% Mean +/- between-subject SD; n=60 each. Effort loss pairs directions across sessions 1 and 3.',
              '% * Holm-adjusted two-sided Wilcoxon vs zero; FMA is a synthetic proxy, not a clinical outcome.']
    (out/'table6_coadapt.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    back=pd.read_csv(out/'table6_coadapt.csv')
    assert np.allclose(back[cols],frame[cols],rtol=1e-12,atol=1e-12)
    parsed=[]
    text=(out/'table6_coadapt.tex').read_text(encoding='utf-8')
    for line in text.splitlines():
        if line.split(' & ')[0] in DISPLAY:
            values=[]
            for cell in line.split(' & ')[1:]:
                cell=cell.replace('$^{*}$','')
                values+=list(map(float,re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?',cell)))
            parsed.append(values)
    assert np.allclose(parsed,expected,rtol=0,atol=1e-12)
    validation=dict(csv_to_summary_allclose=True,tex_to_csv_display_precision_allclose=True,
                    rows=len(frame),numbers_checked=int(np.asarray(expected).size))
    (validation_dir/'validation_table6.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',default='results/coadapt/run_20261012_v2_1')
    ap.add_argument('--fig-dir',default='figures')
    ap.add_argument('--table-dir',default='tables')
    args=ap.parse_args();run=ROOT/args.run;fig=ROOT/args.fig_dir;tab=ROOT/args.table_dir
    fig.mkdir(parents=True,exist_ok=True);tab.mkdir(parents=True,exist_ok=True)
    mpf.FIG=str(fig)
    long=pd.read_csv(run/'trials_long.csv');summ=pd.read_csv(run/'summary.csv')
    tests=pd.read_csv(run/'stats.csv');sessions=pd.read_csv(run/'sessions.csv')
    figure(long,summ,tests,sessions,fig);table(summ,tests,tab,run)
    print(f'Figure/table generated from {run}')

if __name__=='__main__':main()
