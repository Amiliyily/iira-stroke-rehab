"""Subject-level paired coadaptation analysis; no trial pseudoreplication."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

METRICS = ['rmse_p_mm', 'rho_dynamic', 'norm_jerk', 'viol_pct', 'tau_rms',
           'slacking_pct', 'effort_multiplier_drop_pct', 'rho_slope',
           'rho_within_session_slope', 'effort_slope_pct', 'delta_fma', 'fma_slope']

def slope(y, x=None):
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, float)
    xc = x - x.mean()
    return float(xc @ (y-y.mean()) / (xc @ xc)) if xc @ xc > 0 else np.nan

def direction_adjusted_slope(g, col):
    x = g.trial_global - g.groupby('direction').trial_global.transform('mean')
    y = g[col] - g.groupby('direction')[col].transform('mean')
    return float(np.dot(x,y) / np.dot(x,x))

def bootstrap_ci(x, rng):
    x = np.asarray(x, float)
    means = x[rng.integers(len(x), size=(5000,len(x)))].mean(axis=1)
    return np.quantile(means, [.025,.975])

def test_row(x, y, family, metric, a, b, rng):
    x,y = np.asarray(x,float), np.broadcast_to(np.asarray(y,float),np.shape(x))
    d = x-y
    nz = d[np.abs(d)>1e-12]
    if not len(nz):
        W,p,rbc = 0.,1.,0.
    else:
        W,p = stats.wilcoxon(nz, alternative='two-sided', method='auto')
        ranks = stats.rankdata(np.abs(nz))
        rbc = float(np.dot(np.sign(nz),ranks)/ranks.sum())
    lo,hi = bootstrap_ci(d,rng)
    return dict(family=family,metric=metric,model_a=a,model_b=b,n=len(d),
                n_nonzero=len(nz),mean_a=x.mean(),mean_b=y.mean(),
                mean_difference=d.mean(),ci_low=lo,ci_high=hi,
                rank_biserial=rbc,wilcoxon_W=W,p_raw=p)

def summarize(df, output, controllers):
    output=Path(output)
    keys=['model','fold','subject']
    records=[]
    for (m,f,s),g in df.groupby(keys,sort=False):
        g=g.sort_values('trial_global')
        first=g[g.session==1].set_index('direction')
        last=g[g.session==3].set_index('direction').loc[first.index]
        e1,e3=first.patient_vol_rms.to_numpy(),last.patient_vol_rms.to_numpy()
        fmas=g.groupby('session').fma_after_learning.last()
        rec=dict(model=m,fold=f,subject=s,stratum=g.stratum.iloc[0],
                 **{k:g[k].mean() for k in METRICS[:5]},
                 slacking_pct=100*np.mean((e1-e3)/np.maximum(e1,1e-9)),
                 effort_multiplier_drop_pct=100*(g.m_start.iloc[0]-g.m_end.iloc[-1]),
                 rho_slope=direction_adjusted_slope(g,'rho_dynamic'),
                 rho_within_session_slope=np.mean([slope(h.rho_dynamic,h.trial_in_session)
                                                   for _,h in g.groupby('session')]),
                 effort_slope_pct=100*direction_adjusted_slope(g,'patient_vol_rms')/
                                    max(first.patient_vol_rms.mean(),1e-9),
                 delta_fma=fmas.iloc[-1]-g.fma_baseline.iloc[0],
                 fma_slope=slope(fmas),fma_baseline=g.fma_baseline.iloc[0],
                 fma_end=fmas.iloc[-1])
        records.append(rec)
    per=pd.DataFrame(records)
    per.to_csv(output/'per_subject.csv',index=False)
    summary=[]
    for m in controllers:
        g=per[per.model==m]
        row=dict(model=m,n_subjects=len(g))
        for k in METRICS+['fma_baseline','fma_end']:
            row[k+'_mean']=g[k].mean();row[k+'_std']=g[k].std()
        summary.append(row)
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    rng=np.random.default_rng(20261012)
    tests=[]
    for family,metric in [('slacking_vs0','slacking_pct'),('rho_vs0','rho_slope'),
                          ('learning_vs0','delta_fma'),
                          ('rho_within_session_vs0','rho_within_session_slope')]:
        for m in controllers:
            g=per[per.model==m]
            tests.append(test_row(g[metric],0.,family,metric,m,'0',rng))
    for a,b in [('iira','smc'),('iira_tuned','smc'),('iira','iira_noaan'),
                ('iira_tuned','iira')]:
        A=per[per.model==a].set_index(['fold','subject'])
        B=per[per.model==b].set_index(['fold','subject']).loc[A.index]
        for metric in ['rmse_p_mm','rho_dynamic','slacking_pct','delta_fma',
                       'viol_pct','norm_jerk']:
            tests.append(test_row(A[metric],B[metric],'paired',metric,a,b,rng))
    out=pd.DataFrame(tests)
    out['p_holm']=np.nan
    for _,g in out.groupby('family'):
        ps=g.p_raw.to_numpy();order=np.argsort(ps)
        adjusted=np.empty(len(ps));running=0.
        for rank,ix in enumerate(order):
            running=max(running,(len(ps)-rank)*ps[ix])
            adjusted[ix]=min(running,1.)
        out.loc[g.index,'p_holm']=adjusted
    out['significant']=out.p_holm<.05
    out.to_csv(output/'stats.csv',index=False)
    sessions=df.sort_values('trial_global').groupby(keys+['session']).agg(
        fma_baseline=('fma_baseline','first'),fma_proxy=('fma_after_learning','last'),
        rho_dynamic=('rho_dynamic','mean'),patient_vol_rms=('patient_vol_rms','mean'),
        rmse_p_mm=('rmse_p_mm','mean')).reset_index()
    sessions.to_csv(output/'sessions.csv',index=False)
    folds=per.groupby(['model','fold'])[METRICS].mean().reset_index()
    folds.to_csv(output/'per_fold.csv',index=False)
    return per,out
