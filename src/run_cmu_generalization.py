"""SPEC C: frozen controllers on retargeted real CMU movement references.

Real reference motion + simulated arm and adaptive virtual stroke patient.
This is NOT a clinical transfer experiment. No simulation trajectories saved.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from scipy import stats
import dataset as ds
import run_experiment as re
import simulate as sim
import patient_closed_loop as pcl
from run_coadapt import CONTROLLERS,load_controller,TUNED_LAW

ROOT=Path(re.ROOT)
DATA=ROOT/'data_real/cmu_mocap'

def worker(job):
    fold,name,sids,assign=job
    torch.set_num_threads(1)
    subjects,_=ds.load_subjects(str(ROOT))
    z=np.load(DATA/'references.npz')
    ctrl=load_controller(name,fold);rows=[]
    for sid in sids:
        subj=subjects[sid]
        patient=pcl.ClosedLoopPatient(sid,subj.stratum,subj.lam,subj.perm,subj.fma)
        for slot,ri in enumerate(assign[sid]):
            ref={k:z[k][ri].astype(float) for k in ['q_ref','qd_ref','qdd_ref','p_ref','tau_ff']}
            ref.update(t=z['t'],dt=.01,T=2.,d_actual=float(z['d_actual'][ri]))
            seed=re.SEED+1500000+fold*10000+sid*100+slot
            before=patient.snapshot_slow()
            rec=sim.rollout(ctrl,patient,ref,measure_time=True,seed=seed)
            metrics=sim.trial_metrics(rec)
            _,rho=pcl.trial_rhos_cl(patient,rec)
            vol=pcl.crr_rms(rec['tau_vol']);dyn=pcl.crr_rms(rec['tau_a']-rec['g'])
            part=vol/(vol+dyn+1e-12)*rec['patient_state'][:,2].mean()
            after=patient.snapshot_slow();patient.end_trial(part)
            rows.append(dict(model=name,fold=fold,virtual_subject=sid,stratum=subj.stratum,
                             motion_slot=slot+1,reference_index=int(ri),
                             source=str(z['names'][ri]),capture_block=int(z['capture_block'][ri]),
                             seed=seed,**metrics,rho_dynamic=rho,patient_vol_rms=vol,
                             dynamic_assist_rms=dyn,sim_m_start=before['m'],sim_m_end=after['m'],
                             sim_F_start=before['F'],sim_F_end=after['F']))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--output',default='results/external_motion_cmu/run_20261012_1')
    args=ap.parse_args();out=ROOT/args.output
    if (out/'config.json').exists():raise FileExistsError('Choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    z=np.load(DATA/'references.npz');n=len(z['names'])
    assert n>=300,'Insufficient acquired motion records for the larger benchmark'
    rng=np.random.default_rng(20261012)
    slots=np.resize(rng.permutation(n),480).reshape(60,8)
    subjects,fold_of=ds.load_subjects(str(ROOT))
    cfg=dict(protocol='60 held-out virtual subjects, original 5 folds, 8 frozen controllers, 1 session x 8 external reference slots',
             real_source_motion_records=n,used_unique_records=len(set(slots.ravel())),
             source_capture_blocks=len(set(z['capture_block'])),actual_virtual_subjects=60,
             clinically_validated_transfer=False,real_patient_control=False,
             simulated_patient_params=pcl.PARAMS,frozen_weights=True,tuned_law=TUNED_LAW,
             success_criterion='mean RMSE<=50 mm and mean viol<=1%; engineering convention, not clinical',
             reference_sha256=hashlib.sha256((DATA/'references.npz').read_bytes()).hexdigest(),
             seed_assignment=20261012,workers=args.workers,started_local=time.strftime('%Y-%m-%d %H:%M:%S'),
             limitation='CMU subject numbers may alias the same person; statistical units are virtual patients, not clinical patients')
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    t0=time.perf_counter();jobs=[]
    for fold in range(1,6):
        sids=[s.sid for s in subjects if fold_of[s.sid]==fold-1]
        for name in CONTROLLERS:jobs.append((fold,name,sids,slots))
    rows=[]
    with ProcessPoolExecutor(args.workers) as pool:
        fs={pool.submit(worker,j):(j[0],j[1]) for j in jobs}
        for i,f in enumerate(as_completed(fs),1):
            fold,name=fs[f];part=pd.DataFrame(f.result());rows.extend(part.to_dict('records'))
            part.to_csv(out/f'cell_{name}_fold{fold}.csv',index=False)
            print(f'{i}/40 {name}/fold{fold}: {len(part)} trials; {(time.perf_counter()-t0)/60:.2f} min',flush=True)
    df=pd.DataFrame(rows).sort_values(['model','fold','virtual_subject','motion_slot'])
    assert len(df)==3840 and not df.duplicated(['model','virtual_subject','motion_slot']).any()
    assert np.isfinite(df.select_dtypes('number')).all().all()
    assert df.groupby(['virtual_subject','motion_slot']).seed.nunique().eq(1).all()
    df.to_csv(out/'trials_long.csv',index=False)
    metrics=['rmse_p_mm','rho_dynamic','viol_pct','norm_jerk','tau_rms','patient_vol_rms']
    per=df.groupby(['model','fold','virtual_subject'])[metrics].mean().reset_index()
    per.to_csv(out/'per_subject.csv',index=False)
    summary=per.groupby('model')[metrics].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reindex(CONTROLLERS)
    summary['engineering_pass']=(summary.rmse_p_mm_mean<=50)&(summary.viol_pct_mean<=1)
    summary.to_csv(out/'summary.csv')
    A=per[per.model=='iira_tuned'].set_index('virtual_subject')
    B=per[per.model=='iira'].set_index('virtual_subject').loc[A.index]
    tests=[]
    for metric in metrics:
        delta=(A[metric]-B[metric]).to_numpy()
        if np.allclose(delta,0):w,p=0,1
        else:w,p=stats.wilcoxon(delta)
        tests.append(dict(model_a='iira_tuned',model_b='iira',metric=metric,n=60,
                          mean_difference=delta.mean(),wilcoxon_w=w,p_raw=p))
    tests=pd.DataFrame(tests);order=np.argsort(tests.p_raw);prev=0.;adj=np.zeros(6)
    for i,j in enumerate(order):prev=max(prev,(6-i)*tests.p_raw.iloc[j]);adj[j]=min(prev,1.)
    tests['p_holm']=adj;tests.to_csv(out/'stats.csv',index=False)
    cfg.update(rows=len(df),runtime_min=(time.perf_counter()-t0)/60,complete=True,
               saved_simulation_trajectories=0,all_metrics_finite=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)
    print(f'COMPLETE {len(df)} trials; {cfg["runtime_min"]:.2f} min',flush=True)

if __name__=='__main__':main()
