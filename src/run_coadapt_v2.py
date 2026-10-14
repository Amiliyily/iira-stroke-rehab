"""Audited v2 patient experiment. The original coadapt results are preserved."""
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
import dataset as ds
import run_experiment as re
import simulate as sim
import patient_closed_loop as pcl
from run_coadapt import load_controller, CONTROLLERS, TUNED_LAW
from coadapt_analysis import summarize

ROOT=Path(re.ROOT)

def run_cell(job):
    fold,name,sids,params,save_rep=job
    torch.set_num_threads(1)
    subjects,_=ds.load_subjects(str(ROOT))
    dz=np.load(ROOT/'data/dataset.npz')
    refs=[re.ref_from_npz(dz,d*5+2) for d in range(8)]
    ctrl=load_controller(name,fold)
    rows=[];representative=None
    for sid in sids:
        subj=subjects[sid]
        patient=pcl.ClosedLoopPatient(sid,subj.stratum,subj.lam,subj.perm,subj.fma,params)
        for sess in range(1,4):
            for di,ref in enumerate(refs):
                # Controller-independent paired random draws; repetition equals session.
                seed=re.SEED+900000+fold*10000+sid*100+di*10+(sess-1)
                slow0=patient.snapshot_slow()
                rec=sim.rollout(ctrl,patient,ref,measure_time=True,seed=seed)
                metrics=sim.trial_metrics(rec)
                rr,rd=pcl.trial_rhos_cl(patient,rec)
                vol=pcl.crr_rms(rec['tau_vol'])
                dyn=pcl.crr_rms(rec['tau_a']-rec['g'])
                effort=float(rec['patient_state'][:,2].mean())
                participation=vol/(vol+dyn+1e-12)*effort
                slow1=patient.snapshot_slow()
                patient.end_trial(participation)
                rows.append(dict(model=name,fold=fold,subject=sid,stratum=subj.stratum,
                                 session=sess,direction=di,trial_in_session=di+1,
                                 trial_global=(sess-1)*8+di+1,seed=seed,**metrics,
                                 rho_realized=rr,rho_dynamic=rd,patient_vol_rms=vol,
                                 dynamic_assist_rms=dyn,activation_effort=effort,
                                 participation=participation,m_start=slow0['m'],
                                 m_end=slow1['m'],F_start=slow0['F'],F_end=slow1['F'],
                                 kh_mean_start=float(slow0['K_h'].mean()),
                                 kh_mean_after_learning=float(patient.K_h.mean()),
                                 fma_baseline=subj.fma,fma_before_learning=slow1['fma_eff'],
                                 fma_after_learning=patient.fma))
                if save_rep==sid and sess==3 and di==0:
                    keep=['t','q','qd','p','p_ref','q_ref','qd_ref','tau_a','tau_patient',
                          'tau_patient_noiseless','tau_vol','g','patient_state','violations']
                    representative={k:np.asarray(rec[k],np.float32) for k in keep}
                    representative.update(subject=sid,fold=fold,session=sess,direction=di,
                                          dt=rec['dt'],seed=seed)
            patient.end_session()
    return rows,representative

def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def provenance():
    files=list((ROOT/'src').glob('*.py'))+list((ROOT/'src/controllers').glob('*.py'))
    files += [ROOT/'SPEC.md',ROOT/'data/subjects.csv',ROOT/'data/folds.csv',
              ROOT/'data/dataset.npz',ROOT/'results/tuning/grid_fold1.csv']
    for f in range(1,6):
        for name in ('mlp','lstm','iira','iira_noaan'):
            files+=list((ROOT/'results/weights').glob(f'{name}_fold{f}.*'))
            files+=list((ROOT/'results/weights').glob(f'{name}_fold{f}_norm.npz'))
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(set(files))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='results/coadapt/run_20261012_v2_1')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--probe',action='store_true',help='3 severity subjects, 24 trials/model')
    ap.add_argument('--resume',action='store_true',help='resume only this run, matching provenance')
    ap.add_argument('--sensitivity',action='store_true')
    args=ap.parse_args()
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    manifest=provenance()
    cfgpath=output/'config.json'
    if cfgpath.exists():
        if not args.resume:
            raise FileExistsError('Run exists: choose a new --output or explicit --resume')
        old=json.loads(cfgpath.read_text(encoding='utf-8'))
        if old['sha256']!=manifest:
            raise ValueError('Code/weights/config changed: use a new output directory')
    subjects,fold_of=ds.load_subjects(str(ROOT))
    rep=next(s.sid for s in subjects if fold_of[s.sid]==0 and s.stratum=='moderate')
    cfg=dict(version='v2.0-audited-1',patient_params=pcl.PARAMS,controllers=CONTROLLERS,
             tuned_law=TUNED_LAW,sessions=3,dirs=8,folds=5,seed=re.SEED,
             protocol='60 subjects, saved stratified 5 folds; 3 sessions x 8 directions; d=0.18m',
             paired_seed_formula='SEED+900000+fold*10000+sid*100+direction*10+(session-1)',
             frozen_weights=True,neural_retraining=False,torch_threads_per_worker=1,
             workers=args.workers,probe=args.probe,sensitivity=args.sensitivity,
             sha256=manifest,started_local=time.strftime('%Y-%m-%d %H:%M:%S'),
             calibration='Assumed behavioural scales, not fitted clinical parameters',
             limitations=['FMA proxy improvement built into positive learning update',
                          'rho_dynamic total includes noiseless passive spasticity; learning uses volitional only',
                          'fixed task order; within-session slopes exploratory',
                          'tuned deployment law originally selected on fold-1 test subjects; selection bias retained',
                          'noaan saved separately trained main-experiment ablation weights'],
             deviations=['Sensitivity representative only, not full 60-subject CV',
                         'Optional closed-loop fine-tuning omitted; frozen zero-shot weights retained'],
             literature=['https://doi.org/10.1186/1743-0003-6-20',
                         'https://pubmed.ncbi.nlm.nih.gov/18002881/'])
    cfgpath.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    t0=time.perf_counter()
    variants=[('base',{})]
    if args.sensitivity:
        variants=[]
        for param in ('BETA_SLACK','ALPHA_LEARN','K_FATIGUE'):
            for mult in (.5,2.):
                variants.append((f'{param}_{mult:g}x',{param:pcl.PARAMS[param]*mult}))
    jobs=[]
    labels={}
    for variant,params in variants:
        for fold in range(1,6):
            sids=[s.sid for s in subjects if fold_of[s.sid]==fold-1]
            if args.probe or args.sensitivity:
                if fold!=1:continue
                sids=[next(s.sid for s in subjects if fold_of[s.sid]==0 and s.stratum==stratum)
                      for stratum in ('mild','moderate','severe')]
            models=['smc','iira','iira_tuned'] if args.sensitivity else CONTROLLERS
            for name in models:
                label=f'{variant}_{name}_fold{fold}'
                shard=output/f'cell_{label}.csv'
                if args.resume and shard.exists():continue
                job=(fold,name,sids,params,rep if not(args.probe or args.sensitivity) else None)
                jobs.append((label,job))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_cell,job):label for label,job in jobs}
        for ix,future in enumerate(as_completed(futures),1):
            label=futures[future];rows,representative=future.result()
            part=pd.DataFrame(rows)
            part['variant']=label.rsplit('_'+part.model.iloc[0]+'_fold',1)[0]
            part.to_csv(output/f'cell_{label}.csv',index=False)
            if representative is not None:
                np.savez_compressed(output/f'traj_{part.model.iloc[0]}.npz',**representative)
            print(f'{ix}/{len(jobs)} {label}: {len(part)} trials; elapsed {(time.perf_counter()-t0)/60:.2f} min',flush=True)
    df=pd.concat([pd.read_csv(p) for p in sorted(output.glob('cell_*.csv'))],ignore_index=True)
    df=df.sort_values(['variant','model','fold','subject','trial_global'])
    df.to_csv(output/'trials_long.csv',index=False)
    if not args.sensitivity:
        summarize(df,output,CONTROLLERS)
    else:
        from coadapt_analysis import direction_adjusted_slope
        summaries=[]
        for (variant,model),g in df.groupby(['variant','model']):
            values=[]
            for _,h in g.groupby('subject'):
                e1=h[h.session==1].patient_vol_rms.mean()
                e3=h[h.session==3].patient_vol_rms.mean()
                values.append([100*(e1-e3)/e1,direction_adjusted_slope(h,'rho_dynamic'),
                               h.fma_after_learning.iloc[-1]-h.fma_baseline.iloc[0],
                               h.rmse_p_mm.mean(),h.F_end.max()])
            vals=np.array(values).mean(axis=0)
            summaries.append(dict(variant=variant,model=model,n_subjects=3,
                                  slacking_pct=vals[0],rho_slope=vals[1],delta_fma=vals[2],
                                  rmse_p_mm=vals[3],max_fatigue=vals[4]))
        pd.DataFrame(summaries).to_csv(output/'sensitivity_summary.csv',index=False)
    cfg['runtime_min']=(time.perf_counter()-t0)/60
    cfg['rows']=len(df);cfg['complete']=True
    cfg['finished_local']=time.strftime('%Y-%m-%d %H:%M:%S')
    cfgpath.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'COMPLETE rows={len(df)}, runtime={cfg["runtime_min"]:.2f} min',flush=True)

if __name__=='__main__':main()
