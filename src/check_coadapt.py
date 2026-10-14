"""Mechanism and timeline validation for a stateful scientific simulation."""
import json
from pathlib import Path
import numpy as np
import dataset as ds
import run_experiment as re
import simulate as sim
import injury
from controllers.classical import ImpedanceController
from patient_closed_loop import ClosedLoopPatient, PARAMS

def make(**params):
    return ClosedLoopPatient(0,'severe',[.85,.4,2.,.2],[2,0,3,1],30.,params)

def main():
    checks={}
    p=make(BETA_SLACK=0.,K_FATIGUE=0.)
    for k in range(12):
        p.set_feedback(np.ones(6),np.zeros(6),np.zeros(6))
        p.step(np.zeros(4),np.zeros(6),.01,np.zeros(6))
        assert np.all(p.c==0),'Feedback arrived before 120 ms'
    p.step(np.zeros(4),np.zeros(6),.01,np.zeros(6))
    assert np.linalg.norm(p.c)>0
    checks['explicit_120ms_feedback_delay']=True
    high,low=make(K_FATIGUE=0.),make(K_FATIGUE=0.)
    for k in range(1000):
        for patient,assist in [(high,np.ones(6)*100),(low,np.zeros(6))]:
            patient.set_feedback(np.zeros(6),np.zeros(6),assist)
            patient.step(np.ones(4)*.2,np.zeros(6),.01,np.zeros(6))
    assert high.m < low.m and PARAMS['M_MIN']<=high.m<=1 and low.m==1
    checks['more_dynamic_assistance_reduces_effort']=True
    p=make()
    previous=p.F
    for k in range(2000):
        p.step(np.ones(4),np.zeros(6),.01,np.zeros(6))
        assert p.F>=previous and 1-p.lam[0]*p.F>0
        previous=p.F
    p.end_trial(.5)
    assert p.F<previous and p.K_h[0]>8.
    saved=p.snapshot_slow();p.reset(None)
    assert p.m==saved['m'] and p.F==saved['F'] and np.array_equal(p.K_h,saved['K_h'])
    checks['fatigue_monotone_positive_weakness_recovery_and_slow_state_carry']=True
    p0,p1=make(),make()
    for k in range(24):p0.end_trial(0.);p1.end_trial(.5)
    assert p1.fma>p0.fma and np.all(p1.K_h<=p1.K_h0*1.5)
    checks['learning_requires_active_participation']=True
    subjects,_=ds.load_subjects(re.ROOT)
    dz=np.load(Path(re.ROOT)/'data/dataset.npz')
    ref=re.ref_from_npz(dz,2)
    s=subjects[22]
    p=ClosedLoopPatient(s.sid,s.stratum,s.lam,s.perm,s.fma)
    slow=p.snapshot_slow()
    rec=sim.rollout(ImpedanceController(),p,ref,seed=100)
    q_pre=np.vstack([ref['q_ref'][:1],rec['q'][:-1]])
    qd_pre=np.vstack([np.zeros((1,6)),rec['qd'][:-1]])
    tp,tv=p.replay_tau_patient(qd_pre,ref['q_ref']-q_pre,ref['qd_ref']-qd_pre,
                              rec['tau_a']-rec['g'],ref['tau_ff'],ref['dt'],slow)
    assert np.allclose(tp,rec['tau_patient_noiseless'],rtol=0,atol=1e-12)
    assert np.allclose(tv,rec['tau_vol'],rtol=0,atol=1e-12)
    checks['online_noiseless_torque_matches_exact_preintegration_replay']=True
    a=sim.rollout(ImpedanceController(),s,ref,seed=100)
    b=sim.rollout(ImpedanceController(),s,ref,seed=100)
    assert np.array_equal(a['q'],b['q']) and 'patient_state' not in a
    checks['feedforward_subject_still_deterministic_and_unaffected_by_hook']=True
    out=Path(re.ROOT)/'results/coadapt/validation_mechanisms.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
