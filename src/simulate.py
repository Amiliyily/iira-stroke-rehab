"""Closed-loop simulation engine (SPEC 2.2/6): controller x subject x trial.

Semi-implicit Euler, dt = 0.01 s.  The engine is shared by every controller
(classical and neural) for a fair comparison.  Per step:
  1. dynamics quantities M, C qd + g, g, p, J at the current state
  2. patient torque from the injury model (SPEC 3)
  3. controller torque (only measurable signals; tau_patient is exposed solely
     as the interaction-torque sensor analogue used by admittance control)
  4. torque saturation to the actuator limits (demanded torque is logged for
     the safety-violation metric)
  5. semi-implicit Euler integration with viscous + Coulomb friction
"""

import time
import numpy as np
import arm6dof as arm
import injury as inj

# Nominal PD gains of the neural residual controllers = IIRA K_max (SPEC 4.1);
# supervised residual targets are defined relative to this PD + gravity base.
KP_DEF = np.array([60.0, 60.0, 60.0, 12.0, 12.0, 12.0])
KD_DEF = np.array([6.0, 6.0, 6.0, 1.5, 1.5, 1.5])
WS_RADIUS = 0.85


def rollout(controller, subject, ref, measure_time=False, seed=None,
            collect_targets=False):
    """Run one closed-loop trial. Returns a dict of trajectories.

    collect_targets additionally logs g(q) and the supervised residual target
        r = M qdd_ref + (bias - g) + friction - tau_patient - (KP e + KD ed)
    i.e. the torque beyond (default PD + gravity) that would track q_ref
    exactly under the true patient torque.
    """
    dt = ref["dt"]
    N = len(ref["t"])
    q = ref["q_ref"][0].astype(float).copy()
    qd = np.zeros(6)
    rng = np.random.default_rng(seed)
    subject.reset(rng)
    controller.reset()
    tau_a_prev = np.zeros(6)
    g_prev = np.zeros(6)
    adaptive_patient = hasattr(subject, "set_feedback")

    Q = np.zeros((N, 6)); QD = np.zeros((N, 6)); P = np.zeros((N, 3))
    TA = np.zeros((N, 6)); TD = np.zeros((N, 6)); TP = np.zeros((N, 6))
    TV = np.zeros((N, 6))
    VIOL = np.zeros(N, bool)
    G = np.zeros((N, 6)) if collect_targets or adaptive_patient else None
    TP0 = np.zeros((N, 6)) if adaptive_patient else None
    slow_trace = np.zeros((N, 3)) if adaptive_patient else None
    RT = np.zeros((N, 6)) if collect_targets else None
    times = []

    for k in range(N):
        M, bias, g, p, J = arm.engine_dynamics(q, qd)
        c_dem = subject.c_demand(ref["tau_ff"][k])
        # Optional hook for closed-loop patient models (SPEC appendix B):
        # subjects exposing set_feedback receive the current tracking error and
        # the previous assist torque.  v1 injury.Subject has no such method and
        # is completely unaffected.
        if adaptive_patient:
            subject.set_feedback(ref["q_ref"][k] - q, ref["qd_ref"][k] - qd,
                                 tau_a_prev - g_prev)
        tau_pat, tau_vol = subject.step(c_dem, qd, dt, rng.standard_normal(6))
        if adaptive_patient:
            TP0[k] = subject.last_noiseless_tau
            G[k] = g
            slow_trace[k] = [subject.m, subject.F, subject.last_effort]
        ctx = dict(dt=dt, g=g, bias=bias, M=M, p=p, J=J, step=k,
                   tau_patient=tau_pat, tau_ff=ref["tau_ff"][k])
        if measure_time:
            t0 = time.perf_counter()
            tau_cmd = np.asarray(controller.step(ref["t"][k], q, qd,
                                                 ref["q_ref"][k], ref["qd_ref"][k],
                                                 ref["qdd_ref"][k], ctx), float)
            times.append(time.perf_counter() - t0)
        else:
            tau_cmd = np.asarray(controller.step(ref["t"][k], q, qd,
                                                 ref["q_ref"][k], ref["qd_ref"][k],
                                                 ref["qdd_ref"][k], ctx), float)
        viol = bool(np.any(np.abs(tau_cmd) > arm.TAU_LIM + 1e-9))
        tau_a = np.clip(tau_cmd, -arm.TAU_LIM, arm.TAU_LIM)
        # Semi-implicit Euler, dt = 0.01 (SPEC 2.2).  Velocity-proportional
        # dissipation (viscous/Coulomb friction Jacobian and the controller's
        # own damping gain) is treated implicitly - first-order accurate and
        # identical for every controller - because the wrist inertias
        # (~1e-4 kg m^2) make the fully explicit discretisation unstable.
        Dfr = arm.FV + arm.FC * (1.0 / np.cosh(qd / 0.01) ** 2) / 0.01
        Dtot = np.diag(Dfr + subject.spastic_damping(qd))
        Dctrl = getattr(controller, "damping_matrix", None)
        if Dctrl is not None:
            Dtot = Dtot + Dctrl
        fric = arm.FV * qd + arm.FC * np.tanh(qd / 0.01)
        if collect_targets:
            e = ref["q_ref"][k] - q
            ed = ref["qd_ref"][k] - qd
            G[k] = g
            RT[k] = (M @ ref["qdd_ref"][k] + (bias - g) + fric - tau_pat
                     - (KP_DEF * e + KD_DEF * ed))
        qdd = np.linalg.solve(M + dt * Dtot, tau_a + tau_pat - bias - fric)
        qd = qd + qdd * dt
        q = q + qd * dt
        if np.any(np.abs(qd) > arm.QD_LIM) or np.linalg.norm(p) > WS_RADIUS:
            viol = True

        Q[k] = q; QD[k] = qd; P[k] = p; TA[k] = tau_a; TD[k] = tau_cmd
        TP[k] = tau_pat; TV[k] = tau_vol; VIOL[k] = viol
        tau_a_prev = tau_a
        g_prev = g

    rec = dict(t=ref["t"], q=Q, qd=QD, p=P, p_ref=ref["p_ref"],
               q_ref=ref["q_ref"], qd_ref=ref["qd_ref"],
               tau_a=TA, tau_demanded=TD, tau_patient=TP, tau_vol=TV,
               violations=VIOL, step_times=np.array(times),
               d_actual=ref["d_actual"], T=ref["T"], dt=dt)
    if collect_targets:
        rec["g"] = G
        rec["r_target"] = RT
    if adaptive_patient:
        rec.update(g=G, tau_patient_noiseless=TP0, patient_state=slow_trace)
    controller.on_trial_end(rec)
    return rec


def rho_truth(rec, eps=1e-3):
    """Patient contribution ratio (SPEC 4.2): RMS volitional / (RMS patient +
    RMS assist + eps)."""
    rms = lambda x: float(np.sqrt(np.mean(np.sum(x ** 2, axis=1))))
    v, p_, a = rms(rec["tau_vol"]), rms(rec["tau_patient"]), rms(rec["tau_a"])
    return float(np.clip(v / (p_ + a + eps), 0.0, 1.0))


def trial_metrics(rec):
    """The 7 SPEC-6 metrics for one trial (compute time added by caller)."""
    dt = rec["dt"]
    ep = rec["p"] - rec["p_ref"]
    rmse_p = float(np.sqrt(np.mean(np.sum(ep ** 2, axis=1)))) * 1000.0   # mm
    eq = rec["q"] - rec["q_ref"]
    rmse_q = float(np.sqrt(np.mean(eq ** 2))) * 180.0 / np.pi             # deg
    jerk = np.diff(rec["p"], n=3, axis=0) / dt ** 3
    njerk = float(np.sum(jerk ** 2) * dt * rec["T"] ** 5 / max(rec["d_actual"], 1e-6) ** 2)
    tau_rms = float(np.sqrt(np.mean(np.sum(rec["tau_a"] ** 2, axis=1))))  # N m
    viol = float(100.0 * np.mean(rec["violations"]))                      # %
    settle = float(np.mean(np.linalg.norm(ep[-30:], axis=1))) * 1000.0    # mm
    out = dict(rmse_p_mm=rmse_p, rmse_q_deg=rmse_q, norm_jerk=njerk,
               tau_rms=tau_rms, viol_pct=viol, settle_mm=settle)
    if len(rec["step_times"]):
        out["time_ms"] = float(np.median(rec["step_times"])) * 1000.0
    return out
