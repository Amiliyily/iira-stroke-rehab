"""SPEC appendix B: delayed feedback, slacking, fatigue and trial learning.

Parameters express a behavioural hypothesis, not a clinical calibration.
Fast state resets per rollout; m/F/K_h/assist-share EMA persist per patient.
"""
from collections import deque
import numpy as np
import injury as inj

PARAMS = dict(
    K_H0=[8., 8., 8., 2., 2., 2.], K_HV_RATIO=0.1,
    FEEDBACK_DELAY=0.120, ACTIVATION_TAU=0.120, K_H_MAX_MULT=1.5,
    BETA_SLACK=0.15, S_TARGET=0.5, SLACK_EMA_TAU=2.0,
    M_MIN=0.3, M_MAX=1.0, K_FATIGUE=0.03, F_MAX=1.15,
    RECOVER_TRIAL=0.5, RECOVER_SESSION=0.9, M_SESSION_RECOVER=0.5,
    ALPHA_LEARN=0.02, FMA_GAIN=20.0,
)

def crr_rms(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x, float)**2, axis=1))))

class ClosedLoopPatient:
    def __init__(self, sid, stratum, lam, perm, fma, params=None):
        self.sid, self.stratum = int(sid), stratum
        self.lam, self.perm = np.asarray(lam, float), np.asarray(perm, int)
        self.fma_base = float(fma)
        self.P = dict(PARAMS, **(params or {}))
        if self.lam[0] * self.P['F_MAX'] >= 1:
            raise ValueError('F_MAX would reverse volitional torque')
        self.K_h0 = np.array(self.P['K_H0'], float)
        self.K_h = self.K_h0.copy()
        self.m, self.F, self._s_ema = 1., 1., self.P['S_TARGET']
        self.reset(None)

    def reset(self, rng):
        self.rng = rng
        self.c, self.xi = np.zeros(4), np.zeros(6)
        self._e, self._ed, self._tau_dynamic_prev = (np.zeros(6) for _ in range(3))
        self._delay, self._delay_dt = None, None
        self.last_noiseless_tau = np.zeros(6)
        self.last_effort = 0.

    def set_feedback(self, e, ed, tau_dynamic_prev):
        self._e = np.asarray(e, float)
        self._ed = np.asarray(ed, float)
        self._tau_dynamic_prev = np.asarray(tau_dynamic_prev, float)

    def c_demand(self, tau_ff_step):
        return inj.US_PINV @ tau_ff_step

    def _delayed_feedback(self, dt):
        n = int(round(self.P['FEEDBACK_DELAY'] / dt))
        if not np.isclose(n * dt, self.P['FEEDBACK_DELAY']):
            raise ValueError('Feedback delay must be an integer number of steps')
        if self._delay is None or self._delay_dt != dt:
            self._delay = deque([np.zeros(6) for _ in range(n)])
            self._delay_dt = dt
        fb = self.K_h * (self._e + self.P['K_HV_RATIO'] * self._ed)
        self._delay.append(fb.copy())
        return self._delay.popleft()

    def step(self, c_dem, qd, dt, eta):
        lw, lr, ls, ln = self.lam
        p = self.P
        target = c_dem + inj.US_PINV @ self._delayed_feedback(dt)
        self.c += (dt / p['ACTIVATION_TAU']) * (target - self.c)
        np.clip(self.c, -1., 1., out=self.c)
        ct = (1 - lr) * self.c + lr * self.c[self.perm]
        tau_vol = self.m * (1 - lw * self.F) * inj.U_MAX * (inj.S @ ct)
        self.last_effort = self.m * float(np.linalg.norm(self.c)) / 2.
        na, nv = np.linalg.norm(self._tau_dynamic_prev), np.linalg.norm(tau_vol)
        share = na / (na + nv) if na + nv > 1e-9 else p['S_TARGET']
        self._s_ema += dt / (dt + p['SLACK_EMA_TAU']) * (share - self._s_ema)
        self.m = float(np.clip(self.m - p['BETA_SLACK'] *
                              (self._s_ema - p['S_TARGET']) * dt,
                              p['M_MIN'], p['M_MAX']))
        self.F = min(self.F + p['K_FATIGUE'] * self.last_effort * dt, p['F_MAX'])
        tau_sp = ls * inj.B_SP * np.tanh(qd / .5) * np.abs(qd)
        self.last_noiseless_tau = tau_vol - tau_sp
        an = dt / (dt + 1 / (2 * np.pi * inj.NOISE_FC))
        self.xi += an * (inj.NOISE_SCALE * eta - self.xi)
        return self.last_noiseless_tau + ln * self.xi, tau_vol

    def end_trial(self, active_participation):
        p = self.P
        active_participation = float(np.clip(active_participation, 0., 1.))
        self.K_h = np.minimum(self.K_h * (1 + p['ALPHA_LEARN'] * active_participation),
                              p['K_H_MAX_MULT'] * self.K_h0)
        self.F = 1 + (self.F - 1) * (1 - p['RECOVER_TRIAL'])

    def end_session(self):
        p = self.P
        self.F = 1 + (self.F - 1) * (1 - p['RECOVER_SESSION'])
        self.m = 1 - (1 - self.m) * (1 - p['M_SESSION_RECOVER'])

    @property
    def fma(self):
        growth = float(np.mean(self.K_h / self.K_h0) - 1)
        return float(np.clip(self.fma_base + self.P['FMA_GAIN'] * growth, 4., 66.))

    def snapshot_slow(self):
        return dict(m=self.m, F=self.F, K_h=self.K_h.copy(),
                    s_ema=self._s_ema, fma_eff=self.fma)

    def spastic_damping(self, qd):
        x = np.clip(qd / .5, -100, 100)
        return self.lam[2] * inj.B_SP * (2 * np.abs(qd) / np.cosh(x)**2
                                        + np.abs(np.tanh(x)))

    def replay_tau_patient(self, qd_pre, e, ed, tau_dynamic, tau_ff, dt, slow0):
        """Exact noiseless replay using pre-integration state and assist logs."""
        clone = ClosedLoopPatient(self.sid, self.stratum, self.lam, self.perm,
                                  self.fma_base, self.P)
        clone.m, clone.F = slow0['m'], slow0['F']
        clone.K_h = np.array(slow0['K_h'], float)
        clone._s_ema = slow0['s_ema']
        tp, tv, prev = [], [], np.zeros(6)
        for k in range(len(qd_pre)):
            clone.set_feedback(e[k], ed[k], prev)
            a, b = clone.step(clone.c_demand(tau_ff[k]), qd_pre[k], dt, np.zeros(6))
            tp.append(a.copy()); tv.append(b.copy())
            prev = tau_dynamic[k]
        return np.asarray(tp), np.asarray(tv)

def trial_rhos_cl(patient, rec, tau_ff=None, slow0=None):
    """RMS ratio, noiseless patient total and gravity-subtracted assistance."""
    pat = crr_rms(rec['tau_patient_noiseless'])
    return (pat / (pat + crr_rms(rec['tau_a']) + 1e-12),
            pat / (pat + crr_rms(rec['tau_a'] - rec['g']) + 1e-12))
