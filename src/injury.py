"""Injury model and virtual-subject sampling (SPEC 3).

    tau_patient = (1-lam_w) U S c~(t)  -  lam_s B_sp ⊙ tanh(qd/0.5) ⊙ |qd|  +  lam_n xi(t)

with synergy dissolution  c~ = (1-lam_r)c + lam_r Pi c  (Pi fixed per subject),
c(t) a first-order (tau=120 ms) EMG-like filter of the feedforward demand,
xi a 3 Hz low-pass filtered white noise.

U (not expanded in the SPEC formula) is the maximal voluntary torque scale
diag([25,25,20,6,6,4]) N·m (proximal > distal); documented in config snapshot.
"""

import numpy as np

SEED = 20261011

# healthy synergy matrix S (6x4), non-negative, row-normalised (SPEC 3).
# synergies: 1 shoulder, 2 elbow, 3 shoulder-elbow coupling, 4 wrist
S_RAW = np.array([
    [0.70, 0.10, 0.50, 0.00],
    [0.60, 0.60, 0.20, 0.00],
    [0.10, 0.70, 0.50, 0.00],
    [0.05, 0.05, 0.10, 0.50],
    [0.00, 0.05, 0.05, 0.30],
    [0.00, 0.00, 0.05, 0.20],
])
S = S_RAW / S_RAW.sum(axis=1, keepdims=True)

U_MAX = np.array([25.0, 25.0, 20.0, 6.0, 6.0, 4.0])       # N·m per joint
B_SP = np.array([1.0, 1.2, 1.2, 0.3, 0.3, 0.3])
NOISE_SCALE = np.array([10.0, 10.0, 10.0, 2.0, 2.0, 2.0])  # N·m std of xi
NOISE_FC = 3.0          # Hz, cutoff of xi
C_TAU = 0.120           # s, EMG-like first-order lag for c(t)
THETA_MAX = np.array([0.85, 0.60, 3.0, 0.50])              # upper bounds of lam

# (US)^+ for mapping feedforward torque demand to synergy activations
US_PINV = np.linalg.pinv(U_MAX[:, None] * S)

STRATA = {
    "mild":     dict(lw=(0.05, 0.25), lr=(0.00, 0.15), ls=(0.0, 0.8), ln=(0.00, 0.10)),
    "moderate": dict(lw=(0.25, 0.50), lr=(0.15, 0.35), ls=(0.8, 1.8), ln=(0.10, 0.25)),
    "severe":   dict(lw=(0.50, 0.85), lr=(0.35, 0.60), ls=(1.8, 3.0), ln=(0.25, 0.50)),
}


def synthesize_fma(lam, rng):
    """Synthetic Fugl-Meyer proxy (0-66), SPEC 3."""
    fma = 66.0 * (1.0 - 0.50 * lam[0] - 0.20 * lam[1] - 0.067 * lam[2] - 0.20 * lam[3])
    fma += rng.normal(0.0, 1.5)
    return float(np.clip(fma, 4.0, 66.0))


class Subject:
    """Virtual stroke subject with a four-parameter injury theta_inj."""

    def __init__(self, sid, stratum, lam, perm, fma):
        self.sid = int(sid)
        self.stratum = stratum
        self.lam = np.asarray(lam, float)      # (lw, lr, ls, ln)
        self.perm = np.asarray(perm, int)      # synergy permutation Pi
        self.fma = float(fma)
        self.reset(None)

    def reset(self, rng):
        self.rng = rng
        self.c = np.zeros(4)
        self.xi = np.zeros(6)

    def step(self, c_dem, qd, dt, eta):
        """Advance one step; eta ~ N(0,I_6) raw noise draw.
        Returns (tau_patient, tau_volitional)."""
        lw, lr, ls, ln = self.lam
        # EMG-like first-order lag, saturated to amplitude 1
        self.c += (dt / C_TAU) * (np.clip(c_dem, -1.0, 1.0) - self.c)
        np.clip(self.c, -1.0, 1.0, out=self.c)
        # synergy dissolution
        ct = (1.0 - lr) * self.c + lr * self.c[self.perm]
        tau_vol = (1.0 - lw) * U_MAX * (S @ ct)
        # spasticity: velocity-dependent resistance
        tau_sp = ls * B_SP * np.tanh(qd / 0.5) * np.abs(qd)
        # co-contraction / low-frequency tremor noise (3 Hz low-pass)
        a = dt / (dt + 1.0 / (2.0 * np.pi * NOISE_FC))
        self.xi += a * (NOISE_SCALE * eta - self.xi)
        return tau_vol - tau_sp + ln * self.xi, tau_vol

    def c_demand(self, tau_ff_step):
        """Map feedforward torque demand to desired synergy activation."""
        return US_PINV @ tau_ff_step

    def spastic_damping(self, qd):
        """Jacobian of the spastic resistance w.r.t. qd (diagonal, >= 0).
        Treated implicitly by the engine for a stable dt=0.01 discretisation."""
        ls = self.lam[2]
        if ls == 0.0:
            return np.zeros(6)
        x = qd / 0.5
        return ls * B_SP * (2.0 * np.abs(qd) / np.cosh(x) ** 2 + np.abs(np.tanh(x)))


def replay_tau_patient(qd, tau_ff, lam, perm, dt):
    """Deterministic offline replay of the patient torque (SPEC 3).

    Given a recorded trajectory's joint velocities qd (N,6) and the
    reference feedforward torque tau_ff (N,6) of the same trial, replays
    the injury model exactly as Subject.step computes it online - same
    c(t) EMG-like first-order lag (driven by c_dem = US_PINV @ tau_ff,
    saturated), same synergy dissolution, same volitional and spastic
    terms - with the stochastic co-contraction noise xi(t) set to zero
    (eta = 0), so the result is a deterministic function of
    (qd, tau_ff, theta_inj).  c(0) = 0, matching Subject.reset.

    Returns (tau_patient (N,6), tau_vol (N,6), c (N,4)).
    """
    qd = np.asarray(qd, float)
    tau_ff = np.asarray(tau_ff, float)
    lam = np.asarray(lam, float)
    perm = np.asarray(perm, int)
    lw, lr, ls, ln = lam
    N = qd.shape[0]
    TP = np.zeros((N, 6))
    TV = np.zeros((N, 6))
    C = np.zeros((N, 4))
    c = np.zeros(4)
    for k in range(N):
        c_dem = US_PINV @ tau_ff[k]
        c += (dt / C_TAU) * (np.clip(c_dem, -1.0, 1.0) - c)
        np.clip(c, -1.0, 1.0, out=c)
        ct = (1.0 - lr) * c + lr * c[perm]
        tau_vol = (1.0 - lw) * U_MAX * (S @ ct)
        tau_sp = ls * B_SP * np.tanh(qd[k] / 0.5) * np.abs(qd[k])
        C[k] = c
        TV[k] = tau_vol
        TP[k] = tau_vol - tau_sp          # ln*xi term = 0 (deterministic)
    return TP, TV, C


def sample_subjects(seed=SEED):
    """60 subjects: mild/moderate/severe x 20 (SPEC 3)."""
    rng = np.random.default_rng(seed)
    subjects = []
    sid = 0
    for stratum, p in STRATA.items():
        for _ in range(20):
            lam = np.array([
                rng.uniform(*p["lw"]), rng.uniform(*p["lr"]),
                rng.uniform(*p["ls"]), rng.uniform(*p["ln"]),
            ])
            perm = rng.permutation(4)
            fma = synthesize_fma(lam, rng)
            subjects.append(Subject(sid, stratum, lam, perm, fma))
            sid += 1
    return subjects


def make_folds(subjects, n_folds=5, seed=SEED):
    """Stratified subject-level folds: 12 per fold (4/4/4). Returns fold id array."""
    rng = np.random.default_rng(seed + 1)
    fold_of = np.zeros(len(subjects), int)
    for s, stratum in enumerate(STRATA):
        idx = [i for i, sub in enumerate(subjects) if sub.stratum == stratum]
        idx = np.array(idx)[rng.permutation(len(idx))]
        for k, i in enumerate(idx):
            fold_of[i] = k % n_folds
    return fold_of
