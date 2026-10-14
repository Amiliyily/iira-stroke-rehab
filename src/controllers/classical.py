"""Classical baseline controllers 1-11 (SPEC 5), all default/literature configs."""

import numpy as np

import arm6dof as arm
from controllers.base import Controller

KP = np.array([40.0, 40.0, 40.0, 8.0, 8.0, 8.0])
KI = 0.5 * KP * 0.1
KD = np.array([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])


def _fric(qd):
    return arm.FV * qd + arm.FC * np.tanh(qd / 0.01)


class PIDController(Controller):
    name = "pid"
    damping_matrix = np.diag(KD)

    def reset(self):
        self.eint = np.zeros(6)

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        e = q_ref - q
        ed = qd_ref - qd
        self.eint = np.clip(self.eint + e * ctx["dt"], -5.0, 5.0)
        return KP * e + KI * self.eint + KD * ed

    def save(self, path):
        np.savez(path, Kp=KP, Ki=KI, Kd=KD)


class PDGravityController(Controller):
    name = "pd_grav"
    damping_matrix = np.diag(KD)

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        return KP * (q_ref - q) + KD * (qd_ref - qd) + ctx["g"]

    def save(self, path):
        np.savez(path, Kp=KP, Kd=KD)


class ComputedTorqueController(Controller):
    name = "computed_torque"

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        e = q_ref - q
        ed = qd_ref - qd
        self.damping_matrix = ctx["M"] @ np.diag(KD)
        return ctx["M"] @ (qdd_ref + KD * ed + KP * e) + ctx["bias"] + _fric(qd)

    def save(self, path):
        np.savez(path, Kp=KP, Kd=KD, model="nominal exact")


class ImpedanceController(Controller):
    """Hogan 1985, joint space, M_d = I, damping = 2 sqrt(Kp), stiffness = Kp."""
    name = "impedance"
    KD_IMP = 2.0 * np.sqrt(KP)

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        e = q_ref - q
        ed = qd_ref - qd
        self.damping_matrix = ctx["M"] @ np.diag(self.KD_IMP)
        return ctx["M"] @ (qdd_ref + self.KD_IMP * ed + KP * e) + ctx["bias"]

    def save(self, path):
        np.savez(path, Kp=KP, Kd=self.KD_IMP, Md=np.ones(6))


class AdmittanceController(Controller):
    """Outer joint-space admittance (M_a=2I, B_a=10I, K_a=50I) driven by the
    measured interaction torque (tau_patient sensor analogue), inner PD+g."""
    name = "admittance"
    MA, BA, KA = 2.0, 10.0, 50.0

    def reset(self):
        self.x = np.zeros(6)
        self.xd = np.zeros(6)

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        xdd = (ctx["tau_patient"] - self.BA * self.xd - self.KA * self.x) / self.MA
        self.xd = self.xd + xdd * ctx["dt"]
        self.x = self.x + self.xd * ctx["dt"]
        self.x = np.clip(self.x, -0.5, 0.5)
        self.damping_matrix = np.diag(KD)
        return KP * (q_ref + self.x - q) + KD * (qd_ref + self.xd - qd) + ctx["g"]

    def save(self, path):
        np.savez(path, Ma=self.MA, Ba=self.BA, Ka=self.KA, Kp=KP, Kd=KD)


class SMCController(Controller):
    """Sliding-mode, switching gain 15, boundary layer 0.05, Lambda = 5 I."""
    name = "smc"
    ETA = 15.0
    PHI = 0.05
    LAM = 5.0

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        e = q_ref - q
        ed = qd_ref - qd
        s = ed + self.LAM * e
        sat = np.clip(s / self.PHI, -1.0, 1.0)
        bl = (np.abs(s) < self.PHI).astype(float)
        self.damping_matrix = ctx["M"] * self.LAM + (self.ETA / self.PHI) * np.diag(bl)
        return ctx["M"] @ (qdd_ref + self.LAM * ed) + ctx["bias"] + self.ETA * sat

    def save(self, path):
        np.savez(path, eta=self.ETA, phi=self.PHI, lam=self.LAM)


def _linearize_at_q0():
    """Discrete linearisation (Ad, Bd) of the actual engine one-step map
    (semi-implicit Euler with implicit friction, dt = 0.01) about
    x0 = (q0, 0), u0 = g(q0), no patient torque.  Linearising the discrete
    engine map directly keeps the stiff friction modes consistent with the
    simulated plant (a continuous-time linearisation + Euler discretisation
    aliases the ~1e4 rad/s friction pole of the feather-weight wrist)."""
    q0 = arm.Q0_HOME
    dt = 0.01
    x0 = np.concatenate([q0, np.zeros(6)])
    _, _, g0, _, _ = arm.engine_dynamics(q0, np.zeros(6))
    u0 = g0.copy()

    def plant(x, u):
        q, qd = x[:6], x[6:]
        M, bias, g, _, _ = arm.engine_dynamics(q, qd)
        Dfr = arm.FV + arm.FC * (1.0 / np.cosh(qd / 0.01) ** 2) / 0.01
        fric = arm.FV * qd + arm.FC * np.tanh(qd / 0.01)
        qdd = np.linalg.solve(M + dt * np.diag(Dfr), u - bias - fric)
        qd_n = qd + dt * qdd
        return np.concatenate([q + dt * qd_n, qd_n])

    h = 1e-6
    Ad = np.zeros((12, 12))
    Bd = np.zeros((12, 6))
    for j in range(12):
        dx = np.zeros(12)
        dx[j] = h
        Ad[:, j] = (plant(x0 + dx, u0) - plant(x0 - dx, u0)) / (2 * h)
    for j in range(6):
        du = np.zeros(6)
        du[j] = h
        Bd[:, j] = (plant(x0, u0 + du) - plant(x0, u0 - du)) / (2 * h)
    return Ad, Bd


class LQRController(Controller):
    """LQR on the discrete engine-map linearisation at q0,
    Q = diag(10 I6, I6), R = 0.1 I6."""
    name = "lqr"
    _K = None

    @classmethod
    def gain(cls):
        if cls._K is None:
            from scipy.linalg import solve_discrete_are
            Ad, Bd = _linearize_at_q0()
            Q = np.diag([10.0] * 6 + [1.0] * 6)
            R = 0.1 * np.eye(6)
            P = solve_discrete_are(Ad, Bd, Q, R)
            cls._K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        return cls._K

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        x_err = np.concatenate([q - q_ref, qd - qd_ref])
        self.damping_matrix = self.gain()[:, 6:]
        return ctx["g"] - self.gain() @ x_err

    def save(self, path):
        np.savez(path, K=self.gain())


class MPCController(Controller):
    """Unconstrained linear MPC, horizon 10, Q/R as LQR, analytic first move."""
    name = "mpc"
    N = 10
    _F = None

    @classmethod
    def gain(cls):
        if cls._F is None:
            Ad, Bd = _linearize_at_q0()
            n, m, N = 12, 6, cls.N
            Sx = np.zeros((N * n, n))
            Su = np.zeros((N * n, N * m))
            Apow = np.eye(n)
            for i in range(N):
                Apow = Ad @ Apow
                Sx[i * n:(i + 1) * n] = Apow
                Arow = np.eye(n)
                for j in range(i + 1):
                    Arow = np.linalg.matrix_power(Ad, i - j)
                    Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = Arow @ Bd
            Qbar = np.kron(np.eye(N), np.diag([10.0] * 6 + [1.0] * 6))
            # terminal cost from the discrete ARE (standard finite-horizon LQR)
            from scipy.linalg import solve_discrete_are
            P_term = solve_discrete_are(Ad, Bd, np.diag([10.0] * 6 + [1.0] * 6),
                                        0.1 * np.eye(6))
            Qbar[(N - 1) * n:N * n, (N - 1) * n:N * n] = P_term
            Rbar = np.kron(np.eye(N), 0.1 * np.eye(m))
            H = Su.T @ Qbar @ Su + Rbar
            G = Su.T @ Qbar @ Sx
            cls._F = -np.linalg.solve(H, G)[:m]
        return cls._F

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        x_err = np.concatenate([q - q_ref, qd - qd_ref])
        self.damping_matrix = -self.gain()[:, 6:]
        return ctx["g"] + self.gain() @ x_err

    def save(self, path):
        np.savez(path, F=self.gain(), horizon=self.N)


class DOBController(Controller):
    """Disturbance observer + PD, first-order Q filter, bandwidth 20 rad/s."""
    name = "dob"
    WC = 20.0
    damping_matrix = np.diag(KD)

    def reset(self):
        self.dhat = np.zeros(6)
        self.prev = None

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        if self.prev is not None:
            qd_p, tau_p, M_p, b_p = self.prev
            qdd_est = (qd - qd_p) / ctx["dt"]
            fric_p = arm.FV * qd_p + arm.FC * np.tanh(qd_p / 0.01)
            d_tilde = tau_p - (M_p @ qdd_est + b_p + fric_p)
            self.dhat += ctx["dt"] * self.WC * (d_tilde - self.dhat)
        self.prev = (qd.copy(),
                     KP * (q_ref - q) + KD * (qd_ref - qd) + ctx["g"] - self.dhat,
                     ctx["M"].copy(), ctx["bias"].copy())
        return self.prev[1]

    def save(self, path):
        np.savez(path, wc=self.WC, Kp=KP, Kd=KD)


class ILCController(Controller):
    """P-type ILC over repeated trials, learning rate 0.3, per (subject, dir)."""
    name = "ilc"
    LR = 0.3
    damping_matrix = np.diag(KD)

    def __init__(self):
        self.ff = {}          # trial_key -> (N,6) feedforward
        self.trial_key = None
        self._fb = []

    def set_trial_key(self, key):
        self.trial_key = key

    def reset(self):
        self._fb = []
        if self.trial_key not in self.ff:
            self.ff[self.trial_key] = None

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        fb = KP * (q_ref - q) + KD * (qd_ref - qd)
        self._fb.append(fb)
        ff = self.ff[self.trial_key]
        u = fb if ff is None else fb + ff[ctx["step"]]
        return u + ctx["g"]

    def on_trial_end(self, rec):
        if self.trial_key is None or not self._fb:
            return
        fb = np.array(self._fb)
        ff = self.ff[self.trial_key]
        new = self.LR * fb if ff is None else ff[:len(fb)] + self.LR * fb
        self.ff[self.trial_key] = np.clip(new, -arm.TAU_LIM, arm.TAU_LIM)

    def save(self, path):
        keys = list(self.ff.keys())
        arrs = {f"k{i}": self.ff[k] for i, k in enumerate(keys) if self.ff[k] is not None}
        np.savez(path, keys=np.array([str(k) for k in keys]), **arrs)


class RBFAdaptiveController(Controller):
    """Slotine-style RBF adaptation: 25 Gaussian bases per joint on (e, ed),
    width 1.0, adaptation rate 5, composite error s = ed + 5 e."""
    name = "rbf"
    damping_matrix = np.diag(KD)
    GAMMA = 5.0
    LAM = 5.0
    WIDTH = 1.0

    def __init__(self):
        ce = np.linspace(-1.0, 1.0, 5)
        cv = np.linspace(-2.0, 2.0, 5)
        self.centers = np.array([(a, b) for a in ce for b in cv])  # (25,2)
        self.W = np.zeros((6, 25))

    def reset(self):
        self.W = np.zeros((6, 25))   # cold start each trial (fair protocol)

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        e = q_ref - q
        ed = qd_ref - qd
        s = ed + self.LAM * e
        tau = KP * e + KD * ed + ctx["g"].copy()
        for j in range(6):
            d2 = (self.centers[:, 0] - e[j]) ** 2 + (self.centers[:, 1] - ed[j]) ** 2
            phi = np.exp(-d2 / (2 * self.WIDTH ** 2))
            tau[j] += self.W[j] @ phi
            self.W[j] += self.GAMMA * phi * s[j] * ctx["dt"]
        return tau

    def save(self, path):
        np.savez(path, W=self.W, centers=self.centers, gamma=self.GAMMA)


ALL_CLASSICAL = [PIDController, PDGravityController, ComputedTorqueController,
                 ImpedanceController, AdmittanceController, SMCController,
                 LQRController, MPCController, DOBController, ILCController,
                 RBFAdaptiveController]
