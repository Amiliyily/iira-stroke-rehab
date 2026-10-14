"""Neural controllers: baselines 12-15, IIRA (SPEC 4), and 5 ablations.

All neural controllers share the same interface: a sliding window of W=50
feature vectors x_t = [q, qd, e, ed, tau_a] (30 dims, measurable signals only;
tau_patient is never used), normalised with the fold's train statistics.

Baselines 12-15 output a residual assist torque on top of PD + gravity with
base gains KP_MAX/KD_MAX (the same nominal gains IIRA modulates).
IIRA (SPEC 4.1, control law v1.1 - passivity-preserving revision):
temporal encoder (Conv1d k=5 ch=32 + MLP -> h in R^64)
-> synergy bottleneck z in R^4 (tanh) -> heads theta_hat (4, sigmoid x bounds)
and rho_hat (sigmoid) -> AAN gain modulation of the STIFFNESS only,
K_p(t) = K_p_min + (K_p_max - K_p_min)(1 - rho_hat), K_p_min = 0.2 K_p_max,
while damping stays at K_d^max (full damping preserves the passivity margin)
-> residual net f_psi([q,qd,e,ed,theta_hat]), 2x128 tanh x torque limits,
low-passed at tau_c = 0.05 s before entering the torque sum
-> dissipative-safe spasticity compensation
tau_comp = -kappa lam_s_hat B_sp tanh(qd_f/0.5)|qd_f| (kappa=0.8, strictly
damping, cannot become positive feedback)
-> safety layer (velocity damping beyond 2.5 rad/s, torque saturation,
spherical workspace damping at |p| > 0.85 m via J^T).
qd_f is a 20 Hz first-order low-pass of qd; the compensation Jacobian is
added to the engine's implicit damping matrix (same implicit-dissipation
policy as friction and the PD damping).
"""

import numpy as np
import torch
import torch.nn as nn

import arm6dof as arm
import injury as inj
import simulate as sim
from controllers.base import Controller

WINDOW = 50
FEAT_DIM = 30          # q, qd, e, ed, tau_a
KP_MAX = sim.KP_DEF    # K_p^max (SPEC 4.1) = neural residual base gains
KD_MAX = sim.KD_DEF    # K_d^max (v1.1: constant, never down-modulated)
KP_MIN = 0.2 * KP_MAX  # v1.1: AAN stiffness floor
TAU_C = 0.05           # v1.1: residual low-pass time constant (s)
QD_FC = 20.0           # v1.1: joint-velocity low-pass cutoff (Hz)
KAPPA = 0.8            # v1.1: dissipative compensation gain
TSCALE = 50.0          # torque normalisation used in the loss proxies

torch.set_num_threads(4)


# ------------------------------------------------------------------ models
class LSTMResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(FEAT_DIM, 64, batch_first=True)
        self.head = nn.Linear(64, 6)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1])


class GRUResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(FEAT_DIM, 64, batch_first=True)
        self.head = nn.Linear(64, 6)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1])


class TransformerResNet(nn.Module):
    """d=64, single attention head, 2 layers, window 50 (SPEC 5)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(FEAT_DIM, 64)
        self.pos = nn.Parameter(torch.randn(1, WINDOW, 64) * 0.02)
        layer = nn.TransformerEncoderLayer(64, 1, 128, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(64, 6)

    def forward(self, x):
        o = self.enc(self.proj(x) + self.pos)
        return self.head(o[:, -1])


class IIRANet(nn.Module):
    """IIRA trunk; ablation flags switch modules (SPEC 4.1/4.2)."""

    def __init__(self, bottleneck=True, theta_head=True, residual=True):
        super().__init__()
        self.bottleneck = bottleneck
        self.theta_head = theta_head
        self.residual = residual
        self.conv = nn.Conv1d(FEAT_DIM, 32, 5)
        self.fc = nn.Linear(32 * (WINDOW - 4), 64)
        z_dim = 4 if bottleneck else 64
        self.z_lin = nn.Linear(64, 4) if bottleneck else nn.Identity()
        if theta_head:
            self.theta_mlp = nn.Sequential(nn.Linear(z_dim, 32), nn.ReLU(),
                                           nn.Linear(32, 4))
        self.rho_lin = nn.Linear(z_dim, 1)
        if residual:
            in_dim = 24 + (4 if theta_head else 0)
            self.res_net = nn.Sequential(nn.Linear(in_dim, 128), nn.Tanh(),
                                         nn.Linear(128, 128), nn.Tanh(),
                                         nn.Linear(128, 6))
        self.register_buffer("theta_max", torch.tensor(inj.THETA_MAX, dtype=torch.float32))
        self.register_buffer("tau_lim", torch.tensor(arm.TAU_LIM, dtype=torch.float32))

    def forward(self, win, sub=5):
        """win: (B, 50, 30) normalised.  Returns residual at the last step,
        the residual sub-sequence (every `sub`-th step, for the smoothness
        loss), theta_hat (physical units), rho_hat, z."""
        h = torch.relu(self.fc(torch.relu(self.conv(win.transpose(1, 2))).flatten(1)))
        z = torch.tanh(self.z_lin(h)) if self.bottleneck else h
        theta = torch.sigmoid(self.theta_mlp(z)) * self.theta_max if self.theta_head else None
        rho = torch.sigmoid(self.rho_lin(z)).squeeze(-1)
        if self.residual:
            cur = win[:, ::sub, :24]
            B, Wn, _ = cur.shape
            if self.theta_head:
                rin = torch.cat([cur, theta.unsqueeze(1).expand(B, Wn, 4)], dim=-1)
            else:
                rin = cur
            res_seq = torch.tanh(self.res_net(rin)) * self.tau_lim
            res = res_seq[:, -1]
        else:
            res_seq = res = torch.zeros(win.shape[0], 6, device=win.device)
        return dict(res=res, res_seq=res_seq, theta=theta, rho=rho, z=z)


# ------------------------------------------------------------- controllers
class _WindowBase(Controller):
    is_neural = True

    def __init__(self, mean, std):
        self.mean = mean          # (30,)
        self.std = std

    def reset(self):
        self.win = []
        self.tau_prev = np.zeros(6)

    def _features(self, q, qd, q_ref, qd_ref):
        return np.concatenate([q, qd, q_ref - q, qd_ref - qd, self.tau_prev])

    def _window_tensor(self, feat):
        fn = (feat - self.mean) / self.std
        self.win.append(fn)
        if len(self.win) > WINDOW:
            self.win.pop(0)
        arr = np.array(self.win, dtype=np.float32)
        if len(arr) < WINDOW:                       # left-pad with oldest row
            arr = np.concatenate([np.repeat(arr[:1], WINDOW - len(arr), axis=0), arr])
        return torch.from_numpy(arr[None])

    def _finish(self, tau):
        self.tau_prev = np.clip(tau, -arm.TAU_LIM, arm.TAU_LIM)
        return tau


class TorchResController(_WindowBase):
    """Baselines 13-15 (LSTM/GRU/Transformer residual)."""

    def __init__(self, model, mean, std, name):
        super().__init__(mean, std)
        self.model = model.eval()
        self.name = name
        self.damping_matrix = np.diag(KD_MAX)

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        x = self._window_tensor(self._features(q, qd, q_ref, qd_ref))
        with torch.no_grad():
            res = self.model(x).numpy()[0]
        tau = KP_MAX * (q_ref - q) + KD_MAX * (qd_ref - qd) + ctx["g"] + res
        return self._finish(tau)

    def save(self, path):
        torch.save(self.model.state_dict(), path + ".pt")
        np.savez(path + "_norm.npz", mean=self.mean, std=self.std)


class MLPResController(_WindowBase):
    """Baseline 12: sklearn MLP [64,64] tanh on the current-step features."""

    def __init__(self, mlp, feat_mean, feat_std, y_mean, y_std):
        super().__init__(feat_mean, feat_std)
        self.mlp = mlp
        self.name = "mlp"
        self.y_mean = y_mean
        self.y_std = y_std
        self.damping_matrix = np.diag(KD_MAX)

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        feat = np.concatenate([q, qd, q_ref - q, qd_ref - qd])   # 24 dims
        fn = ((feat - self.mean[:24]) / self.std[:24]).astype(np.float32)
        res = self.mlp.predict(fn[None])[0] * self.y_std + self.y_mean
        tau = KP_MAX * (q_ref - q) + KD_MAX * (qd_ref - qd) + ctx["g"] + res
        return self._finish(tau)

    def save(self, path):
        np.savez(path + ".npz",
                 coefs=np.array(self.mlp.coefs_, dtype=object),
                 intercepts=np.array(self.mlp.intercepts_, dtype=object),
                 mean=self.mean, std=self.std, y_mean=self.y_mean, y_std=self.y_std)


class IIRAController(_WindowBase):
    """IIRA (SPEC 4.1, control law v1.1) with ablation switches."""

    def __init__(self, model, mean, std, name="iira", aan=True, safety=True,
                 kp_min_ratio=0.2, kp_scale=1.0, tau_c=TAU_C, kappa=KAPPA):
        super().__init__(mean, std)
        self.model = model.eval()
        self.name = name
        self.aan = aan
        self.safety = safety
        # deployment-law parameters (defaults = v1.1; tunable without retraining)
        self.kp_max = KP_MAX * kp_scale
        self.kp_min = kp_min_ratio * self.kp_max
        self.tau_c = tau_c
        self.kappa = kappa
        self.damping_matrix = np.diag(KD_MAX)

    def reset(self):
        super().reset()
        self.theta_log = []
        self.rho_log = []
        self._rho_s = 0.0
        self._theta_s = None
        self.tau_f = np.zeros(6)      # v1.1: low-passed residual torque
        self.qd_f = np.zeros(6)       # v1.1: 20 Hz low-passed joint velocity

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        dt = ctx["dt"]
        x = self._window_tensor(self._features(q, qd, q_ref, qd_ref))
        with torch.no_grad():
            out = self.model(x)
        res_raw = out["res"].numpy()[0]
        rho_raw = float(out["rho"].numpy()[0])
        theta_raw = out["theta"].numpy()[0] if out["theta"] is not None else None
        # EMA smoothing of the estimates over the trial (they are subject-level
        # quantities; per-step raw heads are noisy)
        self._rho_s = 0.9 * self._rho_s + 0.1 * rho_raw
        if theta_raw is not None:
            self._theta_s = theta_raw if self._theta_s is None \
                else 0.9 * self._theta_s + 0.1 * theta_raw
        self.rho_log.append(self._rho_s)
        if self._theta_s is not None:
            self.theta_log.append(self._theta_s)
        # v1.1: 20 Hz first-order low-pass of the measured velocity
        self.qd_f += dt / (dt + 1.0 / (2.0 * np.pi * QD_FC)) * (qd - self.qd_f)
        # v1.1: low-pass the residual torque before it enters the torque sum
        self.tau_f += dt / (self.tau_c + dt) * (res_raw - self.tau_f)
        if self.aan:
            # v1.1: AAN modulates the stiffness only (floor kp_min);
            # damping stays at K_d^max - full damping preserves passivity.
            kp = self.kp_min + (self.kp_max - self.kp_min) * (1.0 - self._rho_s)
        else:
            kp = self.kp_max
        kd = KD_MAX
        # The Jacobian of the dissipative compensation below is handled
        # implicitly (the engine's standard policy for damping-like terms).
        dvec = kd.astype(float).copy()
        if self._theta_s is not None:
            thB = self.kappa * self._theta_s[2] * inj.B_SP
            x_f = self.qd_f / 0.5
            dvec = dvec + thB * (2.0 * np.abs(self.qd_f) / np.cosh(x_f) ** 2
                                 + np.abs(np.tanh(x_f)))
        self.damping_matrix = np.diag(dvec)
        tau = kp * (q_ref - q) + kd * (qd_ref - qd) + self.tau_f + ctx["g"]
        if self._theta_s is not None:
            # v1.1: dissipative-safe spasticity term (SPEC 4.1e revised):
            # strictly a damper on the filtered velocity, so it can never
            # become positive feedback regardless of the theta_hat error.
            tau = tau - thB * np.tanh(x_f) * np.abs(self.qd_f)
        if self.safety:
            # unchanged safety layer, but evaluated on the filtered velocity
            over = np.abs(self.qd_f) > 2.5
            if np.any(over):
                tau = tau - 2.0 * (np.abs(self.qd_f) - 2.5) * np.sign(self.qd_f) * over
            pn = np.linalg.norm(ctx["p"])
            if pn > 0.85:
                F = -20.0 * (pn - 0.85) * ctx["p"] / pn
                tau = tau + ctx["J"][:3].T @ F
        return self._finish(tau)

    def estimates(self):
        th = np.array(self.theta_log).mean(axis=0) if self.theta_log else None
        return th, float(np.mean(self.rho_log)) if self.rho_log else np.nan

    def save(self, path):
        torch.save(self.model.state_dict(), path + ".pt")
        np.savez(path + "_norm.npz", mean=self.mean, std=self.std)
