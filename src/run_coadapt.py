"""Closed-loop adaptive patient experiment (SPEC appendix B / ROADMAP item 3).

Protocol mirrors the main experiment (subject-level 5-fold; test refs = 8
directions x d=0.18 m; same engine, same seeds discipline), with the v1
feedforward patient replaced by the v2.0 ClosedLoopPatient whose slow state
(effort m, fatigue F, feedback gains K_h) evolves across trials and sessions:
each test subject runs 3 sessions x 8 trials per controller.

Controller subset: pid, impedance, smc (classical), mlp, lstm, iira (v1.1
law), iira_noaan, iira_tuned (tuned deployment law).  Neural models use the
saved per-fold weights WITHOUT retraining - the closed-loop patient shifts
the input distribution; that zero-shot shift is itself a research point and
is reported as such.

Writes (results/coadapt/; nothing else is touched, no trajectories except one
representative trial per controller):
  trials_long.csv    - one row per (controller, fold, subject, session, trial)
  summary.csv        - per-controller aggregates incl. slacking/learning slopes
  stats.csv          - hypothesis tests (slacking, learning; IIRA vs SMC/PID)
  config.json        - patient parameters, tuned law, protocol, runtime
  traj_{model}.npz   - one representative trial per controller (for plotting)
"""

import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats

import dataset as ds
import simulate as sim
import patient_closed_loop as pcl
import run_experiment as re
from controllers import classical as cl
from controllers import neural as neu

ROOT = re.ROOT
SEED = re.SEED
N_SESSIONS = 3
N_DIRS = 8

CONTROLLERS = ["pid", "impedance", "smc", "mlp", "lstm",
               "iira", "iira_noaan", "iira_tuned"]
TUNED_LAW = dict(tau_c=0.02, kappa=0.5, kp_min_ratio=0.3, kp_scale=1.25)


class _MLPShim:
    """Pure-numpy tanh-MLP forward pass matching sklearn MLPRegressor
    (activation='tanh', identity output) for the saved coefs/intercepts."""

    def __init__(self, coefs, intercepts):
        self.coefs_ = list(coefs)
        self.intercepts_ = list(intercepts)

    def predict(self, X):
        h = X
        for W, b in zip(self.coefs_[:-1], self.intercepts_[:-1]):
            h = np.tanh(h @ W + b)
        return h @ self.coefs_[-1] + self.intercepts_[-1]


def load_controller(name, fold):
    """Per-fold controller with the saved weights (classical: fresh instance)."""
    w = os.path.join(ROOT, "results", "weights")
    if name in ("pid", "impedance", "smc"):
        return {c.name: c for c in (cls() for cls in cl.ALL_CLASSICAL)}[name]
    if name == "mlp":
        z = np.load(os.path.join(w, f"mlp_fold{fold}.npz"), allow_pickle=True)
        mlp = _MLPShim(z["coefs"], z["intercepts"])
        return neu.MLPResController(mlp, z["mean"], z["std"],
                                    z["y_mean"], z["y_std"])
    if name == "lstm":
        model = neu.LSTMResNet()
        model.load_state_dict(torch.load(os.path.join(w, f"lstm_fold{fold}.pt")))
        nz = np.load(os.path.join(w, f"lstm_fold{fold}_norm.npz"))
        return neu.TorchResController(model, nz["mean"], nz["std"], "lstm")
    model = neu.IIRANet(bottleneck=True, theta_head=True, residual=True)
    weight_name = 'iira_noaan' if name == 'iira_noaan' else 'iira'
    model.load_state_dict(torch.load(os.path.join(w, f"{weight_name}_fold{fold}.pt")))
    nz = np.load(os.path.join(w, f"{weight_name}_fold{fold}_norm.npz"))
    if name == "iira":
        return neu.IIRAController(model, nz["mean"], nz["std"], name="iira")
    if name == "iira_noaan":
        return neu.IIRAController(model, nz["mean"], nz["std"],
                                  name="iira_noaan", aan=False)
    if name == "iira_tuned":
        c = neu.IIRAController(model, nz["mean"], nz["std"],
                               name="iira_tuned", **TUNED_LAW)
        return c
    raise ValueError(name)



if __name__ == "__main__":
    from run_coadapt_v2 import main
    main()
