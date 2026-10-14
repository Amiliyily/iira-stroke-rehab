"""Tiny end-to-end smoke test of the run_experiment pipeline (not the full run)."""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiment as rx
import simulate as sim
import dataset as ds
import injury as inj
from controllers import classical as cl
from controllers import neural as neu

rx.EPOCHS = 2
rx.BATCH = 256

subjects, fold_of = ds.load_subjects(rx.ROOT)
dz = np.load(os.path.join(rx.ROOT, "data", "dataset.npz"))
rx.refs_global = {i: rx.ref_from_npz(dz, i) for i in [0, 2, 10, 12, 20, 22]}

sids = [0, 1, 20, 21, 40, 41]
data = rx.collect_fold_data(subjects, rx.refs_global, sids, fold=9,
                            train_refs=[0, 10, 20])
print("data:", data["X"].shape, "rho range", data["RHO"].min(), data["RHO"].max())
tr_mask = data["SID"] != 41
va_mask = ~tr_mask

ctrls, (mu, sd) = rx.build_neural_controllers(
    data, tr_mask, va_mask, fold=9, w2=0.02, w3=0.05)
print("trained:", list(ctrls))

controllers = {"pid": cl.PIDController(), "smc": cl.SMCController()}
controllers.update({k: ctrls[k] for k in ("lstm", "iira", "iira_nosafety")})
rows, trajs = rx.evaluate(controllers, subjects, [1, 41], [2], fold=9)
df = pd.DataFrame(rows)
print(df[["model", "subject", "rmse_p_mm", "tau_rms", "time_ms", "viol_pct"]].to_string())
print(df.columns.tolist())
print("SMOKE OK")
