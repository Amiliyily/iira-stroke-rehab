"""Task-1: offline patient-contribution ratios for every stored test trial,
merged into v2 metric tables.

    rho_realized = ||tau_patient|| / (||tau_patient|| + ||tau_a||)
    rho_dynamic  = ||tau_patient|| / (||tau_patient|| + ||tau_a - g(q)||)

per trial, with ||.|| the RMS-of-vector-norm convention of
simulate.rho_truth.  rho_dynamic removes the (passive) gravity-support
component g(q) of the assist torque before scoring -- gravity compensation
is a passive support, not "doing the task for the patient" (standard
practice in rehab-robotics; see README).  g(q) is recomputed frame by frame
from the stored q with arm6dof.rne(q, 0, 0) (identical to g_potential,
cross-checked < 1e-8 in the self-check).

tau_patient is recomputed offline with the deterministic
injury.replay_tau_patient (noise xi = 0; see its docstring), driven by the
recorded qd and the reference feedforward torque tau_ff of the same trial
(data/dataset.npz, test refs = directions x d=0.18 m).

Writes (originals are NOT overwritten; rebuilt from the v1 files so the
script is idempotent):
  results/metrics_per_fold_v2.csv  - metrics_per_fold.csv + rho_realized + rho_dynamic
  results/metrics_summary_v2.csv   - metrics_summary.csv  + both mean/std
  results/stats_v2.csv             - stats.csv + Wilcoxon/Holm rows for both ratios
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm6dof as arm
import injury as inj

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHO_METRICS = ["rho_realized", "rho_dynamic"]
ZERO6 = np.zeros(6)


def rms(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x, float) ** 2, axis=1))))


def trial_rhos(q, qd, tau_a, tau_ff, lam, perm, dt):
    """(rho_realized, rho_dynamic) for one trial."""
    tp, _, _ = inj.replay_tau_patient(qd, tau_ff, lam, perm, dt)
    g = np.array([arm.rne(q[k], ZERO6, ZERO6) for k in range(len(q))])
    denom_dyn = rms(tau_a - g)
    p = rms(tp)
    return p / (p + rms(tau_a)), p / (p + denom_dyn)


def stats_rows(df5, metric):
    """Wilcoxon IIRA vs each model on `metric`, Holm-corrected within metric.
    Cliff's delta: positive => baseline rho HIGHER than IIRA."""
    subj_mean = (df5.groupby(["model", "fold", "subject"])[metric]
                 .mean().reset_index())
    iira = subj_mean[subj_mean.model == "iira"].set_index(["fold", "subject"])
    ps = []
    for m in sorted(subj_mean.model.unique()):
        if m == "iira":
            continue
        v = subj_mean[subj_mean.model == m].set_index(["fold", "subject"])
        a = iira[metric].loc[v.index].to_numpy()
        b = v[metric].to_numpy()
        keep = np.abs(a - b) > 1e-12
        if keep.sum() < 5:
            W, p = np.nan, 1.0
        else:
            W, p = sstats.wilcoxon(a[keep], b[keep])
        gt = sum(np.sum(bi > a) for bi in b)
        lt = sum(np.sum(bi < a) for bi in b)
        ps.append(dict(metric=metric, baseline=m, n=int(keep.sum()),
                       wilcoxon_W=float(W) if np.isfinite(W) else np.nan,
                       p_raw=float(p),
                       cliff_delta=float((gt - lt) / (len(a) * len(b)))))
    order = np.argsort([r["p_raw"] for r in ps])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[idx]["p_raw"])
        ps[idx]["p_holm"] = float(min(running, 1.0))
    for r in ps:
        r["significant"] = bool(r["p_holm"] < 0.05)
    return ps


def main():
    subj = pd.read_csv(os.path.join(ROOT, "data", "subjects.csv"))
    lam_of = {r.subject: np.array([r.lambda_w, r.lambda_r, r.lambda_s, r.lambda_n])
              for r in subj.itertuples()}
    perm_of = {r.subject: np.array([int(v) for v in str(r.perm).split()])
               for r in subj.itertuples()}
    dz = np.load(os.path.join(ROOT, "data", "dataset.npz"))
    tau_ff_all = dz["ref_tau_ff"].astype(float)     # (40, 201, 6)
    dt = float(dz["ref_t"][1] - dz["ref_t"][0])

    pf = pd.read_csv(os.path.join(ROOT, "results", "metrics_per_fold.csv"))

    rho_rows = {}
    tdir = os.path.join(ROOT, "results", "trajectories")
    models = sorted(pf.model.unique())
    for mi, model in enumerate(models):
        for fold in range(1, 6):
            path = os.path.join(tdir, f"{model}_fold{fold}.npz")
            z = np.load(path)
            for i in range(len(z["subject"])):
                sid = int(z["subject"][i])
                di = int(z["direction"][i])
                rep = int(z["rep"][i])
                tau_ff = tau_ff_all[di * 5 + 2]     # test refs: d=0.18 m
                rr, rd = trial_rhos(z["q"][i], z["qd"][i], z["tau_a"][i],
                                    tau_ff, lam_of[sid], perm_of[sid], dt)
                rho_rows[(model, fold, sid, di, rep)] = (rr, rd)
        print(f"[{mi + 1}/{len(models)}] {model} done", flush=True)

    key = list(zip(pf.model, pf.fold, pf.subject, pf.trial, pf.rep))
    pf["rho_realized"] = [rho_rows[k][0] for k in key]
    pf["rho_dynamic"] = [rho_rows[k][1] for k in key]
    pf.to_csv(os.path.join(ROOT, "results", "metrics_per_fold_v2.csv"),
              index=False)

    # summary: same aggregation convention as metrics_summary.csv
    df5 = pf[(pf.model != "ilc") | (pf.rep == 4)]
    grp = df5.groupby("model")[RHO_METRICS]
    rho_summ = pd.concat([grp.mean().add_suffix("_mean"),
                          grp.std().add_suffix("_std")], axis=1).reset_index()
    summ = pd.read_csv(os.path.join(ROOT, "results", "metrics_summary.csv"))
    summ = summ.merge(rho_summ, on="model", how="left")
    summ.to_csv(os.path.join(ROOT, "results", "metrics_summary_v2.csv"),
                index=False)

    # stats: Wilcoxon IIRA vs each model on both ratios, Holm-corrected
    stats = pd.read_csv(os.path.join(ROOT, "results", "stats.csv"))
    extra = []
    for metric in RHO_METRICS:
        extra += stats_rows(df5, metric)
    stats = pd.concat([stats, pd.DataFrame(extra)], ignore_index=True)
    stats.to_csv(os.path.join(ROOT, "results", "stats_v2.csv"), index=False)

    for metric in RHO_METRICS:
        print(f"\n{metric} summary (mean over trials):", flush=True)
        print(rho_summ[["model", metric + "_mean", metric + "_std"]]
              .sort_values(metric + "_mean", ascending=False)
              .to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
