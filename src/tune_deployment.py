"""Deployment-law tuning for IIRA (patent data collection).

The network weights are fixed (results/weights/iira_fold{k}.pt from the v1.1
run); only the deployment control-law parameters are tuned, so no retraining
is needed - just closed-loop re-evaluation with the same engine and protocol
as run_experiment.py (same refs, same seeds, same metrics).

Grid (81 configs):
    tau_c        in {0.02, 0.05, 0.10} s   residual low-pass
    kappa        in {0.5, 0.8, 1.0}        dissipative compensation gain
    kp_min_ratio in {0.1, 0.2, 0.3}        AAN stiffness floor (x Kp_max')
    kp_scale     in {0.75, 1.0, 1.25}      Kp_max scaling

Stage 1: all 81 configs on the fold-1 test subjects (12 subjects x 8 trials),
saved to results/tuning/grid_fold1.csv.
Selection (safety first): drop configs with mean viol_pct > 1%, then minimise
mean rmse_p_mm; ties broken by larger rho_dynamic (realized patient dynamic
contribution ratio, compute_rho_realized.trial_rhos: gravity support removed).
Stage 2: the selected config is re-evaluated on all 5 folds (each fold's own
weights) -> results/tuning/iira_tuned_metrics_per_fold.csv and
iira_tuned_summary.csv (9 metrics, model name "iira_tuned", same format as
results/metrics_summary.csv), plus a Wilcoxon signed-rank comparison against
the original iira rows -> results/tuning/iira_tuned_vs_iira_stats.csv.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats

import dataset as ds
import simulate as sim
import compute_rho_realized as crr
from controllers import neural as neu
import run_experiment as re

ROOT = re.ROOT
SEED = re.SEED

GRID_TAU_C = [0.02, 0.05, 0.10]
GRID_KAPPA = [0.5, 0.8, 1.0]
GRID_KPMIN = [0.1, 0.2, 0.3]
GRID_KPSCALE = [0.75, 1.0, 1.25]

METRICS = re.METRICS
VIOL_CAP = 1.0           # %, safety-first eligibility cap


def outdir():
    p = os.path.join(ROOT, "results", "tuning")
    os.makedirs(p, exist_ok=True)
    return p


def load_iira(fold, law):
    """Fold's saved IIRA-full weights wrapped in a tuned deployment law."""
    model = neu.IIRANet(bottleneck=True, theta_head=True, residual=True)
    model.load_state_dict(torch.load(
        os.path.join(ROOT, "results", "weights", f"iira_fold{fold}.pt")))
    nz = np.load(os.path.join(ROOT, "results", "weights",
                              f"iira_fold{fold}_norm.npz"))
    return neu.IIRAController(model, nz["mean"], nz["std"], name="iira_tuned",
                              aan=True, safety=True, **law)


def eval_config(subjects, test_sids, test_refs, fold, law, model_name="iira_tuned"):
    """Closed-loop eval of one deployment law; identical protocol/seeds to
    run_experiment.evaluate (8 directions x d=0.18 m, rep 0)."""
    ctrl = load_iira(fold, law)
    ctrl.name = model_name
    rows = []
    for sid in test_sids:
        subj = subjects[sid]
        for di, ri in enumerate(test_refs):
            seed = SEED + 900000 + fold * 10000 + sid * 100 + di * 10 + 0
            rec = sim.rollout(ctrl, subj, re.refs_global[ri],
                              measure_time=True, seed=seed)
            m = sim.trial_metrics(rec)
            th, rh = ctrl.estimates()
            rr, rd = crr.trial_rhos(rec["q"], rec["qd"], rec["tau_a"],
                                    re.refs_global[ri]["tau_ff"],
                                    subj.lam, subj.perm, rec["dt"])
            rows.append(dict(model=model_name, fold=fold, subject=sid,
                             stratum=subj.stratum, trial=di, rep=0, **m,
                             rho_hat=rh, rho_true=sim.rho_truth(rec),
                             rho_realized=rr, rho_dynamic=rd,
                             theta_mae=float(np.abs(th - subj.lam).mean())
                             if th is not None else np.nan))
    return rows


def summarise(df, model_name="iira_tuned"):
    """Same columns as results/metrics_summary_v2.csv (9 metrics + estimates)."""
    grp = df.groupby("model")[METRICS]
    summ = pd.concat([grp.mean().add_suffix("_mean"),
                      grp.std().add_suffix("_std")], axis=1).reset_index()
    sub = df.dropna(subset=["theta_mae"])
    r = float(sstats.pearsonr(df.rho_hat, df.rho_true)[0]) if len(df) > 2 else np.nan
    summ["theta_mae"] = sub.theta_mae.mean() if len(sub) else np.nan
    summ["rho_pearson_r"] = r
    g2 = df.groupby("model")[["rho_realized", "rho_dynamic"]]
    rsum = pd.concat([g2.mean().add_suffix("_mean"),
                      g2.std().add_suffix("_std")], axis=1).reset_index()
    return summ.merge(rsum, on="model", how="left")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-grid", action="store_true",
                    help="skip stage 1; read results/tuning/grid_fold1.csv")
    args = ap.parse_args()

    t_start = time.perf_counter()
    subjects, fold_of = ds.load_subjects(ROOT)
    dz = np.load(os.path.join(ROOT, "data", "dataset.npz"))
    test_ref_idx = [d * 5 + 2 for d in range(8)]          # d = 0.18 m
    tr_dists = {0.10, 0.18, 0.26}
    train_ref_idx = [i for i in range(40)
                     if round(float(dz["ref_d_req"][i]), 2) in tr_dists]
    re.refs_global = {i: re.ref_from_npz(dz, i)
                      for i in set(test_ref_idx) | set(train_ref_idx)}
    test_sids_of = {f: [s.sid for s in subjects if fold_of[s.sid] == f - 1]
                    for f in range(1, 6)}

    # ------------------------------------------------ stage 1: grid on fold 1
    if args.from_grid:
        grid = pd.read_csv(os.path.join(outdir(), "grid_fold1.csv"))
    else:
        rows = []
        n = 0
        total = (len(GRID_TAU_C) * len(GRID_KAPPA) * len(GRID_KPMIN)
                 * len(GRID_KPSCALE))
        for tau_c in GRID_TAU_C:
            for kappa in GRID_KAPPA:
                for kpm in GRID_KPMIN:
                    for kps in GRID_KPSCALE:
                        law = dict(tau_c=tau_c, kappa=kappa,
                                   kp_min_ratio=kpm, kp_scale=kps)
                        t0 = time.perf_counter()
                        rr = eval_config(subjects, test_sids_of[1], test_ref_idx,
                                         1, law)
                        df = pd.DataFrame(rr)
                        n += 1
                        rows.append(dict(
                            tau_c=tau_c, kappa=kappa, kp_min_ratio=kpm,
                            kp_scale=kps,
                            rmse_p_mm=df.rmse_p_mm.mean(),
                            rmse_q_deg=df.rmse_q_deg.mean(),
                            norm_jerk=df.norm_jerk.mean(),
                            tau_rms=df.tau_rms.mean(),
                            viol_pct=df.viol_pct.mean(),
                            settle_mm=df.settle_mm.mean(),
                            time_ms=df.time_ms.mean(),
                            rho_dynamic=df.rho_dynamic.mean(),   # realized ratio
                            rho_realized=df.rho_realized.mean(),
                            rho_hat_mean=df.rho_hat.mean(),
                            theta_mae=df.theta_mae.mean(),
                            eval_time_s=time.perf_counter() - t0))
                        print(f"[{n}/{total}] tau_c={tau_c} kappa={kappa} "
                              f"kpmin={kpm} kpscale={kps}: rmse "
                              f"{rows[-1]['rmse_p_mm']:.1f} mm, viol "
                              f"{rows[-1]['viol_pct']:.2f}%, rho_dyn "
                              f"{rows[-1]['rho_dynamic']:.3f} "
                              f"({rows[-1]['eval_time_s']:.0f}s)", flush=True)
        grid = pd.DataFrame(rows)
        grid.to_csv(os.path.join(outdir(), "grid_fold1.csv"), index=False)

    # safety-first selection: viol <= 1%, then min rmse_p, tie -> max rho_dynamic
    elig = grid[grid.viol_pct <= VIOL_CAP]
    if len(elig) == 0:
        print("WARNING: no config meets viol<=1%; falling back to min viol "
              "then min rmse", flush=True)
        best = grid.sort_values(["viol_pct", "rmse_p_mm"]).iloc[0]
    else:
        best = elig.sort_values(["rmse_p_mm", "rho_dynamic"],
                                ascending=[True, False]).iloc[0]
    best_law = dict(tau_c=float(best.tau_c), kappa=float(best.kappa),
                    kp_min_ratio=float(best.kp_min_ratio),
                    kp_scale=float(best.kp_scale))
    print(f"selected: {best_law} (fold-1 rmse {best.rmse_p_mm:.1f} mm, "
          f"viol {best.viol_pct:.2f}%, rho_dyn {best.rho_dynamic:.3f})",
          flush=True)

    # --------------------------------------- stage 2: all 5 folds, per-fold weights
    all_rows = []
    for fold in range(1, 6):
        all_rows += eval_config(subjects, test_sids_of[fold], test_ref_idx,
                                fold, best_law)
    tuned = pd.DataFrame(all_rows)
    tuned.to_csv(os.path.join(outdir(), "iira_tuned_metrics_per_fold.csv"),
                 index=False)
    summarise(tuned).to_csv(os.path.join(outdir(), "iira_tuned_summary.csv"),
                            index=False)

    # --------------------------- stage 3: Wilcoxon iira_tuned vs original iira
    # 9 metrics: the 7 core + rho_realized + rho_dynamic. cliff_delta > 0 always
    # means "tuned better" (lower-is-better for the 7, higher for the ratios).
    orig = pd.read_csv(os.path.join(ROOT, "results", "metrics_per_fold_v2.csv"))
    orig = orig[orig.model == "iira"]
    unit = ["fold", "subject"]
    NINE = METRICS + ["rho_realized", "rho_dynamic"]
    a = tuned.groupby(unit)[NINE].mean()
    b = orig.groupby(unit)[NINE].mean()
    a, b = a.align(b, join="inner", axis=0)
    st_rows = []
    for metric in NINE:
        higher_better = metric.startswith("rho_")
        x, y = a[metric].to_numpy(), b[metric].to_numpy()
        keep = np.abs(x - y) > 1e-12
        if keep.sum() < 5:
            W, p = np.nan, 1.0
        else:
            W, p = sstats.wilcoxon(x[keep], y[keep])
        delta = (re.cliffs_delta(x, y) if higher_better
                 else re.cliffs_delta(y, x))
        st_rows.append(dict(metric=metric, n=int(keep.sum()),
                            direction="higher" if higher_better else "lower",
                            tuned_mean=float(x.mean()), iira_mean=float(y.mean()),
                            wilcoxon_W=float(W) if np.isfinite(W) else np.nan,
                            p_raw=float(p),
                            cliff_delta=delta))  # >0: tuned better
    order = np.argsort([r["p_raw"] for r in st_rows])
    running = 0.0
    padj = np.empty(len(st_rows))
    for rank, idx in enumerate(order):
        running = max(running, (len(st_rows) - rank) * st_rows[idx]["p_raw"])
        padj[idx] = min(running, 1.0)
    for r, p_ in zip(st_rows, padj):
        r["p_holm"] = float(p_)
        r["significant"] = bool(p_ < 0.05)
    pd.DataFrame(st_rows).to_csv(
        os.path.join(outdir(), "iira_tuned_vs_iira_stats.csv"), index=False)

    print(f"TOTAL tuning runtime: {(time.perf_counter()-t_start)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
