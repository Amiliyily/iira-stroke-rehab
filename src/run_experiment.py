"""5-fold subject-level CV experiment + IIRA loss-weight grid search (SPEC 6/4.2).

Usage:
  python run_experiment.py            # full run
  python run_experiment.py --probe    # fold-1 data + 2-epoch timing probe

Reductions vs SPEC defaults (recorded in config_snapshot.json; identical for
all models): 40 -> 24 training rollouts per subject (8 directions x 3 speeds),
60 -> 30 epochs, windows subsampled every WINDOW_STRIDE steps.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats

import arm6dof as arm
import injury as inj
import simulate as sim
import dataset as ds
from controllers.base import Controller
from controllers import classical as cl
from controllers import neural as neu

SEED = 20261011
N_FOLDS = 5
TRAIN_DISTS = [0.10, 0.18, 0.26]      # 8 dir x 3 speeds = 24 rollouts/subject
EPOCHS = 20
BATCH = 512
LR = 1e-3
PATIENCE = 8
WINDOW_STRIDE = 30                    # 6 windows per 201-step trial
N_VAL_SUBJECTS = 10
GRID_W2 = [0.01, 0.02, 0.05]
GRID_W3 = [0.02, 0.05, 0.10]
W_FIXED = dict(w1=1.0, w4=0.5, mu1=1.0, mu2=0.5)

METRICS = ["rmse_p_mm", "rmse_q_deg", "norm_jerk", "tau_rms", "viol_pct",
           "settle_mm", "time_ms"]

IIRA_VARIANTS = {
    "iira":            dict(bottleneck=True,  theta_head=True,  residual=True,  aan=True,  safety=True),
    "iira_nobottleneck": dict(bottleneck=False, theta_head=True,  residual=True,  aan=True,  safety=True),
    "iira_noaan":      dict(bottleneck=True,  theta_head=True,  residual=True,  aan=False, safety=True),
    "iira_noresidual": dict(bottleneck=True,  theta_head=True,  residual=False, aan=True,  safety=True),
    "iira_nosafety":   dict(bottleneck=True,  theta_head=True,  residual=True,  aan=True,  safety=False),
    "iira_notheta":    dict(bottleneck=True,  theta_head=False, residual=True,  aan=True,  safety=True),
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def outdir(*parts):
    p = os.path.join(ROOT, *parts)
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------- behavior policy
class BehaviorPD(Controller):
    """Gain-scheduled PD + gravity with exploration noise (SPEC 4.3)."""

    name = "behavior_pd"

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.gs = rng.uniform(0.6, 1.3, 6)
        self.rng = np.random.default_rng(seed + 7)
        self.nstate = np.zeros(6)
        self.nsig = np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])
        # base gains = the neural controllers' nominal gains (K_max), so the
        # behaviour data live in the same closed-loop regime as evaluation
        self.damping_matrix = np.diag(sim.KD_DEF * self.gs)

    def reset(self):
        self.nstate = np.zeros(6)

    def step(self, t, q, qd, q_ref, qd_ref, qdd_ref, ctx):
        a = ctx["dt"] / (ctx["dt"] + 1.0 / (2 * np.pi * 5.0))
        self.nstate += a * (self.nsig * self.rng.standard_normal(6) - self.nstate)
        return (ctx["g"] + sim.KP_DEF * self.gs * (q_ref - q)
                + sim.KD_DEF * self.gs * (qd_ref - qd) + self.nstate)


# ---------------------------------------------------------- data collection
def collect_fold_data(subjects, refs, train_sids, fold, train_refs=None):
    """Closed-loop rollouts with the behavior policy; supervised targets."""
    if train_refs is None:
        dz = np.load(os.path.join(ROOT, "data", "dataset.npz"))
        tr_dists = {0.10, 0.18, 0.26}
        train_refs = [i for i in range(40)
                      if round(float(dz["ref_d_req"][i]), 2) in tr_dists]
    t_idx = np.arange(neu.WINDOW - 1, 201, WINDOW_STRIDE)
    X, R, G, TH, RHO, SID = [], [], [], [], [], []
    for sid in train_sids:
        subj = subjects[sid]
        for ri in train_refs:
            ref = refs[ri]
            pol = BehaviorPD(SEED + fold * 100000 + sid * 100 + ri)
            rec = sim.rollout(pol, subj, ref, collect_targets=True,
                              seed=SEED + fold * 100000 + sid * 100 + ri)
            e = rec["q_ref"] - rec["q"]
            ed = rec["qd_ref"] - rec["qd"]
            X.append(np.concatenate([rec["q"], rec["qd"], e, ed, rec["tau_a"]],
                                    axis=1))
            R.append(rec["r_target"])
            G.append(rec["g"])
            TH.append(subj.lam)
            RHO.append(sim.rho_truth(rec))
            SID.append(sid)
    return dict(X=np.stack(X).astype(np.float32), R=np.stack(R).astype(np.float32),
                G=np.stack(G).astype(np.float32),
                TH=np.stack(TH).astype(np.float32),
                RHO=np.array(RHO, np.float32), SID=np.array(SID, int),
                t_idx=t_idx)


def norm_stats(data, train_mask):
    X = data["X"][train_mask]
    mu = X.reshape(-1, 30).mean(axis=0)
    sd = X.reshape(-1, 30).std(axis=0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def stratified_val_split(train_sids, subjects, fold):
    rng = np.random.default_rng(SEED + 31 + fold)
    val = []
    counts = {"mild": 3, "moderate": 4, "severe": 3}     # 10 of 48, stratified
    for stratum in inj.STRATA:
        ids = [s for s in train_sids if subjects[s].stratum == stratum]
        ids = list(np.array(ids)[rng.permutation(len(ids))])
        val += ids[:counts[stratum]]
    val = sorted(val)
    tr = [s for s in train_sids if s not in val]
    return tr, val


# ------------------------------------------------------------------ batches
class WindowSampler:
    """Index pairs (trial, t) gathered from the rollout arrays."""

    def __init__(self, data, mask):
        self.data = data
        self.trials = np.where(mask)[0]
        self.pairs = np.array([(tr, t) for tr in self.trials
                               for t in data["t_idx"]], dtype=np.int64)

    def __len__(self):
        return len(self.pairs)

    def batch(self, idx):
        d = self.data
        tr = self.pairs[idx, 0]
        t = self.pairs[idx, 1]
        steps = t[:, None] + np.arange(-(neu.WINDOW - 1), 1)[None, :]
        win = d["X"][tr[:, None], steps]                    # (B,50,30) raw
        return dict(win_raw=win,
                    r=d["R"][tr, t], g=d["G"][tr, t],
                    theta=d["TH"][tr], rho=d["RHO"][tr])


def to_torch(b, mu, sd):
    raw = torch.from_numpy(b["win_raw"])
    win = (raw - mu) / sd
    return dict(win=win, win_raw=raw,
                r=torch.from_numpy(b["r"]), g=torch.from_numpy(b["g"]),
                theta=torch.from_numpy(b["theta"]), rho=torch.from_numpy(b["rho"]))


# -------------------------------------------------------------------- losses
def res_loss_fn(model, bt):
    pred = model(bt["win"])
    return (pred - bt["r"]).pow(2).mean(), {}


def iira_loss_fn(w1, w2, w3, w4, mu1, mu2):
    def loss(model, bt):
        out = model(bt["win"])
        res = out["res"]
        L = {}
        if model.residual:
            L["res"] = w1 * (res - bt["r"]).pow(2).mean()
            seq = out["res_seq"]
            jerk = (seq[:, 2:] - 2 * seq[:, 1:-1] + seq[:, :-2]) / (0.05 ** 2 * neu.TSCALE)
            L["jerk"] = w2 * jerk.pow(2).mean()
            e = bt["win_raw"][:, -1, 12:18]
            ed = bt["win_raw"][:, -1, 18:24]
            tau_est = (torch.from_numpy(neu.KP_MAX).float() * e
                       + torch.from_numpy(neu.KD_MAX).float() * ed
                       + bt["g"] + res)
            L["tau"] = w3 * (tau_est / neu.TSCALE).pow(2).mean()
            lim = torch.from_numpy(arm.TAU_LIM).float()
            L["safe"] = w4 * torch.relu(tau_est.abs() / lim - 1.0).mean()
        if model.theta_head:
            tmax = torch.from_numpy(inj.THETA_MAX).float()
            L["theta"] = mu1 * ((out["theta"] - bt["theta"]) / tmax).pow(2).mean()
        L["rho"] = mu2 * torch.nn.functional.binary_cross_entropy(
            out["rho"], bt["rho"])
        total = sum(L.values())
        return total, {k: float(v) for k, v in L.items()}
    return loss


# ------------------------------------------------------------------ trainer
def train_torch(model, loss_fn, data, tr_mask, va_mask, mu, sd, log_path,
                seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tr = WindowSampler(data, tr_mask)
    va = WindowSampler(data, va_mask)
    mut, sdt = torch.from_numpy(mu), torch.from_numpy(sd)
    best_val, best_state, bad = np.inf, None, 0
    rows = []
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(len(tr))
        tl, nb = 0.0, 0
        for i in range(0, len(perm), BATCH):
            bt = to_torch(tr.batch(perm[i:i + BATCH]), mut, sdt)
            opt.zero_grad()
            loss, _ = loss_fn(model, bt)
            loss.backward()
            opt.step()
            tl += float(loss)
            nb += 1
        model.eval()
        vl, nb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va), BATCH):
                bt = to_torch(va.batch(np.arange(i, min(i + BATCH, len(va)))), mut, sdt)
                loss, parts = loss_fn(model, bt)
                vl += float(loss)
                nb += 1
        tl /= max(nb, 1)
        vl /= max(nb, 1)
        rows.append(dict(epoch=ep, train_loss=tl, val_loss=vl))
        if vl < best_val - 1e-5:
            best_val, best_state, bad = vl, {k: v.clone() for k, v in
                                             model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(rows).to_csv(log_path, index=False)
    return model, rows


def train_mlp(data, tr_mask, va_mask, mu, sd, log_path, seed):
    from sklearn.neural_network import MLPRegressor
    d = data
    tr = WindowSampler(d, tr_mask)
    va = WindowSampler(d, va_mask)
    Xtr = d["X"][tr.pairs[:, 0], tr.pairs[:, 1], :24]
    ytr = d["R"][tr.pairs[:, 0], tr.pairs[:, 1]]
    Xn = (Xtr - mu[:24]) / sd[:24]
    y_mu, y_sd = ytr.mean(axis=0), ytr.std(axis=0) + 1e-6
    yn = (ytr - y_mu) / y_sd
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                       solver="adam", learning_rate_init=LR, batch_size=256,
                       max_iter=EPOCHS, early_stopping=True,
                       validation_fraction=0.15, n_iter_no_change=PATIENCE,
                       random_state=seed)
    mlp.fit(Xn, yn)
    rows = [dict(epoch=i, train_loss=l,
                 val_loss=(mlp.validation_scores_[i] if hasattr(mlp, "validation_scores_")
                           and i < len(mlp.validation_scores_) else np.nan))
            for i, l in enumerate(mlp.loss_curve_)]
    pd.DataFrame(rows).to_csv(log_path, index=False)
    return mlp, y_mu.astype(np.float32), y_sd.astype(np.float32)


# ---------------------------------------------------------------- evaluation
def ref_from_npz(refs_npz, i):
    return dict(t=refs_npz["ref_t"], p_ref=refs_npz["ref_p"][i].astype(float),
                q_ref=refs_npz["ref_q"][i].astype(float),
                qd_ref=refs_npz["ref_qd"][i].astype(float),
                qdd_ref=refs_npz["ref_qdd"][i].astype(float),
                tau_ff=refs_npz["ref_tau_ff"][i].astype(float),
                d_actual=float(refs_npz["ref_d_actual"][i]), T=2.0, dt=0.01)


def evaluate(controllers, subjects, test_sids, test_refs, fold):
    """Run the eval protocol; returns (metric rows, trajectory store)."""
    rows = []
    trajs = {}
    for name, ctrl in controllers.items():
        store = {k: [] for k in ("q", "qd", "p", "p_ref", "tau_a")}
        meta = {"subject": [], "direction": [], "rep": []}
        for sid in test_sids:
            subj = subjects[sid]
            for di, ri in enumerate(test_refs):
                ref = refs_global[ri]
                reps = range(5) if name == "ilc" else [0]
                if name == "ilc":
                    ctrl.set_trial_key((sid, di))
                for rep in reps:
                    seed = SEED + 900000 + fold * 10000 + sid * 100 + di * 10 + rep
                    rec = sim.rollout(ctrl, subj, ref, measure_time=True, seed=seed)
                    m = sim.trial_metrics(rec)
                    row = dict(model=name, fold=fold, subject=sid,
                               stratum=subj.stratum, trial=di, rep=rep, **m)
                    if isinstance(ctrl, neu.IIRAController):
                        th, rh = ctrl.estimates()
                        row["rho_hat"] = rh
                        row["rho_true"] = sim.rho_truth(rec)
                        if th is not None:
                            row["theta_mae"] = float(np.abs(th - subj.lam).mean())
                    rows.append(row)
                    for k in store:
                        store[k].append(rec[k if k in rec else k])
                    meta["subject"].append(sid)
                    meta["direction"].append(di)
                    meta["rep"].append(rep)
        trajs[name] = (store, meta)
    return rows, trajs


def save_trajectories(path, store, meta, t):
    np.savez_compressed(
        path,
        t=t.astype(np.float32),
        q=np.stack(store["q"]).astype(np.float32),
        qd=np.stack(store["qd"]).astype(np.float32),
        p=np.stack(store["p"]).astype(np.float32),
        p_ref=np.stack(store["p_ref"]).astype(np.float32),
        tau_a=np.stack(store["tau_a"]).astype(np.float32),
        subject=np.array(meta["subject"]), direction=np.array(meta["direction"]),
        rep=np.array(meta["rep"]))


# ------------------------------------------------------------------- models
def build_neural_controllers(data, tr_mask, va_mask, fold, w2, w3):
    """Train all neural models for one fold; return {name: controller}."""
    mu, sd = norm_stats(data, tr_mask)
    logs = outdir("results", "logs")
    ctrls = {}

    mlp, y_mu, y_sd = train_mlp(data, tr_mask, va_mask, mu, sd,
                                os.path.join(logs, f"mlp_fold{fold}.csv"),
                                SEED + fold)
    ctrls["mlp"] = neu.MLPResController(mlp, mu, sd, y_mu, y_sd)

    for name, cls in (("lstm", neu.LSTMResNet), ("gru", neu.GRUResNet),
                      ("transformer", neu.TransformerResNet)):
        model = cls()
        model, _ = train_torch(model, res_loss_fn, data, tr_mask, va_mask, mu, sd,
                               os.path.join(logs, f"{name}_fold{fold}.csv"),
                               SEED + fold)
        ctrls[name] = neu.TorchResController(model, mu, sd, name)

    for name, fl in IIRA_VARIANTS.items():
        model = neu.IIRANet(bottleneck=fl["bottleneck"],
                            theta_head=fl["theta_head"], residual=fl["residual"])
        loss_fn = iira_loss_fn(W_FIXED["w1"], w2, w3, W_FIXED["w4"],
                               W_FIXED["mu1"], W_FIXED["mu2"])
        model, _ = train_torch(model, loss_fn, data, tr_mask, va_mask, mu, sd,
                               os.path.join(logs, f"{name}_fold{fold}.csv"),
                               SEED + fold)
        ctrls[name] = neu.IIRAController(model, mu, sd, name=name,
                                         aan=fl["aan"], safety=fl["safety"])
    return ctrls, (mu, sd)


def grid_search(data, tr_mask, va_mask, subjects, val_sids, fold=1):
    """w2 x w3 grid on fold-1 (SPEC 4.2); selected by val-subject closed-loop
    end-effector RMSE (4 directions x val subjects)."""
    logs = outdir("results", "logs")
    mu, sd = norm_stats(data, tr_mask)
    val_refs = [d * 5 + 2 for d in (0, 2, 5, 7)]    # 4 directions, d=0.18
    rows = []
    for w2 in GRID_W2:
        for w3 in GRID_W3:
            t0 = time.perf_counter()
            model = neu.IIRANet()
            loss_fn = iira_loss_fn(W_FIXED["w1"], w2, w3, W_FIXED["w4"],
                                   W_FIXED["mu1"], W_FIXED["mu2"])
            model, ep_rows = train_torch(
                model, loss_fn, data, tr_mask, va_mask, mu, sd,
                os.path.join(logs, f"grid_w2{w2}_w3{w3}_fold1.csv"),
                SEED + 77)
            ctrl = neu.IIRAController(model, mu, sd)
            errs = []
            for sid in val_sids:
                for ri in val_refs:
                    rec = sim.rollout(ctrl, subjects[sid], refs_global[ri],
                                      seed=SEED + 555 + sid * 10 + ri)
                    errs.append(sim.trial_metrics(rec)["rmse_p_mm"])
            rmse = float(np.mean(errs))
            rows.append(dict(w2=w2, w3=w3, val_rmse_mm=rmse,
                             epochs_run=len(ep_rows),
                             final_val_loss=ep_rows[-1]["val_loss"],
                             train_time_s=time.perf_counter() - t0))
            print(f"  grid w2={w2} w3={w3}: val RMSE {rmse:.2f} mm "
                  f"({rows[-1]['train_time_s']:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(logs, "grid_search_fold1.csv"), index=False)
    best = df.loc[df.val_rmse_mm.idxmin()]
    print(f"grid search selected: w2={best.w2}, w3={best.w3} "
          f"(val RMSE {best.val_rmse_mm:.2f} mm)", flush=True)
    return float(best.w2), float(best.w3)


# -------------------------------------------------------------------- stats
def cliffs_delta(x, y):
    """delta = P(x > y) - P(x < y); positive => x (baseline) worse than y (IIRA)
    for lower-is-better metrics."""
    gt = lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))


def run_stats(df_per_fold):
    """Wilcoxon signed-rank IIRA vs each other model per metric, Holm-corrected."""
    subj_mean = (df_per_fold.groupby(["model", "fold", "subject"])[METRICS]
                 .mean().reset_index())
    iira = subj_mean[subj_mean.model == "iira"].set_index(["fold", "subject"])
    rows = []
    for metric in METRICS:
        others = [m for m in subj_mean.model.unique() if m != "iira"]
        ps = []
        for m in others:
            v = subj_mean[subj_mean.model == m].set_index(["fold", "subject"])
            a = iira[metric].loc[v.index].to_numpy()
            b = v[metric].to_numpy()
            keep = np.abs(a - b) > 1e-12
            if keep.sum() < 5:
                p = 1.0
                W = np.nan
            else:
                W, p = sstats.wilcoxon(a[keep], b[keep])
            ps.append(dict(metric=metric, baseline=m, n=int(keep.sum()),
                           wilcoxon_W=float(W) if np.isfinite(W) else np.nan,
                           p_raw=float(p), cliff_delta=cliffs_delta(b, a)))
        # Holm correction within metric
        order = np.argsort([r["p_raw"] for r in ps])
        padj = np.empty(len(ps))
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (len(ps) - rank) * ps[idx]["p_raw"])
            padj[idx] = min(running, 1.0)
        for r, p_ in zip(ps, padj):
            r["p_holm"] = float(p_)
            r["significant"] = bool(p_ < 0.05)
        rows += ps
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main
refs_global = None


def main():
    global refs_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    t_start = time.perf_counter()
    subjects, fold_of = ds.load_subjects(ROOT)
    dz = np.load(os.path.join(ROOT, "data", "dataset.npz"))
    test_ref_idx = [d * 5 + 2 for d in range(8)]            # d = 0.18 m
    tr_dists = {0.10, 0.18, 0.26}
    train_ref_idx = [i for i in range(40)
                     if round(float(dz["ref_d_req"][i]), 2) in tr_dists]
    refs_global = {i: ref_from_npz(dz, i)
                   for i in set(test_ref_idx) | set(train_ref_idx)}

    all_rows = []
    best_w2 = best_w3 = None
    for fold in range(1, N_FOLDS + 1):
        t0 = time.perf_counter()
        test_sids = [s.sid for s in subjects if fold_of[s.sid] == fold - 1]
        train_sids = [s.sid for s in subjects if fold_of[s.sid] != fold - 1]
        tr_sids, va_sids = stratified_val_split(train_sids, subjects, fold)
        print(f"[fold {fold}] collecting training rollouts: "
              f"{len(train_sids)} subjects x {len(train_ref_idx)} trials", flush=True)
        data = collect_fold_data(subjects, refs_global, train_sids, fold)
        tr_mask = np.isin(data["SID"], tr_sids)
        va_mask = np.isin(data["SID"], va_sids)

        if fold == 1:
            print("[grid] IIRA loss-weight grid search on fold-1", flush=True)
            best_w2, best_w3 = grid_search(data, tr_mask, va_mask, subjects, va_sids)

        print(f"[fold {fold}] training neural models", flush=True)
        neural_ctrls, (mu, sd) = build_neural_controllers(data, tr_mask, va_mask,
                                                          fold, best_w2, best_w3)
        if args.probe:
            print(f"[probe] fold-1 pipeline done in {time.perf_counter()-t0:.0f}s")
            return

        controllers = {c.name: c for c in (cls() for cls in cl.ALL_CLASSICAL)}
        controllers.update(neural_ctrls)

        # save weights
        wdir = outdir("results", "weights")
        for name, ctrl in controllers.items():
            ctrl.save(os.path.join(wdir, f"{name}_fold{fold}"))

        print(f"[fold {fold}] evaluating {len(controllers)} controllers", flush=True)
        rows, trajs = evaluate(controllers, subjects, test_sids, test_ref_idx, fold)
        all_rows += rows
        tdir = outdir("results", "trajectories")
        tvec = dz["ref_t"]
        for name, (store, meta) in trajs.items():
            save_trajectories(os.path.join(tdir, f"{name}_fold{fold}.npz"),
                              store, meta, tvec)
        pd.DataFrame(all_rows).to_csv(
            os.path.join(outdir("results"), "metrics_per_fold.csv"), index=False)
        print(f"[fold {fold}] done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(outdir("results"), "metrics_per_fold.csv"), index=False)

    # summary over subject-fold units (ILC: 5th repetition per SPEC 6)
    df5 = df[(df.model != "ilc") | (df.rep == 4)]
    grp = df5.groupby("model")[METRICS]
    summ = pd.concat([grp.mean().add_suffix("_mean"),
                      grp.std().add_suffix("_std")], axis=1).reset_index()
    extra = []
    for m in IIRA_VARIANTS:
        sub = df5[df5.model == m].dropna(subset=["theta_mae"])
        subr = df5[df5.model == m].dropna(subset=["rho_hat"])
        r = float(sstats.pearsonr(subr.rho_hat, subr.rho_true)[0]) if len(subr) > 2 else np.nan
        extra.append(dict(model=m, theta_mae=sub.theta_mae.mean() if len(sub) else np.nan,
                          rho_pearson_r=r))
    summ = summ.merge(pd.DataFrame(extra), on="model", how="left")
    summ.to_csv(os.path.join(outdir("results"), "metrics_summary.csv"), index=False)

    stats_df = run_stats(df5)
    stats_df.to_csv(os.path.join(outdir("results"), "stats.csv"), index=False)

    snapshot = dict(
        seed=SEED, python="3.12.14", numpy=np.__version__, torch=torch.__version__,
        controller_version="v1.1 passivity-preserving (SPEC appendix A)",
        v11_changes="AAN modulates stiffness only: Kp(t)=0.2Kp_max+0.8Kp_max(1-rho_hat), "
                    "Kd=Kd_max constant; residual torque low-passed (tau_c=0.05 s); "
                    "dissipative-safe spasticity term -0.8*lam_s_hat*B_sp*tanh(qd_f/0.5)"
                    "|qd_f| with qd_f a 20 Hz low-pass of qd (its Jacobian joins the "
                    "engine's implicit damping matrix); safety velocity damping "
                    "evaluated on qd_f",
        ablation_semantics_v11="noaan=fixed Kp_max; notheta=no injury-estimation head "
                               "hence no tau_comp; noresidual=no residual net (nor its "
                               "low-pass); nobottleneck/nosafety unchanged",
        dt=0.01, n_folds=N_FOLDS, subjects=60,
        train_rollouts_per_subject=len(train_ref_idx),
        train_speeds_m=sorted(tr_dists),
        spec_default_rollouts=40, spec_default_epochs=60,
        epochs=EPOCHS, batch=BATCH, lr=LR, patience=PATIENCE,
        window=neu.WINDOW, window_stride=WINDOW_STRIDE,
        n_val_subjects=N_VAL_SUBJECTS,
        grid=dict(w2=GRID_W2, w3=GRID_W3, selected_w2=best_w2, selected_w3=best_w3,
                  w_fixed=W_FIXED),
        test_protocol="8 directions x d=0.18 m, T=2.0 s; ILC 5 repetitions, rep 5 reported",
        reductions_vs_spec="train rollouts 40->24, epochs 60->20, window stride 30 "
                           "(6 windows/trial); identical for all neural models",
        integration="semi-implicit Euler dt=0.01 with implicit treatment of "
                    "velocity-proportional dissipation (friction Jacobian, controller "
                    "damping gain, patient spasticity Jacobian) for numerical "
                    "stability with the feather-weight wrist links",
        behavior_policy="gain-scheduled PD+gravity (scale U(0.6,1.3)) + 5 Hz "
                        "low-pass exploration noise",
        residual_target="r = M qdd_ref + C qd + friction - tau_patient - (Kpmax e + Kdmax ed)",
        rho_eps=1e-3, noise_scale_Nm=inj.NOISE_SCALE.tolist(),
        U_max_Nm=inj.U_MAX.tolist(),
        total_runtime_min=(time.perf_counter() - t_start) / 60.0,
    )
    with open(os.path.join(outdir("results"), "config_snapshot.json"), "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"TOTAL runtime: {(time.perf_counter()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
