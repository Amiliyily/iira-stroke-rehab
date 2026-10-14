"""Round-2 transfer experiment (transfer_v2).

A. Classic clinical pipeline: engineered per-trial kinematic features
   (data_real/real_trial_features.csv, see features_classic.py) +
   Ridge / RandomForest / GradientBoosting regressors.

B. IIRA v2: bottleneck tanh replaced by linear + LayerNorm (de-saturation),
   short recalibration on simulation data (encoder frozen, LN + theta head
   trained, 300 steps) so theta_hat stays meaningful, then full fine-tuning
   on real windows with a CORAL domain-alignment loss between simulation and
   real encoder activations h (lambda grid {0.1, 0.3, 1.0}, selected by val
   window MSE).

Same protocol as round 1: subject-level stratified 5-fold CV (seed
20261011, identical fold assignment), FMA regression + threshold-derived
3-class severity, pooled subject-level metrics.

Folds run in parallel (ProcessPoolExecutor, 5 workers x 2 torch threads).

Outputs under results/transfer_v2/: weights/, logs/, metrics_per_fold.csv,
metrics_summary.csv, metrics_foldlevel.csv, predictions.csv,
metrics_by_severity.csv, latency.csv, stats_transfer_v2.csv, mmd_v2.json,
v2_diagnostics.json, config_snapshot.json.  Round-1 rows are copied into the
summary tables (not recomputed); round-1 artifacts are untouched.
"""
import io
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'transfer'))
from run_transfer import (SEED, W, FEAT5, ELBOW, CLASSES, THR, MAX_EPOCHS,
                          PATIENCE, BS, N_VAL_SUBJ, IIRANet, Head, map30,
                          cls_from_fma, fma_from_theta, reg_metrics,
                          cls_metrics, latency_ms, mmd2_rbf)

# NOTE: run_transfer sets sys.stdout to a fresh UTF-8 TextIOWrapper at import
# time.  Do NOT reassign sys.stdout here: the wrapper owns the shared buffer,
# and dropping its last reference closes the buffer for everyone.

OUT = os.path.join(ROOT, 'results', 'transfer_v2')
V1 = os.path.join(ROOT, 'results', 'transfer')
os.makedirs(os.path.join(OUT, 'weights'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)

V1_MODELS = ['iira_zero_shot', 'iira_ft_head', 'iira_ft_full', 'mlp', 'lstm',
             'gru', 'transformer']
NEW_MODELS = ['ridge', 'rf', 'gbt', 'iira_v2_zero_shot', 'iira_v2_ft']
ALL_MODELS = V1_MODELS + NEW_MODELS
LAMBDAS = [0.1, 0.3, 1.0]
RECALIB_STEPS = 300

T0 = time.time()


# ------------------------------------------------------------------ models
class IIRANetV2(nn.Module):
    """IIRA trunk with the bottleneck de-saturated: z = LayerNorm(z_lin(h))
    instead of tanh(z_lin(h)).  Loads round-1 sim weights with strict=False
    (the LayerNorm parameters are new)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(30, 32, 5)
        self.fc = nn.Linear(32 * (W - 4), 64)
        self.z_lin = nn.Linear(64, 4)
        self.z_ln = nn.LayerNorm(4)
        self.theta_mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU(),
                                       nn.Linear(32, 4))
        self.rho_lin = nn.Linear(4, 1)
        self.register_buffer("theta_max", torch.tensor([0.85, 0.6, 3.0, 0.5]))

    def encode_h(self, win):
        return torch.relu(self.fc(torch.relu(
            self.conv(win.transpose(1, 2))).flatten(1)))

    def encode(self, win):
        return self.z_ln(self.z_lin(self.encode_h(win)))

    def theta(self, win):
        return torch.sigmoid(self.theta_mlp(self.encode(win))) * self.theta_max


class ComboV2(nn.Module):
    def __init__(self, iira):
        super().__init__()
        self.iira = iira
        self.head = Head()

    def forward(self, x):
        return self.head(self.iira.encode(x))


def coral_loss(a, b):
    d = a.shape[1]
    ca = torch.cov(a.T)
    cb = torch.cov(b.T)
    return ((ca - cb) ** 2).sum() / (4.0 * d * d)


# ------------------------------------------------------------------ data
def load_real():
    d = np.load(os.path.join(ROOT, 'data_real', 'real_windows.npz'),
                allow_pickle=True)
    return (d['X'].astype(np.float32), d['subject'].astype(int),
            d['fma'].astype(float))


def sim_windows_fold(k):
    """Sim windows + subject ids from the fold-k test trajectories."""
    traj = np.load(os.path.join(ROOT, 'results', 'trajectories',
                                f'iira_fold{k}.npz'))
    ds = np.load(os.path.join(ROOT, 'data', 'dataset.npz'))
    dirn = traj['direction'].astype(int)
    subj = traj['subject'].astype(int)
    Xs, Ss = [], []
    for i in range(traj['q'].shape[0]):
        ref = ds['ref_q'][dirn[i] * 5 + 2, :, ELBOW]
        refd = ds['ref_qd'][dirn[i] * 5 + 2, :, ELBOW]
        F = np.stack([traj['q'][i, :, ELBOW], traj['qd'][i, :, ELBOW],
                      ref - traj['q'][i, :, ELBOW],
                      refd - traj['qd'][i, :, ELBOW],
                      traj['tau_a'][i, :, ELBOW]], axis=1)
        for s0 in range(0, len(F) - W + 1, 10):
            Xs.append(F[s0:s0 + W])
            Ss.append(subj[i])
    return np.array(Xs, np.float32), np.array(Ss, int), ds


def fold_norm30(k):
    n = np.load(os.path.join(ROOT, 'results', 'weights',
                             f'iira_fold{k}_norm.npz'))
    return n['mean'], np.where(n['std'] < 1e-6, 1.0, n['std'])


def load_v2_backbone(k):
    m = IIRANetV2()
    sd = torch.load(os.path.join(ROOT, 'results', 'weights',
                                 f'iira_fold{k}.pt'),
                    map_location='cpu', weights_only=True)
    sd = {kk: vv for kk, vv in sd.items()
          if not kk.startswith(('res_net', 'tau_lim'))}
    m.load_state_dict(sd, strict=False)
    return m


# ------------------------------------------------------------- fold worker
def run_fold(k, tr_sub, te_sub, va_sub, fit_sub):
    torch.set_num_threads(2)
    torch.manual_seed(SEED + k)
    rng = np.random.default_rng(SEED + k)
    X5, subs, fma = load_real()
    tr_m = np.isin(subs, tr_sub)
    te_m = np.isin(subs, te_sub)
    fit_m = np.isin(subs, fit_sub)
    va_m = np.isin(subs, va_sub)
    mean30, std30 = fold_norm30(k)
    X30 = map30(X5, mean30, std30).astype(np.float32)
    ym, ys = fma[fit_m].mean(), fma[fit_m].std() + 1e-8
    yz = ((fma - ym) / ys).astype(np.float32)

    # sim recalibration data
    Xsim5, ssub, ds = sim_windows_fold(k)
    Xsim = map30(Xsim5, mean30, std30).astype(np.float32)
    th_true = torch.from_numpy(ds['theta'][ssub].astype(np.float32))

    # ---------------- recalibration (freeze encoder, train LN + theta head)
    iira = load_v2_backbone(k)
    for p in iira.parameters():
        p.requires_grad_(False)
    for p in list(iira.z_ln.parameters()) + list(iira.theta_mlp.parameters()):
        p.requires_grad_(True)
    xsim = torch.from_numpy(Xsim)
    with torch.no_grad():
        z_pre = iira.encode(xsim[:512]).numpy()
        th_pre = iira.theta(xsim).numpy()
    tmax = iira.theta_max
    mae_pre = float(np.mean(np.abs(th_pre - th_true.numpy())))
    opt = torch.optim.Adam([p for p in iira.parameters() if p.requires_grad],
                           lr=1e-3)
    lossf = nn.MSELoss()
    log = []
    for step in range(RECALIB_STEPS):
        b = torch.randint(0, len(xsim), (128,))
        opt.zero_grad()
        loss = lossf(iira.theta(xsim[b]) / tmax, th_true[b] / tmax)
        loss.backward()
        opt.step()
        log.append((step, float(loss)))
    with open(os.path.join(OUT, 'logs', f'iira_v2_recalib_fold{k}.csv'),
              'w') as f:
        f.write('step,loss\n')
        for s, l in log:
            f.write(f'{s},{l:.6f}\n')
    iira.eval()
    with torch.no_grad():
        z_post = iira.encode(xsim[:512]).numpy()
        th_post = iira.theta(xsim).numpy()
    mae_post = float(np.mean(np.abs(th_post - th_true.numpy())))
    torch.save(iira.state_dict(), os.path.join(
        OUT, 'weights', f'iira_v2_recalib_fold{k}.pt'))
    diag = dict(z_std_pre=z_pre.std(0).round(6).tolist(),
                z_std_post=z_post.std(0).round(6).tolist(),
                theta_mae_pre=mae_pre, theta_mae_post=mae_post)

    # ---------------- iira_v2_zero_shot
    ths = []
    with torch.no_grad():
        for i in range(0, te_m.sum(), 512):
            ths.append(iira.theta(torch.from_numpy(X30[te_m][i:i + 512]))
                       .numpy())
    fma_zs = fma_from_theta(np.concatenate(ths))
    lat_zs = latency_ms(lambda x: iira.theta(x),
                        torch.from_numpy(X30[te_m][:1]))

    # ---------------- iira_v2_ft (CORAL alignment, lambda grid)
    xfit = torch.from_numpy(X30[fit_m])
    yfit = torch.from_numpy(yz[fit_m])
    xva = torch.from_numpy(X30[va_m])
    yva = torch.from_numpy(yz[va_m])
    best = None
    for lam in LAMBDAS:
        torch.manual_seed(SEED + 1000 + k)
        combo = ComboV2(load_v2_backbone(k))
        combo.load_state_dict(
            {('iira.' + kk): vv for kk, vv in iira.state_dict().items()},
            strict=False)
        params = [dict(params=combo.head.parameters(), lr=1e-3),
                  dict(params=combo.iira.parameters(), lr=1e-4)]
        opt = torch.optim.Adam(params)
        best_val, best_state, bad, logs = np.inf, None, 0, []
        for ep in range(MAX_EPOCHS):
            combo.train()
            perm = torch.randperm(len(xfit))
            tot = 0.0
            for i in range(0, len(xfit), BS):
                b = perm[i:i + BS]
                sb = torch.randint(0, len(xsim), (len(b),))
                opt.zero_grad()
                pred = combo(xfit[b])
                loss = lossf(pred, yfit[b]) + lam * coral_loss(
                    combo.iira.encode_h(xfit[b]),
                    combo.iira.encode_h(xsim[sb]))
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            combo.eval()
            with torch.no_grad():
                vl = float(lossf(combo(xva), yva))
            logs.append((ep, tot / len(xfit), vl))
            if vl < best_val - 1e-6:
                best_val, bad = vl, 0
                best_state = {kk: vv.detach().clone()
                              for kk, vv in combo.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        with open(os.path.join(OUT, 'logs',
                               f'iira_v2_ft_fold{k}_lam{lam}.csv'), 'w') as f:
            f.write('epoch,train_loss,val_loss\n')
            for ep, tr, vl in logs:
                f.write(f'{ep},{tr:.6f},{vl:.6f}\n')
        if best is None or best_val < best[0]:
            best = (best_val, lam, best_state)
    lam_best = best[1]
    combo = ComboV2(load_v2_backbone(k))
    combo.load_state_dict(best[2])
    combo.eval()
    torch.save(combo.state_dict(),
               os.path.join(OUT, 'weights', f'iira_v2_ft_fold{k}.pt'))
    np.savez(os.path.join(OUT, 'weights', f'iira_v2_ft_fold{k}_norm.npz'),
             mean30=mean30, std30=std30, y_mean=ym, y_std=ys,
             lam_coral=lam_best)
    with torch.no_grad():
        p_ft = np.concatenate([combo(torch.from_numpy(X30[te_m][i:i + 512]))
                               .numpy()
                               for i in range(0, te_m.sum(), 512)])
    fma_ft = p_ft * ys + ym
    lat_ft = latency_ms(lambda x: combo(x), torch.from_numpy(X30[te_m][:1]))

    # ---------------- subject-level rows
    subs_te, fma_te = subs[te_m], fma[te_m]
    rows, mets = [], []
    for m, fp in [('iira_v2_zero_shot', fma_zs), ('iira_v2_ft', fma_ft)]:
        for s in te_sub:
            sm = subs_te == s
            yt, yp = fma_te[sm][0], float(fp[sm].mean())
            rows.append(dict(model=m, fold=k, subject=int(s),
                             fma_true=float(yt), fma_pred=yp,
                             class_true=int(cls_from_fma(yt)),
                             class_pred=int(cls_from_fma(yp)),
                             n_windows=int(sm.sum())))
        ys_t = np.array([fma_te[subs_te == s][0] for s in te_sub])
        ps_t = np.array([fp[subs_te == s].mean() for s in te_sub])
        met = reg_metrics(ys_t, ps_t)
        met.update(cls_metrics(cls_from_fma(ys_t), cls_from_fma(ps_t), ps_t))
        for key, v in met.items():
            mets.append(dict(model=m, fold=k, metric=key, value=v))
    lat = [dict(model='iira_v2_zero_shot', fold=k, latency_ms=lat_zs),
           dict(model='iira_v2_ft', fold=k, latency_ms=lat_ft)]
    return k, rows, mets, lat, diag, lam_best


# --------------------------------------------------------- classic models
def run_classic(folds, usub):
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import RidgeCV
    df = pd.read_csv(os.path.join(ROOT, 'data_real',
                                  'real_trial_features.csv'))
    feats = ['dur_s', 'mean_speed', 'peak_speed', 'std_speed', 'n_vel_peaks',
             'path_ratio', 'endpoint_err', 'rmse_target', 'dim_jerk', 'sal']
    Xf = df[feats].values
    Xf[:, 8] = np.log10(Xf[:, 8])       # log dimensionless jerk
    subs_t = df['subject'].values
    fma_t = df['fma'].values.astype(float)
    rows, mets, lat = [], [], []
    builders = dict(
        ridge=lambda s: RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
        rf=lambda s: RandomForestRegressor(n_estimators=300, random_state=s,
                                           n_jobs=2),
        gbt=lambda s: GradientBoostingRegressor(n_estimators=200,
                                                random_state=s))
    for name, build in builders.items():
        for k, (tr_sub, te_sub) in enumerate(folds, start=1):
            tr_m = np.isin(subs_t, tr_sub)
            te_m = np.isin(subs_t, te_sub)
            mu, sd = Xf[tr_m].mean(0), Xf[tr_m].std(0) + 1e-8
            mdl = build(SEED + k)
            mdl.fit((Xf[tr_m] - mu) / sd, fma_t[tr_m])
            p = mdl.predict((Xf[te_m] - mu) / sd)
            with open(os.path.join(OUT, 'weights', f'{name}_fold{k}.pkl'),
                      'wb') as f:
                pickle.dump(dict(model=mdl, mean=mu, std=sd), f)
            t0 = time.perf_counter()
            for _ in range(200):
                mdl.predict(((Xf[te_m] - mu) / sd)[:1])
            lat.append(dict(model=name, fold=k,
                            latency_ms=(time.perf_counter() - t0) / 200 * 1e3))
            subs_te, fma_te = subs_t[te_m], fma_t[te_m]
            for s in te_sub:
                sm = subs_te == s
                yt, yp = fma_te[sm][0], float(p[sm].mean())
                rows.append(dict(model=name, fold=k, subject=int(s),
                                 fma_true=float(yt), fma_pred=yp,
                                 class_true=int(cls_from_fma(yt)),
                                 class_pred=int(cls_from_fma(yp)),
                                 n_windows=int(sm.sum())))
            ys_t = np.array([fma_te[subs_te == s][0] for s in te_sub])
            ps_t = np.array([p[subs_te == s].mean() for s in te_sub])
            met = reg_metrics(ys_t, ps_t)
            met.update(cls_metrics(cls_from_fma(ys_t), cls_from_fma(ps_t),
                                   ps_t))
            for key, v in met.items():
                mets.append(dict(model=name, fold=k, metric=key, value=v))
    return rows, mets, lat


# ------------------------------------------------------------------ main
def main():
    from sklearn.model_selection import StratifiedKFold
    _, subs, fma = load_real()
    usub = np.unique(subs)
    ufma = np.array([fma[subs == s][0] for s in usub])
    ucls = cls_from_fma(ufma)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds_idx = list(skf.split(usub, ucls))
    folds = [(usub[tr], usub[te]) for tr, te in folds_idx]

    # classic models (fast, serial)
    c_rows, c_mets, c_lat = run_classic(folds, usub)
    print('classic done')

    # IIRA v2 folds in parallel
    jobs = []
    for k, (tr_sub, te_sub) in enumerate(folds, start=1):
        r2 = np.random.default_rng(SEED + 100 + k)
        order = r2.permutation(len(tr_sub))
        va_sub = tr_sub[order[:N_VAL_SUBJ]]
        fit_sub = tr_sub[order[N_VAL_SUBJ:]]
        jobs.append((k, tr_sub, te_sub, va_sub, fit_sub))
    all_rows, all_mets, all_lat, diags, lams = [], [], [], {}, {}
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(run_fold, *j) for j in jobs]
        for fut in futs:
            k, rows, mets, lat, diag, lam = fut.result()
            all_rows += rows
            all_mets += mets
            all_lat += lat
            diags[k] = diag
            lams[k] = lam
            print(f'fold {k} done (lam={lam}, theta MAE '
                  f'{diag["theta_mae_pre"]:.3f}->{diag["theta_mae_post"]:.3f})',
                  flush=True)

    json.dump({'recalib': diags, 'lam_coral': lams,
               'z_note': 'v1 z std ~ 0 (tanh saturation); v2 = linear + '
                         'LayerNorm bottleneck, std should be O(1)'},
              open(os.path.join(OUT, 'v2_diagnostics.json'), 'w'), indent=1)

    # ---------------- merge with round-1 rows
    pf1 = pd.read_csv(os.path.join(V1, 'metrics_per_fold.csv'))
    fl1 = pd.read_csv(os.path.join(V1, 'metrics_foldlevel.csv'))
    lat1 = pd.read_csv(os.path.join(V1, 'latency.csv'))
    pf = pd.concat([pf1, pd.DataFrame(all_rows + c_rows)],
                   ignore_index=True)
    fl = pd.concat([fl1, pd.DataFrame(all_mets + c_mets)], ignore_index=True)
    lat = pd.concat([lat1, pd.DataFrame(all_lat + c_lat)], ignore_index=True)
    pf.to_csv(os.path.join(OUT, 'metrics_per_fold.csv'), index=False)
    fl.to_csv(os.path.join(OUT, 'metrics_foldlevel.csv'), index=False)
    lat.to_csv(os.path.join(OUT, 'latency.csv'), index=False)

    # ---------------- pooled + summary
    summary, pred_rows, sev_rows = [], [], []
    for m in ALL_MODELS:
        d = pf[pf.model == m].sort_values('subject')
        y, p = d.fma_true.values, d.fma_pred.values
        pred_rows.extend(dict(model=m, subject=int(s), fma_true=float(a),
                              fma_pred=float(b))
                         for s, a, b in zip(d.subject, y, p))
        pmet = reg_metrics(y, p)
        pmet.update(cls_metrics(cls_from_fma(y), cls_from_fma(p), p))
        for key in ['mae', 'rmse', 'r2', 'pearson', 'spearman', 'acc',
                    'macro_f1', 'kappa', 'auc_ovr']:
            vals = fl[(fl.model == m) & (fl.metric == key)].value.values
            vals = vals[np.isfinite(vals)]
            summary.append(dict(model=m, metric=key,
                                fold_mean=float(np.mean(vals)),
                                fold_std=float(np.std(vals)),
                                pooled=float(pmet[key])))
        lt = lat[lat.model == m].latency_ms.values
        summary.append(dict(model=m, metric='latency_ms',
                            fold_mean=float(np.mean(lt)),
                            fold_std=float(np.std(lt)),
                            pooled=float(np.mean(lt))))
        for ci, cname in enumerate(CLASSES):
            msk = cls_from_fma(y) == ci
            sev_rows.append(dict(model=m, severity=cname,
                                 mae=float(np.mean(np.abs(y[msk] - p[msk]))),
                                 acc=float(np.mean(cls_from_fma(p[msk]) == ci)),
                                 n=int(msk.sum())))
    pd.DataFrame(summary).to_csv(os.path.join(OUT, 'metrics_summary.csv'),
                                 index=False)
    pd.DataFrame(pred_rows).to_csv(os.path.join(OUT, 'predictions.csv'),
                                   index=False)
    pd.DataFrame(sev_rows).to_csv(os.path.join(OUT, 'metrics_by_severity.csv'),
                                  index=False)

    # ---------------- paired stats (Wilcoxon, fold level) vs best model
    from scipy.stats import wilcoxon
    piv = fl.pivot_table(index=['model', 'fold'], columns='metric',
                         values='value')
    f1m = [r['pooled'] for r in summary if r['metric'] == 'macro_f1']
    best_m = ALL_MODELS[int(np.argmax(f1m))]
    rows = []
    for m in ALL_MODELS:
        if m == best_m:
            continue
        for metric in ['mae', 'macro_f1']:
            a = piv.loc[best_m, metric].values
            b = piv.loc[m, metric].values
            try:
                stat, pv = wilcoxon(a, b)
            except ValueError:
                stat, pv = np.nan, np.nan
            rows.append(dict(reference=best_m, model=m, metric=metric,
                             ref_mean=float(np.mean(a)),
                             model_mean=float(np.mean(b)),
                             wilcoxon_stat=float(stat), p_value=float(pv),
                             n_folds=len(a)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'stats_transfer_v2.csv'),
                              index=False)

    # ---------------- domain shift (v1 copy + v2 h-space)
    mmd = json.load(open(os.path.join(V1, 'mmd.json')))
    iira1 = IIRANetV2()
    sd1 = torch.load(os.path.join(OUT, 'weights', 'iira_v2_recalib_fold1.pt'),
                     map_location='cpu', weights_only=True)
    iira1.load_state_dict(sd1)
    iira1.eval()
    mean30, std30 = fold_norm30(1)
    X5, _, _ = load_real()
    Xsim5, _, _ = sim_windows_fold(1)

    def hv(A5):
        H = []
        with torch.no_grad():
            for i in range(0, len(A5), 512):
                H.append(iira1.encode_h(torch.from_numpy(
                    map30(A5[i:i + 512], mean30, std30))).numpy())
        return np.concatenate(H)

    Hs, Hr = hv(Xsim5), hv(X5)
    r3 = np.random.default_rng(SEED + 7)
    nsub = 1500
    obs, pv, gam = mmd2_rbf(Hs[r3.choice(len(Hs), nsub, replace=False)],
                            Hr[r3.choice(len(Hr), nsub, replace=False)], r3)
    mmd['iira_v2_h'] = dict(mmd2=obs, p_value=pv, gamma=gam, n=nsub,
                            note='fold-1 v2 encoder after sim recalibration')
    json.dump(mmd, open(os.path.join(OUT, 'mmd_v2.json'), 'w'), indent=1)

    json.dump(dict(
        seed=SEED, round=2,
        protocol='identical folds/metrics as round 1 (StratifiedKFold(5, '
                 'seed 20261011), pooled subject-level headline metrics)',
        new_models=dict(
            classic='Ridge(RidgeCV alphas .01-100) / RF(300 trees) / '
                    'GBT(200) on 10 engineered per-trial features '
                    '(real_trial_features.csv), trial-level, subject pred = '
                    'mean of 4 trials, trained on ALL train subjects (no '
                    'early-stopping holdout needed - documented difference '
                    'from the torch models)',
            iira_v2='linear+LayerNorm bottleneck; 300-step sim '
                    'recalibration (encoder frozen, LN+theta head, MSE on '
                    'theta/theta_max); full fine-tune lr 1e-4/1e-3 with '
                    'CORAL loss on h, lambda grid {0.1,0.3,1.0} selected by '
                    'val window MSE'),
        parallel='ProcessPoolExecutor, 5 fold workers x 2 torch threads',
        wall_time_s=round(time.time() - T0, 1)),
        open(os.path.join(OUT, 'config_snapshot.json'), 'w'), indent=1)
    print('done ->', OUT, f'({time.time() - T0:.0f} s)')


if __name__ == '__main__':
    main()
