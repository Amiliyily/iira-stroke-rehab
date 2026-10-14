"""Round-2 transfer experiment, final protocol (repeated CV + balanced
weights).  All artifacts go to results/transfer_v2/ (round-1 results in
results/transfer/ are untouched).

Protocol deltas vs the single-shot v2 run:
  - repeated subject-level stratified 5-fold CV x 5 repeats (fold-split seed
    20261011 + 1000*r; repeat 0 reproduces the round-1 fold assignment);
  - class-balanced sample weights (severity class counts 10/4/3 among the 17
    subjects; window/trial weight = N_fit_subjects / (3 * n_class_in_fit));
  - subject prediction = mean over the subject's windows (deep models) or
    over its 4 trials (classic models); classification derived from the
    predicted FMA via the documented thresholds [28.5, 42.5];
  - repeat-level pooled metrics with mean +/- std and t-based 95% CI
    (t_4 = 2.776);
  - confusion matrices at subject level (17 subjects), counts + row
    normalised, from repeat-averaged subject predictions.

Models (12): round-1 seven (iira_zero_shot, iira_ft_head, iira_ft_full, mlp,
lstm, gru, transformer - the five trainable ones retrained with balanced
weights), classic engineered-feature pipeline (ridge, rf, gbt), and IIRA v2
(iira_v2_zero_shot after sim recalibration, iira_v2_ft with CORAL domain
alignment, lambda grid {0.1, 0.3, 1.0}).

Parallelism: ProcessPoolExecutor over the 25 (repeat, fold) jobs,
6 workers x 2 torch threads.
"""
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
from run_transfer import (SEED, W, ELBOW, CLASSES, THR, MAX_EPOCHS, PATIENCE,
                          BS, N_VAL_SUBJ, IIRANet, Head, LSTMReg, GRUReg,
                          TransformerReg, map30, cls_from_fma, fma_from_theta,
                          reg_metrics, cls_metrics, latency_ms, mmd2_rbf)
from run_transfer_v2 import (IIRANetV2, ComboV2, coral_loss, load_real,
                             sim_windows_fold, fold_norm30, load_v2_backbone,
                             LAMBDAS, RECALIB_STEPS)

OUT = os.path.join(ROOT, 'results', 'transfer_v2')
os.makedirs(os.path.join(OUT, 'weights'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'confusion'), exist_ok=True)

N_REPEATS = 5
T_CRIT = 2.776          # t_{0.975, 4}
N_WORKERS = 6
MODELS = ['iira_zero_shot', 'iira_ft_head', 'iira_ft_full', 'mlp', 'lstm',
          'gru', 'transformer', 'ridge', 'rf', 'gbt', 'iira_v2_zero_shot',
          'iira_v2_ft']
METRICS = ['mae', 'rmse', 'r2', 'pearson', 'spearman', 'acc', 'macro_f1',
           'auc_ovr', 'kappa']

T0 = time.time()


def class_weights(subj_fma, fit_subjects, subs_arr):
    """Balanced sample weight per window/trial: w = N_fit/(3 * n_c)."""
    fc = cls_from_fma(np.array([subj_fma[int(s)] for s in fit_subjects]))
    n = np.bincount(fc, minlength=3).astype(float)
    n = np.where(n == 0, 1, n)
    w_cls = len(fit_subjects) / (3.0 * n)
    fma_arr = np.array([subj_fma[int(s)] for s in subs_arr])
    return w_cls[cls_from_fma(fma_arr)].astype(np.float32)


def wmse(pred, y, w):
    return (w * (pred - y) ** 2).sum() / w.sum()


def train_torch_bal(module, X, yz, w, Xva, yzva, log_path, seed,
                    params=None, coral=None):
    """Balanced-weight MSE training, early stop on (unweighted) val MSE.
    coral: optional (Xsim_tensor, lambda) adding a CORAL term on h."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params if params is not None
                           else module.parameters(), lr=1e-3)
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(yz.astype(np.float32))
    wt = torch.from_numpy(w)
    xv = torch.from_numpy(Xva)
    yv = torch.from_numpy(yzva.astype(np.float32))
    lossf = nn.MSELoss()
    best_val, best_state, bad, logs = np.inf, None, 0, []
    for ep in range(MAX_EPOCHS):
        module.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), BS):
            b = perm[i:i + BS]
            opt.zero_grad()
            loss = wmse(module(xt[b]), yt[b], wt[b])
            if coral is not None:
                xs, lam = coral
                sb = torch.randint(0, len(xs), (len(b),))
                loss = loss + lam * coral_loss(
                    module.iira.encode_h(xt[b]),
                    module.iira.encode_h(xs[sb]))
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        module.eval()
        with torch.no_grad():
            vl = float(lossf(module(xv), yv))
        logs.append((ep, tot / len(xt), vl))
        if vl < best_val - 1e-6:
            best_val, bad = vl, 0
            best_state = {kk: vv.detach().clone()
                          for kk, vv in module.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        for ep, tr, vl in logs:
            f.write(f'{ep},{tr:.6f},{vl:.6f}\n')
    return module, best_val


def predict_module(module, X, bs=512):
    module.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(module(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(out)


# ------------------------------------------------------------- one job
def run_job(r, k, tr_sub, te_sub, va_sub, fit_sub):
    torch.set_num_threads(2)
    seed = SEED + 1000 * r + k
    X5, subs, fma = load_real()
    subj_fma = {int(s): float(fma[subs == s][0]) for s in np.unique(subs)}
    tr_m = np.isin(subs, tr_sub)
    te_m = np.isin(subs, te_sub)
    fit_m = np.isin(subs, fit_sub)
    va_m = np.isin(subs, va_sub)
    mean30, std30 = fold_norm30(k)
    X30 = map30(X5, mean30, std30).astype(np.float32)
    mu = X5[fit_m].reshape(-1, 5).mean(0)
    sd = X5[fit_m].reshape(-1, 5).std(0) + 1e-8
    Xn = ((X5 - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)
    ym, ys = fma[fit_m].mean(), fma[fit_m].std() + 1e-8
    yz = ((fma - ym) / ys).astype(np.float32)
    w_all = class_weights(subj_fma, fit_sub, subs)
    tag = f'r{r}f{k}'
    preds = {}
    lat = []

    # ---------------- iira_zero_shot (deterministic, sim fold-k weights)
    iira = IIRANet()
    iira.load_state_dict(torch.load(
        os.path.join(ROOT, 'results', 'weights', f'iira_fold{k}.pt'),
        map_location='cpu', weights_only=True))
    iira.eval()
    ths = []
    with torch.no_grad():
        for i in range(0, te_m.sum(), 512):
            ths.append(iira.theta(torch.from_numpy(X30[te_m][i:i + 512]))
                       .numpy())
    preds['iira_zero_shot'] = fma_from_theta(np.concatenate(ths))
    if r == 0:
        lat.append(dict(model='iira_zero_shot', latency_ms=latency_ms(
            lambda x: iira.theta(x), torch.from_numpy(X30[te_m][:1]))))

    # ---------------- iira_v2_zero_shot (recalibrated)
    iira2 = load_v2_backbone(k)
    for p in iira2.parameters():
        p.requires_grad_(False)
    for p in list(iira2.z_ln.parameters()) + list(iira2.theta_mlp.parameters()):
        p.requires_grad_(True)
    Xsim5, ssub, ds = sim_windows_fold(k)
    Xsim = map30(Xsim5, mean30, std30).astype(np.float32)
    xsim = torch.from_numpy(Xsim)
    th_true = torch.from_numpy(ds['theta'][ssub].astype(np.float32))
    opt = torch.optim.Adam([p for p in iira2.parameters() if p.requires_grad],
                           lr=1e-3)
    lossf = nn.MSELoss()
    tmax = iira2.theta_max
    for step in range(RECALIB_STEPS):
        b = torch.randint(0, len(xsim), (128,))
        opt.zero_grad()
        loss = lossf(iira2.theta(xsim[b]) / tmax, th_true[b] / tmax)
        loss.backward()
        opt.step()
    iira2.eval()
    ths = []
    with torch.no_grad():
        for i in range(0, te_m.sum(), 512):
            ths.append(iira2.theta(torch.from_numpy(X30[te_m][i:i + 512]))
                       .numpy())
    preds['iira_v2_zero_shot'] = fma_from_theta(np.concatenate(ths))
    if r == 0:
        torch.save(iira2.state_dict(), os.path.join(
            OUT, 'weights', f'iira_v2_recalib_fold{k}.pt'))
        with torch.no_grad():
            z_post = iira2.encode(xsim[:512]).numpy()
            th_post = iira2.theta(xsim).numpy()
        diag = dict(z_std_post=z_post.std(0).round(6).tolist(),
                    theta_mae_post=float(np.mean(np.abs(
                        th_post - th_true.numpy()))))
        lat.append(dict(model='iira_v2_zero_shot', latency_ms=latency_ms(
            lambda x: iira2.theta(x), torch.from_numpy(X30[te_m][:1]))))
    else:
        diag = None

    # ---------------- torch models trained on real windows
    def fit_torch(name, module, X, params=None, coral=None):
        logp = os.path.join(OUT, 'logs', f'{name}_{tag}.csv')
        module, bv = train_torch_bal(module, X[fit_m], yz[fit_m],
                                     w_all[fit_m], X[va_m], yz[va_m],
                                     logp, seed, params=params, coral=coral)
        torch.save(module.state_dict(),
                   os.path.join(OUT, 'weights', f'{name}_{tag}.pt'))
        p = predict_module(module, X[te_m]) * ys + ym
        if r == 0:
            lat.append(dict(model=name, latency_ms=latency_ms(
                lambda x: module(x), torch.from_numpy(X[te_m][:1]))))
        return p, bv

    base = IIRANet()
    base.load_state_dict(torch.load(
        os.path.join(ROOT, 'results', 'weights', f'iira_fold{k}.pt'),
        map_location='cpu', weights_only=True))

    class ComboV1(nn.Module):
        def __init__(self, ii, freeze):
            super().__init__()
            self.iira = ii
            self.head = Head()
            if freeze:
                for p in self.iira.parameters():
                    p.requires_grad_(False)

        def forward(self, x):
            h = torch.relu(self.iira.fc(torch.relu(
                self.iira.conv(x.transpose(1, 2))).flatten(1)))
            return self.head(torch.tanh(self.iira.z_lin(h)))

    torch.manual_seed(seed)
    c = ComboV1(base, True)
    preds['iira_ft_head'], _ = fit_torch(
        'iira_ft_head', c, X30, params=c.head.parameters())

    torch.manual_seed(seed)
    base2 = IIRANet()
    base2.load_state_dict(torch.load(
        os.path.join(ROOT, 'results', 'weights', f'iira_fold{k}.pt'),
        map_location='cpu', weights_only=True))
    c = ComboV1(base2, False)
    preds['iira_ft_full'], _ = fit_torch(
        'iira_ft_full', c, X30,
        params=[dict(params=c.head.parameters(), lr=1e-3),
                dict(params=c.iira.parameters(), lr=1e-4)])

    for name, cls in [('lstm', LSTMReg), ('gru', GRUReg),
                      ('transformer', TransformerReg)]:
        torch.manual_seed(seed)
        preds[name], _ = fit_torch(name, cls(), Xn)

    # ---------------- iira_v2_ft (CORAL, lambda grid)
    xsim_c = torch.from_numpy(Xsim)
    best = None
    for lam in LAMBDAS:
        torch.manual_seed(seed)
        combo = ComboV2(load_v2_backbone(k))
        combo.load_state_dict(
            {('iira.' + kk): vv for kk, vv in iira2.state_dict().items()},
            strict=False)
        logp = os.path.join(OUT, 'logs', f'iira_v2_ft_{tag}_lam{lam}.csv')
        combo, bv = train_torch_bal(
            combo, X30[fit_m], yz[fit_m], w_all[fit_m], X30[va_m], yz[va_m],
            logp, seed,
            params=[dict(params=combo.head.parameters(), lr=1e-3),
                    dict(params=combo.iira.parameters(), lr=1e-4)],
            coral=(xsim_c, lam))
        if best is None or bv < best[0]:
            best = (bv, lam, {kk: vv.detach().clone()
                              for kk, vv in combo.state_dict().items()})
    combo = ComboV2(load_v2_backbone(k))
    combo.load_state_dict(best[2])
    torch.save(combo.state_dict(),
               os.path.join(OUT, 'weights', f'iira_v2_ft_{tag}.pt'))
    preds['iira_v2_ft'] = predict_module(combo, X30[te_m]) * ys + ym
    if r == 0:
        lat.append(dict(model='iira_v2_ft', latency_ms=latency_ms(
            lambda x: combo(x), torch.from_numpy(X30[te_m][:1]))))

    # ---------------- mlp on window summary features
    from sklearn.neural_network import MLPRegressor
    summ = np.concatenate([Xn.mean(1), Xn.std(1)], axis=1)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh',
                       max_iter=500, early_stopping=True, random_state=seed)
    try:
        mlp.fit(summ[fit_m], yz[fit_m],
                sample_weight=w_all[fit_m])
    except TypeError:
        mlp.fit(summ[fit_m], yz[fit_m])
    preds['mlp'] = mlp.predict(summ[te_m]) * ys + ym
    if r == 0:
        with open(os.path.join(OUT, 'weights', f'mlp_{tag}.pkl'), 'wb') as f:
            pickle.dump(mlp, f)
        t0 = time.perf_counter()
        for _ in range(200):
            mlp.predict(summ[te_m][:1])
        lat.append(dict(model='mlp',
                        latency_ms=(time.perf_counter() - t0) / 200 * 1e3))

    # ---------------- classic engineered-feature models
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import RidgeCV
    df = pd.read_csv(os.path.join(ROOT, 'data_real',
                                  'real_trial_features.csv'))
    feats = ['dur_s', 'mean_speed', 'peak_speed', 'std_speed', 'n_vel_peaks',
             'path_ratio', 'endpoint_err', 'rmse_target', 'dim_jerk', 'sal']
    Xf = df[feats].values
    Xf[:, 8] = np.log10(Xf[:, 8])
    subs_t = df['subject'].values
    fma_t = df['fma'].values.astype(float)
    tr_f = np.isin(subs_t, tr_sub)
    te_f = np.isin(subs_t, te_sub)
    mu_f, sd_f = Xf[tr_f].mean(0), Xf[tr_f].std(0) + 1e-8
    w_f = class_weights(subj_fma, tr_sub, subs_t)
    builders = dict(
        ridge=lambda: RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
        rf=lambda: RandomForestRegressor(n_estimators=300, random_state=seed,
                                         n_jobs=1),
        gbt=lambda: GradientBoostingRegressor(n_estimators=200,
                                              random_state=seed))
    for name, build in builders.items():
        mdl = build()
        try:
            mdl.fit((Xf[tr_f] - mu_f) / sd_f, fma_t[tr_f],
                    sample_weight=w_f[tr_f])
        except TypeError:
            mdl.fit((Xf[tr_f] - mu_f) / sd_f, fma_t[tr_f])
        p = mdl.predict((Xf[te_f] - mu_f) / sd_f)
        # subject-level: mean over the subject's trials
        subs_te_f = subs_t[te_f]
        sp = np.array([p[subs_te_f == s].mean() for s in te_sub])
        if r == 0:
            with open(os.path.join(OUT, 'weights', f'{name}_{tag}.pkl'),
                      'wb') as f:
                pickle.dump(dict(model=mdl, mean=mu_f, std=sd_f), f)
            t0 = time.perf_counter()
            for _ in range(200):
                mdl.predict(((Xf[te_f] - mu_f) / sd_f)[:1])
            lat.append(dict(model=name,
                            latency_ms=(time.perf_counter() - t0) / 200 * 1e3))
        # expand to window-row convention: one row per subject
        preds[name] = dict(zip(te_sub.tolist(), sp.tolist()))

    # ---------------- subject-level rows
    subs_te, fma_te = subs[te_m], fma[te_m]
    rows = []
    for m, fp in preds.items():
        if isinstance(fp, dict):
            for s in te_sub:
                yt, yp = subj_fma[int(s)], fp[int(s)]
                rows.append(dict(model=m, repeat=r, fold=k, subject=int(s),
                                 fma_true=yt, fma_pred=yp,
                                 class_true=int(cls_from_fma(yt)),
                                 class_pred=int(cls_from_fma(yp)),
                                 n_windows=4))
        else:
            for s in te_sub:
                sm = subs_te == s
                yt, yp = fma_te[sm][0], float(fp[sm].mean())
                rows.append(dict(model=m, repeat=r, fold=k, subject=int(s),
                                 fma_true=float(yt), fma_pred=yp,
                                 class_true=int(cls_from_fma(yt)),
                                 class_pred=int(cls_from_fma(yp)),
                                 n_windows=int(sm.sum())))
    return r, k, rows, lat, diag, best[1]


# ------------------------------------------------------------------ main
def main():
    from sklearn.model_selection import StratifiedKFold
    _, subs, fma = load_real()
    usub = np.unique(subs)
    ufma = np.array([fma[subs == s][0] for s in usub])
    ucls = cls_from_fma(ufma)

    jobs = []
    for r in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=SEED + 1000 * r)
        for k, (tr_i, te_i) in enumerate(skf.split(usub, ucls), start=1):
            tr_sub, te_sub = usub[tr_i], usub[te_i]
            r2 = np.random.default_rng(SEED + 100 + k)
            order = r2.permutation(len(tr_sub))
            va_sub = tr_sub[order[:N_VAL_SUBJ]]
            fit_sub = tr_sub[order[N_VAL_SUBJ:]]
            jobs.append((r, k, tr_sub, te_sub, va_sub, fit_sub))

    all_rows, all_lat, diags, lams = [], [], {}, {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_job, *j) for j in jobs]
        for fut in futs:
            r, k, rows, lat, diag, lam = fut.result()
            all_rows += rows
            all_lat += lat
            if diag is not None:
                diags[k] = diag
            lams[f'{r},{k}'] = lam
            print(f'rep {r} fold {k} done (lam={lam})', flush=True)

    pf = pd.DataFrame(all_rows)
    pf.to_csv(os.path.join(OUT, 'metrics_per_fold.csv'), index=False)
    pd.DataFrame(all_lat).to_csv(os.path.join(OUT, 'latency.csv'),
                                 index=False)
    json.dump({'recalib_rep0': diags, 'lam_coral': lams},
              open(os.path.join(OUT, 'v2_diagnostics.json'), 'w'), indent=1)

    # ---------------- repeat-level pooled metrics
    rep_rows = []
    for m in MODELS:
        for r in range(N_REPEATS):
            d = pf[(pf.model == m) & (pf.repeat == r)]
            y, p = d.fma_true.values, d.fma_pred.values
            met = reg_metrics(y, p)
            met.update(cls_metrics(cls_from_fma(y), cls_from_fma(p), p))
            for key, v in met.items():
                rep_rows.append(dict(model=m, repeat=r, metric=key, value=v))
    rp = pd.DataFrame(rep_rows)
    rp.to_csv(os.path.join(OUT, 'metrics_per_repeat.csv'), index=False)

    # ---------------- summary with 95% CI across repeats
    summ = []
    for m in MODELS:
        for key in METRICS:
            v = rp[(rp.model == m) & (rp.metric == key)].value.values
            v = v[np.isfinite(v)]
            ci = T_CRIT * np.std(v, ddof=1) / np.sqrt(len(v))
            summ.append(dict(model=m, metric=key, rep_mean=float(np.mean(v)),
                             rep_std=float(np.std(v, ddof=1)), ci95=float(ci),
                             n_repeats=len(v)))
        lt = pd.DataFrame(all_lat)
        lv = lt[lt.model == m].latency_ms.values
        summ.append(dict(model=m, metric='latency_ms',
                         rep_mean=float(np.mean(lv)),
                         rep_std=float(np.std(lv, ddof=1)) if len(lv) > 1
                         else 0.0,
                         ci95=0.0, n_repeats=len(lv)))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, 'metrics_summary.csv'),
                              index=False)

    # ---------------- predictions (per repeat + repeat-averaged)
    pred = pf[['model', 'repeat', 'subject', 'fma_true', 'fma_pred']].copy()
    pred.to_csv(os.path.join(OUT, 'predictions.csv'), index=False)
    pm = (pred.groupby(['model', 'subject'])
          .agg(fma_true=('fma_true', 'first'),
               fma_pred=('fma_pred', 'mean')).reset_index())
    pm.to_csv(os.path.join(OUT, 'predictions_repmean.csv'), index=False)

    # ---------------- confusion matrices (subject level, repeat-averaged)
    cc, cn = [], []
    for m in MODELS:
        d = pm[pm.model == m]
        yt = cls_from_fma(d.fma_true.values)
        yp = cls_from_fma(d.fma_pred.values)
        C = np.zeros((3, 3), int)
        for a, b in zip(yt, yp):
            C[a, b] += 1
        rn = C / np.maximum(C.sum(1, keepdims=True), 1)
        for i in range(3):
            for j in range(3):
                cc.append(dict(model=m, true=CLASSES[i], pred=CLASSES[j],
                               count=int(C[i, j])))
                cn.append(dict(model=m, true=CLASSES[i], pred=CLASSES[j],
                               prop=float(rn[i, j])))
    pd.DataFrame(cc).to_csv(os.path.join(OUT, 'confusion',
                                         'confusion_counts.csv'), index=False)
    pd.DataFrame(cn).to_csv(os.path.join(OUT, 'confusion',
                                         'confusion_norm.csv'), index=False)

    # ---------------- per-severity metrics (repeat-averaged predictions)
    sev_rows = []
    for m in MODELS:
        d = pm[pm.model == m]
        y, p = d.fma_true.values, d.fma_pred.values
        for ci, cname in enumerate(CLASSES):
            msk = cls_from_fma(y) == ci
            sev_rows.append(dict(model=m, severity=cname,
                                 mae=float(np.mean(np.abs(y[msk] - p[msk]))),
                                 acc=float(np.mean(cls_from_fma(p[msk]) == ci)),
                                 n=int(msk.sum())))
    pd.DataFrame(sev_rows).to_csv(os.path.join(OUT, 'metrics_by_severity.csv'),
                                  index=False)

    # ---------------- Wilcoxon on repeat-level values vs best macro-F1
    from scipy.stats import wilcoxon
    f1m = {m: rp[(rp.model == m) & (rp.metric == 'macro_f1')].value.mean()
           for m in MODELS}
    best_m = max(f1m, key=f1m.get)
    rows = []
    for m in MODELS:
        if m == best_m:
            continue
        for metric in ['mae', 'macro_f1']:
            a = rp[(rp.model == best_m) & (rp.metric == metric)].value.values
            b = rp[(rp.model == m) & (rp.metric == metric)].value.values
            try:
                stat, pv = wilcoxon(a, b)
            except ValueError:
                stat, pv = np.nan, np.nan
            rows.append(dict(reference=best_m, model=m, metric=metric,
                             ref_mean=float(np.mean(a)),
                             model_mean=float(np.mean(b)),
                             wilcoxon_stat=float(stat), p_value=float(pv),
                             n_repeats=len(a)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'stats_transfer_v2.csv'),
                              index=False)

    json.dump(dict(
        seed=SEED, round='2-final',
        protocol=dict(
            cv='repeated subject-level stratified 5-fold x 5 repeats; '
               'fold-split seed = 20261011 + 1000*repeat (repeat 0 = the '
               'round-1 fold assignment)',
            ci='95% t-based CI across repeats (t_4 = 2.776)',
            class_weights='balanced sample weights N_fit/(3*n_class) on '
                          'windows/trials (severity counts 10/4/3)',
            thresholds='severity from FMA-UE: mild>=43, moderate 29-42, '
                       'severe<=28 (Woytowicz 2017); predicted classes from '
                       'predicted FMA at boundaries 42.5/28.5',
            aggregation='subject prediction = mean of window predictions '
                        '(deep) / mean of 4 trial predictions (classic); '
                        'confusion matrices on repeat-averaged subject '
                        'predictions',
            early_stop_val='3 train subjects per fold (same seed formula as '
                           'round 1)'),
        wall_time_s=round(time.time() - T0, 1)),
        open(os.path.join(OUT, 'config_snapshot.json'), 'w'), indent=1)
    print('done ->', OUT, f'({time.time() - T0:.0f} s)')


if __name__ == '__main__':
    main()
