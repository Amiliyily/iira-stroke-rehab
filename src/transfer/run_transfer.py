"""Sim-to-real transfer experiment: IIRA vs trained-from-scratch baselines on
the Gurari PLOS2018 stroke elbow position-matching data.

Task A: 3-class severity classification (mild / moderate / severe; Woytowicz
2017 thresholds on FMA-UE).  Task B: FMA-UE regression.  All class labels are
derived from (predicted) FMA via the thresholds [28.5, 42.5] - documented as
an adaptation.  Subject-level stratified 5-fold CV (seed 20261011); every
subject is tested exactly once (pooled predictions over 17 subjects).

Models:
  iira_zero_shot  sim-trained IIRA encoder+theta head (iira_fold{k}.pt),
                  FMA from the inverse of the synthetic FMA formula
                  (injury.synthesize_fma without the noise term).
  iira_ft_head    frozen conv/fc/z_lin encoder, new z->MLP(8)->FMA head.
  iira_ft_full    same head, whole network fine-tuned at lr 1e-4.
  mlp             sklearn MLPRegressor [64,64] tanh on window summary
                  features (per-channel mean+std, 10 dims - documented
                  adaptation: the sim MLP uses instantaneous features,
                  which do not exist as a window for sequence models).
  lstm/gru/transformer  same trunk classes as the sim baselines, input dim
                  5, scalar regression head.

Outputs under results/transfer/: weights/, logs/, metrics_per_fold.csv,
metrics_summary.csv, predictions.csv, metrics_by_severity.csv, mmd.json,
pca_data.npz, config_snapshot.json.
"""
import io
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'results', 'transfer')
SEED = 20261011
W = 50
FEAT5 = 5           # [q, qd, e, ed, tau] of the elbow
FEAT30 = 30
ELBOW = 2           # UR5 elbow joint index in the sim feature vector
CLASSES = ['severe', 'moderate', 'mild']   # index 0/1/2
THR = (28.5, 42.5)  # severe|moderate, moderate|mild boundaries
MAX_EPOCHS = 100
PATIENCE = 10
BS = 64
N_VAL_SUBJ = 3

os.makedirs(os.path.join(OUT, 'weights'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)


# ------------------------------------------------------------------ models
class IIRANet(nn.Module):
    """Copy of controllers.neural.IIRANet (trunk used for transfer)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(FEAT30, 32, 5)
        self.fc = nn.Linear(32 * (W - 4), 64)
        self.z_lin = nn.Linear(64, 4)
        self.theta_mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU(),
                                       nn.Linear(32, 4))
        self.rho_lin = nn.Linear(4, 1)
        self.res_net = nn.Sequential(nn.Linear(28, 128), nn.Tanh(),
                                     nn.Linear(128, 128), nn.Tanh(),
                                     nn.Linear(128, 6))
        self.register_buffer("theta_max", torch.tensor([0.85, 0.6, 3.0, 0.5]))
        self.register_buffer("tau_lim", torch.zeros(6))

    def encode(self, win):
        """win (B, 50, 30) normalised -> z (B, 4)."""
        h = torch.relu(self.fc(torch.relu(self.conv(win.transpose(1, 2))).flatten(1)))
        return torch.tanh(self.z_lin(h))

    def theta(self, win):
        return torch.sigmoid(self.theta_mlp(self.encode(win))) * self.theta_max


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, z):
        return self.mlp(z).squeeze(-1)


class LSTMReg(nn.Module):
    def __init__(self, d=FEAT5):
        super().__init__()
        self.rnn = nn.LSTM(d, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(-1)


class GRUReg(nn.Module):
    def __init__(self, d=FEAT5):
        super().__init__()
        self.rnn = nn.GRU(d, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(-1)


class TransformerReg(nn.Module):
    def __init__(self, d=FEAT5):
        super().__init__()
        self.proj = nn.Linear(d, 64)
        self.pos = nn.Parameter(torch.randn(1, W, 64) * 0.02)
        layer = nn.TransformerEncoderLayer(64, 1, 128, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        o = self.enc(self.proj(x) + self.pos)
        return self.head(o[:, -1]).squeeze(-1)


# ------------------------------------------------------------------ helpers
def cls_from_fma(f):
    f = np.asarray(f, float)
    return np.where(f >= THR[1], 2, np.where(f >= THR[0], 1, 0))


def map30(X5, mean, std):
    """(N, 50, 5) elbow features -> normalised (N, 50, 30) IIRA input."""
    N = X5.shape[0]
    F = np.zeros((N, W, FEAT30), np.float32)
    F[:, :, ELBOW] = X5[:, :, 0]
    F[:, :, 6 + ELBOW] = X5[:, :, 1]
    F[:, :, 12 + ELBOW] = X5[:, :, 2]
    F[:, :, 18 + ELBOW] = X5[:, :, 3]
    F[:, :, 24 + ELBOW] = X5[:, :, 4]
    return (F - mean[None, None, :]) / std[None, None, :]


def fma_from_theta(th):
    """Inverse of injury.synthesize_fma without the noise term."""
    lw, lr, ls, ln = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    return np.clip(66.0 * (1.0 - 0.5 * lw - 0.2 * lr - 0.067 * ls - 0.2 * ln),
                   4.0, 66.0)


def train_module(module, Xtr, ytr, Xva, yva, lr, log_path, seed, params=None):
    """MSE training with early stopping (val window loss); restores the best
    epoch's weights. `params` may be an optimizer param-group list."""
    torch.manual_seed(seed)
    module.train()
    opt = torch.optim.Adam(
        params if params is not None else module.parameters(), lr=lr)
    lossf = nn.MSELoss()
    xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr.astype(np.float32))
    xv = torch.from_numpy(Xva)
    yv = torch.from_numpy(yva.astype(np.float32))
    best_val, best_state, bad, logs = np.inf, None, 0, []
    for ep in range(MAX_EPOCHS):
        module.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), BS):
            b = perm[i:i + BS]
            opt.zero_grad()
            loss = lossf(module(xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        module.eval()
        with torch.no_grad():
            vl = float(lossf(module(xv), yv))
        logs.append((ep, tot / len(xt), vl))
        if vl < best_val - 1e-6:
            best_val, bad = vl, 0
            best_state = {k: v.detach().clone()
                          for k, v in module.state_dict().items()}
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
    return module


class Combo(nn.Module):
    """IIRA encoder + regression head (ft_head / ft_full)."""

    def __init__(self, iira, freeze):
        super().__init__()
        self.iira = iira
        self.head = Head()
        if freeze:
            for p in self.iira.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        return self.head(self.iira.encode(x))


def predict_module(module, X, bs=512):
    module.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(module(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(out)


def reg_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    ss = 1.0 - np.sum((y - p) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)
    pr = float(pearsonr(y, p)[0]) if np.std(p) > 1e-12 and np.std(y) > 1e-12 else np.nan
    sr = float(spearmanr(y, p)[0]) if np.std(p) > 1e-12 and np.std(y) > 1e-12 else np.nan
    return dict(mae=mae, rmse=rmse, r2=float(ss), pearson=pr, spearman=sr)


def auc_ovr(yt, fma_pred):
    """Macro one-vs-rest AUC from monotone scores of the predicted FMA
    (documented adaptation): mild: +fma, severe: -fma,
    moderate: -|fma - 35.5|.  Computed as the mean of the three binary
    class-vs-rest AUCs."""
    yt = np.asarray(yt)
    if len(np.unique(yt)) < 3:
        return np.nan
    scores = {1: -np.abs(fma_pred - 35.5), 2: fma_pred, 0: -fma_pred}
    return float(np.mean([roc_auc_score((yt == c).astype(int), s)
                          for c, s in scores.items()]))


def cls_metrics(yt, yp, fma_pred):
    return dict(acc=float(accuracy_score(yt, yp)),
                macro_f1=float(f1_score(yt, yp, average='macro',
                                        labels=[0, 1, 2], zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp)),
                auc_ovr=auc_ovr(yt, fma_pred))


def latency_ms(fn, x):
    with torch.no_grad():
        for _ in range(20):
            fn(x)
        t0 = time.perf_counter()
        for _ in range(200):
            fn(x)
    return (time.perf_counter() - t0) / 200 * 1e3


def mmd2_rbf(A, B, rng, n_perm=200):
    """Biased MMD^2 with RBF kernel (median heuristic) + permutation p."""
    X = np.concatenate([A, B])
    D2 = np.sum((X[:, None] - X[None]) ** 2, axis=-1)
    gamma = 1.0 / max(np.median(D2[D2 > 0]), 1e-12)
    K = np.exp(-gamma * D2)
    n, m = len(A), len(B)

    def stat(idx_a):
        idx_b = np.setdiff1d(np.arange(n + m), idx_a)
        return (K[np.ix_(idx_a, idx_a)].mean() + K[np.ix_(idx_b, idx_b)].mean()
                - 2 * K[np.ix_(idx_a, idx_b)].mean())

    obs = stat(np.arange(n))
    cnt = 1
    for _ in range(n_perm):
        cnt += stat(rng.permutation(n + m)[:n]) >= obs
    return float(obs), float(cnt / (n_perm + 1)), float(gamma)


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    d = np.load(os.path.join(ROOT, 'data_real', 'real_windows.npz'),
                allow_pickle=True)
    X5 = d['X'].astype(np.float32)
    subs = d['subject'].astype(int)
    fma = d['fma'].astype(float)
    ycls = cls_from_fma(fma)
    usub = np.unique(subs)
    ufma = np.array([fma[subs == s][0] for s in usub])
    ucls = cls_from_fma(ufma)
    print(f'windows={X5.shape}, subjects={len(usub)}, '
          f'class counts={np.bincount(ucls, minlength=3)}')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(usub, ucls))
    print('fold sizes:', [len(t) for _, t in folds])

    per_fold_rows, pred_rows, summ_rows, sev_rows = [], [], [], []
    lat_rows = []
    models_order = ['iira_zero_shot', 'iira_ft_head', 'iira_ft_full', 'mlp',
                    'lstm', 'gru', 'transformer']
    pooled = {m: dict(y=[], p=[]) for m in models_order}

    for k, (tr_i, te_i) in enumerate(folds, start=1):
        tr_sub, te_sub = usub[tr_i], usub[te_i]
        tr_m = np.isin(subs, tr_sub)
        te_m = np.isin(subs, te_sub)
        # ~25% of train subjects for early stopping
        r2 = np.random.default_rng(SEED + 100 + k)
        order = r2.permutation(len(tr_sub))
        va_sub = tr_sub[order[:N_VAL_SUBJ]]
        fit_sub = tr_sub[order[N_VAL_SUBJ:]]
        fit_m = np.isin(subs, fit_sub)
        va_m = np.isin(subs, va_sub)

        # fold normalisation (fit windows only), per channel
        mu = X5[fit_m].reshape(-1, FEAT5).mean(0)
        sd = X5[fit_m].reshape(-1, FEAT5).std(0) + 1e-8
        Xn = (X5 - mu[None, None, :]) / sd[None, None, :]
        ym, ys = fma[fit_m].mean(), fma[fit_m].std() + 1e-8
        yz = ((fma - ym) / ys).astype(np.float32)

        norm = np.load(os.path.join(ROOT, 'results', 'weights',
                                    f'iira_fold{k}_norm.npz'))
        mean30, std30 = norm['mean'], np.where(norm['std'] < 1e-6, 1.0, norm['std'])
        X30 = map30(X5, mean30, std30).astype(np.float32)

        print(f'--- fold {k}: test subjects {list(te_sub)} '
              f'(val {list(va_sub)})')

        # ---------------- iira_zero_shot
        iira = IIRANet()
        iira.load_state_dict(torch.load(
            os.path.join(ROOT, 'results', 'weights', f'iira_fold{k}.pt'),
            map_location='cpu', weights_only=True))
        iira.eval()
        ths = []
        with torch.no_grad():
            for i in range(0, te_m.sum(), 512):
                ths.append(iira.theta(torch.from_numpy(X30[te_m][i:i + 512])).numpy())
        fma_zs = fma_from_theta(np.concatenate(ths))
        lat = latency_ms(lambda x: iira.theta(x),
                         torch.from_numpy(X30[te_m][:1]))
        fold_pred = {'iira_zero_shot': fma_zs}
        lat_rows.append(dict(model='iira_zero_shot', fold=k, latency_ms=lat))

        # ---------------- iira_ft_head / iira_ft_full
        for name, freeze, lr in [('iira_ft_head', True, 1e-3),
                                 ('iira_ft_full', False, 1e-4)]:
            torch.manual_seed(SEED + k)
            base = IIRANet()
            base.load_state_dict(torch.load(
                os.path.join(ROOT, 'results', 'weights', f'iira_fold{k}.pt'),
                map_location='cpu', weights_only=True))
            combo = Combo(base, freeze)
            params = (combo.head.parameters() if freeze
                      else [dict(params=combo.head.parameters(), lr=1e-3),
                            dict(params=combo.iira.parameters(), lr=lr)])
            logp = os.path.join(OUT, 'logs', f'{name}_fold{k}.csv')
            train_module(combo, X30[fit_m], yz[fit_m], X30[va_m], yz[va_m],
                         1e-3, logp, SEED + k, params=params)
            p = predict_module(combo, X30[te_m]) * ys + ym
            fold_pred[name] = p
            torch.save(combo.state_dict(),
                       os.path.join(OUT, 'weights', f'{name}_fold{k}.pt'))
            np.savez(os.path.join(OUT, 'weights', f'{name}_fold{k}_norm.npz'),
                     mean30=mean30, std30=std30, y_mean=ym, y_std=ys,
                     mean5=mu, std5=sd)
            lat_rows.append(dict(model=name, fold=k, latency_ms=latency_ms(
                lambda x: combo(x), torch.from_numpy(X30[te_m][:1]))))

        # ---------------- mlp (window summary features)
        summ = np.concatenate([Xn.mean(1), Xn.std(1)], axis=1)  # (N, 10)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh',
                           max_iter=500, early_stopping=True,
                           random_state=SEED + k)
        mlp.fit(summ[fit_m], yz[fit_m])
        fold_pred['mlp'] = mlp.predict(summ[te_m]) * ys + ym
        with open(os.path.join(OUT, 'weights', f'mlp_fold{k}.pkl'), 'wb') as f:
            pickle.dump(dict(mlp=mlp, mean5=mu, std5=sd, y_mean=ym, y_std=ys), f)
        t0 = time.perf_counter()
        for _ in range(200):
            mlp.predict(summ[te_m][:1])
        lat_rows.append(dict(model='mlp', fold=k,
                             latency_ms=(time.perf_counter() - t0) / 200 * 1e3))

        # ---------------- lstm / gru / transformer
        for name, cls in [('lstm', LSTMReg), ('gru', GRUReg),
                          ('transformer', TransformerReg)]:
            torch.manual_seed(SEED + k)
            net = cls()
            logp = os.path.join(OUT, 'logs', f'{name}_fold{k}.csv')
            train_module(net, Xn[fit_m], yz[fit_m], Xn[va_m], yz[va_m],
                         1e-3, logp, SEED + k)
            fold_pred[name] = predict_module(net, Xn[te_m]) * ys + ym
            torch.save(net.state_dict(),
                       os.path.join(OUT, 'weights', f'{name}_fold{k}.pt'))
            np.savez(os.path.join(OUT, 'weights', f'{name}_fold{k}_norm.npz'),
                     mean5=mu, std5=sd, y_mean=ym, y_std=ys)
            lat_rows.append(dict(model=name, fold=k, latency_ms=latency_ms(
                lambda x: net(x), torch.from_numpy(Xn[te_m][:1]))))

        # ---------------- subject-level fold metrics
        subs_te, fma_te = subs[te_m], fma[te_m]
        for m in models_order:
            fp = fold_pred[m]
            for s in te_sub:
                sm = subs_te == s
                yt, yp = fma_te[sm][0], float(fp[sm].mean())
                per_fold_rows.append(dict(
                    model=m, fold=k, subject=int(s), fma_true=float(yt),
                    fma_pred=yp, class_true=int(cls_from_fma(yt)),
                    class_pred=int(cls_from_fma(yp)), n_windows=int(sm.sum())))
                pooled[m]['y'].append(yt)
                pooled[m]['p'].append(yp)
            ys_t = np.array([fma_te[subs_te == s][0] for s in te_sub])
            ps_t = np.array([fp[subs_te == s].mean() for s in te_sub])
            met = reg_metrics(ys_t, ps_t)
            met.update(cls_metrics(cls_from_fma(ys_t), cls_from_fma(ps_t), ps_t))
            for key, v in met.items():
                summ_rows.append(dict(model=m, fold=k, metric=key, value=v))

    # ------------------------------------------------------------- outputs
    import csv
    pf = os.path.join(OUT, 'metrics_per_fold.csv')
    with open(pf, 'w', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=list(per_fold_rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(per_fold_rows)

    # pooled + summary
    summary = []
    for m in models_order:
        y = np.array(pooled[m]['y'])
        p = np.array(pooled[m]['p'])
        pred_rows.extend(dict(model=m, subject=int(s), fma_true=float(a),
                              fma_pred=float(b))
                         for s, a, b in zip(usub, y, p))
        pmet = reg_metrics(y, p)
        pmet.update(cls_metrics(cls_from_fma(y), cls_from_fma(p), p))
        for key in ['mae', 'rmse', 'r2', 'pearson', 'spearman', 'acc',
                    'macro_f1', 'kappa', 'auc_ovr']:
            vals = [r['value'] for r in summ_rows
                    if r['model'] == m and r['metric'] == key]
            vals = [v for v in vals if np.isfinite(v)]
            summary.append(dict(model=m, metric=key,
                                fold_mean=float(np.mean(vals)),
                                fold_std=float(np.std(vals)),
                                pooled=float(pmet[key])))
        lat = [r['latency_ms'] for r in lat_rows if r['model'] == m]
        summary.append(dict(model=m, metric='latency_ms',
                            fold_mean=float(np.mean(lat)),
                            fold_std=float(np.std(lat)),
                            pooled=float(np.mean(lat))))
        for ci, cname in enumerate(CLASSES):
            msk = cls_from_fma(y) == ci
            sev_rows.append(dict(model=m, severity=cname,
                                 mae=float(np.mean(np.abs(y[msk] - p[msk]))),
                                 acc=float(np.mean(cls_from_fma(p[msk]) == ci)),
                                 n=int(msk.sum())))

    with open(os.path.join(OUT, 'metrics_summary.csv'), 'w', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=['model', 'metric', 'fold_mean',
                                             'fold_std', 'pooled'])
        wcsv.writeheader()
        wcsv.writerows(summary)
    with open(os.path.join(OUT, 'metrics_foldlevel.csv'), 'w', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=['model', 'fold', 'metric', 'value'])
        wcsv.writeheader()
        wcsv.writerows(summ_rows)
    with open(os.path.join(OUT, 'predictions.csv'), 'w', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=['model', 'subject', 'fma_true',
                                             'fma_pred'])
        wcsv.writeheader()
        wcsv.writerows(pred_rows)
    with open(os.path.join(OUT, 'metrics_by_severity.csv'), 'w', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=['model', 'severity', 'mae', 'acc',
                                             'n'])
        wcsv.writeheader()
        wcsv.writerows(sev_rows)
    with open(os.path.join(OUT, 'latency.csv'), 'w', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=['model', 'fold', 'latency_ms'])
        wcsv.writeheader()
        wcsv.writerows(lat_rows)

    # ------------------------------------------------- domain shift analysis
    traj = np.load(os.path.join(ROOT, 'results', 'trajectories',
                                'iira_fold1.npz'))
    ds = np.load(os.path.join(ROOT, 'data', 'dataset.npz'))
    dirn = traj['direction'].astype(int)
    qs = traj['q'][:, :, ELBOW]
    qds = traj['qd'][:, :, ELBOW]
    taus = traj['tau_a'][:, :, ELBOW]
    simw = []
    for i in range(qs.shape[0]):
        ref = ds['ref_q'][dirn[i] * 5 + 2, :, ELBOW]
        refd = ds['ref_qd'][dirn[i] * 5 + 2, :, ELBOW]
        F = np.stack([qs[i], qds[i], ref - qs[i], refd - qds[i], taus[i]],
                     axis=1)
        for s0 in range(0, len(F) - W + 1, 10):
            simw.append(F[s0:s0 + W])
    simw = np.array(simw, np.float32)
    print('sim windows for MMD:', simw.shape)

    def summ10(A):
        return np.concatenate([A.mean(1), A.std(1)], axis=1)

    Ss, Rs = summ10(simw), summ10(X5)
    nsub = min(1500, len(Ss), len(Rs))
    r3 = np.random.default_rng(SEED + 7)
    obs, pv, gam = mmd2_rbf(Ss[r3.choice(len(Ss), nsub, replace=False)],
                            Rs[r3.choice(len(Rs), nsub, replace=False)], r3)
    mmd = {'raw_summary': dict(mmd2=obs, p_value=pv, gamma=gam, n=nsub)}

    iira1 = IIRANet()
    iira1.load_state_dict(torch.load(
        os.path.join(ROOT, 'results', 'weights', 'iira_fold1.pt'),
        map_location='cpu', weights_only=True))
    iira1.eval()
    n1 = np.load(os.path.join(ROOT, 'results', 'weights',
                              'iira_fold1_norm.npz'))
    m30 = n1['mean']
    s30 = np.where(n1['std'] < 1e-6, 1.0, n1['std'])

    def hspace(A5):
        H, Z = [], []
        with torch.no_grad():
            for i in range(0, len(A5), 512):
                x = torch.from_numpy(map30(A5[i:i + 512], m30, s30))
                h = torch.relu(iira1.fc(torch.relu(
                    iira1.conv(x.transpose(1, 2))).flatten(1)))
                H.append(h.numpy())
                Z.append(torch.tanh(iira1.z_lin(h)).numpy())
        return np.concatenate(H), np.concatenate(Z)

    Hs, Zs = hspace(simw[:2000])
    Hr, Zr = hspace(X5)
    # The trained sim bottleneck z is saturated (tanh +-1, std ~ 0 on both
    # domains), so z-space MMD is degenerate; the latent MMD uses the
    # pre-bottleneck representation h (64-d) instead.
    mmd['z_saturation'] = dict(
        sim_z_std_per_dim=Zs.std(0).round(8).tolist(),
        real_z_std_per_dim=Zr.std(0).round(8).tolist())
    obs, pv, gam = mmd2_rbf(Hs[r3.choice(len(Hs), nsub, replace=False)],
                            Hr[r3.choice(len(Hr), nsub, replace=False)], r3)
    mmd['iira_h'] = dict(mmd2=obs, p_value=pv, gamma=gam, n=nsub)
    json.dump(mmd, open(os.path.join(OUT, 'mmd.json'), 'w'), indent=1)
    print('MMD:', json.dumps(mmd, indent=1))

    # PCA on the 10-dim summary features (fit on sim+real)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scl = StandardScaler().fit(np.concatenate([Ss, Rs]))
    pca = PCA(n_components=2, random_state=SEED).fit(
        scl.transform(np.concatenate([Ss, Rs])))
    np.savez(os.path.join(OUT, 'pca_data.npz'),
             sim=pca.transform(scl.transform(Ss)),
             real=pca.transform(scl.transform(Rs)),
             real_severity=ycls, real_subject=subs,
             explained=pca.explained_variance_ratio_)

    json.dump(dict(
        seed=SEED,
        data=dict(
            source='Gurari, Drogos & Dewald, PLOS ONE 2018;13(10):e0206518, '
                   'Harvard Dataverse PA_Pos_Percept, CC0',
            subjects=17, excluded='S6 (no single-arm data)',
            trials_per_subject=[1, 5, 9, 13],
            window=W, stride=10, fs_hz=100,
            features=['q_elb', 'qd_elb', 'e=tgt-q', 'ed=-qd', 'tau=0'],
            angle_convention='anat_deg = 135 - |device_deg|; '
                             'qd_anat = -sign(dev)*qd_dev',
            active_phases=[11, 12, 13],
            fma_source='paper Table 1 (Europe PMC PMC6205610 full-text XML)',
            severity_thresholds='Woytowicz 2017: mild>=43, moderate 29-42, '
                                'severe<=28 (boundaries 42.5 / 28.5)'),
        models=models_order,
        adaptations=[
            'single elbow DoF mapped to UR5 elbow (joint index 2); other '
            'sim channels zero-padded before sim-norm normalisation',
            'tau channel = 0 (no robot assistance in the real trials)',
            'classification derived from predicted FMA via thresholds '
            '[28.5, 42.5]',
            'MLP baseline uses window summary features (mean+std, 10 dims) '
            'instead of instantaneous features',
            'OvR AUC from monotone scores of predicted FMA (mild:+fma, '
            'severe:-fma, moderate:-|fma-35.5|)',
            'zero-shot FMA = inverse synthetic-FMA formula on theta_hat'],
        training=dict(max_epochs=MAX_EPOCHS, patience=PATIENCE, batch=BS,
                      lr=dict(head=1e-3, full_finetune=1e-4, baselines=1e-3),
                      val_subjects=N_VAL_SUBJ,
                      target_standardised=True),
        sim_weights='results/weights/iira_fold{1..5}.pt (+_norm.npz)',
        cv='StratifiedKFold(5, shuffle, seed 20261011) over 17 subjects',
        cv_rationale=('5-fold stratified subject-level CV over 17 subjects '
                      '(class counts 10/4/3 -> folds 4/4/3/3/3); every '
                      'subject is tested exactly once, headline numbers are '
                      'pooled over the 17 subjects'),
        post_run_notes=[
            'trained sim IIRA bottleneck z is saturated (tanh +-1, std~0 on '
            'both domains); zero-shot theta->FMA is therefore a near-'
            'constant predictor (consistent with sim-side theta_mae ~= 0.30)',
            'z-space MMD degenerate; latent MMD reported on pre-bottleneck '
            'h (64-d)',
            't-SNE (perplexity 30, seed 20261011) added to pca_data.npz '
            '(tsne_sim/tsne_real); paired Wilcoxon fold-level stats in '
            'stats_transfer.csv']),
        open(os.path.join(OUT, 'config_snapshot.json'), 'w'), indent=1)
    print('done ->', OUT)


if __name__ == '__main__':
    main()
