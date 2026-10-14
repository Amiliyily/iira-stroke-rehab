"""Merge the LOSO calibration variants (calib_loso.csv) into the round-2
transfer_v2 tables: metrics_summary.csv, metrics_per_repeat.csv,
predictions_repmean.csv, confusion/{confusion_counts,confusion_norm}.csv,
metrics_by_severity.csv, latency.csv, and config_snapshot.json.

The calib models are single LOSO passes (no repeats), so summary rows carry
n_repeats=1, rep_std=0.  ci95 is a subject-level t-CI (t_16 = 2.120) where
the metric decomposes per subject (mae, acc); 0.0 otherwise - the semantics
differ from the repeated-CV rows and are documented in config_snapshot.json.
Idempotent: existing rows for the calib models are replaced.

Run:  python src/transfer/merge_calib.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'transfer'))
from run_transfer import (SEED, map30, cls_from_fma, reg_metrics, cls_metrics,
                          CLASSES)
from run_transfer_v2 import IIRANetV2, fold_norm30, load_real  # noqa: F401

OUT = os.path.join(ROOT, 'results', 'transfer_v2')
CALIB_MODELS = ['iira_v2_loso_global', 'iira_v2_calib1t', 'iira_v2_calib2t']
T16 = 2.120  # t_{0.975, 16} for the 17-subject LOSO CI
METRICS = ['mae', 'rmse', 'r2', 'pearson', 'spearman', 'acc', 'macro_f1',
           'auc_ovr', 'kappa', 'latency_ms']


def measure_latency():
    """Per-window ms for frozen-encoder encoding + ridge predict (CPU)."""
    d = np.load(os.path.join(ROOT, 'data_real', 'real_windows.npz'),
                allow_pickle=True)
    X5 = d['X'].astype(np.float32)
    iira = IIRANetV2()
    iira.load_state_dict(torch.load(
        os.path.join(OUT, 'weights', 'iira_v2_recalib_fold1.pt'),
        map_location='cpu', weights_only=True))
    iira.eval()
    mean30, std30 = fold_norm30(1)
    mdl = Ridge(alpha=1.0)
    mdl.fit(np.zeros((4, 4)), np.zeros(4))  # warm sklearn predict path
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(X5), 512):
            z = iira.encode(torch.from_numpy(
                map30(X5[i:i + 512], mean30, std30))).numpy()
            mdl.predict(z)
    return (time.perf_counter() - t0) / len(X5) * 1000.0


def sev_of(fma):
    return np.where(fma >= 43, 'mild', np.where(fma >= 29, 'moderate',
                                                'severe'))


def main():
    cal = pd.read_csv(os.path.join(OUT, 'calib_loso.csv'))
    latency = measure_latency()
    print(f'calib latency {latency:.4f} ms/window')

    summ_rows, rep_rows, pred_rows, sev_rows, cc_rows = [], [], [], [], []
    for name in CALIB_MODELS:
        d_m = cal[cal.model == name]
        yt, yp = d_m.fma_true.values, d_m.fma_pred.values
        met = reg_metrics(yt, yp)
        met.update(cls_metrics(cls_from_fma(yt), cls_from_fma(yp), yp))
        met['latency_ms'] = latency
        # subject-level CI where the metric decomposes per subject
        mae_s = np.abs(yt - yp)
        acc_s = (cls_from_fma(yt) == cls_from_fma(yp)).astype(float)
        ci = {k: 0.0 for k in METRICS}
        ci['mae'] = T16 * mae_s.std(ddof=1) / np.sqrt(len(mae_s))
        ci['acc'] = T16 * acc_s.std(ddof=1) / np.sqrt(len(acc_s))
        for k in METRICS:
            summ_rows.append(dict(model=name, metric=k,
                                  rep_mean=met[k], rep_std=0.0, ci95=ci[k],
                                  n_repeats=1))
            rep_rows.append(dict(model=name, repeat=0, metric=k,
                                 value=met[k]))
        for _, r in d_m.iterrows():
            pred_rows.append(dict(model=name, subject=int(r.subject),
                                  fma_true=r.fma_true, fma_pred=r.fma_pred))
        sv, pc = sev_of(yt), cls_from_fma(yp)
        tc = cls_from_fma(yt)
        for ci, c in enumerate(CLASSES):
            m = sv == c
            sev_rows.append(dict(model=name, severity=c,
                                 mae=float(np.abs(yt[m] - yp[m]).mean()),
                                 acc=float((cls_from_fma(yt[m]) == pc[m])
                                           .mean()),
                                 n=int(m.sum())))
        for ti, t in enumerate(CLASSES):
            for pi, p in enumerate(CLASSES):
                cc_rows.append(dict(model=name, true=t, pred=p,
                                    count=int(((tc == ti) & (pc == pi))
                                              .sum())))

    def _append(fname, new, keys=('model',)):
        path = os.path.join(OUT, fname)
        old = pd.read_csv(path)
        old = old[~old.model.isin(CALIB_MODELS)]
        pd.concat([old, pd.DataFrame(new)], ignore_index=True).to_csv(
            path, index=False)

    _append('metrics_summary.csv', summ_rows)
    _append('metrics_per_repeat.csv', rep_rows)
    _append('predictions_repmean.csv', pred_rows)
    _append('metrics_by_severity.csv', sev_rows)
    _append(os.path.join('confusion', 'confusion_counts.csv'), cc_rows)

    # row-normalized confusion
    cc = pd.DataFrame(cc_rows)
    tot = cc.groupby(['model', 'true'])['count'].transform('sum')
    cn = cc.assign(prop=np.where(tot > 0, cc['count'] / tot, 0.0))
    _append(os.path.join('confusion', 'confusion_norm.csv'),
            cn.to_dict('records'))

    lat_path = os.path.join(OUT, 'latency.csv')
    lat = pd.read_csv(lat_path)
    lat = lat[~lat.model.isin(CALIB_MODELS)]
    pd.concat([lat, pd.DataFrame([dict(model=m, latency_ms=latency)
                                  for m in CALIB_MODELS])],
              ignore_index=True).to_csv(lat_path, index=False)

    cfg_path = os.path.join(OUT, 'config_snapshot.json')
    cfg = json.load(open(cfg_path))
    cfg['calib_protocol'] = (
        'LOSO over 17 subjects with frozen de-saturated IIRA-v2 encoder '
        '(sim-recalibrated fold-1 weights). iira_v2_loso_global: ridge '
        'z->FMA fit on the other 16 subjects, no calibration. '
        'iira_v2_calib{1,2}t: same global ridge plus a per-subject bias '
        'offset from the subject calibration trial(s) 01 / 01+05, evaluated '
        'on the strictly separated remaining trials. Calibration trials '
        'carry the true FMA (oracle one-measurement-per-patient analogue, '
        'an upper bound, not a label-free predictor). Summary rows: '
        'n_repeats=1, rep_std=0; ci95 is a subject-level t-CI (t_16=2.120) '
        'for mae/acc and 0.0 for non-decomposable metrics - different '
        'semantics from the repeated-CV rows.')
    json.dump(cfg, open(cfg_path, 'w'), indent=1)
    print('merged', CALIB_MODELS, 'into transfer_v2 tables')


if __name__ == '__main__':
    main()
