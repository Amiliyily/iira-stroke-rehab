"""Per-subject calibration variant (roadmap's lowest-cost transfer fix).

Frozen de-saturated IIRA-v2 encoder (sim-recalibrated, fold-1 weights).  In
a LOSO protocol over the 17 subjects, a global ridge head (z -> FMA) is fit
on the other 16 subjects; the test subject then contributes 1 or 2
calibration trials (trials 01 / 01+05) from which a per-subject bias offset
is estimated; evaluation is on the subject's REMAINING trials (strictly
separated).

IMPORTANT (documented caveat): the calibration trials carry the subject's
true FMA label, so this is an oracle-style "one clinical measurement per
patient" calibration (the deployment analogue: one therapist-scored FMA at
intake), NOT a label-free predictor.  Because FMA is a per-subject constant
in this dataset, a direct per-subject z->FMA ridge would be degenerate
(MAE ~ 0 by construction); the bias-corrected form used here instead
measures how consistently the encoder + global head represent a subject
across trials - an upper bound, reported honestly as such.

Outputs (merged into results/transfer_v2/ by merge_calib.py):
  results/transfer_v2/calib_loso.csv  (per subject x n_calib)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'transfer'))
from run_transfer import (SEED, map30, cls_from_fma, reg_metrics, cls_metrics,
                          CLASSES)
from run_transfer_v2 import IIRANetV2, fold_norm30, load_real

OUT = os.path.join(ROOT, 'results', 'transfer_v2')
TRIAL_ORDER = ['S{:02d}_t{:02d}'.format(s, t) for s in range(1, 19)
               for t in (1, 5, 9, 13)]
T0 = time.time()


def main():
    torch.manual_seed(SEED)
    d = np.load(os.path.join(ROOT, 'data_real', 'real_windows.npz'),
                allow_pickle=True)
    X5 = d['X'].astype(np.float32)
    subs = d['subject'].astype(int)
    fma = d['fma'].astype(float)
    trials = d['trial'].astype(str)

    iira = IIRANetV2()
    iira.load_state_dict(torch.load(
        os.path.join(OUT, 'weights', 'iira_v2_recalib_fold1.pt'),
        map_location='cpu', weights_only=True))
    iira.eval()
    mean30, std30 = fold_norm30(1)

    # encode all windows once (frozen encoder)
    Z = []
    with torch.no_grad():
        for i in range(0, len(X5), 512):
            Z.append(iira.encode(torch.from_numpy(
                map30(X5[i:i + 512], mean30, std30))).numpy())
    Z = np.concatenate(Z)
    print('encoded', Z.shape)

    # Direct per-subject ridge on z -> FMA is DEGENERATE here: FMA is a
    # per-subject constant, so the fit memorises the constant and scores
    # MAE ~ 0 by construction.  The informative calibration variant is a
    # LOSO global ridge (fit on the other 16 subjects) plus a per-subject
    # bias offset estimated from the subject's calibration trial(s):
    #   pred_calib = pred_global + (fma_true - mean(pred_global on calib))
    # Its error equals |mean global error on calib trials - mean global
    # error on test trials|, i.e. it measures how consistently the encoder
    # + global head represent a subject ACROSS trials - exactly the
    # quantity that decides whether one intake FMA measurement can
    # calibrate a deployed predictor.
    subjects = np.unique(subs)
    rows = []
    for s in subjects:
        m_s = subs == s
        m_o = ~m_s
        mu, sd = Z[m_o].mean(0), Z[m_o].std(0) + 1e-8
        mdl = Ridge(alpha=1.0)
        mdl.fit((Z[m_o] - mu) / sd, fma[m_o])
        p_s = mdl.predict((Z[m_s] - mu) / sd)
        tr_s = trials[m_s]
        yt = float(fma[m_s][0])
        rows.append(dict(model='iira_v2_loso_global', subject=int(s),
                         fma_true=yt, fma_pred=float(p_s.mean()),
                         n_calib_windows=0, n_test_windows=int(m_s.sum())))
        for n_calib, calib_t in [(1, ['t01']), (2, ['t01', 't05'])]:
            name = f'iira_v2_calib{n_calib}t'
            c_mask = np.isin(tr_s, [f'S{s:02d}_{t}' for t in calib_t])
            t_mask = ~c_mask
            if c_mask.sum() == 0 or t_mask.sum() == 0:
                continue
            offset = yt - p_s[c_mask].mean()
            rows.append(dict(model=name, subject=int(s), fma_true=yt,
                             fma_pred=float(p_s[t_mask].mean() + offset),
                             n_calib_windows=int(c_mask.sum()),
                             n_test_windows=int(t_mask.sum())))

    for name in ['iira_v2_loso_global', 'iira_v2_calib1t', 'iira_v2_calib2t']:
        d_m = pd.DataFrame([r for r in rows if r['model'] == name])
        met = reg_metrics(d_m.fma_true.values, d_m.fma_pred.values)
        met.update(cls_metrics(cls_from_fma(d_m.fma_true.values),
                               cls_from_fma(d_m.fma_pred.values),
                               d_m.fma_pred.values))
        print(name, {k: round(v, 3) for k, v in met.items()})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'calib_loso.csv'), index=False)
    print('saved calib_loso.csv', df.shape, f'({time.time() - T0:.0f} s)')


if __name__ == '__main__':
    main()
