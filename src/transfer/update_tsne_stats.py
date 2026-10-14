"""Add t-SNE embedding to the domain-shift data and write cross-model
paired statistics (stats_transfer.csv).

- t-SNE (perplexity 30, seed 20261011) on the same standardised 10-d window
  summary features used for the PCA; appended to pca_data.npz as
  tsne_sim / tsne_real.
- stats_transfer.csv: Wilcoxon signed-rank tests on the 5 fold-level MAE and
  macro-F1 values, every model vs the best-pooled-F1 model (iira_ft_full)
  and vs the best baseline (gru).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_transfer import SEED
from update_mmd import sim_windows

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'results', 'transfer')


def summ10(A):
    return np.concatenate([A.mean(1), A.std(1)], axis=1)


def main():
    # ---------------- t-SNE on the same features as the PCA
    d = np.load(os.path.join(ROOT, 'data_real', 'real_windows.npz'),
                allow_pickle=True)
    X5 = d['X'].astype(np.float32)
    simw = sim_windows()
    Ss, Rs = summ10(simw), summ10(X5)
    scl = StandardScaler().fit(np.concatenate([Ss, Rs]))
    Z = np.concatenate([scl.transform(Ss), scl.transform(Rs)])
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED,
                init='pca', learning_rate='auto')
    E = tsne.fit_transform(Z)
    p = dict(np.load(os.path.join(OUT, 'pca_data.npz')))
    p['tsne_sim'] = E[:len(Ss)]
    p['tsne_real'] = E[len(Ss):]
    np.savez(os.path.join(OUT, 'pca_data.npz'), **p)
    print('t-SNE appended to pca_data.npz')

    # ---------------- paired fold-level statistics
    fl = pd.read_csv(os.path.join(OUT, 'metrics_foldlevel.csv'))
    piv = fl.pivot_table(index=['model', 'fold'], columns='metric',
                         values='value')
    models = sorted(piv.index.get_level_values(0).unique())
    rows = []
    for ref in ['iira_ft_full', 'gru']:
        for m in models:
            if m == ref:
                continue
            for metric in ['mae', 'macro_f1']:
                a = piv.loc[ref, metric].values
                b = piv.loc[m, metric].values
                try:
                    stat, pv = wilcoxon(a, b)
                except ValueError:
                    stat, pv = np.nan, np.nan
                rows.append(dict(reference=ref, model=m, metric=metric,
                                 ref_mean=float(np.mean(a)),
                                 model_mean=float(np.mean(b)),
                                 wilcoxon_stat=float(stat), p_value=float(pv),
                                 n_folds=len(a)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'stats_transfer.csv'),
                              index=False)
    print('stats_transfer.csv saved')
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == '__main__':
    main()
