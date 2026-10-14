"""Recompute the domain-shift analysis with a non-degenerate latent space.

The trained sim IIRA bottleneck z is saturated (tanh output +-1, std ~ 1e-6
on both domains - an existing property of the sim weights, consistent with
the sim-side theta_mae ~= 0.30).  z-space MMD is therefore degenerate; this
script replaces it with the pre-bottleneck representation h (64-d), which
has genuine variance, and records the z-saturation diagnostic in mmd.json.

Rewrites results/transfer/mmd.json and pca_data.npz only.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_transfer import IIRANet, map30, mmd2_rbf, SEED, W, ELBOW

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'results', 'transfer')


def sim_windows():
    traj = np.load(os.path.join(ROOT, 'results', 'trajectories',
                                'iira_fold1.npz'))
    ds = np.load(os.path.join(ROOT, 'data', 'dataset.npz'))
    dirn = traj['direction'].astype(int)
    out = []
    for i in range(traj['q'].shape[0]):
        ref = ds['ref_q'][dirn[i] * 5 + 2, :, ELBOW]
        refd = ds['ref_qd'][dirn[i] * 5 + 2, :, ELBOW]
        F = np.stack([traj['q'][i, :, ELBOW], traj['qd'][i, :, ELBOW],
                      ref - traj['q'][i, :, ELBOW],
                      refd - traj['qd'][i, :, ELBOW],
                      traj['tau_a'][i, :, ELBOW]], axis=1)
        for s0 in range(0, len(F) - W + 1, 10):
            out.append(F[s0:s0 + W])
    return np.array(out, np.float32)


def main():
    rng = np.random.default_rng(SEED + 7)
    d = np.load(os.path.join(ROOT, 'data_real', 'real_windows.npz'),
                allow_pickle=True)
    X5 = d['X'].astype(np.float32)
    simw = sim_windows()

    def summ10(A):
        return np.concatenate([A.mean(1), A.std(1)], axis=1)

    Ss, Rs = summ10(simw), summ10(X5)
    nsub = min(1500, len(Ss), len(Rs))
    obs, pv, gam = mmd2_rbf(Ss[rng.choice(len(Ss), nsub, replace=False)],
                            Rs[rng.choice(len(Rs), nsub, replace=False)], rng)
    mmd = {'raw_summary': dict(mmd2=obs, p_value=pv, gamma=gam, n=nsub)}

    iira = IIRANet()
    iira.load_state_dict(torch.load(
        os.path.join(ROOT, 'results', 'weights', 'iira_fold1.pt'),
        map_location='cpu', weights_only=True))
    iira.eval()
    n1 = np.load(os.path.join(ROOT, 'results', 'weights',
                              'iira_fold1_norm.npz'))
    m30 = n1['mean']
    s30 = np.where(n1['std'] < 1e-6, 1.0, n1['std'])

    def reps(A5):
        Hs, Zs = [], []
        with torch.no_grad():
            for i in range(0, len(A5), 512):
                x = torch.from_numpy(map30(A5[i:i + 512], m30, s30))
                h = torch.relu(iira.fc(torch.relu(
                    iira.conv(x.transpose(1, 2))).flatten(1)))
                Hs.append(h.numpy())
                Zs.append(torch.tanh(iira.z_lin(h)).numpy())
        return np.concatenate(Hs), np.concatenate(Zs)

    Hs_, Zs_ = reps(simw[:2000])
    Hr_, Zr_ = reps(X5)
    mmd['z_saturation'] = dict(
        sim_z_std_per_dim=Zs_.std(0).round(8).tolist(),
        real_z_std_per_dim=Zr_.std(0).round(8).tolist(),
        note='bottleneck z is saturated (tanh +-1, std ~ 0) on BOTH domains; '
             'z-space MMD is degenerate, so the latent MMD below uses the '
             'pre-bottleneck representation h (64-d)')
    obs, pv, gam = mmd2_rbf(Hs_[rng.choice(len(Hs_), nsub, replace=False)],
                            Hr_[rng.choice(len(Hr_), nsub, replace=False)], rng)
    mmd['iira_h'] = dict(mmd2=obs, p_value=pv, gamma=gam, n=nsub)
    json.dump(mmd, open(os.path.join(OUT, 'mmd.json'), 'w'), indent=1)
    print(json.dumps(mmd, indent=1))

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scl = StandardScaler().fit(np.concatenate([Ss, Rs]))
    pca = PCA(n_components=2, random_state=SEED).fit(
        scl.transform(np.concatenate([Ss, Rs])))
    # severity class index (same convention as run_transfer: 0 severe,
    # 1 moderate, 2 mild)
    fma = d['fma'].astype(float)
    sev = np.where(fma >= 42.5, 2, np.where(fma >= 28.5, 1, 0))
    np.savez(os.path.join(OUT, 'pca_data.npz'),
             sim=pca.transform(scl.transform(Ss)),
             real=pca.transform(scl.transform(Rs)),
             real_severity=sev, real_subject=d['subject'].astype(int),
             explained=pca.explained_variance_ratio_)
    print('pca explained:', pca.explained_variance_ratio_.round(3))


if __name__ == '__main__':
    main()
