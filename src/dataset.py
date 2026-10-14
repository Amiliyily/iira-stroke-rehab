"""Dataset generation and persistence (SPEC 2.3/3/6).

Writes:
  data/subjects.csv  - 60 subjects: id, stratum, theta_inj, FMA
  data/folds.csv     - stratified 5-fold split (4/4/4 per fold)
  data/dataset.npz   - S, U_MAX, B_SP, subject arrays, and all 40 reference
                       trajectories (q_ref/qd_ref/qdd_ref/p_ref/tau_ff, meta)
"""

import os
import numpy as np
import pandas as pd

import injury as inj
import tasks


def generate(root, seed=inj.SEED):
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    subjects = inj.sample_subjects(seed)
    fold_of = inj.make_folds(subjects, 5, seed)

    df = pd.DataFrame(dict(
        subject=[s.sid for s in subjects],
        stratum=[s.stratum for s in subjects],
        lambda_w=[s.lam[0] for s in subjects],
        lambda_r=[s.lam[1] for s in subjects],
        lambda_s=[s.lam[2] for s in subjects],
        lambda_n=[s.lam[3] for s in subjects],
        perm=[" ".join(map(str, s.perm)) for s in subjects],
        fma=[s.fma for s in subjects],
        fold=fold_of,
    ))
    df.to_csv(os.path.join(data_dir, "subjects.csv"), index=False)
    df[["subject", "stratum", "fold"]].to_csv(
        os.path.join(data_dir, "folds.csv"), index=False)

    refs = tasks.build_all_references()
    N = len(refs[0]["t"])
    R = len(refs)
    np.savez_compressed(
        os.path.join(data_dir, "dataset.npz"),
        seed=np.int64(seed),
        S=inj.S, U_MAX=inj.U_MAX, B_SP=inj.B_SP, NOISE_SCALE=inj.NOISE_SCALE,
        theta=np.array([s.lam for s in subjects]),
        fma=np.array([s.fma for s in subjects]),
        perm=np.array([s.perm for s in subjects]),
        fold=fold_of,
        ref_t=refs[0]["t"],
        ref_p=np.stack([r["p_ref"] for r in refs]).astype(np.float32),
        ref_q=np.stack([r["q_ref"] for r in refs]).astype(np.float32),
        ref_qd=np.stack([r["qd_ref"] for r in refs]).astype(np.float32),
        ref_qdd=np.stack([r["qdd_ref"] for r in refs]).astype(np.float32),
        ref_tau_ff=np.stack([r["tau_ff"] for r in refs]).astype(np.float32),
        ref_direction=np.array([r["direction"] for r in refs]),
        ref_d_req=np.array([r["d_req"] for r in refs]),
        ref_d_actual=np.array([r["d_actual"] for r in refs]),
        ref_reachable=np.array([r["reachable"] for r in refs]),
    )
    n_short = sum(1 for r in refs if not r["reachable"])
    print(f"dataset: {len(subjects)} subjects, {R} references "
          f"({n_short} shortened for reachability), N={N} steps")
    for stratum in inj.STRATA:
        f = df[df.stratum == stratum].fma
        print(f"  FMA {stratum:8s}: {f.mean():5.1f} +/- {f.std():4.1f} "
              f"[{f.min():5.1f}, {f.max():5.1f}]")
    return df


def load_subjects(root):
    df = pd.read_csv(os.path.join(root, "data", "subjects.csv"))
    subjects = []
    for _, row in df.iterrows():
        lam = [row.lambda_w, row.lambda_r, row.lambda_s, row.lambda_n]
        perm = np.array([int(v) for v in str(row.perm).split()])
        subjects.append(inj.Subject(row.subject, row.stratum, lam, perm, row.fma))
    return subjects, df["fold"].to_numpy()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    generate(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
