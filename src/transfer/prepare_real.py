"""Prepare real-data windows from the Gurari PLOS2018 stroke dataset.

Source files (per subject Sxx, single-arm experiment, paretic-arm-as-testing-arm
trials Par_arm_trialNN.txt): raw text, ~1 kHz, 18 columns. Column semantics were
reverse-engineered and validated against the paper's Table 1 (see
data_real/qa_columns.json):
  col 0  time (s)
  col 1  trial phase index (1..14; 11 = active position-matching phase,
         12-13 = post-match hold, 6 = passive hold at the reference target)
  col 3  device-commanded (reference) elbow angle, rad
  col 7/10  measured elbow angle, rad (col 10 = filtered duplicate of col 7;
         corr(col7,col10) > 0.999; we use col 10)
  col 8/11  elbow angular velocity, rad/s (col 11 pairs with col 10)
Anatomical elbow angle = 135 deg - |device angle| (the offset cancels in
e = ref - q; validated against Table 1 min/max matching angles, see
qa_columns.json).

Output: data_real/real_windows.npz
  X        (N, 50, 5) float32  [q, qd, e, ed, tau] @ 100 Hz, W=50, stride 10
           (tau = 0: no robot assistance in these trials)
  subject  (N,) int            stroke subject id (1..18, S6 excluded: no data)
  trial    (N,) str            source trial id
  fma      (N,) float          subject FMA-UE score (Table 1 of the paper)
  severity (N,) str            mild / moderate / severe
Also writes data_real/qa_columns.json and data_real/subjects_real.csv.
"""
import json, os, sys, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data_real')
SRC = os.path.join(BASE, 'gurari_stroke')
SEED = 20261011
FS_OUT = 100.0          # Hz after x10 decimation
W, STRIDE = 50, 10

# FMA-UE scores from Table 1 of Gurari, Drogos & Dewald, PLOS ONE 13(10):e0206518
# (2018), extracted from the full-text XML (Europe PMC, PMC6205610).
FMA = {1: 11, 2: 50, 3: 28, 4: 32, 5: 39, 6: 28, 7: 59, 8: 17, 9: 35,
       10: 12, 11: 56, 12: 28, 13: 30, 14: 20, 15: 25, 16: 26, 17: 13, 18: 15}
# Table 1 min/max single-arm position-matching angle of the paretic arm (deg)
# (used only for QA of the angle convention; S10/S15/S18 contain obvious typos
#  in the paper table, e.g. "1,294.1 deg" -> skipped)
ROM_MATCH = {1: (68.6, 115.0), 2: (75.9, 108.8), 3: (73.2, 105.1),
             4: (70.8, 111.1), 5: (72.0, 113.1), 7: (67.9, 112.2),
             8: (76.1, 101.1), 9: (72.8, 110.4), 11: (70.7, 115.0),
             12: (78.2, 109.1), 13: (70.3, 110.6), 14: (75.0, 111.5),
             16: (71.0, 118.4), 17: (81.7, 104.8)}
SEV = lambda f: 'mild' if f >= 43 else ('moderate' if f >= 29 else 'severe')
OFFSET = 3.0 * np.pi / 4.0    # 135 deg; anatomical rad = OFFSET - |device rad|
TRIALS = ['01', '05', '09', '13']


def anat(dev):
    """Device angle (rad) -> anatomical elbow angle (rad)."""
    return OFFSET - np.abs(dev)


def load_trial(path):
    d = np.loadtxt(path)
    return d[::10]            # 1 kHz -> 100 Hz box slicing (signal is smooth)


def trial_target(d):
    """Reference target = held device angle during phase 6 (fallback: the
    longest constant segment of col3 before phase 11)."""
    ph = d[:, 1].astype(int)
    if (ph == 6).sum() > 200:
        return float(np.median(d[ph == 6, 3]))
    ref = d[:, 3]
    pre = np.where(ph < 11)[0]
    if len(pre) == 0:
        return float(np.median(ref))
    # longest run of (nearly) constant col3
    best_len, best_val, cur, start = 0, np.median(ref[pre]), 0, 0
    r = ref[pre]
    for i in range(1, len(r)):
        if abs(r[i] - r[i - 1]) > 1e-4:
            if i - start > best_len:
                best_len, best_val = i - start, np.median(r[start:i])
            start = i
    return float(best_val)


def main():
    Xs, subs, trs, fmas, sevs = [], [], [], [], []
    qa = {}
    for sid in sorted(FMA):
        sdir = os.path.join(SRC, f'S{sid:02d}')
        if not os.path.isdir(sdir):
            continue
        matched_deg = []
        nwin_s = 0
        for tn in TRIALS:
            p = os.path.join(sdir, f'Par_arm_trial{tn}.txt')
            if not os.path.exists(p):
                continue
            d = load_trial(p)
            ph = d[:, 1].astype(int)
            tgt = trial_target(d)
            q = anat(d[:, 10])              # anatomical rad
            qd = -np.sign(d[:, 10]) * d[:, 11]   # d(anat)/dt = -sign(dev)*d(dev)/dt
            act = np.where((ph >= 11) & (ph <= 13))[0]
            if len(act) < W + 5 * STRIDE:
                print(f'S{sid} trial{tn}: active phase too short ({len(act)}), skipped')
                continue
            seg = act[0], act[-1] + 1
            qs, qds = q[seg[0]:seg[1]], qd[seg[0]:seg[1]]
            # matched angle = median q over the last 2 s of the segment
            matched_deg.append(float(np.degrees(np.median(qs[-200:]))))
            e = anat(tgt) - qs
            ed = -qds
            tau = np.zeros_like(qs)
            F = np.stack([qs, qds, e, ed, tau], axis=1).astype(np.float32)
            for s in range(0, len(F) - W + 1, STRIDE):
                Xs.append(F[s:s + W])
                subs.append(sid)
                trs.append(f'S{sid:02d}_t{tn}')
                fmas.append(FMA[sid])
                sevs.append(SEV(FMA[sid]))
                nwin_s += 1
        qa[sid] = dict(n_windows=nwin_s, matched_deg=matched_deg,
                       table1_minmax_deg=ROM_MATCH.get(sid))
        print(f'S{sid:02d}: {nwin_s} windows, matched(deg)='
              + ','.join(f'{v:.1f}' for v in matched_deg)
              + f'  table1={ROM_MATCH.get(sid)}')
    X = np.stack(Xs)
    out = os.path.join(BASE, 'real_windows.npz')
    np.savez_compressed(out, X=X, subject=np.array(subs), trial=np.array(trs),
                        fma=np.array(fmas, float), severity=np.array(sevs))
    json.dump(qa, open(os.path.join(BASE, 'qa_columns.json'), 'w'), indent=1)
    # subject table
    with open(os.path.join(BASE, 'subjects_real.csv'), 'w') as f:
        f.write('subject,fma_ue,severity,source\n')
        for sid in sorted(FMA):
            f.write(f'{sid},{FMA[sid]},{SEV(FMA[sid])},gurari2018_plosone_table1\n')
    print('saved', out, X.shape)


if __name__ == '__main__':
    main()
