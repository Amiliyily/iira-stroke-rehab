"""Engineered per-trial kinematic features (standard stroke motor-assessment
practice) from the Gurari 2018 elbow position-matching trials.

Per trial (active matching segment, phases 11-13, 100 Hz decimation - the
same segment used for the v1 windows), on the anatomical elbow angle q
(135 deg - |device angle|) and its velocity:
  dur_s          movement time of the active segment (s)
  mean_speed     mean |qd| (rad/s)
  peak_speed     max |qd| (rad/s)
  std_speed      std of qd
  n_vel_peaks    number of speed peaks (prominence >= 10% of peak speed)
  path_ratio     total angular path / |net displacement|
  endpoint_err   |matched angle - target angle| (rad)
  rmse_target    RMS tracking error to the target over the segment (rad)
  dim_jerk       dimensionless jerk  sqrt(0.5 * int(jerk^2) dt * T^5 / D^2)
  sal            spectral arc length of the speed profile (Balasubramanian
                 2012; 0-10 Hz, 200-bin normalised spectrum; more negative =
                 less smooth)

Output: data_real/real_trial_features.csv (68 rows = 17 subjects x 4 trials)
with subject/fma/severity columns.
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data_real')
SRC = os.path.join(BASE, 'gurari_stroke')
FS = 100.0
OFFSET = 3.0 * np.pi / 4.0
TRIALS = ['01', '05', '09', '13']

FMA = {1: 11, 2: 50, 3: 28, 4: 32, 5: 39, 6: 28, 7: 59, 8: 17, 9: 35,
       10: 12, 11: 56, 12: 28, 13: 30, 14: 20, 15: 25, 16: 26, 17: 13, 18: 15}
SEV = lambda f: 'mild' if f >= 43 else ('moderate' if f >= 29 else 'severe')

FEATURES = ['dur_s', 'mean_speed', 'peak_speed', 'std_speed', 'n_vel_peaks',
            'path_ratio', 'endpoint_err', 'rmse_target', 'dim_jerk', 'sal']


def trial_target(d):
    ph = d[:, 1].astype(int)
    if (ph == 6).sum() > 20:
        return float(np.median(d[ph == 6, 3]))
    return float(np.median(d[d[:, 1].astype(int) < 11, 3]))


def spectral_arc_length(v, fs, fc=10.0, n_bins=200):
    """Balasubramanian 2012 SAL of the speed profile (negative scalar).
    The spectrum is normalised by its DC component, so the mean is NOT
    subtracted beforehand."""
    n = len(v)
    if n < 8:
        return np.nan
    nfft = int(round(n_bins * fs / fc))
    V = np.abs(np.fft.rfft(v, n=nfft))[:n_bins]
    if V[0] <= 0:
        return np.nan
    V = V / V[0]
    fn = np.linspace(0.0, 1.0, n_bins)
    return float(-np.sum(np.sqrt(np.diff(fn) ** 2 + np.diff(V) ** 2)))


def trial_features(path):
    d = np.loadtxt(path)[::10]          # 1 kHz -> 100 Hz
    ph = d[:, 1].astype(int)
    dev = d[:, 10]
    q = OFFSET - np.abs(dev)            # anatomical rad
    qd = -np.sign(dev) * d[:, 11]
    tgt = OFFSET - abs(trial_target(d))
    act = np.where((ph >= 11) & (ph <= 13))[0]
    if len(act) < 60:
        return None
    q, qd = q[act[0]:act[-1] + 1], qd[act[0]:act[-1] + 1]
    dt = 1.0 / FS
    T = len(q) * dt
    speed = np.abs(qd)
    D = float(np.sum(np.abs(np.diff(q))))          # total angular path
    net = abs(q[-1] - q[0])
    matched = float(np.median(q[-200:]))           # last 2 s
    jerk = np.gradient(np.gradient(np.gradient(q, dt), dt), dt)
    dim_jerk = np.sqrt(0.5 * np.sum(jerk ** 2) * dt * T ** 5
                       / max(D, 1e-9) ** 2)
    peaks, _ = find_peaks(speed, prominence=0.1 * max(speed.max(), 1e-9))
    return dict(dur_s=T, mean_speed=float(speed.mean()),
                peak_speed=float(speed.max()), std_speed=float(qd.std()),
                n_vel_peaks=int(len(peaks)),
                path_ratio=float(D / max(net, 1e-9)),
                endpoint_err=abs(matched - tgt),
                rmse_target=float(np.sqrt(np.mean((q - tgt) ** 2))),
                dim_jerk=float(dim_jerk),
                sal=spectral_arc_length(speed, FS))


def main():
    rows = []
    for sid in sorted(FMA):
        if sid == 6:
            continue
        for tn in TRIALS:
            p = os.path.join(SRC, f'S{sid:02d}', f'Par_arm_trial{tn}.txt')
            if not os.path.exists(p):
                continue
            f = trial_features(p)
            if f is None:
                print(f'S{sid} trial{tn}: segment too short, skipped')
                continue
            f.update(subject=sid, trial=f'S{sid:02d}_t{tn}', fma=FMA[sid],
                     severity=SEV(FMA[sid]))
            rows.append(f)
    df = pd.DataFrame(rows)
    out = os.path.join(BASE, 'real_trial_features.csv')
    df.to_csv(out, index=False)
    print(f'saved {out}: {df.shape}')
    print(df[FEATURES].describe().round(3).to_string())


if __name__ == '__main__':
    main()
