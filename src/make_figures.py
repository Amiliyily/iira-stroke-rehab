"""Figures 1-6 (SPEC 7, v1.1).  PNG 300 dpi + PDF, Okabe-Ito colour-blind
palette, fonts >= 7 pt.  Every number comes from results/*.csv or the stored
trajectory npz files; theta_hat/rho_hat series are re-inferred offline from
the saved fold weights + stored test trajectories (exact replay of the
IIRAController estimate path, validated against metrics_per_fold.csv).

Run:  python src/make_figures.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import arm6dof as arm
import injury as inj

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colour-blind safe palette
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", pink="#CC79A7", black="#000000",
          grey="#999999")
STRATUM_C = {"mild": OI["sky"], "moderate": OI["orange"], "severe": OI["verm"]}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 110, "savefig.dpi": 300, "axes.linewidth": 0.7,
    "lines.linewidth": 1.2, "font.family": "DejaVu Sans",
})

DISPLAY = dict(pid="PID", pd_grav="PD+grav", computed_torque="Comp. torque",
               impedance="Impedance", admittance="Admittance", smc="SMC",
               lqr="LQR", mpc="MPC", dob="DOB", ilc="ILC", rbf="RBF- adap.",
               mlp="MLP", lstm="LSTM", gru="GRU", transformer="Transformer",
               iira="IIRA (ours)",
               iira_nobottleneck="w/o bottleneck", iira_noaan="w/o AAN",
               iira_noresidual="w/o residual", iira_nosafety="w/o safety",
               iira_notheta="w/o $\\hat{\\theta}$ head")
MAIN16 = ["pid", "pd_grav", "computed_torque", "impedance", "admittance",
          "smc", "lqr", "mpc", "dob", "ilc", "rbf", "mlp", "lstm", "gru",
          "transformer", "iira"]
ABL6 = ["iira", "iira_nobottleneck", "iira_noaan", "iira_noresidual",
        "iira_nosafety", "iira_notheta"]


def load_all():
    summ = pd.read_csv(os.path.join(ROOT, "results", "metrics_summary_v2.csv"))
    stats = pd.read_csv(os.path.join(ROOT, "results", "stats_v2.csv"))
    pf = pd.read_csv(os.path.join(ROOT, "results", "metrics_per_fold_v2.csv"))
    subj = pd.read_csv(os.path.join(ROOT, "data", "subjects.csv"))
    return summ, stats, pf, subj


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png/.pdf saved", flush=True)


def traj(model, fold, archive=False):
    base = os.path.join(ROOT, "results_archive_v1_naive" if archive
                        else "results", "trajectories")
    return np.load(os.path.join(base, f"{model}_fold{fold}.npz"))


def find_trial(z, subject, direction, rep=0):
    m = ((z["subject"] == subject) & (z["direction"] == direction)
         & (z["rep"] == rep))
    return int(np.where(m)[0][0])


# ---------------------------------------------------------------
# offline re-inference of IIRA theta_hat / rho_hat on stored test trials
# ---------------------------------------------------------------
def iira_estimates():
    """Replay the IIRAController estimate path on every stored IIRA test
    trial (5 folds x 96 trials).  Returns dict with per-trial theta_hat
    (EMA-smoothed trial mean, as Controller.estimates()), rho_hat series,
    and trial meta.  Cached to results/iira_estimates_cache.npz."""
    cache = os.path.join(ROOT, "results", "iira_estimates_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.keys()}

    import torch
    from controllers import neural as neu

    dz = np.load(os.path.join(ROOT, "data", "dataset.npz"))
    W = neu.WINDOW
    th_out, rho_s_out, meta = [], [], []
    for fold in range(1, 6):
        model = neu.IIRANet()
        model.load_state_dict(torch.load(
            os.path.join(ROOT, "results", "weights", f"iira_fold{fold}.pt")))
        model.eval()
        nm = np.load(os.path.join(ROOT, "results", "weights",
                                  f"iira_fold{fold}_norm.npz"))
        mu, sd = nm["mean"], nm["std"]
        z = traj("iira", fold)
        N = len(z["t"])
        for i in range(len(z["subject"])):
            di = int(z["direction"][i])
            q_ref = dz["ref_q"][di * 5 + 2].astype(float)
            qd_ref = dz["ref_qd"][di * 5 + 2].astype(float)
            q, qd, ta = (z["q"][i].astype(float), z["qd"][i].astype(float),
                         z["tau_a"][i].astype(float))
            ta_prev = np.vstack([np.zeros(6), ta[:-1]])
            feat = np.concatenate([q, qd, q_ref - q, qd_ref - qd, ta_prev],
                                  axis=1)
            fn = ((feat - mu) / sd).astype(np.float32)
            pad = np.concatenate([np.repeat(fn[:1], W - 1, axis=0), fn])
            wins = np.lib.stride_tricks.sliding_window_view(pad, (W, 30)
                                                            )[:, 0]
            with torch.no_grad():
                out = model(torch.from_numpy(wins))
            rho_raw = out["rho"].numpy()
            th_raw = out["theta"].numpy()
            rho_s = np.zeros(N)
            th_s = np.zeros((N, 4))
            rs, ts = 0.0, None
            for k in range(N):
                rs = 0.9 * rs + 0.1 * rho_raw[k]
                ts = th_raw[k] if ts is None else 0.9 * ts + 0.1 * th_raw[k]
                rho_s[k] = rs
                th_s[k] = ts
            th_out.append(th_s.mean(axis=0))
            rho_s_out.append(rho_s)
            meta.append((fold, int(z["subject"][i]), di))
        print(f"  [iira estimates] fold {fold} replayed", flush=True)
    out = dict(theta_hat=np.array(th_out), rho_series=np.array(rho_s_out),
               meta=np.array(meta))
    np.savez_compressed(cache, **out)
    return out


# =============================================================== fig 1
def fig1_principle():
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 115); ax.set_ylim(0, 64); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=7.6, ec="0.25", lw=0.9, style="round,pad=0.35"):
        b = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs)
        return (x, y, w, h)

    def arrow(p1, p2, text=None, color="0.2", lw=1.2, style="-|>", ls="-",
              tpos=0.5, tdx=0.0, tdy=1.0, fs=6.8, connectionstyle="arc3,rad=0"):
        a = FancyArrowPatch(p1, p2, arrowstyle=style, color=color, lw=lw,
                            linestyle=ls, mutation_scale=9,
                            connectionstyle=connectionstyle)
        ax.add_patch(a)
        if text:
            ax.text((p1[0] + p2[0]) / 2 + tdx, (p1[1] + p2[1]) / 2 + tdy,
                    text, fontsize=fs, ha="center", color=color)

    # physical layer (left)
    b_pat = box(2, 44, 22, 12, "Patient\ninjury model $\\theta_{inj}="
                "(\\lambda_w,\\lambda_r,\\lambda_s,\\lambda_n)$",
                "#FDECEA", fs=7.2)
    b_arm = box(2, 22, 22, 12, "6-DOF robot arm\n$M(q)\\ddot q+C(q,\\dot q)"
                "\\dot q+g(q)=\\tau$\nRNE, semi-implicit Euler", "#EAF2FB",
                fs=7.2)
    b_safe = box(2, 4, 23, 10, "Safety layer\nvel. damping ($\\dot q_f$), torque sat.,\n"
                 "workspace sphere $|p|\\leq0.85$ m", "#EDEDED", fs=6.6)
    arrow((13, 44), (13, 34), "$\\tau_{patient}(t)$", tdx=9.0, tdy=-0.5)
    a = FancyArrowPatch((6, 22), (6, 44), arrowstyle="-", color="0.55",
                        lw=0.9, linestyle=":")
    ax.add_patch(a)
    ax.text(4.6, 39, "$q,\\dot q$", fontsize=7.0, rotation=90, color="0.4")
    arrow((24, 26), (30, 42), "$q,\\dot q$ (measurable)", lw=1.1, tdx=3.0,
          tdy=-4.5, fs=6.8, connectionstyle="arc3,rad=0.2")
    arrow((25, 7), (38, 4.5), "", color="0.35", ls="--", lw=1.0,
          connectionstyle="arc3,rad=0.15")
    ax.text(33.5, 5.6, "guards $\\tau_a$", fontsize=6.6, color="0.35")

    # perception / estimation (middle)
    b_win = box(30, 40, 15, 9, "sliding window\n$x_t\\in\\mathbb{R}^{50\\times30}$\n"
                "$[q,\\dot q,e,\\dot e,\\tau_a]$", "#F4F4F4", fs=7.0)
    b_enc = box(50, 40, 17, 9, "temporal encoder\nConv1D($k$=5, 32 ch) + MLP\n"
                "$h\\in\\mathbb{R}^{64}$", "#E8F4FD", fs=7.0)
    b_z = box(72, 40, 15, 9, "synergy bottleneck\n$z\\in\\mathbb{R}^{4}$, tanh\n"
              "(4 muscle synergies)", "#DFF3EC", fs=7.0)
    b_th = box(92, 50, 20, 9, "$\\hat\\theta_{inj}$ head\nMLP($z$)$\\cdot\\sigma$"
               "$\\times$ bounds\n$\\Rightarrow$ dissipative comp.", "#FFF3D6", fs=7.0)
    b_rho = box(92, 31, 20, 9, "$\\hat\\rho$ head\n$\\sigma(w^\\top z)$\n"
                "patient contribution", "#FFF3D6", fs=7.0)

    arrow((45, 44.5), (50, 44.5))
    arrow((67, 44.5), (72, 44.5))
    arrow((87, 46.5), (92, 54), "", connectionstyle="arc3,rad=-0.2")
    arrow((87, 42.5), (92, 36), "", connectionstyle="arc3,rad=0.2")

    # control law (right)
    b_aan = box(52, 20, 27, 11,
                "AAN stiffness modulation (v1.1)\n$K_p(t)=K_p^{min}+(K_p^{max}-K_p^{min})(1-\\hat\\rho)$\n"
                "$K_d\\equiv K_d^{max}$  (full damping)", "#DFF3EC", fs=7.0)
    b_res = box(30, 8, 17, 9, "residual net $f_\\psi$\n$\\to$ low-pass\n"
                "$\\tau_c=0.05$ s", "#F4E8F4", fs=7.0)
    b_sum = box(90, 14, 11, 11, "$\\Sigma$\n$\\tau_a$", "#FFFFFF", fs=8.5,
                style="circle,pad=0.4")
    b_comp = box(52, 8, 33, 8,
                 "dissipative-safe spasticity comp. (v1.1)\n$\\tau_{comp}=-\\kappa\\hat\\lambda_s B_{sp}"
                 "\\odot\\tanh(\\dot q_f/0.5)\\odot|\\dot q_f|$",
                 "#FFF3D6", fs=7.0)

    arrow((92, 35.5), (65.5, 31), "$\\hat\\rho$", tdx=-1, tdy=1.2)
    # lambda_s_hat feeds the dissipative compensation (route around rho box)
    arrow((112, 52), (101.5, 22), "", connectionstyle="arc3,rad=-0.35",
          lw=1.0)
    ax.text(114.2, 38, "$\\hat\\lambda_s$", fontsize=6.8, rotation=90,
            va="center")
    arrow((68.5, 20), (68.5, 16), "", lw=1.0)
    arrow((85, 12), (90, 16), "", lw=1.0)
    arrow((47, 12.5), (52, 12.5), "", lw=1.0)
    arrow((79, 22), (92, 20), connectionstyle="arc3,rad=0.2",
          text="$K_p(t)e+K_d\\dot e$", tdx=-2.0, tdy=4.2)
    arrow((95.5, 14), (95.5, 4), "", lw=1.2)
    arrow((95.5, 4), (18, 4), "", lw=1.2)
    arrow((18, 14), (18, 22), "", lw=1.2)
    ax.text(60, 1.6, "$\\tau_a$ (saturated)", fontsize=7.0, ha="center")

    # passivity callout
    ax.text(2, 61.5, "v1.1 passivity-preserving triad:", fontsize=7.4,
            fontweight="bold", color=OI["verm"])
    ax.text(2, 58.7, "(1) $K_d$ never down-modulated  (2) residual torque low-passed  "
            "(3) compensation strictly dissipative", fontsize=7.0,
            color=OI["verm"])
    ax.text(60, 52.5, "estimation (measurable signals only)", fontsize=7.2,
            style="italic", color="0.35")
    savefig(fig, "fig1_principle")


# =============================================================== fig 2
def fig2_engineering():
    fig = plt.figure(figsize=(11.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.9], wspace=0.28)

    # (a) side-view engineering schematic ----------------------------------
    # Stylised planar layout using the nominal DH lengths (the real wrist
    # offsets leave the x-z plane, so the chain is drawn schematically).
    axa = fig.add_subplot(gs[0, 0])
    d1, a2, a3, d4, d5, d6 = 0.0892, 0.425, 0.392, 0.1093, 0.0948, 0.0823
    th2, th3 = np.deg2rad(58), np.deg2rad(-9)
    P = [(0.0, 0.0)]                                     # J1 (base)
    P.append((0.0, d1))                                  # J2
    P.append((P[-1][0] + a2 * np.cos(th2), P[-1][1] + a2 * np.sin(th2)))
    P.append((P[-1][0] + a3 * np.cos(th3), P[-1][1] + a3 * np.sin(th3)))
    P.append((P[-1][0] + d4 * np.cos(th3), P[-1][1] + d4 * np.sin(th3)))
    P.append((P[-1][0], P[-1][1] - d5))                  # J6 (wrist jog)
    P.append((P[-1][0] + d6 * np.cos(th3), P[-1][1] + d6 * np.sin(th3)))
    xs = [p[0] for p in P]; zs = [p[1] for p in P]
    axa.plot(xs, zs, "-", color=OI["blue"], lw=2.6, zorder=2)
    axa.plot(xs[:6], zs[:6], "o", ms=7, mfc="white", mec=OI["blue"],
             mew=1.6, zorder=3)
    axa.plot(xs[6], zs[6], "s", ms=6, color=OI["blue"], zorder=3)
    axa.add_patch(plt.Rectangle((-0.10, -0.05), 0.20, 0.05, fc="0.75",
                                ec="0.3", zorder=1))
    for i, lab in enumerate(["J1", "J2", "J3", "J4", "J5", "J6"]):
        dx, dy = [(-16, -4), (-14, 2), (-4, 8), (-4, -13), (7, 2),
                  (7, -9)][i]
        axa.annotate(lab, (xs[i], zs[i]), textcoords="offset points",
                     xytext=(dx, dy), fontsize=7.6, fontweight="bold",
                     color=OI["verm"])
    dims = [(0, 1, "$d_1$=0.0892", (-42, 0)),
            (1, 2, "$a_2$=0.425", (-46, 6)),
            (2, 3, "$a_3$=0.392", (-12, 12)),
            (3, 4, "$d_4$=0.1093", (-16, 11)),
            (4, 5, "$d_5$=0.0948", (10, 2)),
            (5, 6, "$d_6$=0.0823", (-12, -15))]
    for i, j, lab, off in dims:
        mx, mz = (xs[i] + xs[j]) / 2, (zs[i] + zs[j]) / 2
        axa.annotate(lab, (mx, mz), textcoords="offset points",
                     xytext=off, fontsize=7.0, color="0.2")
    axa.annotate("end-effector", (xs[6], zs[6]), textcoords="offset points",
                 xytext=(10, -3), fontsize=7.0, style="italic")
    axa.annotate("", xy=(0.16, 0.0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=OI["green"], lw=1.2))
    axa.annotate("", xy=(0.0, 0.16), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=OI["verm"], lw=1.2))
    axa.text(0.17, 0.005, "$x_0$", fontsize=7, color=OI["green"])
    axa.text(-0.042, 0.155, "$z_0$", fontsize=7, color=OI["verm"], ha="right")
    axa.set_xlabel("x (m)"); axa.set_ylabel("z (m)")
    axa.set_aspect("equal"); axa.set_title("(a) 6-DOF arm, side view (DH)")
    axa.grid(alpha=0.3, lw=0.4)
    axa.set_xlim(-0.3, 1.15); axa.set_ylim(-0.08, 0.85)

    # (b) Monte-Carlo reachable workspace ----------------------------------
    rng = np.random.default_rng(7)
    ns = 24000
    qs = rng.uniform(-np.pi, np.pi, (ns, 6))
    pts = np.empty((ns, 3))
    for i in range(ns):
        pts[i], _ = arm.fkine(qs[i])
    axb = fig.add_subplot(gs[0, 1], projection="3d")
    sc = axb.scatter(pts[::3, 0], pts[::3, 1], pts[::3, 2], s=2.5,
                     c=pts[::3, 2], cmap="viridis", alpha=0.5, lw=0)
    axb.set_xlabel("x (m)", fontsize=7); axb.set_ylabel("y (m)", fontsize=7)
    axb.set_title("(b) reachable workspace (Monte-Carlo FK)")
    axb.view_init(elev=18, azim=-58)
    axb.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axb.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axb.set_zticks([-0.5, 0.0, 0.5, 1.0])
    axb.tick_params(labelsize=6)
    cb = fig.colorbar(sc, ax=axb, shrink=0.55, pad=0.08)
    cb.ax.set_title("z (m)", fontsize=6.5, pad=3)
    cb.ax.tick_params(labelsize=6)
    r = np.linalg.norm(pts[:, :2], axis=1)
    axb.text2D(0.02, 0.93, f"N={ns}, reach $\\lesssim$0.92 m",
               transform=axb.transAxes, fontsize=6.5)

    # (c) DH table ----------------------------------------------------------
    axc = fig.add_subplot(gs[0, 2]); axc.axis("off")
    rows = [["1", "0", "$+\\pi/2$", "0.0892", "$q_1$"],
            ["2", "$-0.425$", "0", "0", "$q_2$"],
            ["3", "$-0.392$", "0", "0", "$q_3$"],
            ["4", "0", "$+\\pi/2$", "0.1093", "$q_4$"],
            ["5", "0", "$-\\pi/2$", "0.0948", "$q_5$"],
            ["6", "0", "0", "0.0823", "$q_6$"]]
    tab = axc.table(cellText=rows,
                    colLabels=["$i$", "$a_{i-1}$ (m)", "$\\alpha_{i-1}$",
                               "$d_i$ (m)", "$\\theta_i$"],
                    loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(7.2); tab.scale(1.0, 1.35)
    axc.set_title("(c) standard DH parameters", fontsize=8.5)
    axc.text(0.5, 0.13, "masses (kg): 3.70, 8.39, 2.28, 1.22, 1.22, 0.22\n"
             "$\\tau_{lim}$=[150,150,150,28,28,28] N$\\cdot$m, "
             "$|\\dot q|\\leq3$ rad/s",
             transform=axc.transAxes, ha="center", fontsize=6.8)
    savefig(fig, "fig2_engineering")


# =============================================================== fig 3
def fig3_data(summ, stats, pf, subj):
    dz = np.load(os.path.join(ROOT, "data", "dataset.npz"))
    tau_ff_all = dz["ref_tau_ff"].astype(float)
    dt = 0.01
    z = traj("iira", 1)
    t = z["t"]
    demo = {"mild": 0, "moderate": 24, "severe": 47}     # fold-1 test subjects
    di = 0                                                # same direction
    lam_of = {r.subject: np.array([r.lambda_w, r.lambda_r, r.lambda_s,
                                   r.lambda_n])
              for r in subj.itertuples()}
    perm_of = {r.subject: np.array([int(v) for v in str(r.perm).split()])
               for r in subj.itertuples()}

    fig = plt.figure(figsize=(11.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.15, 0.95], wspace=0.3)

    # (a) end-effector trajectories vs reference
    axa = fig.add_subplot(gs[0, 0])
    axa.plot(z["p_ref"][0, :, 0], z["p_ref"][0, :, 1], "k--", lw=1.4,
             label="reference")
    for s, sid in demo.items():
        i = find_trial(z, sid, di)
        axa.plot(z["p"][i, :, 0], z["p"][i, :, 1], color=STRATUM_C[s],
                 lw=1.3, label=s)
    axa.set_xlabel("x (m)"); axa.set_ylabel("y (m)")
    axa.set_title("(a) reaching trajectories, IIRA, direction 0")
    axa.legend(frameon=False, loc="best"); axa.grid(alpha=0.3, lw=0.4)

    # (b) tau_patient (offline replay) vs tau_a, moderate subject
    axb = fig.add_subplot(gs[0, 1])
    i = find_trial(z, demo["moderate"], di)
    tp, tv, _ = inj.replay_tau_patient(z["qd"][i], tau_ff_all[di * 5 + 2],
                                       lam_of[demo["moderate"]],
                                       perm_of[demo["moderate"]], dt)
    axb.plot(t, np.linalg.norm(tp, axis=1), color=OI["green"], lw=1.3,
             label="$\\|\\tau_{patient}\\|$ (replay)")
    axb.plot(t, np.linalg.norm(z["tau_a"][i], axis=1), color=OI["blue"],
             lw=1.3, label="$\\|\\tau_a\\|$ (robot)")
    axb.plot(t, np.linalg.norm(tv, axis=1), color=OI["green"], lw=0.9,
             ls=":", label="$\\|\\tau_{volitional}\\|$")
    axb.set_xlabel("t (s)"); axb.set_ylabel("torque norm (N$\\cdot$m)")
    axb.set_title(f"(b) torque sharing, moderate subject "
                  f"(FMA={subj.loc[subj.subject == demo['moderate'], 'fma'].iloc[0]:.0f})")
    axb.legend(frameon=False, loc="best"); axb.grid(alpha=0.3, lw=0.4)

    # (c) FMA histogram by stratum
    axc = fig.add_subplot(gs[0, 2])
    bins = np.arange(2, 70, 3)
    for s in ["severe", "moderate", "mild"]:
        f = subj.loc[subj.stratum == s, "fma"]
        axc.hist(f, bins=bins, color=STRATUM_C[s], alpha=0.75,
                 label=f"{s} (n={len(f)}, $\\bar x$={f.mean():.1f})",
                 edgecolor="white", lw=0.4)
    axc.set_xlabel("synthetic FMA-UE score (0-66)")
    axc.set_ylabel("subjects")
    axc.set_title("(c) virtual cohort (60 subjects)")
    axc.legend(frameon=False, loc="upper left")
    axc.grid(alpha=0.3, lw=0.4, axis="y")
    savefig(fig, "fig3_data")


# =============================================================== fig 4
def fig4_main(summ, stats, pf, subj):
    s16 = summ[summ.model.isin(MAIN16)].set_index("model").loc[MAIN16]
    order = s16.sort_values("rmse_p_mm_mean").index.tolist()
    sig = {(r.metric, r.baseline): bool(r.significant)
           for r in stats.itertuples()}

    fig = plt.figure(figsize=(11.5, 7.6))
    gs = fig.add_gridspec(2, 10, height_ratios=[1.05, 1.0], hspace=0.52,
                          wspace=0.62)
    panels = [("rmse_p_mm", "RMSE (mm)", False),
              ("norm_jerk", "norm. jerk (log)", True),
              ("viol_pct", "violations (%)", False),
              ("rho_realized", "$\\rho_{realized}$", False),
              ("rho_dynamic", "$\\rho_{dynamic}$", False)]
    x = np.arange(16)
    for j, (m, ylab, logy) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        vals = s16.loc[order, m + "_mean"].to_numpy()
        errs = s16.loc[order, m + "_std"].to_numpy()
        cols = [OI["verm"] if mo == "iira" else OI["sky"] for mo in order]
        ax.bar(x, vals, yerr=errs, color=cols, capsize=1.6,
               error_kw=dict(lw=0.6), width=0.75)
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        star_ys = []
        for xi_, mo in zip(x, order):
            if mo != "iira" and sig.get((m, mo), False):
                i = list(order).index(mo)
                sy = (vals[i] * 1.35 if logy else
                      vals[i] + errs[i] * 1.1 + 0.02 * vals.max())
                ax.text(xi_, sy, "*", ha="center", fontsize=9, color="k")
                star_ys.append(sy)
        if star_ys:                    # keep stars inside the axes
            etop = (vals + errs).max()
            if logy:
                ax.set_ylim(top=max(etop, max(star_ys)) * 1.5)
            else:
                ax.set_ylim(top=max(etop * 1.12,
                                    max(star_ys) + 0.05 * vals.max()))
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[m] for m in order], rotation=80,
                           ha="right", fontsize=6.0)
        ax.set_ylabel(ylab, fontsize=6.8, labelpad=1)
        ax.grid(alpha=0.3, lw=0.4, axis="y")
        if j == 0:
            ax.set_title("(a) benchmark, 16 controllers "
                         "(mean$\\pm$SD; * $p_{Holm}$<0.05 vs IIRA)",
                         fontsize=8, loc="left")

    # (b) radar, normalised (higher = better on every axis)
    axr = fig.add_subplot(gs[1, :4], projection="polar")
    axes_def = [("tracking", "rmse_p_mm", False),
                ("smoothness", "norm_jerk", False),
                ("low effort", "tau_rms", False),
                ("safety", "viol_pct", False),
                ("settling", "settle_mm", False),
                ("$\\rho_{dynamic}$", "rho_dynamic", True)]
    norm = {}
    for _, m, higher in axes_def:
        v = s16[m + "_mean"].to_numpy(float)
        lo, hi = v.min(), v.max()
        nv = (v - lo) / (hi - lo + 1e-12)
        norm[m] = nv if higher else 1.0 - nv
    radar_models = ["iira", "mlp", "lstm", "impedance", "smc"]
    cols_r = [OI["verm"], OI["blue"], OI["sky"], OI["green"], OI["orange"]]
    ang = np.linspace(0, 2 * np.pi, len(axes_def), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    for mo, c in zip(radar_models, cols_r):
        v = np.array([norm[m][list(s16.index).index(mo)] for _, m, _ in
                      axes_def])
        v = np.concatenate([v, v[:1]])
        axr.plot(ang_c, v, color=c, lw=1.4, label=DISPLAY[mo])
        axr.fill(ang_c, v, color=c, alpha=0.07)
    axr.set_xticks(ang)
    axr.set_xticklabels([a for a, _, _ in axes_def], fontsize=7)
    axr.tick_params(axis="x", pad=9)
    axr.set_yticks([0.25, 0.5, 0.75, 1.0]); axr.set_yticklabels(
        ["0.25", "0.5", "0.75", "1"], fontsize=6)
    for tl in axr.get_yticklabels():
        tl.set_bbox(dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                         alpha=0.75))
    axr.set_title("(b) normalised profile (1 = best of 16)", fontsize=8,
                  pad=14)
    axr.legend(frameon=False, loc="lower right",
               bbox_to_anchor=(1.38, -0.12), fontsize=6.8)

    # (c) Pareto: RMSE vs rho_dynamic (gravity-compensated assist removed)
    axp = fig.add_subplot(gs[1, 4:])
    xv = s16["rmse_p_mm_mean"].to_numpy()
    yv = s16["rho_dynamic_mean"].to_numpy()
    # label placement: compact offsets in points (y-range is compressed, so
    # the central cluster labels are spread by hand); LQR and MPC sit on the
    # identical operating point and get one shared label.
    off = dict(smc=(8, -2), ilc=(-6, 9), rbf=(8, -3),
               transformer=(12, 8), impedance=(8, 10), iira=(10, 9),
               admittance=(-16, -13), dob=(-4, -13), pid=(8, 3),
               computed_torque=(-21, -13))
    lead = {"mlp": (55, 0.4838), "pd_grav": (55, 0.4882),
            "gru": (108, 0.4718), "lstm": (127, 0.4683)}
    for mo, xx, yy in zip(s16.index, xv, yv):
        c = OI["verm"] if mo == "iira" else OI["blue"]
        axp.scatter(xx, yy, s=46 if mo == "iira" else 26,
                    color=c, zorder=3, edgecolor="white", lw=0.5)
        if mo in ("lqr", "mpc"):
            continue                      # identical operating point
        if mo in lead:
            axp.annotate(DISPLAY[mo], (xx, yy), xytext=lead[mo], fontsize=6.4,
                         arrowprops=dict(arrowstyle="-", lw=0.5,
                                         color="0.55", shrinkA=0, shrinkB=2))
        else:
            axp.annotate(DISPLAY[mo], (xx, yy), textcoords="offset points",
                         xytext=off[mo], fontsize=6.4)
    axp.annotate("LQR = MPC", (xv[list(s16.index).index("lqr")],
                               yv[list(s16.index).index("lqr")]),
                 xytext=(268, 0.4865), fontsize=6.4)
    axp.set_xlabel("end-eff. RMSE (mm)")
    axp.set_ylabel("$\\rho_{dynamic}$")
    axp.set_xlim(-28, 448); axp.set_ylim(0.42, 0.505)
    axp.set_title("(c) accuracy–patient-contribution Pareto map", fontsize=8)
    axp.grid(alpha=0.3, lw=0.4)
    axp.annotate("upper-left = desirable", xy=(0.44, 0.05),
                 xycoords="axes fraction", fontsize=6.8, style="italic",
                 color="0.35")
    savefig(fig, "fig4_main")


# =============================================================== fig 5
def fig5_training(summ, stats, pf, subj):
    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.5,
                          wspace=0.4)

    # (a) training / validation loss curves (torch models)
    axa = fig.add_subplot(gs[0, :2])
    for name, c in [("iira", OI["verm"]), ("lstm", OI["blue"]),
                    ("gru", OI["sky"]), ("transformer", OI["green"])]:
        tr, vl = [], []
        for fold in range(1, 6):
            lg = pd.read_csv(os.path.join(ROOT, "results", "logs",
                                          f"{name}_fold{fold}.csv"))
            tr.append(lg.train_loss.to_numpy())
            vl.append(lg.val_loss.to_numpy())
        ep = np.arange(len(tr[0]))
        trm = np.mean(tr, axis=0); vlm = np.mean(vl, axis=0)
        axa.plot(ep, trm, color=c, lw=1.3, label=f"{DISPLAY[name]} train")
        axa.plot(ep, vlm, color=c, lw=1.3, ls="--",
                 label=f"{DISPLAY[name]} val")
    axa.set_yscale("log")
    axa.set_xlabel("epoch"); axa.set_ylabel("loss (log)")
    axa.set_title("(a) training / validation loss (mean over 5 folds)",
                  fontsize=8)
    axa.legend(frameon=False, ncol=2, fontsize=6.6)
    axa.grid(alpha=0.3, lw=0.4, which="both")

    # (c) ablation bars
    axc = fig.add_subplot(gs[0, 2:])
    abl = summ[summ.model.isin(ABL6)].set_index("model").loc[ABL6]
    mets = [("rmse_p_mm", "RMSE (mm)", 1.0),
            ("norm_jerk", "norm. jerk", 1e-7),
            ("viol_pct", "viol. (%)", 1.0),
            ("rho_realized", "$\\rho_{realized}$", 1.0)]
    x = np.arange(6); w = 0.19
    cols_m = [OI["blue"], OI["sky"], OI["orange"], OI["green"]]
    for k, (m, lab, sc) in enumerate(mets):
        v = abl[m + "_mean"].to_numpy() * sc
        vn = v / v.max()
        axc.bar(x + (k - 1.5) * w, vn, w, color=cols_m[k], label=lab)
    axc.set_xticks(x)
    axc.set_xticklabels([DISPLAY[m] for m in ABL6], rotation=30, ha="right",
                        fontsize=6.6)
    axc.set_ylabel("normalised value (per metric)")
    axc.set_title("(c) IIRA ablations (each bar normalised to its worst "
                  "variant)", fontsize=8, pad=22)
    axc.legend(frameon=False, fontsize=6.6, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), columnspacing=0.9,
               handlelength=1.3)
    axc.grid(alpha=0.3, lw=0.4, axis="y")

    # (b) theta_hat vs theta, 4 subplots
    est = iira_estimates()
    th_hat = est["theta_hat"]            # (480, 4)
    meta = est["meta"]                   # (fold, subject, direction)
    lam_true = np.array([subj.loc[subj.subject == s,
                                  ["lambda_w", "lambda_r", "lambda_s",
                                   "lambda_n"]].iloc[0].to_numpy(float)
                         for _, s, _ in meta])
    names = ["$\\lambda_w$ (weakness)", "$\\lambda_r$ (synergy diss.)",
             "$\\lambda_s$ (spasticity)", "$\\lambda_n$ (noise)"]
    # sanity check against stored per-trial theta_mae
    pf_iira = pf[pf.model == "iira"]
    stored = pf_iira.sort_values(["fold", "subject", "trial"])
    ord_idx = np.lexsort((meta[:, 2], meta[:, 1], meta[:, 0]))
    mae_rep = np.abs(th_hat[ord_idx] - lam_true[ord_idx]).mean(axis=1)
    dmax = np.abs(mae_rep - stored.theta_mae.to_numpy()).max()
    # trajectories are stored in float32, so the replay differs from the
    # online (float64) estimates by <= ~2e-3 per trial; overall MAE matches
    # metrics_summary (0.2988) to 4 decimals.
    assert dmax < 5e-3, f"theta_hat replay mismatch vs metrics_per_fold ({dmax})"
    for j in range(4):
        ax = fig.add_subplot(gs[1, j])
        xx, yy = lam_true[:, j], th_hat[:, j]
        for s, c in STRATUM_C.items():
            m = np.array([subj.loc[subj.subject == s_, "stratum"].iloc[0]
                          == s for _, s_, _ in meta])
            ax.scatter(xx[m], yy[m], s=7, color=c, alpha=0.65, lw=0,
                       label=s if j == 0 else None)
        lo, hi = 0, inj.THETA_MAX[j]
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        r2 = 1 - np.sum((xx - yy) ** 2) / np.sum((xx - xx.mean()) ** 2)
        mae = np.abs(xx - yy).mean()
        ax.text(0.04, 0.92, f"$R^2$={r2:.2f}\nMAE={mae:.3f}",
                transform=ax.transAxes, fontsize=6.8, va="top")
        ax.set_xlabel(f"true {names[j]}"); ax.set_ylabel("estimated")
        ax.grid(alpha=0.3, lw=0.4)
        if j == 0:
            ax.set_title("(b) IIRA injury-parameter estimates "
                         "($\\hat\\theta$ vs $\\theta$, all test trials)",
                         fontsize=8, loc="left")
            ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    savefig(fig, "fig5_training")


# =============================================================== fig 6
def fig6_stability(summ, stats, pf, subj):
    summ_v1 = pd.read_csv(os.path.join(ROOT, "results_archive_v1_naive",
                                       "metrics_summary.csv"))
    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    # (a) v1 chattering trial vs v1.1, same subject/direction
    z1 = traj("iira", 5, archive=True)
    z2 = traj("iira", 5)
    i1 = find_trial(z1, 11, 7)
    i2 = find_trial(z2, 11, 7)
    t = z1["t"]
    axa = fig.add_subplot(gs[0, 0])
    axa.plot(t, np.linalg.norm(z1["qd"][i1], axis=1), color=OI["verm"],
             lw=0.9, label="v1.0 (naive AAN)")
    axa.plot(t, np.linalg.norm(z2["qd"][i2], axis=1), color=OI["blue"],
             lw=0.9, label="v1.1 (passivity fix)")
    axa.set_ylabel("$\\|\\dot q\\|$ (rad/s)")
    axa.set_title("(a) chattering trial: IIRA v1.0 vs v1.1 "
                  "(subject 11, direction 7)", fontsize=8)
    axa.legend(frameon=True, framealpha=0.95, edgecolor="0.85",
               loc="upper left", fontsize=6.8)
    axa.grid(alpha=0.3, lw=0.4)
    axa2 = axa.axes.twinx()
    axa2.plot(t, np.linalg.norm(z1["tau_a"][i1], axis=1), color=OI["verm"],
              lw=0.7, ls=":", alpha=0.85)
    axa2.plot(t, np.linalg.norm(z2["tau_a"][i2], axis=1), color=OI["blue"],
              lw=0.7, ls=":", alpha=0.85)
    axa2.set_ylabel("$\\|\\tau_a\\|$ (N$\\cdot$m, dotted)", fontsize=7)
    axa2.tick_params(labelsize=6.5)
    axa.set_xlabel("t (s)")

    # (b) jerk: v1 vs v1.1 per IIRA variant (log)
    axb = fig.add_subplot(gs[0, 1])
    v1 = summ_v1.set_index("model").loc[ABL6, "norm_jerk_mean"].to_numpy()
    v2 = summ.set_index("model").loc[ABL6, "norm_jerk_mean"].to_numpy()
    x = np.arange(6); w = 0.38
    axb.bar(x - w / 2, v1, w, color=OI["verm"], label="v1.0")
    axb.bar(x + w / 2, v2, w, color=OI["blue"], label="v1.1")
    axb.set_yscale("log")
    axb.set_xticks(x)
    axb.set_xticklabels([DISPLAY[m] for m in ABL6], rotation=30, ha="right",
                        fontsize=6.6)
    axb.set_ylabel("normalised jerk (log)")
    axb.set_title("(b) closed-loop jerk, IIRA variants: v1.0 vs v1.1",
                  fontsize=8)
    axb.legend(frameon=False); axb.grid(alpha=0.3, lw=0.4, axis="y",
                                        which="both")

    # (c) strata x representative models, RMSE
    axc = fig.add_subplot(gs[1, 0])
    reps = ["pid", "impedance", "lstm", "iira"]
    df5 = pf[(pf.model != "ilc") | (pf.rep == 4)]
    g = (df5[df5.model.isin(reps)].groupby(["stratum", "model"])
         ["rmse_p_mm"].mean().unstack())
    g = g.loc[["mild", "moderate", "severe"], reps]
    x = np.arange(3); w = 0.2
    for k, mo in enumerate(reps):
        axc.bar(x + (k - 1.5) * w, g[mo].to_numpy(), w,
                color=[OI["grey"], OI["green"], OI["sky"], OI["verm"]][k],
                label=DISPLAY[mo])
    axc.set_xticks(x); axc.set_xticklabels(["mild", "moderate", "severe"])
    axc.set_ylabel("end-eff. RMSE (mm)")
    axc.set_title("(c) tracking error by impairment stratum", fontsize=8)
    axc.legend(frameon=False, fontsize=6.8)
    axc.grid(alpha=0.3, lw=0.4, axis="y")

    # (d) AAN assistance-as-needed: rho_hat(t) and K_p(t)
    axd = fig.add_subplot(gs[1, 1])
    est = iira_estimates()
    sel = None
    for k, (fold, sid, di) in enumerate(est["meta"]):
        if fold == 1 and sid == 24 and di == 0:
            sel = k
    rho_t = est["rho_series"][sel]
    t = np.arange(len(rho_t)) * 0.01
    kp1 = neu_kp(rho_t)[:, 0]
    axd.plot(t, rho_t, color=OI["green"], lw=1.4, label="$\\hat\\rho(t)$")
    axd.set_xlabel("t (s)"); axd.set_ylabel("$\\hat\\rho$", color=OI["green"])
    axd.set_ylim(0, 1)
    axd.tick_params(axis="y", labelcolor=OI["green"])
    axd2 = axd.twinx()
    axd2.plot(t, kp1, color=OI["verm"], lw=1.4, ls="--",
              label="$K_p(t)$ joint 1")
    axd2.set_ylabel("$K_p(t)$ (N$\\cdot$m/rad)", color=OI["verm"])
    axd2.tick_params(axis="y", labelcolor=OI["verm"], labelsize=6.5)
    axd.set_title("(d) AAN modulation, moderate subject (IIRA, fold 1)",
                  fontsize=8)
    axd.grid(alpha=0.3, lw=0.4)
    h1, l1 = axd.get_legend_handles_labels()
    h2, l2 = axd2.get_legend_handles_labels()
    axd.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.8, loc="center right")
    savefig(fig, "fig6_stability")


def neu_kp(rho):
    """K_p(t) for a rho_hat series (N,) -> (N, 6), v1.1 stiffness schedule."""
    from controllers import neural as neu
    return neu.KP_MIN[None, :] + (neu.KP_MAX - neu.KP_MIN)[None, :] * \
        (1.0 - np.asarray(rho)[:, None])


def main():
    print("loading results ...", flush=True)
    summ, stats, pf, subj = load_all()
    fig1_principle()
    fig2_engineering()
    fig3_data(summ, stats, pf, subj)
    fig4_main(summ, stats, pf, subj)
    fig5_training(summ, stats, pf, subj)
    fig6_stability(summ, stats, pf, subj)
    print("all figures done.")


if __name__ == "__main__":
    main()
