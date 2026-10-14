"""fig7_transfer: six-panel sim-to-real transfer figure.

Reads only results/transfer/*.csv / *.npz / mmd.json (no re-computation).
PNG 300 dpi + PDF, Okabe-Ito palette.

Panels:
  a  transfer pipeline schematic
  b  PCA of window summary features, sim vs real (shape = domain,
     colour = severity for the real windows)
  c  true vs predicted FMA-UE, pooled subject-level CV predictions
  d  confusion matrix of the best model (by pooled macro-F1)
  e  four key metrics (MAE, R2, macro-F1, Acc), fold mean +/- std
  f  per-severity accuracy (pooled)
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "transfer")
FIG = os.path.join(ROOT, "figures")

OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", pink="#CC79A7", black="#000000",
          grey="#999999")
SEV_C = {"mild": OI["sky"], "moderate": OI["orange"], "severe": OI["verm"]}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 110, "savefig.dpi": 300, "axes.linewidth": 0.7,
    "lines.linewidth": 1.2, "font.family": "DejaVu Sans",
})

MODELS = ["iira_zero_shot", "iira_ft_head", "iira_ft_full", "mlp", "lstm",
          "gru", "transformer"]
DISP = dict(iira_zero_shot="IIRA zero-shot", iira_ft_head="IIRA ft-head",
            iira_ft_full="IIRA ft-full", mlp="MLP", lstm="LSTM", gru="GRU",
            transformer="Transformer")
MC = dict(zip(MODELS, [OI["blue"], OI["sky"], OI["green"], OI["orange"],
                       OI["verm"], OI["pink"], OI["yellow"]]))
CLASSES = ["severe", "moderate", "mild"]


def panel_schematic(ax):
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=OI["black"], lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=7.5)

    def arrow(x1, y1, x2, y2, text="", dy=0.25):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=OI["black"], lw=1.1))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center",
                    fontsize=7, style="italic")

    box(0.2, 3.6, 2.9, 1.9, "UR5 6-DOF simulator\n60 virtual subjects\n"
        "IIRA trained (5-fold)", OI["sky"] + "44")
    box(0.2, 0.4, 2.9, 1.9, "Real stroke kinematics\nGurari 2018 (CC0)\n"
        "17 subjects, elbow", OI["verm"] + "33")
    box(4.1, 2.3, 2.2, 1.5, "IIRA encoder\nconv+fc+$z^4$", "#EEEEEE")
    box(7.2, 3.8, 2.5, 1.3, "$\\hat{\\theta}\\!\\rightarrow\\!$ FMA\n"
        "(zero-shot)", OI["blue"] + "33")
    box(7.2, 1.0, 2.5, 1.3, "new head\n(fine-tune)", OI["green"] + "44")
    arrow(3.1, 4.55, 4.1, 3.4, "pre-train")
    arrow(3.1, 1.35, 4.1, 2.7)
    ax.text(4.4, 0.22, "W=50 windows @100 Hz\n$[q,\\dot q,e,\\dot e,\\tau]$",
            ha="center", va="center", fontsize=6.5, style="italic")
    arrow(6.3, 3.35, 7.2, 4.45)
    arrow(6.3, 2.75, 7.2, 1.65)
    ax.text(8.45, 3.05, "5-fold\nsubject CV", ha="center", fontsize=7,
            style="italic")
    ax.text(5.0, 5.7, "sim-to-real transfer: FMA-UE regression & "
            "severity classification", ha="center", fontsize=8)


def panel_pca(subfig, pca):
    ax, ax2 = subfig.subplots(1, 2, gridspec_kw=dict(wspace=0.35))
    sim, real = pca["sim"], pca["real"]
    sev = pca["real_severity"]

    def draw(a, S, R, ttl):
        a.scatter(S[:, 0], S[:, 1], s=3, c=OI["grey"], marker="^",
                  alpha=0.35, linewidths=0, label="sim")
        for ci, cname in enumerate(CLASSES):
            m = sev == ci
            a.scatter(R[m, 0], R[m, 1], s=5, c=SEV_C[cname], marker="o",
                      alpha=0.6, linewidths=0, label=f"real ({cname})")
        a.set_title(ttl, fontsize=8)

    ex = pca["explained"]
    draw(ax, sim, real, "PCA")
    ax.set_xlabel(f"PC1 ({ex[0] * 100:.0f}%)")
    ax.set_ylabel(f"PC2 ({ex[1] * 100:.0f}%)")
    ax.legend(frameon=False, markerscale=2, loc="best", fontsize=6)
    draw(ax2, pca["tsne_sim"], pca["tsne_real"], "t-SNE")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    mmd = json.load(open(os.path.join(OUT, "mmd.json")))
    subfig.suptitle(f"MMD$^2$ = {mmd['raw_summary']['mmd2']:.3f} "
                    f"(p = {mmd['raw_summary']['p_value']:.3f}); "
                    f"$h$-space {mmd['iira_h']['mmd2']:.3f} "
                    f"(p = {mmd['iira_h']['p_value']:.3f})",
                    fontsize=7.5, y=0.99)


def panel_scatter(ax, pred):
    lim = (0, 70)
    ax.plot(lim, lim, "--", c=OI["grey"], lw=0.9)
    for m in MODELS:
        d = pred[pred.model == m]
        ax.scatter(d.fma_true, d.fma_pred, s=16, c=MC[m], alpha=0.8,
                   label=DISP[m], linewidths=0.3, edgecolors="white")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("true FMA-UE")
    ax.set_ylabel("predicted FMA-UE")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")


def panel_confusion(ax, pred, summ):
    f1 = summ[summ.metric == "macro_f1"].set_index("model")["pooled"]
    best = f1.idxmax()
    d = pred[pred.model == best]
    yt = np.digitize(d.fma_true, [28.5, 42.5])
    yp = np.digitize(d.fma_pred, [28.5, 42.5])
    C = np.zeros((3, 3), int)
    for a, b in zip(yt, yp):
        C[a, b] += 1
    ax.imshow(C, cmap="Blues", vmin=0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, C[i, j], ha="center", va="center",
                    color="white" if C[i, j] > C.max() / 2 else OI["black"],
                    fontsize=10)
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"best: {DISP[best]} (pooled macro-F1 = {f1.max():.2f})",
                 fontsize=7.5)


def panel_metrics(subfig, summ):
    from matplotlib.ticker import MaxNLocator
    axes = subfig.subplots(1, 4, gridspec_kw=dict(wspace=0.6))
    spec = [("mae", "MAE", False), ("r2", "$R^2$", True),
            ("macro_f1", "macro-F1", True), ("acc", "accuracy", True)]
    short = dict(iira_zero_shot="ZS", iira_ft_head="FTH", iira_ft_full="FTF",
                 mlp="MLP", lstm="LSTM", gru="GRU", transformer="TRF")
    for ax, (key, label, higher) in zip(axes, spec):
        d = summ[summ.metric == key].set_index("model").loc[MODELS]
        x = np.arange(len(MODELS))
        ax.bar(x, d.fold_mean, yerr=d.fold_std, capsize=2,
               color=[MC[m] for m in MODELS], ec=OI["black"], lw=0.4,
               error_kw=dict(lw=0.7))
        ax.set_xticks(x, [short[m] for m in MODELS], rotation=45)
        ax.set_title(label, fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="y", labelsize=6, pad=1)
        ax.tick_params(axis="x", labelsize=6)


def panel_severity(ax, sev):
    width = 0.11
    x = np.arange(3)
    for i, m in enumerate(MODELS):
        d = sev[sev.model == m].set_index("severity").loc[CLASSES]
        ax.bar(x + (i - 3) * width, d.acc, width * 0.92, color=MC[m],
               ec=OI["black"], lw=0.3, label=DISP[m])
    ax.set_xticks(x, [f"{c}\n(n={int(sev[(sev.model == 'mlp') & (sev.severity == c)].n.iloc[0])})"
                      for c in CLASSES])
    ax.set_ylabel("accuracy (pooled)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=5.6, ncol=2)


def main():
    pred = pd.read_csv(os.path.join(OUT, "predictions.csv"))
    summ = pd.read_csv(os.path.join(OUT, "metrics_summary.csv"))
    sev = pd.read_csv(os.path.join(OUT, "metrics_by_severity.csv"))
    pca = np.load(os.path.join(OUT, "pca_data.npz"))

    fig = plt.figure(figsize=(10.5, 9.5))
    sf = fig.subfigures(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.08,
                        wspace=0.12)
    labels = "abcdef"
    for i, s in enumerate(sf.flat):
        s.suptitle(f"({labels[i]})", x=0.02, y=0.99, ha="left", fontsize=10,
                   fontweight="bold")
    panel_schematic(sf[0, 0].subplots())
    panel_pca(sf[0, 1], pca)
    panel_scatter(sf[1, 0].subplots(), pred)
    panel_confusion(sf[1, 1].subplots(), pred, summ)
    panel_metrics(sf[2, 0], summ)
    panel_severity(sf[2, 1].subplots(), sev)

    fig.savefig(os.path.join(FIG, "fig7_transfer.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig7_transfer.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig7_transfer.png/.pdf saved")


if __name__ == "__main__":
    main()
