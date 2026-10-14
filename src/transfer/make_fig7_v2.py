"""fig7_transfer_v2 (final): six-panel transfer figure for round 2.

Panel (a) follows the user's flowchart template (same renderer language as
src/make_patent_flow.py: bordered title bar, light-blue ellipse terminals,
white black-edged rectangles, orange-edged diamond decision, orthogonal
snake arrows, bottom-centred caption).

Data sources (read-only): results/transfer_v2/{predictions_repmean.csv,
metrics_summary.csv, metrics_by_severity.csv, confusion/confusion_counts.csv,
pca_data.npz, mmd_v2.json}.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "transfer_v2")
FIG = os.path.join(ROOT, "figures")

OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", pink="#CC79A7", black="#000000",
          grey="#999999")
SEV_C = {"mild": OI["sky"], "moderate": OI["orange"], "severe": OI["verm"]}
LIGHT_BLUE = "#D6E4F0"
ORANGE_E = "#E36C0A"
CLASSES = ["severe", "moderate", "mild"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 110, "savefig.dpi": 300, "axes.linewidth": 0.7,
    "lines.linewidth": 1.2, "font.family": "DejaVu Sans",
})

MODELS = ["iira_zero_shot", "iira_ft_head", "iira_ft_full",
          "iira_v2_zero_shot", "iira_v2_ft", "iira_v2_loso_global",
          "iira_v2_calib1t", "iira_v2_calib2t",
          "ridge", "rf", "gbt", "mlp", "lstm", "gru", "transformer"]
ORACLE = {"iira_v2_calib1t", "iira_v2_calib2t"}  # use true FMA of calib trial(s)
DISP = dict(iira_zero_shot="IIRA zero-shot", iira_ft_head="IIRA ft-head",
            iira_ft_full="IIRA ft-full",
            iira_v2_zero_shot="IIRA-v2 zero-shot", iira_v2_ft="IIRA-v2 ft",
            iira_v2_loso_global="IIRA-v2 LOSO global",
            iira_v2_calib1t="IIRA-v2 calib-1t (oracle)",
            iira_v2_calib2t="IIRA-v2 calib-2t (oracle)",
            ridge="Ridge", rf="RF", gbt="GBT", mlp="MLP", lstm="LSTM",
            gru="GRU", transformer="Transformer")
MC = dict(zip(MODELS, [OI["blue"], OI["sky"], OI["green"], "#332288",
                       "#117733", "#882255", "#44AA99", "#CC6677",
                       OI["orange"], "#DDCC77", "#AA4499",
                       OI["verm"], OI["pink"], OI["yellow"], OI["grey"]]))
SHORT = dict(iira_zero_shot="ZS", iira_ft_head="FTH", iira_ft_full="FTF",
             iira_v2_zero_shot="ZSv2", iira_v2_ft="FTv2",
             iira_v2_loso_global="LOSO", iira_v2_calib1t="C1t",
             iira_v2_calib2t="C2t", ridge="Ridge", rf="RF", gbt="GBT",
             mlp="MLP", lstm="LSTM", gru="GRU", transformer="TRF")


# ------------------------------------------------- panel (a): template flow
def _orth_arrow(ax, pts, lw=1.1):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color="black", lw=lw, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color="black", lw=lw,
                                mutation_scale=11, shrinkA=0, shrinkB=0))


def panel_flow(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")
    fs = 6.3
    rw, rh = 25, 13
    ew, eh = 12, 8
    dw, dh = 22, 16

    def ellipse(cx, cy, text):
        ax.add_patch(Ellipse((cx, cy), ew, eh, fc=LIGHT_BLUE, ec="black",
                             lw=1.0))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def rect(cx, cy, text, w=None):
        w = w or rw
        ax.add_patch(Rectangle((cx - w / 2, cy - rh / 2), w, rh, fc="white",
                               ec="black", lw=1.0))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def diamond(cx, cy, text):
        pts = [(cx - dw / 2, cy), (cx, cy + dh / 2), (cx + dw / 2, cy),
               (cx, cy - dh / 2)]
        ax.add_patch(Polygon(pts, closed=True, fc="white", ec=ORANGE_E,
                             lw=1.5))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    # bordered title bar
    ax.add_patch(Rectangle((10, 64.5), 80, 5.4, fc="white", ec="black",
                           lw=1.1))
    ax.text(50, 67.2, "Sim-to-real transfer protocol (round 2)",
            ha="center", va="center", fontsize=8.0, fontweight="bold")

    # row 1: start -> r1 -> r2 -> diamond
    y1 = 53
    xs, x1, x2, xd = 8, 30, 58, 82
    ellipse(xs, y1, "Start")
    rect(x1, y1, "Sim pre-training\nIIRA, 60 virtual\nsubjects (5 folds)")
    rect(x2, y1, "De-saturated\nbottleneck (linear +\nLayerNorm) + sim\n"
                 "recalibration")
    diamond(xd, y1, "zero-shot\nonly?")
    _orth_arrow(ax, [(xs + ew / 2, y1), (x1 - rw / 2, y1)])
    _orth_arrow(ax, [(x1 + rw / 2, y1), (x2 - rw / 2, y1)])
    _orth_arrow(ax, [(x2 + rw / 2, y1), (xd - dw / 2, y1)])

    # YES branch: diamond bottom -> r3 -> r4 -> end (row 2, right-to-left)
    y2 = 33
    rect(xd, y2, "$\\hat{\\theta}\\!\\rightarrow\\!$ FMA\n"
                 "(inverse formula)")
    rect(45, y2, "5 repeats x 5-fold\nsubject-level CV\nbalanced weights")
    ellipse(xs, y2, "End")
    _orth_arrow(ax, [(xd, y1 - dh / 2), (xd, y2 + rh / 2)])
    ax.text(xd + 1.6, (y1 - dh / 2 + y2 + rh / 2) / 2, "Yes", fontsize=fs,
            ha="left", va="center")
    _orth_arrow(ax, [(xd - rw / 2, y2), (45 + rw / 2, y2)])
    _orth_arrow(ax, [(45 - rw / 2, y2), (xs + ew / 2, y2)])

    # NO branch: diamond right -> down -> r5 -> up into r4 bottom edge
    y3 = 13.5
    rect(80, y3, "fine-tune + CORAL\ndomain alignment\n($\\lambda$ grid)")
    x_out = 96.5
    _orth_arrow(ax, [(xd + dw / 2, y1), (x_out, y1), (x_out, y3),
                     (80 + rw / 2, y3)])
    ax.text(x_out - 0.2, (y1 + y3) / 2, "No", fontsize=fs, ha="right",
            va="center")
    x_fb = 51
    _orth_arrow(ax, [(80 - rw / 2, y3), (x_fb, y3), (x_fb, y2 - rh / 2)])

    # bottom-centred caption
    ax.text(50, 2.5, "Transfer flow: sim pre-training, bottleneck "
                     "de-saturation + recalibration, optional CORAL-aligned "
                     "fine-tuning;\nrepeated subject-level cross-validation "
                     "with class-balanced weights; LOSO per-subject "
                     "calibration (oracle)\nas an upper-bound reference.",
            ha="center", va="center", fontsize=6.2)


# --------------------------------------------------- other panels
def panel_domain(subfig, pca):
    ax, ax2 = subfig.subplots(1, 2, gridspec_kw=dict(wspace=0.35))
    sev_idx = pca["real_severity"]

    def draw(a, S, R, ttl):
        a.scatter(S[:, 0], S[:, 1], s=3, c=OI["grey"], marker="^",
                  alpha=0.35, linewidths=0, label="sim")
        for ci, cname in enumerate(CLASSES):
            m = sev_idx == ci
            a.scatter(R[m, 0], R[m, 1], s=5, c=SEV_C[cname], marker="o",
                      alpha=0.6, linewidths=0, label=f"real ({cname})")
        a.set_title(ttl, fontsize=8)

    ex = pca["explained"]
    draw(ax, pca["sim"], pca["real"], "PCA")
    ax.set_xlabel(f"PC1 ({ex[0] * 100:.0f}%)")
    ax.set_ylabel(f"PC2 ({ex[1] * 100:.0f}%)")
    ax.legend(frameon=False, markerscale=2, loc="upper right", fontsize=6)
    draw(ax2, pca["tsne_sim"], pca["tsne_real"], "t-SNE")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    mmd = json.load(open(os.path.join(OUT, "mmd_v2.json")))
    subfig.suptitle(f"MMD$^2$: raw {mmd['raw_summary']['mmd2']:.3f}, "
                    f"$h$ {mmd['iira_h']['mmd2']:.3f}, "
                    f"$h$_v2 {mmd['iira_v2_h']['mmd2']:.3f}",
                    fontsize=7.5, y=1.00)


def panel_scatter(ax, pm):
    lim = (0, 70)
    ax.plot(lim, lim, "--", c=OI["grey"], lw=0.9)
    for m in MODELS:
        d = pm[pm.model == m]
        ax.scatter(d.fma_true, d.fma_pred, s=13, c=MC[m], alpha=0.75,
                   label=DISP[m], linewidths=0.3, edgecolors="white")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("true FMA-UE")
    ax.set_ylabel("predicted FMA-UE (repeat mean)")
    ax.legend(frameon=False, fontsize=5.0, ncol=3, loc="upper left",
              columnspacing=0.6, handletextpad=0.2)


def panel_confusion(subfig, cc, summ):
    axl, axr = subfig.subplots(1, 2, gridspec_kw=dict(wspace=0.45))
    f1 = summ[summ.metric == "macro_f1"].set_index("model")["rep_mean"]
    f1_labelfree = f1.drop(index=[m for m in ORACLE if m in f1.index])
    best = f1_labelfree.idxmax()

    def matrix(model):
        d = cc[cc.model == model]
        C = np.zeros((3, 3), int)
        for _, row in d.iterrows():
            C[CLASSES.index(row["true"]), CLASSES.index(row["pred"])] = \
                row["count"]
        return C

    C_l = matrix(best)                      # best label-free (repeat-avg.)
    C_r = matrix("iira_v2_calib1t")         # per-subject calibration, LOSO
    vmax = max(C_l.max(), C_r.max())        # shared colour scale

    def draw(ax, C, title, ylabel=False):
        ax.imshow(C, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(C[i, j]), ha="center", va="center",
                        color="white" if C[i, j] > vmax / 2 else OI["black"],
                        fontsize=9.5, fontweight="bold")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES if ylabel else [""] * 3)
        ax.set_xlabel("predicted")
        if ylabel:
            ax.set_ylabel("true")
        ax.set_title(title, fontsize=7.2)

    draw(axl, C_l, f"best label-free: {DISP[best]}\n(macro-F1 = "
                   f"{f1_labelfree.max():.2f}, repeat-averaged)",
         ylabel=True)
    draw(axr, C_r, "per-subject calib-1t$^\\dagger$\n(macro-F1 = "
                   f"{f1['iira_v2_calib1t']:.2f})")
    subfig.text(0.5, 0.015, "$^\\dagger$oracle: the calibration trial "
                            "carries the subject's intake FMA (one clinical "
                            "measurement per patient); subject-level "
                            "matrices, shared colour scale.",
                ha="center", va="bottom", fontsize=6,
                transform=subfig.transSubfigure)


def panel_metrics(subfig, summ):
    from matplotlib.ticker import MaxNLocator
    axes = subfig.subplots(1, 4, gridspec_kw=dict(wspace=0.6))
    spec = [("mae", "MAE"), ("r2", "$R^2$"), ("macro_f1", "macro-F1"),
            ("acc", "accuracy")]
    for ax, (key, label) in zip(axes, spec):
        d = summ[summ.metric == key].set_index("model").loc[MODELS]
        x = np.arange(len(MODELS))
        ax.bar(x, d.rep_mean, yerr=d.ci95, capsize=1.5,
               color=[MC[m] for m in MODELS], ec=OI["black"], lw=0.4,
               error_kw=dict(lw=0.6))
        ax.set_xticks(x, [""] * len(MODELS))
        ax.set_title(label, fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="y", labelsize=6, pad=1)
    axes[0].set_xlabel("bar colours follow the model legend in panel (f)",
                       fontsize=6)


def panel_severity(ax, sev):
    width = 0.8 / len(MODELS)
    x = np.arange(3)
    for i, m in enumerate(MODELS):
        d = sev[sev.model == m].set_index("severity").loc[CLASSES]
        ax.bar(x + (i - (len(MODELS) - 1) / 2) * width, d.acc, width * 0.92,
               color=MC[m], ec=OI["black"], lw=0.3, label=DISP[m])
    ax.set_xticks(x, [f"{c}\n(n={int(sev[(sev.model == 'mlp') & (sev.severity == c)].n.iloc[0])})"
                      for c in CLASSES])
    ax.set_ylabel("accuracy (repeat-averaged)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=5.0, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), columnspacing=0.7,
              handletextpad=0.3)


def main():
    pm = pd.read_csv(os.path.join(OUT, "predictions_repmean.csv"))
    summ = pd.read_csv(os.path.join(OUT, "metrics_summary.csv"))
    sev = pd.read_csv(os.path.join(OUT, "metrics_by_severity.csv"))
    cc = pd.read_csv(os.path.join(OUT, "confusion", "confusion_counts.csv"))
    pca = np.load(os.path.join(OUT, "pca_data.npz"))

    fig = plt.figure(figsize=(10.5, 9.8))
    sf = fig.subfigures(3, 2, height_ratios=[1.05, 1.0, 1.0], hspace=0.08,
                        wspace=0.12)
    labels = "abcdef"
    for i, s in enumerate(sf.flat):
        s.suptitle(f"({labels[i]})", x=0.02, y=0.99, ha="left", fontsize=10,
                   fontweight="bold")
    panel_flow(sf[0, 0].subplots())
    panel_domain(sf[0, 1], pca)
    panel_scatter(sf[1, 0].subplots(), pm)
    panel_confusion(sf[1, 1], cc, summ)
    panel_metrics(sf[2, 0], summ)
    panel_severity(sf[2, 1].subplots(), sev)

    fig.savefig(os.path.join(FIG, "fig7_transfer_v2.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig7_transfer_v2.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("fig7_transfer_v2.png/.pdf saved")


if __name__ == "__main__":
    main()
