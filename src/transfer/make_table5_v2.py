"""table5_transfer_v2 (final): 15-model booktabs table for round 2, repeated
5x5-fold protocol (12 repeated-CV models) plus 3 single-pass LOSO
calibration variants.  Cells show the repeat-level mean with the t-based 95%
CI half-width; best mean per column bolded.  allclose verification of tex vs
csv (on the rendered mean values).

Run:  python src/transfer/make_table5_v2.py
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "transfer_v2")
TAB = os.path.join(ROOT, "tables")
os.makedirs(TAB, exist_ok=True)

MODELS = ["iira_zero_shot", "iira_ft_head", "iira_ft_full",
          "iira_v2_zero_shot", "iira_v2_ft", "iira_v2_loso_global",
          "iira_v2_calib1t", "iira_v2_calib2t",
          "ridge", "rf", "gbt", "mlp", "lstm", "gru", "transformer"]
DISP = dict(iira_zero_shot="IIRA zero-shot", iira_ft_head="IIRA ft-head",
            iira_ft_full="IIRA ft-full",
            iira_v2_zero_shot="IIRA-v2 zero-shot", iira_v2_ft="IIRA-v2 ft",
            iira_v2_loso_global="IIRA-v2 LOSO global",
            iira_v2_calib1t="IIRA-v2 calib-1t$^{\\dagger}$",
            iira_v2_calib2t="IIRA-v2 calib-2t$^{\\dagger}$",
            ridge="Ridge (eng. feat.)", rf="RF (eng. feat.)",
            gbt="GBT (eng. feat.)", mlp="MLP", lstm="LSTM", gru="GRU",
            transformer="Transformer")
COLS = [("mae", "MAE$\\downarrow$", 2, False),
        ("rmse", "RMSE$\\downarrow$", 2, False),
        ("r2", "$R^2$$\\uparrow$", 3, True),
        ("pearson", "Pearson$\\uparrow$", 3, True),
        ("spearman", "Spearman$\\uparrow$", 3, True),
        ("acc", "Acc$\\uparrow$", 3, True),
        ("macro_f1", "macro-F1$\\uparrow$", 3, True),
        ("auc_ovr", "AUC$\\uparrow$", 3, True),
        ("kappa", "$\\kappa$$\\uparrow$", 3, True),
        ("latency_ms", "lat. (ms)$\\downarrow$", 2, False)]


def main():
    s = pd.read_csv(os.path.join(OUT, "metrics_summary.csv"))
    pm = s.pivot_table(index="model", columns="metric",
                       values="rep_mean").loc[MODELS]
    pc = s.pivot_table(index="model", columns="metric", values="ci95").loc[
        MODELS]

    csv_df = pd.DataFrame({"model": [DISP[m] for m in MODELS]})
    for key, _, dec, _ in COLS:
        csv_df[key] = pm[key].round(dec).values
        csv_df[key + "_ci95"] = pc[key].round(dec).values

    best = {}
    for key, _, _, higher in COLS:
        v = pm[key].values
        best[key] = (np.nanmax(v) if higher else np.nanmin(v))

    header = ["Model"] + [h for _, h, _, _ in COLS]
    rows = []
    for m in MODELS:
        r = [DISP[m]]
        for key, _, dec, higher in COLS:
            v = pm.loc[m, key]
            cell = f"{v:.{dec}f}$\\pm${pc.loc[m, key]:.{dec}f}"
            if abs(v - best[key]) < 10 ** (-(dec + 1)):
                cell = "\\textbf{%s}" % cell
            r.append(cell)
        rows.append(r)

    align = "l" + "r" * len(COLS)
    tex = ["% table5_transfer_v2: round-2 transfer, repeated subject-level "
           "5-fold CV x 5 repeats, class-balanced weights.  Cells: repeat "
           "mean +/- t-based 95% CI half-width.  Latency measured once "
           "(repeat 0).  Dagger: LOSO per-subject calibration variants use "
           "the subject's true FMA of the calibration trial(s) (oracle "
           "upper bound); their CI is subject-level (t_16), n_repeats=1.",
           "\\begin{tabular}{%s}" % align, "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    tex += [" & ".join(r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]

    with open(os.path.join(TAB, "table5_transfer_v2.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    csv_df.to_csv(os.path.join(TAB, "table5_transfer_v2.csv"), index=False)

    # ---- allclose verification: tex mean values vs csv mean columns
    nums = []
    for row in tex:
        if row.startswith(("\\", "%")):
            continue
        cells = row.split(" & ")
        if cells[0].strip() not in DISP.values():
            continue
        for cell in cells[1:]:
            mnum = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
            mnum = mnum.split("$\\pm$")[0].strip().rstrip("\\").strip()
            nums.append(float(mnum))
    ref = np.array([round(float(pm.loc[m, key]), dec)
                    for m in MODELS for key, _, dec, _ in COLS])
    assert len(nums) == len(ref), (len(nums), len(ref))
    assert np.allclose(nums, ref, rtol=1e-3, atol=5 * 10 ** -3), \
        "tex/csv mismatch"
    print("table5_transfer_v2.tex/.csv saved; allclose check passed")


if __name__ == "__main__":
    main()
