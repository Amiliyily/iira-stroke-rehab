"""table5_transfer: model x full-metric booktabs table for the transfer
experiment. Numbers come from results/transfer/metrics_summary.csv (pooled
subject-level CV predictions over the 17 subjects; fold mean +/- std are in
metrics_summary.csv). Best value per column is bolded in the .tex.

Also runs an allclose verification: every number rendered in the .tex must
match the .csv (rtol=1e-3 on the rounded values).

Run:  python src/transfer/make_table5.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "transfer")
TAB = os.path.join(ROOT, "tables")
os.makedirs(TAB, exist_ok=True)

MODELS = ["iira_zero_shot", "iira_ft_head", "iira_ft_full", "mlp", "lstm",
          "gru", "transformer"]
DISP = dict(iira_zero_shot="IIRA zero-shot", iira_ft_head="IIRA ft-head",
            iira_ft_full="IIRA ft-full", mlp="MLP", lstm="LSTM", gru="GRU",
            transformer="Transformer")
# (metric key, column header, decimals, higher-is-better)
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
    piv = s.pivot_table(index="model", columns="metric", values="pooled")
    piv = piv.loc[MODELS]

    csv_df = pd.DataFrame({"model": [DISP[m] for m in MODELS]})
    for key, _, dec, _ in COLS:
        csv_df[key] = piv[key].round(dec).values

    best = {}
    for key, _, _, higher in COLS:
        v = piv[key].values
        best[key] = (np.nanmax(v) if higher else np.nanmin(v))

    header = ["Model"] + [h for _, h, _, _ in COLS]
    rows = []
    for m in MODELS:
        r = [DISP[m]]
        for key, _, dec, higher in COLS:
            v = piv.loc[m, key]
            cell = f"{v:.{dec}f}"
            if abs(v - best[key]) < 10 ** (-(dec + 1)):
                cell = "\\textbf{%s}" % cell
            r.append(cell)
        rows.append(r)

    align = "l" + "r" * len(COLS)
    tex = ["% table5_transfer: sim-to-real transfer to Gurari 2018 stroke "
           "kinematics (17 subjects, subject-level 5-fold CV).",
           "% Numbers are pooled subject-level metrics; fold mean +/- std in "
           "results/transfer/metrics_summary.csv.",
           "\\begin{tabular}{%s}" % align, "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    tex += [" & ".join(r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]

    with open(os.path.join(TAB, "table5_transfer.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    csv_df.to_csv(os.path.join(TAB, "table5_transfer.csv"), index=False)

    # ---- allclose verification: tex cells vs csv values
    tex_txt = "\n".join(tex)
    nums = []
    for row in tex:
        if row.startswith(("\\", "%")):
            continue
        cells = row.split(" & ")
        if cells[0].strip() not in DISP.values():
            continue
        for cell in cells[1:]:
            mnum = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell).strip().rstrip("\\").strip()
            nums.append(float(mnum))
    ref = np.array([round(float(piv.loc[m, key]), dec)
                    for m in MODELS for key, _, dec, _ in COLS])
    assert len(nums) == len(ref), (len(nums), len(ref))
    assert np.allclose(nums, ref, rtol=1e-3, atol=5 * 10 ** -3), \
        "tex/csv mismatch"
    print("table5_transfer.tex/.csv saved; allclose check passed")


if __name__ == "__main__":
    main()
