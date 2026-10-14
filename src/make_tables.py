"""Tables 1-4 (SPEC 8): booktabs three-line .tex + .csv pairs.

Every number comes from results/metrics_summary_v2.csv, results/stats_v2.csv,
results/config_snapshot.json, data/subjects.csv or paper/citations.md.

Run:  python src/make_tables.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
TAB = os.path.join(ROOT, "tables")
os.makedirs(TAB, exist_ok=True)

DISPLAY = dict(pid="PID", pd_grav="PD + gravity", computed_torque="Computed torque",
               impedance="Impedance", admittance="Admittance", smc="SMC",
               lqr="LQR", mpc="MPC", dob="DOB + PD", ilc="ILC",
               rbf="RBF adaptive", mlp="MLP residual", lstm="LSTM residual",
               gru="GRU residual", transformer="Transformer residual",
               iira="\\textbf{IIRA (ours)}",
               iira_nobottleneck="IIRA w/o synergy bottleneck",
               iira_noaan="IIRA w/o AAN (fixed $K_p^{max}$)",
               iira_noresidual="IIRA w/o residual net",
               iira_nosafety="IIRA w/o safety layer",
               iira_notheta="IIRA w/o injury head")
MAIN16 = ["pid", "pd_grav", "computed_torque", "impedance", "admittance",
          "smc", "lqr", "mpc", "dob", "ilc", "rbf", "mlp", "lstm", "gru",
          "transformer", "iira"]
ABL6 = ["iira", "iira_nobottleneck", "iira_noaan", "iira_noresidual",
        "iira_nosafety", "iira_notheta"]


def emit(name, tex_lines, csv_df):
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    csv_df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    print(f"  {name}.tex/.csv saved", flush=True)


def booktabs(header, rows, align):
    out = ["\\begin{tabular}{%s}" % align, "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return out


# ---------------------------------------------------------------- table 1
def table1_benchmark():
    """Industry/state-of-the-art comparison; bibliographic data verified in
    paper/citations.md (PDF first pages + web)."""
    rows = [
        ["Fugl-Meyer 1975 \\cite{fuglmeyer1975}", "Scand J Rehabil Med",
         "clinical scale", "--", "standardised motor assessment",
         "--", "0--66 UE motor score; de-facto primary endpoint"],
        ["Hogan 1985 \\cite{hogan1985a}", "J Dyn Syst Meas Control",
         "-- (theory)", "general", "impedance control",
         "--", "foundation of rehabilitation-robot interaction control"],
        ["Krebs 1998 \\cite{krebs1998}", "IEEE Trans Rehabil Eng",
         "MIT-Manus", "2 (planar)", "impedance-based assistive",
         "pilot clinical", "reduced motor impairment vs. control exercise"],
        ["Kwakkel 2008 \\cite{kwakkel2008}", "Neurorehabil Neural Repair",
         "review (ARM, MIT-Manus et al.)", "--", "systematic review",
         "8 RCTs", "arm-function gain, no ADL gain; small trials"],
        ["Lo 2010 (VA) \\cite{lo2010}", "N Engl J Med",
         "MIT-Manus", "2 (planar)", "assist-as-needed robotic therapy",
         "127 (RCT)", "no benefit over usual care at 12 wk (primary)"],
        ["Mehrholz 2018 \\cite{mehrholz2018arm}", "Cochrane Database Syst Rev",
         "review (all arm robots)", "--", "systematic review + meta-analysis",
         "45 trials, 1619", "small ADL/arm-function benefit; low-quality evidence"],
        ["RATULS 2019 \\cite{rodgers2019ratuls}", "Lancet",
         "InMotion (MIT-Manus lineage)", "2 (planar)", "assistive robotic training",
         "770 (3-arm RCT)", "negative primary ADL outcome at 3 months"],
        ["PREP2 2017 \\cite{stinear2017prep2}", "Ann Clin Transl Neurol",
         "-- (clinical)", "--", "biomarker algorithm (SAFE+TMS/MRI)",
         "prospective cohort", "predicts UL outcome; no robot control loop"],
        ["StrokeRehab 2022 \\cite{kaku2022strokerehab}", "NeurIPS D\\&B",
         "-- (video/IMU benchmark)", "--", "action-identification benchmark",
         "51 subjects", "rehab-motion benchmark; no robotic assistance"],
        ["COBRA 2024 \\cite{yu2024cobra}", "npj Digit Med",
         "-- (video)", "--", "AI impairment score trained on healthy",
         "healthy + stroke", "quantifies impairment; no assistance control"],
        ["REHAB 2026 \\cite{lv2026rehab}", "Sci Data",
         "wearable IMU", "--", "standardised-task kinematic dataset",
         "120 patients", "open dataset; no closed-loop control"],
        ["\\textbf{This work}", "-- (simulation)",
         "6-DOF arm (UR5-scale)", "6", "IIRA: injury-inspired residual + AAN",
         "60 virtual, 5-fold CV", "16-controller benchmark with patient "
         "contribution $\\rho$ and passivity guarantees"],
    ]
    csv = pd.DataFrame(rows, columns=["study", "venue", "platform", "dof",
                                      "control_or_method", "sample",
                                      "main_outcome_or_limitation"])
    csv = csv.replace({r"\\cite\{[^}]*\}": "", r"\\textbf\{([^}]*)\}": r"\1"},
                      regex=True)
    tex = ["% table1: benchmark comparison (sources: paper/citations.md)",
           "\\begin{scriptsize}"]
    tex += booktabs(["Study", "Venue", "Platform", "DOF", "Control / method",
                     "Sample", "Main outcome / limitation"],
                    rows, "lllllll")
    tex += ["\\end{scriptsize}"]
    emit("table1_benchmark", tex, csv)


# ---------------------------------------------------------------- table 2
def fmt(v, nd):
    return f"{v:.{nd}f}"


def table2_main(summ, stats):
    s16 = summ[summ.model.isin(MAIN16)].set_index("model").loc[MAIN16]
    sig = {(r.metric, r.baseline): bool(r.significant)
           for r in stats.itertuples()}
    # column: (metric, header, decimals, scale, lower_better)
    cols = [("rmse_p_mm", "RMSE$_{pos}$ (mm)", 1, 1.0, True),
            ("rmse_q_deg", "RMSE$_{q}$ (deg)", 2, 1.0, True),
            ("norm_jerk", "norm jerk ($10^{7}$)", 2, 1e-7, True),
            ("tau_rms", "$\\|\\tau_a\\|$ (N$\\cdot$m)", 2, 1.0, True),
            ("viol_pct", "viol (\\%)", 2, 1.0, True),
            ("settle_mm", "settle (mm)", 1, 1.0, True),
            ("time_ms", "time (ms)", 3, 1.0, True),
            ("rho_realized", "$\\rho_{realized}$", 3, 1.0, False),
            ("rho_dynamic", "$\\rho_{dynamic}$", 3, 1.0, False)]
    best = {}
    for m, _, _, sc, low in cols:
        v = s16[m + "_mean"].to_numpy() * sc
        best[m] = v.min() if low else v.max()
    rows = []
    csv_rows = []
    for mo in s16.index:
        cells = [DISPLAY[mo]]
        rec = {"model": mo}
        for m, _, nd, sc, low in cols:
            mean = s16.loc[mo, m + "_mean"] * sc
            std = s16.loc[mo, m + "_std"] * sc
            cell = f"${mean:.{nd}f}\\pm{std:.{nd}f}$"
            if np.isclose(mean, best[m]):
                cell = "$\\mathbf{" + f"{mean:.{nd}f}" + "}\\pm" + \
                       f"{std:.{nd}f}$"
            if mo != "iira" and sig.get((m, mo), False):
                cell += "$^{\\ast}$"
            cells.append(cell)
            rec[m + "_mean"] = s16.loc[mo, m + "_mean"]
            rec[m + "_std"] = s16.loc[mo, m + "_std"]
        rows.append(cells)
        csv_rows.append(rec)
    tex = ["% table2: 16 controllers x 9 metrics (mean +/- SD over trials;",
           "% bold = best per column; * = Wilcoxon+Holm p<0.05 vs IIRA;",
           "% rho_dynamic = rho with gravity compensation removed from tau_a)"]
    tex += booktabs(["Controller"] + [c[1] for c in cols], rows,
                    "l" + "r" * len(cols))
    emit("table2_main", tex, pd.DataFrame(csv_rows))


# ---------------------------------------------------------------- table 3
def table3_ablation(summ):
    abl = summ[summ.model.isin(ABL6)].set_index("model").loc[ABL6]
    cols = [("rmse_p_mm", "RMSE$_{pos}$ (mm)", 1, 1.0),
            ("norm_jerk", "norm jerk ($10^{7}$)", 2, 1e-7),
            ("viol_pct", "viol (\\%)", 2, 1.0),
            ("settle_mm", "settle (mm)", 1, 1.0),
            ("rho_realized", "$\\rho_{realized}$", 3, 1.0)]
    rows = []
    for mo in abl.index:
        cells = [DISPLAY[mo]]
        for m, _, nd, sc in cols:
            cells.append(f"${abl.loc[mo, m + '_mean'] * sc:.{nd}f}"
                         f"\\pm{abl.loc[mo, m + '_std'] * sc:.{nd}f}$")
        tm = abl.loc[mo, "theta_mae"]
        cells.append(f"{tm:.3f}" if np.isfinite(tm) else "--")
        cells.append(f"{abl.loc[mo, 'rho_pearson_r']:.3f}")
        rows.append(cells)
    csv = abl.reset_index()[
        ["model", "rmse_p_mm_mean", "rmse_p_mm_std", "norm_jerk_mean",
         "norm_jerk_std", "viol_pct_mean", "viol_pct_std", "settle_mm_mean",
         "settle_mm_std", "rho_realized_mean", "rho_realized_std",
         "theta_mae", "rho_pearson_r"]]
    tex = ["% table3: IIRA ablations (6 variants)"]
    tex += booktabs(["Variant"] + [c[1] for c in cols] +
                    ["$\\hat\\theta$ MAE", "$\\hat\\rho$ Pearson $r$"],
                    rows, "l" + "r" * (len(cols) + 2))
    emit("table3_ablation", tex, csv)


# ---------------------------------------------------------------- table 4
def table4_config():
    import arm6dof as arm
    import injury as inj
    from controllers import neural as neu
    with open(os.path.join(ROOT, "results", "config_snapshot.json")) as f:
        cfg = json.load(f)
    g = cfg["grid"]

    def flist(a):
        return "[" + ", ".join(f"{float(v):g}" for v in a) + "]"

    rows = [
        ["controller version", "v1.1 passivity-preserving (SPEC App. A)"],
        ["global seed", str(cfg["seed"])],
        ["integration", "semi-implicit Euler, dt = %.2f s" % cfg["dt"]],
        ["implicit dissipation", "friction + damping Jacobians (all controllers)"],
        ["subjects / folds", f"{cfg['subjects']} / {cfg['n_folds']}-fold "
         "subject-level stratified CV"],
        ["training rollouts", f"{cfg['train_rollouts_per_subject']}/subject "
         f"(8 dir x {cfg['train_speeds_m']} m)"],
        ["test protocol", cfg["test_protocol"]],
        ["epochs / batch / lr", f"{cfg['epochs']} / {cfg['batch']} / "
         f"{cfg['lr']}"],
        ["early stop patience", str(cfg["patience"])],
        ["window / stride", f"{cfg['window']} / {cfg['window_stride']}"],
        ["validation subjects", str(cfg["n_val_subjects"])],
        ["loss weights", f"$w_1$={g['w_fixed']['w1']}, $w_2$="
         f"{g['selected_w2']}, $w_3$={g['selected_w3']}, $w_4$="
         f"{g['w_fixed']['w4']}, $\\mu_1$={g['w_fixed']['mu1']}, "
         f"$\\mu_2$={g['w_fixed']['mu2']}"],
        ["grid searched", f"$w_2\\in{g['w2']}$, $w_3\\in{g['w3']}$ (fold-1)"],
        ["$K_p^{max}$ / $K_d^{max}$",
         f"{flist(neu.KP_MAX)} / {flist(neu.KD_MAX)} N$\\cdot$m"],
        ["$K_p^{min}$ (AAN floor)", "0.2 $K_p^{max}$"],
        ["residual low-pass $\\tau_c$", f"{neu.TAU_C} s"],
        ["dissipative comp.", f"$\\kappa$={neu.KAPPA}, $\\dot q_f$ low-pass "
         f"{neu.QD_FC:g} Hz"],
        ["torque limits", f"{flist(arm.TAU_LIM)} N$\\cdot$m; "
         f"$|\\dot q|\\leq${arm.QD_LIM:g} rad/s"],
        ["$U$ (volitional scale)", f"{flist(inj.U_MAX)} N$\\cdot$m"],
        ["$B_{sp}$", flist(inj.B_SP)],
        ["injury noise", f"{flist(inj.NOISE_SCALE)} N$\\cdot$m, "
         f"{inj.NOISE_FC:g} Hz low-pass; $c(t)$ lag {inj.C_TAU*1000:.0f} ms"],
        ["safety layer", "$|\\dot q_f|$>2.5 rad/s damping; $|p|$>0.85 m "
         "workspace sphere"],
        ["$\\rho_{realized}$ (task 1)", "offline deterministic replay of the "
         "injury model ($\\xi$=0) on stored test trajectories"],
        ["total runtime", f"{cfg['total_runtime_min']:.0f} min (CPU)"],
    ]
    csv = pd.DataFrame(rows, columns=["parameter", "value"])
    csv["value"] = csv["value"].str.replace("$", "", regex=False)
    tex = ["% table4: simulation / training hyper-parameters "
           "(results/config_snapshot.json + module constants)"]
    tex += booktabs(["Parameter", "Value"], rows, "ll")
    emit("table4_config", tex, csv)


def main():
    summ = pd.read_csv(os.path.join(ROOT, "results", "metrics_summary_v2.csv"))
    stats = pd.read_csv(os.path.join(ROOT, "results", "stats_v2.csv"))
    table1_benchmark()
    table2_main(summ, stats)
    table3_ablation(summ)
    table4_config()
    print("all tables done.")


if __name__ == "__main__":
    main()
