"""Patent-application-style flowcharts (snake layout, orthogonal arrows).

  fig_patent_flow  - Chinese method flowchart (IIRA), patent style:
                     bordered title bar, light-blue start/end ellipses,
                     white rectangles, orange-edged diamond decision with
                     yes/no branches, orthogonal black arrows, snake reflow.
  fig1_flowstyle   - English flowstyle redraw of fig1_principle (same
                     layout language; the original fig1_principle is kept).

Run:  python src/make_patent_flow.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

LIGHT_BLUE = "#D6E4F0"      # start/end ellipses
ORANGE = "#E36C0A"          # decision diamond edge
BLACK = "#000000"


def orth_arrow(ax, pts, lw=1.2):
    """Orthogonal multi-segment arrow: line through pts[:-1], arrow head
    on the last segment."""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=BLACK, lw=lw, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=lw,
                                mutation_scale=13, shrinkA=0, shrinkB=0))


def node_ellipse(ax, cx, cy, w, h, text, fs):
    ax.add_patch(Ellipse((cx, cy), w, h, fc=LIGHT_BLUE, ec=BLACK, lw=1.1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def node_rect(ax, cx, cy, w, h, text, fs):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white",
                           ec=BLACK, lw=1.1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def node_diamond(ax, cx, cy, w, h, text, fs):
    pts = [(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy),
           (cx, cy - h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=ORANGE, lw=1.6))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def draw_flow(cfg):
    plt.rcParams.update({"font.sans-serif": cfg["fonts"],
                         "font.family": "sans-serif",
                         "axes.unicode_minus": False})
    L = cfg["layout"]
    fig, ax = plt.subplots(figsize=cfg["figsize"])
    ax.set_xlim(0, L["xlim"]); ax.set_ylim(0, 72); ax.axis("off")
    fs_n, fs_b = cfg["fs_node"], cfg["fs_branch"]
    rw, rh = cfg["rect_w"], cfg["rect_h"]
    ew, eh = 11, 7                      # start/end ellipse
    dw, dh = cfg["dia_w"], 13           # decision diamond

    # bordered title bar
    bx0, bx1 = L["title_bar"]
    ax.add_patch(Rectangle((bx0, 65), bx1 - bx0, 5.2, fc="white", ec=BLACK,
                           lw=1.2))
    ax.text((bx0 + bx1) / 2, 67.6, cfg["title"], ha="center", va="center",
            fontsize=cfg["fs_title"], fontweight="bold")

    # row 1: start -> R1 -> R2 -> diamond (left to right)
    y1 = 55
    node_ellipse(ax, L["start"], y1, ew, eh, cfg["start"], fs_n)
    node_rect(ax, L["r1"], y1, rw, rh, cfg["r1"], fs_n)
    node_rect(ax, L["r2"], y1, rw, rh, cfg["r2"], fs_n)
    node_diamond(ax, L["d"], y1, dw, dh, cfg["diamond"], fs_n)
    orth_arrow(ax, [(L["start"] + ew / 2, y1), (L["r1"] - rw / 2, y1)])
    orth_arrow(ax, [(L["r1"] + rw / 2, y1), (L["r2"] - rw / 2, y1)])
    orth_arrow(ax, [(L["r2"] + rw / 2, y1), (L["d"] - dw / 2, y1)])

    # YES branch: diamond bottom -> R3 -> R4 -> end (row 2, right-to-left)
    y2 = 36
    node_rect(ax, L["r3"], y2, rw, rh, cfg["r3"], fs_n)
    node_rect(ax, L["r4"], y2, rw, rh, cfg["r4"], fs_n)
    node_ellipse(ax, L["end"], y2, ew, eh, cfg["end"], fs_n)
    orth_arrow(ax, [(L["d"], y1 - dh / 2), (L["r3"], y2 + rh / 2)])
    ax.text(L["d"] + 1.6, (y1 - dh / 2 + y2 + rh / 2) / 2, cfg["yes"],
            fontsize=fs_b, ha="left", va="center")
    orth_arrow(ax, [(L["r3"] - rw / 2, y2), (L["r4"] + rw / 2, y2)])
    orth_arrow(ax, [(L["r4"] - rw / 2, y2), (L["end"] + ew / 2, y2)])

    # NO branch: diamond right vertex -> down -> R5 -> up through the gap
    # between the R4 and R3 columns -> into R2 bottom edge (no box crossing)
    y3 = 17
    nw = cfg["rect_w_no"]
    node_rect(ax, L["r5"], y3, nw, rh, cfg["r5"], fs_n)
    x_out = L["d"] + dw / 2 + 6
    orth_arrow(ax, [(L["d"] + dw / 2, y1), (x_out, y1),
                    (x_out, y3), (L["r5"] + nw / 2, y3)])
    ax.text(x_out + 1.2, (y1 + y3) / 2, cfg["no"], fontsize=fs_b,
            ha="left", va="center")
    x_fb = L["x_fb"]
    orth_arrow(ax, [(L["r5"] - nw / 2, y3), (x_fb, y3),
                    (x_fb, y1 - rh / 2)])

    # caption
    ax.text(L["caption_x"], 4, cfg["caption"], ha="center", va="center",
            fontsize=cfg["fs_caption"])

    fig.savefig(os.path.join(FIG, cfg["name"] + ".png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIG, cfg["name"] + ".pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {cfg['name']}.png/.pdf saved", flush=True)


CN = dict(
    name="fig_patent_flow",
    fonts=["Microsoft YaHei", "SimHei"],
    figsize=(10.5, 7.6),
    fs_title=13.5, fs_node=10.5, fs_branch=10.5, fs_caption=11,
    rect_w=19, rect_h=10, rect_w_no=21, dia_w=20,
    layout=dict(xlim=100, title_bar=(15, 85), caption_x=50,
                start=10, r1=30, r2=54, d=80,
                r3=80, r4=49, end=11, r5=80, x_fb=62),
    title="(一)损伤启发的自适应康复辅助方法流程",
    start="开始", end="结束",
    r1="关节角度/速度采集\n滑动窗口特征提取",
    r2="时序编码与协同瓶颈\n损伤参数在线估计",
    diamond="辅助性能稳定\n是或否",
    r3="按需辅助刚度调制\n残余力矩补偿与安全层",
    r4="机械臂辅助执行\n患者主动参与训练",
    r5="分析失稳/失败场景\n调整增益与网络权重",
    yes="是", no="否",
    caption="图1 损伤启发的脑卒中上肢康复辅助方法流程。",
)

EN = dict(
    name="fig1_flowstyle",
    fonts=["DejaVu Sans"],
    figsize=(11.2, 7.4),
    fs_title=11.5, fs_node=8.4, fs_branch=9.5, fs_caption=10,
    rect_w=24, rect_h=10, rect_w_no=26, dia_w=22,
    layout=dict(xlim=110, title_bar=(10, 100), caption_x=55,
                start=10, r1=31, r2=59, d=86,
                r3=86, r4=52, end=10, r5=86, x_fb=69),
    title="(a) Injury-inspired adaptive residual assistance (IIRA): method flow",
    start="Start", end="End",
    r1="Joint-state acquisition\nsliding-window features",
    r2="Synergy-bottleneck encoder\nonline injury estimation",
    diamond="Tracking & safety\nstable?",
    r3="AAN stiffness modulation\nresidual comp. + safety layer",
    r4="Robot-assisted execution\nactive patient training",
    r5="Analyse failure cases\nretune gains and weights",
    yes="Yes", no="No",
    caption="Fig. 1  Flowchart of the IIRA upper-limb rehabilitation "
            "assistance method.",
)


def main():
    draw_flow(CN)
    draw_flow(EN)


if __name__ == "__main__":
    main()
