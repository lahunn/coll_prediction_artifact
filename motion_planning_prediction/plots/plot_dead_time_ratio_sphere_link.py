#!/usr/bin/env python3
"""Plot grouped stacked bars for dead-time ratio (top) and effective ratio (bottom)."""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib import font_manager as fm
# --- 统一绘图风格配置 ---
sns.set_theme(style="white")
sns.set_style("white")
sns.set_palette("colorblind")

# 字体加载与配置
font_path = os.path.expanduser("~/.local/share/fonts/simsun.ttc")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

# 字体大小变量
FONT_SIZE = 16  # 其它字体大小（如标签、标题等）
TICK_FONT_SIZE = 12  # 坐标轴刻度字体大小
LEGEND_FONT_SIZE = 12  # legend字体大小

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "SimSun",
    "WenQuanYi Micro Hei",
    "STSong",
    "Songti SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

palette = sns.color_palette("colorblind")
LINK_COLOR = palette[0]
SPHERE_COLOR = palette[1]
DEAD_COLOR = palette[7]


def get_csv_path(algorithm):
    if algorithm == "bit_star":
        return "../result_files/sphere_link_comparison_results_bit_star.csv"
    if algorithm == "gnnmp":
        return "../result_files/sphere_link_comparison_results_gnnmp.csv"
    print(f"Warning: unknown algorithm '{algorithm}', using bit_star csv by default.")
    return "../result_files/sphere_link_comparison_results_bit_star.csv"


def get_cjk_font():
    candidates = [
        "/home/lanh/.local/share/fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return fm.FontProperties(fname=path)
    return None


def safe_get_ratio(row):
    ratio_col = "Dead_Time_Total_Ratio"
    if ratio_col in row and pd.notna(row[ratio_col]):
        return float(row[ratio_col])

    dead_cycles = float(row.get("Dead_Time_Total_Cycles", 0.0))
    total_cycles = float(row.get("Total_Pred_Cycles", 0.0))
    if total_cycles <= 0:
        return 0.0
    return (dead_cycles / total_cycles) * 100.0


def extract_dead_ratios(df, difficulties):
    link_dead = []
    sphere_dead = []

    for diff in difficulties:
        l_row = df[(df["Difficulty"] == diff) & (df["Strategy"] == "link_coord")]
        s_row = df[(df["Difficulty"] == diff) & (df["Strategy"] == "sphere_coord")]

        if l_row.empty:
            link_dead.append(0.0)
        else:
            link_dead.append(safe_get_ratio(l_row.iloc[0]))

        if s_row.empty:
            sphere_dead.append(0.0)
        else:
            sphere_dead.append(safe_get_ratio(s_row.iloc[0]))

    return np.array(link_dead), np.array(sphere_dead)


def annotate_dead(ax, x_values, effective_values, dead_values):
    for x_val, eff, dead in zip(x_values, effective_values, dead_values):
        if dead <= 0:
            continue
        ax.annotate(
            f"{dead:.1f}%",
            xy=(x_val, eff + dead),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def draw_stacked_bars(ax, x_pos, width, effective, dead, base_color, label):
    ax.bar(x_pos, effective, width, color=base_color, label=label)
    ax.bar(x_pos, dead, width, bottom=effective, color=DEAD_COLOR, alpha=0.8)


def main():
    algorithm = sys.argv[1] if len(sys.argv) > 1 else "bit_star"
    csv_path = get_csv_path(algorithm)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    difficulties = ["G1", "G2", "G3", "G4", "G5"]
    cjk_font = get_cjk_font()
    link_dead, sphere_dead = extract_dead_ratios(df, difficulties)
    link_effective = 100.0 - link_dead
    sphere_effective = 100.0 - sphere_dead

    x = np.arange(len(difficulties))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    x_link = x - width / 2
    x_sphere = x + width / 2

    draw_stacked_bars(
        ax,
        x_link,
        width,
        link_effective,
        link_dead,
        LINK_COLOR,
        "连杆级推测调度",
    )
    draw_stacked_bars(
        ax,
        x_sphere,
        width,
        sphere_effective,
        sphere_dead,
        SPHERE_COLOR,
        "球体级推测调度",
    )

    annotate_dead(ax, x_link, link_effective, link_dead)
    annotate_dead(ax, x_sphere, sphere_effective, sphere_dead)

    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=FONT_SIZE)
    ax.set_ylabel("周期占比 (%)", fontsize=FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    legend_handles = [
        Patch(facecolor=LINK_COLOR, label="连杆级推测调度"),
        Patch(facecolor=SPHERE_COLOR, label="球体级推测调度"),
        Patch(facecolor=DEAD_COLOR, alpha=0.8, label="死区占比"),
    ]
    legend = ax.legend(handles=legend_handles, fontsize=LEGEND_FONT_SIZE)
    ax.set_ylim(0, 108)

    for xi in x:
        ax.axvline(x=xi + 0.5, color="#d9d9d9", linewidth=0.8, zorder=0)

    plt.tight_layout()

    output_dir = "figs"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"dead_time_ratio_sphere_link_{algorithm}.pdf")
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
