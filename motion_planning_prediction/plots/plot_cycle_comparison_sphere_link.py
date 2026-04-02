#!/usr/bin/env python3
import math
import sys
import os
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
from matplotlib.ticker import LogLocator, ScalarFormatter

# --- 统一绘图风格配置 ---
sns.set_theme(style="whitegrid")
sns.set_style("white")
sns.set_palette("colorblind")

# 字体加载与配置
font_path = os.path.expanduser("~/.local/share/fonts/simsun.ttc")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

plt.rcParams.update({
    'font.sans-serif': ['SimSun', 'NSimSun', 'STSong', 'Songti SC', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Arial Unicode MS', 'sans-serif'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# 统一配色方案（使用 seaborn colorblind 调色板）
palette = sns.color_palette("colorblind")
BASELINE_COLOR = palette[7]
LINK_COLOR = palette[0]
SPHERE_COLOR = palette[1]
ORACLE_COLOR = palette[2]

# 允许通过命令行参数指定算法

if len(sys.argv) > 1:
    algorithm = sys.argv[1]
else:
    algorithm = "bit_star"

if algorithm == "bit_star":
    csv_file = "../result_files/sphere_link_comparison_results_bit_star.csv"
elif algorithm == "gnnmp":
    csv_file = "../result_files/sphere_link_comparison_results_gnnmp.csv"
else:
    print(f"Warning: unknown algorithm '{algorithm}', using bit_star csv by default.")
    csv_file = "../result_files/sphere_link_comparison_results_bit_star.csv"

# 用于图片文件名的tag
algorithm_tag = f"_{algorithm}" if algorithm else ""

output_dir = "figs"
os.makedirs(output_dir, exist_ok=True)


def extract_metrics(df, difficulties):
    sphere_cycles = []
    link_cycles = []
    oracle_cycles = []
    sphere_queries = []
    link_queries = []
    oracle_queries = []
    sphere_utilization = []
    link_utilization = []
    total_checks = []
    baseline_cycles = []

    for diff in difficulties:
        s_row = df[(df["Difficulty"] == diff) & (df["Strategy"] == "sphere_coord")]
        l_row = df[(df["Difficulty"] == diff) & (df["Strategy"] == "link_coord")]

        sphere_cycles.append(
            s_row["Total_Pred_Cycles"].values[0] if not s_row.empty else 0
        )
        link_cycles.append(
            l_row["Total_Pred_Cycles"].values[0] if not l_row.empty else 0
        )

        if "Total_Oracle_Cycles" in df.columns:
            if not s_row.empty and "Total_Oracle_Cycles" in s_row.columns:
                oracle_cycles.append(s_row["Total_Oracle_Cycles"].values[0])
            elif not l_row.empty and "Total_Oracle_Cycles" in l_row.columns:
                oracle_cycles.append(l_row["Total_Oracle_Cycles"].values[0])
            else:
                oracle_cycles.append(0)
        else:
            oracle_cycles.append(0)

        sphere_queries.append(
            s_row["Total_Pred_Queries"].values[0] if not s_row.empty else 0
        )
        link_queries.append(
            l_row["Total_Pred_Queries"].values[0] if not l_row.empty else 0
        )

        # Total_Oracle_Queries（若存在，优先从 sphere 行，其次 link 行）
        if "Total_Oracle_Queries" in df.columns:
            if not s_row.empty and "Total_Oracle_Queries" in s_row.columns:
                oracle_queries.append(s_row["Total_Oracle_Queries"].values[0])
            elif not l_row.empty and "Total_Oracle_Queries" in l_row.columns:
                oracle_queries.append(l_row["Total_Oracle_Queries"].values[0])
            else:
                oracle_queries.append(0)
        else:
            oracle_queries.append(0)

        sphere_utilization.append(
            s_row["OOCD_Utilization"].values[0] if not s_row.empty else 0
        )
        link_utilization.append(
            l_row["OOCD_Utilization"].values[0] if not l_row.empty else 0
        )

        # Total_Checks 作为基线（优先从 sphere 行取值，其次 link 行）
        current_checks = 0
        if "Total_Checks" in df.columns:
            if not s_row.empty and "Total_Checks" in s_row.columns:
                current_checks = s_row["Total_Checks"].values[0]
            elif not l_row.empty and "Total_Checks" in l_row.columns:
                current_checks = l_row["Total_Checks"].values[0]
        
        total_checks.append(current_checks)
        # 估算基准周期: (Checks * Cost) / Num_OOCDs = (Checks * 15) / 8
        baseline_cycles.append((current_checks * 15) / 8)

    return (
        sphere_cycles,
        link_cycles,
        sphere_queries,
        link_queries,
        oracle_queries,
        sphere_utilization,
        link_utilization,
        oracle_cycles,
        total_checks,
        baseline_cycles,
    )


def place_diff_label(ax, xpos, base_height, text, color, pixel_offset=12):
    ax.annotate(
        text,
        xy=(xpos, base_height),
        xycoords="data",
        xytext=(0, pixel_offset),
        textcoords="offset pixels",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=color,
    )


def autolabel(rects, labels=None):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        text = labels[i] if labels is not None else f"{int(height):,}"
        plt.annotate(
            text,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )


def plot_total_prediction_cycles(difficulties, link_cycles, sphere_cycles, oracle_cycles, baseline_cycles):
    x = np.arange(len(difficulties))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 在对数坐标系下无法显示0，给极小值添加轻微抬升避免警告
    min_positive = 1.0
    safe_baseline = [max(v, min_positive) for v in baseline_cycles]
    safe_link = [max(v, min_positive) for v in link_cycles]
    safe_sphere = [max(v, min_positive) for v in sphere_cycles]
    safe_oracle = [max(v, min_positive) for v in oracle_cycles]

    rects0 = ax.bar(
        x - 1.5 * width, safe_baseline, width, label="基准检测周期", color=BASELINE_COLOR
    )
    rects1 = ax.bar(
        x - 0.5 * width, safe_link, width, label="连杆级", color=LINK_COLOR
    )
    rects2 = ax.bar(
        x + 0.5 * width, safe_sphere, width, label="球体级", color=SPHERE_COLOR
    )
    rects3 = ax.bar(
        x + 1.5 * width,
        safe_oracle,
        width,
        label="理想情况",
        color="none",
        edgecolor=ORACLE_COLOR,
        hatch="//",
        linewidth=1.5,
    )

    ax.set_xlabel("难度等级")
    ax.set_ylabel("总预测周期 (对数尺度)")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_yscale("log")
    
    # 增加主次刻度，提升读数可读性
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(np.arange(2, 10) * 0.1).tolist()))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="y", which="major", length=6)
    ax.tick_params(axis="y", which="minor", length=3)
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"cycle_comparison_sphere_link{algorithm_tag}.pdf")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def plot_total_prediction_queries(
    difficulties, link_queries, sphere_queries, total_checks, oracle_queries
):
    x = np.arange(len(difficulties))
    baseline_available = total_checks is not None and any(tc > 0 for tc in total_checks)
    oracle_available = oracle_queries is not None and any(
        oq > 0 for oq in oracle_queries
    )

    # 构建按顺序的系列：[Baseline, Link, Sphere, Oracle]
    series = []
    if baseline_available:
        series.append(("基准检测次数", total_checks, BASELINE_COLOR, {}))
    series.append(("连杆级", link_queries, LINK_COLOR, {}))
    series.append(("球体级", sphere_queries, SPHERE_COLOR, {}))
    if oracle_available:
        series.append(
            (
                "理想情况",
                oracle_queries,
                "none",
                {"edgecolor": ORACLE_COLOR, "hatch": "//", "linewidth": 1.5},
            )
        )

    bars_count = len(series)
    if bars_count >= 4:
        width = 0.2
    elif bars_count == 3:
        width = 0.25
    else:
        width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # 以顺序居中排列，保证左到右顺序即 series 顺序
    offsets = [(i - (bars_count - 1) / 2) * width for i in range(bars_count)]
    rects_map = {}
    # 在对数坐标系下无法显示0，给极小值添加轻微抬升避免警告
    min_positive = 1.0
    for (label, values, color, kw), dx in zip(series, offsets):
        safe_values = [max(v, min_positive) for v in values]
        rects = ax.bar(x + dx, safe_values, width, label=label, color=color, **kw)
        rects_map[label] = rects

    ax.set_xlabel("难度等级")
    ax.set_ylabel("总预测查询次数 (对数尺度)")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_yscale("log")
    # 增加主次刻度，提升读数可读性
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(np.arange(2, 10) * 0.1).tolist()))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="y", which="major", length=6)
    ax.tick_params(axis="y", which="minor", length=3)
    ax.legend()
    # grid removed per project style

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"query_comparison_sphere_link{algorithm_tag}.pdf")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def plot_oocd_utilization(difficulties, link_utilization, sphere_utilization):
    x = np.arange(len(difficulties))
    width = 0.35

    plt.figure(figsize=(10, 6))
    rects5 = plt.bar(
        x - width / 2, link_utilization, width, label="连杆级", color=LINK_COLOR
    )
    rects6 = plt.bar(
        x + width / 2,
        sphere_utilization,
        width,
        label="球体级",
        color=SPHERE_COLOR,
    )

    plt.xlabel("难度等级")
    plt.ylabel("OOCD 利用率 (%)")
    plt.xticks(x, difficulties)
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"utilization_comparison_sphere_link{algorithm_tag}.pdf")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def main():
    difficulties = ["G1", "G2", "G3", "G4", "G5"]
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
        return

    (
        sphere_cycles,
        link_cycles,
        sphere_queries,
        link_queries,
        oracle_queries,
        sphere_utilization,
        link_utilization,
        oracle_cycles,
        total_checks,
        baseline_cycles,
    ) = extract_metrics(df, difficulties)

    plot_total_prediction_cycles(difficulties, link_cycles, sphere_cycles, oracle_cycles, baseline_cycles)
    plot_total_prediction_queries(
        difficulties, link_queries, sphere_queries, total_checks, oracle_queries
    )
    plot_oocd_utilization(difficulties, link_utilization, sphere_utilization)


if __name__ == "__main__":
    main()
