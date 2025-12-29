#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import os

# 统一绘图样式
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.style.use("seaborn-v0_8-whitegrid")
font = {
    "family": "serif",
    "weight": "normal",
    "size": 16,
}
plt.rc("font", **font)

# 统一配色方案（便于黑白打印的区分）
BASELINE_COLOR = "#BBBBBB"
LINK_COLOR = "#0072B2"
SPHERE_COLOR = "#D55E00"
ORACLE_COLOR = "#009E73"


import sys

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
        if "Total_Checks" in df.columns:
            if not s_row.empty and "Total_Checks" in s_row.columns:
                total_checks.append(s_row["Total_Checks"].values[0])
            elif not l_row.empty and "Total_Checks" in l_row.columns:
                total_checks.append(l_row["Total_Checks"].values[0])
            else:
                total_checks.append(0)
        else:
            total_checks.append(0)

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


def plot_total_prediction_cycles(difficulties, link_cycles, sphere_cycles, oracle_cycles):
    x = np.arange(len(difficulties))
    width = 0.25

    plt.figure(figsize=(10, 6))
    rects1 = plt.bar(
        x - width, link_cycles, width, label="Link-based", color=LINK_COLOR
    )
    rects2 = plt.bar(
        x, sphere_cycles, width, label="Sphere-based", color=SPHERE_COLOR
    )
    rects3 = plt.bar(
        x + width,
        oracle_cycles,
        width,
        label="Oracle (ideal)",
        color="none",
        edgecolor=ORACLE_COLOR,
        hatch="//",
        linewidth=1.5,
    )

    plt.xlabel("Difficulty Level")
    plt.ylabel("Total Prediction Cycles")
    plt.xticks(x, difficulties)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i in range(len(difficulties)):
        l_val = link_cycles[i]
        s_val = sphere_cycles[i]
        if l_val > 0:
            diff_pct = (s_val - l_val) / l_val * 100
            max_height = max(l_val, s_val)
            text_color = "black"
            sign = "+" if diff_pct > 0 else ""
            place_diff_label(
                plt.gca(), i, max_height, f"{sign}{diff_pct:.1f}%", text_color
            )

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"cycle_comparison_sphere_link{algorithm_tag}.png")
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
        series.append(("Baseline Checks", total_checks, BASELINE_COLOR, {}))
    series.append(("Link-based", link_queries, LINK_COLOR, {}))
    series.append(("Sphere-based", sphere_queries, SPHERE_COLOR, {}))
    if oracle_available:
        series.append(
            (
                "Oracle (ideal)",
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

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Total Prediction Queries (Log Scale)")
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
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # 移除相对 Baseline 的百分比标注，保持图面简洁

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"query_comparison_sphere_link{algorithm_tag}.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def plot_oocd_utilization(difficulties, link_utilization, sphere_utilization):
    x = np.arange(len(difficulties))
    width = 0.35

    plt.figure(figsize=(10, 6))
    rects5 = plt.bar(
        x - width / 2, link_utilization, width, label="Link-based", color=LINK_COLOR
    )
    rects6 = plt.bar(
        x + width / 2,
        sphere_utilization,
        width,
        label="Sphere-based",
        color=SPHERE_COLOR,
    )

    plt.xlabel("Difficulty Level")
    plt.ylabel("OOCD Utilization (%)")
    plt.xticks(x, difficulties)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in rects5:
        height = rect.get_height()
        plt.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for rect in rects6:
        height = rect.get_height()
        plt.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for i in range(len(difficulties)):
        l_val = link_utilization[i]
        s_val = sphere_utilization[i]
        if l_val > 0:
            diff_pct = s_val - l_val
            max_height = max(l_val, s_val)
            text_color = "green" if diff_pct > 0 else "red"
            sign = "+" if diff_pct > 0 else ""
            place_diff_label(
                plt.gca(),
                i,
                max_height,
                f"{sign}{diff_pct:.1f}pp",
                text_color,
                pixel_offset=10,
            )

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"utilization_comparison_sphere_link{algorithm_tag}.png")
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
    ) = extract_metrics(df, difficulties)

    plot_total_prediction_cycles(difficulties, link_cycles, sphere_cycles, oracle_cycles)
    plot_total_prediction_queries(
        difficulties, link_queries, sphere_queries, total_checks, oracle_queries
    )
    plot_oocd_utilization(difficulties, link_utilization, sphere_utilization)


if __name__ == "__main__":
    main()
