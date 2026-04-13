import math
import sys

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib

# --- 统一绘图风格配置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")
import os
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")

import os
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")


# Unified plotting style
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
sns.set_style("white")
# use colorblind-friendly palette
sns.set_palette("colorblind")




def plot_global_oocd_cycle_comparison():
    """Compare Total Cycles across four Global OOCD configurations"""
    # use seaborn white style and colorblind palette
    sns.set_style("white")
    sns.set_palette("colorblind")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result_files")

    files = {
        "Dual Port (Pred=8)": "global_oocd_dual_port_pred8_results.csv",
        "Dual Port (Pred=16)": "global_oocd_dual_port_pred16_results.csv",
        "Multi-Bank (Pred=8)": "global_oocd_multi_bank_pred8_results.csv",
        "Multi-Bank (Pred=16)": "global_oocd_multi_bank_pred16_results.csv",
    }

    data = {}
    scenes = None

    print("Reading Global OOCD data files...")
    for label, filename in files.items():
        filepath = os.path.join(result_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue

        try:
            df = pd.read_csv(filepath)
            df["Scene_Num"] = df["Scene"].str.extract(r"(\d+)").astype(int)
            df = df.sort_values("Scene_Num")
            if scenes is None:
                scenes = df["Scene"].tolist()
            data[label] = df["Total_Cycles"].tolist()
            print(f"Loaded {label}: {len(df)} scenes")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if not data or scenes is None:
        print("No data or scenes loaded. Exiting.")
        return

    x = np.arange(len(scenes))
    num_strategies = len(data)
    width = 0.18

    colors = {
        "Dual Port (Pred=8)": colors[0],
        "Dual Port (Pred=16)": "#1f4e79",
        "Multi-Bank (Pred=8)": "darkgreen",
        "Multi-Bank (Pred=16)": "#2e8b57",
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (strategy_name, cycles) in enumerate(data.items()):
        offset = width * (i - num_strategies / 2 + 0.5)
        ax.bar(
            x + offset,
            cycles,
            width,
            label=strategy_name,
            color=colors.get(strategy_name, "#999999"),
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )

    # Annotate difference between Dual Port (Pred=8) and Multi-Bank (Pred=16)
    if "Dual Port (Pred=8)" in data and "Multi-Bank (Pred=16)" in data:
        dual_port_8 = data["Dual Port (Pred=8)"]
        multi_bank_16 = data["Multi-Bank (Pred=16)"]

        # 计算柱子的偏移量
        dp8_offset = width * (0 - num_strategies / 2 + 0.5)
        mb16_offset = width * (3 - num_strategies / 2 + 0.5)

        for idx, scene in enumerate(scenes):
            dp8 = dual_port_8[idx]
            mb16 = multi_bank_16[idx]
            reduction = dp8 - mb16
            reduction_pct = (reduction / dp8) * 100 if dp8 > 0 else 0
            
            # 计算标注位置 (在两个柱子之间的上方)
            mid_x = x[idx] + (dp8_offset + mb16_offset) / 2
            max_height = max(dp8, mb16)

            if reduction > 0:
                # 绘制连线
                ax.plot(
                    [x[idx] + dp8_offset, x[idx] + mb16_offset],
                    [dp8, mb16],
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=10,
                )

                # 添加文本标注
                ax.text(
                    mid_x,
                    max_height * 1.05,
                    f"-{reduction_pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="darkgreen",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="darkgreen",
                        linewidth=1.2,
                        alpha=0.9,
                    ),
                )

    ax.set_ylabel("Total Cycles", fontsize=14, fontweight="bold")
    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.tick_params(axis="y", labelsize=11)

    # grid removed per project style
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        fontsize=11,
        frameon=True,
        shadow=True,
        ncol=2,
        columnspacing=1.5,
    )

    plt.tight_layout()

    output_dir = os.path.join(current_dir, "figs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "global_oocd_cycle_comparison.pdf")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    print(f"Strategies compared: {list(data.keys())}")
    print(f"Scenes: {scenes}")


def plot_global_oocd_utilization_comparison():
    """Compare OOCD Utilization across four configurations"""
    # use seaborn white style and colorblind palette
    sns.set_style("white")
    sns.set_palette("colorblind")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result_files")

    files = {
        "Dual Port (Pred=8)": "global_oocd_dual_port_pred8_results.csv",
        "Dual Port (Pred=16)": "global_oocd_dual_port_pred16_results.csv",
        "Multi-Bank (Pred=8)": "global_oocd_multi_bank_pred8_results.csv",
        "Multi-Bank (Pred=16)": "global_oocd_multi_bank_pred16_results.csv",
    }

    data = {}
    scenes = None

    print("Reading Global OOCD utilization data...")
    for label, filename in files.items():
        filepath = os.path.join(result_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue

        try:
            df = pd.read_csv(filepath)
            df["Scene_Num"] = df["Scene"].str.extract(r"(\d+)").astype(int)
            df = df.sort_values("Scene_Num")
            if scenes is None:
                scenes = df["Scene"].tolist()
            # Convert percentage string to float
            utilization = df["Utilization"].str.rstrip("%").astype(float)
            data[label] = utilization.tolist()
            print(f"Loaded {label}: {len(df)} scenes")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if not data or scenes is None:
        print("No data or scenes loaded. Exiting.")
        return

    x = np.arange(len(scenes))
    num_strategies = len(data)
    width = 0.18

    colors = {
        "Dual Port (Pred=8)": colors[0],
        "Dual Port (Pred=16)": "#1f4e79",
        "Multi-Bank (Pred=8)": "darkgreen",
        "Multi-Bank (Pred=16)": "#2e8b57",
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (strategy_name, utilization) in enumerate(data.items()):
        offset = width * (i - num_strategies / 2 + 0.5)
        ax.bar(
            x + offset,
            utilization,
            width,
            label=strategy_name,
            color=colors.get(strategy_name, "#999999"),
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )

    ax.set_ylabel("OOCD Utilization (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)

    # grid removed per project style
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower right",
        fontsize=11,
        frameon=True,
        shadow=True,
        ncol=2,
        columnspacing=1.5,
    )

    plt.tight_layout()

    output_dir = os.path.join(current_dir, "figs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "global_oocd_utilization_comparison.pdf")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    print(f"Strategies compared: {list(data.keys())}")


def plot_global_oocd_throughput_comparison():
    """Compare Throughput across four configurations"""
    # use seaborn white style and colorblind palette
    sns.set_style("white")
    sns.set_palette("colorblind")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result_files")

    files = {
        "Dual Port (Pred=8)": "global_oocd_dual_port_pred8_results.csv",
        "Dual Port (Pred=16)": "global_oocd_dual_port_pred16_results.csv",
        "Multi-Bank (Pred=8)": "global_oocd_multi_bank_pred8_results.csv",
        "Multi-Bank (Pred=16)": "global_oocd_multi_bank_pred16_results.csv",
    }

    data = {}
    scenes = None

    print("Reading Global OOCD throughput data...")
    for label, filename in files.items():
        filepath = os.path.join(result_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue

        try:
            df = pd.read_csv(filepath)
            df["Scene_Num"] = df["Scene"].str.extract(r"(\d+)").astype(int)
            df = df.sort_values("Scene_Num")
            if scenes is None:
                scenes = df["Scene"].tolist()
            data[label] = df["Throughput"].tolist()
            print(f"Loaded {label}: {len(df)} scenes")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if not data or scenes is None:
        print("No data or scenes loaded. Exiting.")
        return

    x = np.arange(len(scenes))
    num_strategies = len(data)
    width = 0.18

    colors = {
        "Dual Port (Pred=8)": "#2E86AB",
        "Dual Port (Pred=16)": "#A23B72",
        "Multi-Bank (Pred=8)": "#F18F01",
        "Multi-Bank (Pred=16)": "#C73E1D",
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (strategy_name, throughput) in enumerate(data.items()):
        offset = width * (i - num_strategies / 2 + 0.5)
        ax.bar(
            x + offset,
            throughput,
            width,
            label=strategy_name,
            color=colors.get(strategy_name, "#999999"),
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )

    # Add reference line at throughput = 1.0
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Ideal (1.0)")

    ax.set_ylabel("Throughput (Queries/Cycle)", fontsize=14, fontweight="bold")
    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)

    # grid removed per project style
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        fontsize=11,
        frameon=True,
        shadow=True,
        ncol=2,
        columnspacing=1.5,
    )

    plt.tight_layout()

    output_dir = os.path.join(current_dir, "figs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "global_oocd_throughput_comparison.pdf")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    print(f"Strategies compared: {list(data.keys())}")


def plot_global_oocd_pred_impact():
    """Compare impact of prediction buffer size (8 vs 16) for both CHT types"""
    # use seaborn white style and colorblind palette
    sns.set_style("white")
    sns.set_palette("colorblind")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result_files")

    files = {
        "Dual Port (Pred=8)": "global_oocd_dual_port_pred8_results.csv",
        "Dual Port (Pred=16)": "global_oocd_dual_port_pred16_results.csv",
        "Multi-Bank (Pred=8)": "global_oocd_multi_bank_pred8_results.csv",
        "Multi-Bank (Pred=16)": "global_oocd_multi_bank_pred16_results.csv",
    }

    cycles_data = {}
    queries_data = {}
    scenes = None

    print("Reading data for prediction impact comparison...")
    for label, filename in files.items():
        filepath = os.path.join(result_dir, filename)
        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath)
            df["Scene_Num"] = df["Scene"].str.extract(r"(\d+)").astype(int)
            df = df.sort_values("Scene_Num")
            if scenes is None:
                scenes = df["Scene"].tolist()
            cycles_data[label] = df["Total_Cycles"].tolist()
            queries_data[label] = df["Total_Queries"].tolist()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if not cycles_data or scenes is None:
        print("No data loaded. Exiting.")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    x = np.arange(len(scenes))
    width = 0.35

    colors_dp = {"Pred=8": "#2E86AB", "Pred=16": "#A23B72"}
    colors_mb = {"Pred=8": "#F18F01", "Pred=16": "#C73E1D"}

    # Dual Port Cycles
    if "Dual Port (Pred=8)" in cycles_data and "Dual Port (Pred=16)" in cycles_data:
        ax1.bar(
            x - width / 2,
            cycles_data["Dual Port (Pred=8)"],
            width,
            label="Pred=8",
            color=colors_dp["Pred=8"],
            edgecolor="black",
            linewidth=1.2,
        )
        ax1.bar(
            x + width / 2,
            cycles_data["Dual Port (Pred=16)"],
            width,
            label="Pred=16",
            color=colors_dp["Pred=16"],
            edgecolor="black",
            linewidth=1.2,
        )
        ax1.set_ylabel("Total Cycles", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenes, fontsize=11)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
        # grid removed per project style
        ax1.legend()

    # Dual Port Queries
    if "Dual Port (Pred=8)" in queries_data and "Dual Port (Pred=16)" in queries_data:
        ax2.bar(
            x - width / 2,
            queries_data["Dual Port (Pred=8)"],
            width,
            label="Pred=8",
            color=colors_dp["Pred=8"],
            edgecolor="black",
            linewidth=1.2,
        )
        ax2.bar(
            x + width / 2,
            queries_data["Dual Port (Pred=16)"],
            width,
            label="Pred=16",
            color=colors_dp["Pred=16"],
            edgecolor="black",
            linewidth=1.2,
        )
        ax2.set_ylabel("Total Queries", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenes, fontsize=11)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
        # grid removed per project style
        ax2.legend()

    # Multi-Bank Cycles
    if "Multi-Bank (Pred=8)" in cycles_data and "Multi-Bank (Pred=16)" in cycles_data:
        ax3.bar(
            x - width / 2,
            cycles_data["Multi-Bank (Pred=8)"],
            width,
            label="Pred=8",
            color=colors_mb["Pred=8"],
            edgecolor="black",
            linewidth=1.2,
        )
        ax3.bar(
            x + width / 2,
            cycles_data["Multi-Bank (Pred=16)"],
            width,
            label="Pred=16",
            color=colors_mb["Pred=16"],
            edgecolor="black",
            linewidth=1.2,
        )
        ax3.set_ylabel("Total Cycles", fontsize=12, fontweight="bold")
        ax3.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=12, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenes, fontsize=11)
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
        # grid removed per project style
        ax3.legend()

    # Multi-Bank Queries
    if "Multi-Bank (Pred=8)" in queries_data and "Multi-Bank (Pred=16)" in queries_data:
        ax4.bar(
            x - width / 2,
            queries_data["Multi-Bank (Pred=8)"],
            width,
            label="Pred=8",
            color=colors_mb["Pred=8"],
            edgecolor="black",
            linewidth=1.2,
        )
        ax4.bar(
            x + width / 2,
            queries_data["Multi-Bank (Pred=16)"],
            width,
            label="Pred=16",
            color=colors_mb["Pred=16"],
            edgecolor="black",
            linewidth=1.2,
        )
        ax4.set_ylabel("Total Queries", fontsize=12, fontweight="bold")
        ax4.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=12, fontweight="bold")
        ax4.set_xticks(x)
        ax4.set_xticklabels(scenes, fontsize=11)
        ax4.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
        # grid removed per project style
        ax4.legend()

    plt.tight_layout()

    output_dir = os.path.join(current_dir, "figs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "global_oocd_pred_impact.pdf")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    plot_global_oocd_cycle_comparison()
    plot_global_oocd_utilization_comparison()
    plot_global_oocd_throughput_comparison()
    plot_global_oocd_pred_impact()
