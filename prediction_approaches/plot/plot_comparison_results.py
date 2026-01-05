#!/usr/bin/env python3
"""
OBB与Sphere碰撞预测性能对比绘图脚本
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import matplotlib
import os
import sys

# Ensure local package modules are importable when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import add_bar_labels

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

font = {
    "family": "serif",
    "weight": "normal",
    "size": 28,
}

# ===== 全局配色方案 =====
LINK_COLOR = "#0072B2"  # Link-based 方法主色
SPHERE_COLOR = "#D55E00"  # Sphere-based 方法主色
# 供多曲线使用的渐变色（由深到浅）
LINK_COLOR_SET = ["#003366", "#0072B2", "#88CCEE"]  # Dark Blue, Base Blue, Light Cyan
SPHERE_COLOR_SET = [
    "#663300",
    "#D55E00",
    "#FFCC99",
]  # Dark Brown, Base Orange, Light Peach
# Precision: Blue, Recall: Vermilion (Orange-Red), SpeedUp: Bluish Green
METRIC_COLORS = ["#0072B2", "#D55E00", "#009E73"]
# ======================

# ===== 全局字号配置参数 =====
FONT_SIZE_TITLE = 22  # 子图标题字号
FONT_SIZE_LABEL = 22  # 坐标轴标签字号（X/Y Label）
FONT_SIZE_TICK = 18  # 坐标轴刻度字号
FONT_SIZE_LEGEND = 24  # 图例字号
FONT_SIZE_TEXT = 18  # 数值标签、注释等字号5
# =========================


def plot_accuracy_recall_comparison():
    """
    图1: 不同密度场景下OBB和Sphere策略的准确率和召回率对比
    统一取QuantBits=4(OBB)和QuantBits=4(Sphere)，对不同Threshold和SampleRate计算平均值
    输出两张独立图: 精确率和召回率
    """
    # 读取OBB详细结果
    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    # 读取Sphere详细结果
    sphere_data = pd.read_csv(
        "../result_files/sphere_hashing_cost_results.csv",
        header=0,
    )

    # 提取每个密度级别的结果
    densities = ["dens3", "dens6", "dens9", "dens12"]
    density_labels = ["Density 3", "Density 6", "Density 9", "Density 12"]

    obb_precision = []
    obb_recall = []
    sphere_precision = []
    sphere_recall = []

    for density in densities:
        # OBB: 固定QuantBits=4，对所有Threshold和SampleRate求平均
        obb_density = obb_data[
            (obb_data["Density"] == density) & (obb_data["QuantBits"] == 4)
        ]
        obb_precision.append(obb_density["PosePrecision"].mean())
        obb_recall.append(obb_density["PoseRecall"].mean())

        # Sphere: 固定QuantBits=4，对所有Threshold和SampleRate求平均
        sphere_density = sphere_data[
            (sphere_data["Density"] == density) & (sphere_data["QuantBits"] == 4)
        ]
        sphere_precision.append(sphere_density["PosePrecision"].mean())
        sphere_recall.append(sphere_density["PoseRecall"].mean())

    x = np.arange(len(densities))
    width = 0.35

    # === 精确率图 ===
    fig_prec, ax_prec = plt.subplots(figsize=(10, 7))
    plt.rc("font", **font)
    fig_prec.patch.set_facecolor("white")
    ax_prec.set_facecolor("white")

    bars1 = ax_prec.bar(
        x - width / 2,
        obb_precision,
        width,
        label="Link-based",
        color=LINK_COLOR,
    )
    bars2 = ax_prec.bar(
        x + width / 2,
        sphere_precision,
        width,
        label="Sphere-based",
        color=SPHERE_COLOR,
    )

    ax_prec.set_ylabel("Precision (%)", fontsize=FONT_SIZE_LABEL)
    ax_prec.set_xticks(x)
    ax_prec.set_xticklabels(["3", "6", "9", "12"], fontsize=FONT_SIZE_TICK)
    ax_prec.set_ylim((0, 115))
    ax_prec.grid(axis="y", alpha=0.2, linestyle="--")
    ax_prec.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax_prec.set_xlabel("Obstacle Density", fontsize=FONT_SIZE_LABEL)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_prec.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_TEXT,
            )

    ax_prec.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig("figs/fig_obb_sphere_precision.png", dpi=300)
    print("✅ 图1-Precision已保存: figs/fig_obb_sphere_precision.png")
    plt.close()

    # === 召回率图 ===
    fig_rec, ax_rec = plt.subplots(figsize=(10, 7))
    plt.rc("font", **font)
    fig_rec.patch.set_facecolor("white")
    ax_rec.set_facecolor("white")

    bars3 = ax_rec.bar(
        x - width / 2,
        obb_recall,
        width,
        label="Link-based",
        color=LINK_COLOR,
    )
    bars4 = ax_rec.bar(
        x + width / 2,
        sphere_recall,
        width,
        label="Sphere-based",
        color=SPHERE_COLOR,
    )

    ax_rec.set_ylabel("Recall (%)", fontsize=FONT_SIZE_LABEL)
    ax_rec.set_xticks(x)
    ax_rec.set_xticklabels(["3", "6", "9", "12"], fontsize=FONT_SIZE_TICK)
    ax_rec.set_ylim((0, 115))
    ax_rec.grid(axis="y", alpha=0.2, linestyle="--")
    ax_rec.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax_rec.set_xlabel("Obstacle Density", fontsize=FONT_SIZE_LABEL)

    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax_rec.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_TEXT,
            )

    ax_rec.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig("figs/fig_obb_sphere_recall.png", dpi=300)
    print("✅ 图1-Recall已保存: figs/fig_obb_sphere_recall.png")
    plt.close()


def plot_cost_comparison():
    """
    图2: 不同密度场景下OBB和Sphere策略的SpeedUp_Pct对比
    从对应CSV直接提取SpeedUp_Pct并进行比较（每个密度取最大值）
    """
    # 读取优化结果
    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    sphere_data = pd.read_csv(
        "../result_files/sphere_hashing_cost_results.csv",
        header=0,
    )

    densities = ["dens3", "dens6", "dens9", "dens12"]
    density_labels = ["Density 3", "Density 6", "Density 9", "Density 12"]

    obb_speedups = []
    sphere_speedups = []

    for density in densities:
        # 直接比较SpeedUp_Pct：为每个密度选择中位数（稳健统计）
        obb_density = obb_data[obb_data["Density"] == density]
        sphere_density = sphere_data[sphere_data["Density"] == density]

        obb_speed = (
            obb_density["SpeedUp_Pct"].median() if not obb_density.empty else 0.0
        )
        sphere_speed = (
            sphere_density["SpeedUp_Pct"].median() if not sphere_density.empty else 0.0
        )

        obb_speedups.append(obb_speed)
        sphere_speedups.append(sphere_speed)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.rc("font", **font)
    # 白底与轻网格
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(densities))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        obb_speedups,
        width,
        label="Link-based",
        color=LINK_COLOR,
    )
    bars2 = ax.bar(
        x + width / 2,
        sphere_speedups,
        width,
        label="Sphere-based",
        color=SPHERE_COLOR,
    )

    # SpeedUp_Pct 表示相较于无预测基线的相对计算量（百分比）
    ax.set_ylabel("Relative Computation Cost (%)", fontsize=FONT_SIZE_LABEL - 2)
    ax.set_xticks(x)
    ax.set_xticklabels(["3", "6", "9", "12"], fontsize=FONT_SIZE_TICK)
    # 动态上限，为数值标签留空间
    max_val = (
        max([*obb_speedups, *sphere_speedups])
        if (obb_speedups + sphere_speedups)
        else 0
    )
    ax.set_ylim((0, max_val * 1.15 if max_val > 0 else 100))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

    # 在柱状图上添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_TEXT + 1,
            )

    # 共享图例置顶，避免遮挡
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )
    # 统一X轴标签
    fig.text(0.5, 0.01, "Obstacle Density", ha="center", fontsize=FONT_SIZE_LABEL - 2)

    plt.tight_layout(rect=(0, 0.03, 1, 0.93))
    plt.savefig("figs/fig_obb_sphere_cost.png", dpi=300)
    print("✅ 图2已保存: figs/fig_obb_sphere_cost.png")
    plt.close()


def plot_threshold_comparison(density="dens6"):
    """
    图3: 不同阈值下,OBB和Sphere策略的准确率和召回率对比
    为每个QuantBits/QuantBits生成独立的图表

    Args:
        density: 密度级别 ('dens3', 'dens6', 'dens9', 'dens12')
    """
    # 读取详细结果
    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    sphere_data = pd.read_csv(
        "../result_files/sphere_hashing_cost_results.csv",
        header=0,
    )

    quant_bits = [3, 4, 5]
    coord_bits = [3, 4, 5]

    # 为每个QuantBits生成OBB图表
    for qb in quant_bits:
        obb_fixed = obb_data[
            (obb_data["Density"] == density) & (obb_data["QuantBits"] == qb)
        ].sort_values("Threshold")

        if len(obb_fixed) == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        plt.rc("font", **font)
        fig.patch.set_facecolor("white")
        for ax in axes:
            ax.set_facecolor("white")

        thresholds = obb_fixed["Threshold"].values
        threshold_labels = [f"S={t:.2f}" for t in thresholds]
        x = np.arange(len(thresholds))

        # 子图1: OBB精确率
        axes[0].plot(
            x,
            obb_fixed["PosePrecision"].values,
            "o-",
            linewidth=2.5,
            markersize=8,
            color=LINK_COLOR,
            label="Precision",
        )
        axes[0].set_ylabel("Precision (%)", fontsize=FONT_SIZE_LABEL)
        axes[0].set_title(
            f"Link-based (Q={qb}) - {density.upper()}", fontsize=FONT_SIZE_TITLE
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(
            threshold_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK
        )
        axes[0].grid(alpha=0.2, linestyle="--")
        axes[0].set_ylim([0, 100])
        axes[0].tick_params(axis="y", labelsize=FONT_SIZE_TICK)

        # 子图2: OBB召回率
        axes[1].plot(
            x,
            obb_fixed["PoseRecall"].values,
            "s-",
            linewidth=2.5,
            markersize=8,
            color=LINK_COLOR,
            label="Recall",
        )
        axes[1].set_ylabel("Recall (%)", fontsize=FONT_SIZE_LABEL)
        axes[1].set_title(
            f"Link-based (Q={qb}) - {density.upper()}", fontsize=FONT_SIZE_TITLE
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            threshold_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK
        )
        axes[1].grid(alpha=0.2, linestyle="--")
        axes[1].set_ylim([0, 100])
        axes[1].tick_params(axis="y", labelsize=FONT_SIZE_TICK)

        # 共享图例置于顶部
        handles0, labels0 = axes[0].get_legend_handles_labels()
        fig.legend(
            handles0,
            labels0,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            fontsize=FONT_SIZE_LEGEND,
        )

        plt.tight_layout()
        plt.savefig(f"figs/fig_threshold_comparison_{density}_obb_q{qb}.png", dpi=300)
        print(f"✅ 图3已保存: figs/fig_threshold_comparison_{density}_obb_q{qb}.png")
        plt.close()

    # 为每个QuantBits生成Sphere图表
    for cb in coord_bits:
        sphere_fixed = sphere_data[
            (sphere_data["Density"] == density)
            & (sphere_data["QuantBits"] == cb)
            & (sphere_data["RadiusBits"] == 1)
        ].sort_values("Threshold")

        if len(sphere_fixed) == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        plt.rc("font", **font)
        fig.patch.set_facecolor("white")
        for ax in axes:
            ax.set_facecolor("white")

        thresholds = sphere_fixed["Threshold"].values
        threshold_labels = [f"S={t:.2f}" for t in thresholds]
        x = np.arange(len(thresholds))

        # 子图1: Sphere精确率
        axes[0].plot(
            x,
            sphere_fixed["PosePrecision"].values,
            "o-",
            linewidth=2.5,
            markersize=8,
            color=SPHERE_COLOR,
            label="Precision",
        )
        axes[0].set_ylabel("Precision (%)", fontsize=FONT_SIZE_LABEL)
        axes[0].set_title(
            f"Sphere-based (C={cb}) - {density.upper()}", fontsize=FONT_SIZE_TITLE
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(
            threshold_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK
        )
        axes[0].grid(alpha=0.2, linestyle="--")
        axes[0].set_ylim([0, 100])
        axes[0].tick_params(axis="y", labelsize=FONT_SIZE_TICK)

        # 子图2: Sphere召回率
        axes[1].plot(
            x,
            sphere_fixed["PoseRecall"].values,
            "s-",
            linewidth=2.5,
            markersize=8,
            color=SPHERE_COLOR,
            label="Recall",
        )
        axes[1].set_ylabel("Recall (%)", fontsize=FONT_SIZE_LABEL)
        axes[1].set_title(
            f"Sphere-based (C={cb}) - {density.upper()}", fontsize=FONT_SIZE_TITLE
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            threshold_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK
        )
        axes[1].grid(alpha=0.2, linestyle="--")
        axes[1].set_ylim([0, 100])
        axes[1].tick_params(axis="y", labelsize=FONT_SIZE_TICK)

        # 共享图例置于顶部
        handles0, labels0 = axes[0].get_legend_handles_labels()
        fig.legend(
            handles0,
            labels0,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            fontsize=FONT_SIZE_LEGEND,
        )

        plt.tight_layout()
        plt.savefig(
            f"figs/fig_threshold_comparison_{density}_sphere_c{cb}.png", dpi=300
        )
        print(f"✅ 图3已保存: figs/fig_threshold_comparison_{density}_sphere_c{cb}.png")
        plt.close()


def plot_combined_threshold_comparison():
    """
    图3综合版: 在同一张图中对比OBB和Sphere在不同阈值下的表现
    为每个QuantBits/QuantBits绘制单独的曲线
    """
    # 读取详细结果
    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    sphere_data = pd.read_csv(
        "../result_files/sphere_hashing_cost_results.csv",
        header=0,
    )

    densities = ["dens3", "dens6", "dens9", "dens12"]
    density_labels = {
        "dens3": "Density 3",
        "dens6": "Density 6",
        "dens9": "Density 9",
        "dens12": "Density 12",
    }

    quant_bits = [3, 4, 5]
    coord_bits = [3, 4, 5]

    # 颜色映射
    obb_colors = LINK_COLOR_SET
    sphere_colors = SPHERE_COLOR_SET
    markers_obb = ["o", "s", "^"]
    markers_sphere = ["o", "s", "^"]

    # 创建4行2列的子图
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    plt.rc("font", **font)
    fig.patch.set_facecolor("white")
    for axrow in axes:
        for ax in axrow:
            ax.set_facecolor("white")

    for idx, density in enumerate(densities):
        # 获取唯一阈值（所有配置共享相同的阈值）
        thresholds = sorted(
            obb_data[obb_data["Density"] == density]["Threshold"].unique()
        )
        threshold_labels = [f"{t:.2f}" for t in thresholds]
        x = np.arange(len(thresholds))

        # 左列: 精确率对比
        ax_prec = axes[idx, 0]

        # 绘制不同QuantBits的OBB曲线
        for qb_idx, qb in enumerate(quant_bits):
            obb_subset = obb_data[
                (obb_data["Density"] == density) & (obb_data["QuantBits"] == qb)
            ].sort_values("Threshold")
            ax_prec.plot(
                x,
                obb_subset["PosePrecision"].values,
                marker=markers_obb[qb_idx],
                linestyle="-",
                linewidth=2.5,
                markersize=7,
                color=obb_colors[qb_idx],
                label=f"Link-based (Q={qb})",
                alpha=0.8,
            )

        # 绘制不同QuantBits的Sphere曲线
        for cb_idx, cb in enumerate(coord_bits):
            sphere_subset = sphere_data[
                (sphere_data["Density"] == density)
                & (sphere_data["QuantBits"] == cb)
                & (sphere_data["RadiusBits"] == 1)
            ].sort_values("Threshold")
            ax_prec.plot(
                x,
                sphere_subset["PosePrecision"].values,
                marker=markers_sphere[cb_idx],
                linestyle="--",
                linewidth=2.5,
                markersize=7,
                color=sphere_colors[cb_idx],
                label=f"Sphere-based (C={cb})",
                alpha=0.8,
            )

        ax_prec.set_ylabel("Precision (%)", fontsize=FONT_SIZE_LABEL)
        ax_prec.set_title(
            f"{density_labels[density]} - Precision", fontsize=FONT_SIZE_TITLE
        )
        ax_prec.set_xticks(x)
        ax_prec.set_xticklabels(
            threshold_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK
        )
        ax_prec.grid(alpha=0.2, linestyle="--")
        ax_prec.set_ylim([0, 100])
        ax_prec.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

        # 右列: 召回率对比
        ax_rec = axes[idx, 1]

        # 绘制不同QuantBits的OBB曲线
        for qb_idx, qb in enumerate(quant_bits):
            obb_subset = obb_data[
                (obb_data["Density"] == density) & (obb_data["QuantBits"] == qb)
            ].sort_values("Threshold")
            ax_rec.plot(
                x,
                obb_subset["PoseRecall"].values,
                marker=markers_obb[qb_idx],
                linestyle="-",
                linewidth=2.5,
                markersize=7,
                color=obb_colors[qb_idx],
                label=f"Link-based (Q={qb})",
                alpha=0.8,
            )

        # 绘制不同QuantBits的Sphere曲线
        for cb_idx, cb in enumerate(coord_bits):
            sphere_subset = sphere_data[
                (sphere_data["Density"] == density)
                & (sphere_data["QuantBits"] == cb)
                & (sphere_data["RadiusBits"] == 1)
            ].sort_values("Threshold")
            ax_rec.plot(
                x,
                sphere_subset["PoseRecall"].values,
                marker=markers_sphere[cb_idx],
                linestyle="--",
                linewidth=2.5,
                markersize=7,
                color=sphere_colors[cb_idx],
                label=f"Sphere-based (C={cb})",
                alpha=0.8,
            )

        ax_rec.set_ylabel("Recall (%)", fontsize=FONT_SIZE_LABEL)
        ax_rec.set_title(
            f"{density_labels[density]} - Recall", fontsize=FONT_SIZE_TITLE
        )
        ax_rec.set_xticks(x)
        ax_rec.set_xticklabels(
            threshold_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK
        )
        ax_rec.grid(alpha=0.2, linestyle="--")
        ax_rec.set_ylim([0, 100])
        ax_rec.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

    # 添加x轴总标签
    # 顶部共享图例：汇总所有线条标签，避免每图重复图例
    handles, labels = [], []
    for axrow in axes:
        for ax in axrow:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
    # 去重
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    fig.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )

    fig.text(0.5, 0.01, "Threshold Value (S)", ha="center", fontsize=FONT_SIZE_LABEL)

    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    plt.savefig("figs/fig_threshold_comparison_combined.png", dpi=300)
    print("✅ 图3综合版已保存: figs/fig_threshold_comparison_combined.png")
    plt.close()


def plot_pr_curves():
    """
    图4: 不同障碍物密度下,OBB和Sphere策略的P-R曲线
    P-R曲线显示在不同阈值下,精确率(Precision)和召回率(Recall)之间的权衡关系
    """
    # 读取详细结果
    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    sphere_data = pd.read_csv(
        "../result_files/sphere_hashing_cost_results.csv",
        header=0,
    )

    densities = ["dens3", "dens6", "dens9", "dens12"]
    density_labels = {
        "dens3": "Density 3",
        "dens6": "Density 6",
        "dens9": "Density 9",
        "dens12": "Density 12",
    }

    # 创建图表 - 4个子图分别对应4种密度
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    plt.rc("font", **font)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    for idx, density in enumerate(densities):
        ax = axes[idx]

        # 筛选当前密度的数据，Sphere只考虑RadiusBits=1的情况
        obb_density = obb_data[obb_data["Density"] == density]
        sphere_density = sphere_data[
            (sphere_data["Density"] == density) & (sphere_data["RadiusBits"] == 1)
        ]

        # 按召回率排序(从低到高),以便P-R曲线更清晰
        obb_density = obb_density.sort_values("PoseRecall")
        sphere_density = sphere_density.sort_values("PoseRecall")

        # 绘制Link-based的P-R曲线
        ax.plot(
            obb_density["PoseRecall"].values,
            obb_density["PosePrecision"].values,
            "o-",
            linewidth=3,
            markersize=8,
            color=LINK_COLOR,
            label="Link-based",
            alpha=0.8,
        )

        # 绘制Sphere-based的P-R曲线
        ax.plot(
            sphere_density["PoseRecall"].values,
            sphere_density["PosePrecision"].values,
            "s-",
            linewidth=3,
            markersize=8,
            color=SPHERE_COLOR,
            label="Sphere-based",
            alpha=0.8,
        )

        # 标注几个关键点的阈值
        # 选择3个代表性的点进行标注
        n_points = len(obb_density)
        if n_points >= 3:
            indices_to_label = [0, n_points // 2, n_points - 1]
            for i in indices_to_label:
                obb_row = obb_density.iloc[i]
                threshold = obb_row["Threshold"]
                ax.annotate(
                    f"S={threshold:.2f}",
                    xy=(obb_row["PoseRecall"], obb_row["PosePrecision"]),
                    xytext=(10, -15),
                    textcoords="offset points",
                    fontsize=16,
                    color=LINK_COLOR,
                    alpha=0.7,
                    arrowprops=dict(arrowstyle="->", color=LINK_COLOR, alpha=0.5),
                )

        n_points_sphere = len(sphere_density)
        if n_points_sphere >= 3:
            indices_to_label = [0, n_points_sphere // 2, n_points_sphere - 1]
            for i in indices_to_label:
                sphere_row = sphere_density.iloc[i]
                threshold = sphere_row["Threshold"]
                ax.annotate(
                    f"S={threshold:.2f}",
                    xy=(sphere_row["PoseRecall"], sphere_row["PosePrecision"]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=16,
                    color=SPHERE_COLOR,
                    alpha=0.7,
                    arrowprops=dict(arrowstyle="->", color=SPHERE_COLOR, alpha=0.5),
                )

        # 设置坐标轴
        ax.set_xlabel("Recall (%)", fontsize=28)
        ax.set_ylabel("Precision (%)", fontsize=FONT_SIZE_LABEL)
        ax.set_title(density_labels[density], fontsize=FONT_SIZE_TITLE)
        ax.set_xlim([0, 105])
        ax.set_ylim([0, 105])
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
        ax.grid(alpha=0.2, linestyle="--")

        # 添加对角线参考线(表示Precision = Recall)
        ax.plot([0, 100], [0, 100], "k--", alpha=0.2, linewidth=1)

    # 顶部共享图例（2项），避免遮挡
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig("figs/fig_obb_sphere_pr_curves.png", dpi=300)
    print("✅ 图4已保存: figs/fig_obb_sphere_pr_curves.png")
    plt.close()


def plot_cost_vs_threshold():
    """
    图5: 四种密度场景下OBB和Sphere计算成本随阈值S的变化
    OBB和Sphere的成本曲线在同一张图上进行对比
    """
    # 读取详细结果
    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    sphere_data = pd.read_csv(
        "../result_files/sphere_hashing_cost_results.csv",
        header=0,
    )

    densities = ["dens3", "dens6", "dens9", "dens12"]
    density_labels = {
        "dens3": "Density 3",
        "dens6": "Density 6",
        "dens9": "Density 9",
        "dens12": "Density 12",
    }

    # 创建图表 - 4个子图分别对应4种密度
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    plt.rc("font", **font)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    for idx, density in enumerate(densities):
        ax = axes[idx]

        # 筛选当前密度的数据，Sphere只考虑RadiusBits=1的情况
        obb_density = obb_data[obb_data["Density"] == density]
        sphere_density = sphere_data[
            (sphere_data["Density"] == density) & (sphere_data["RadiusBits"] == 1)
        ]

        # 按阈值排序
        obb_density = obb_density.sort_values("Threshold")
        sphere_density = sphere_density.sort_values("Threshold")

        # 提取阈值和成本（确保为NumPy数组）
        obb_thresholds = np.array(obb_density["Threshold"].values)
        obb_costs = np.array(obb_density["PredCost"].values)
        sphere_thresholds = np.array(sphere_density["Threshold"].values)
        sphere_costs = np.array(sphere_density["PredCost"].values)

        # 创建对数坐标映射:将0映射到一个小的负值,其他值保持不变
        # 这样可以在对数坐标上正确显示
        min_nonzero = min([t for t in obb_thresholds if t > 0])
        log_offset = min_nonzero / 10  # 用最小非零值的1/10作为0的显示位置

        obb_x_positions = np.array(
            [log_offset if t == 0 else t for t in obb_thresholds]
        )
        sphere_x_positions = np.array(
            [log_offset if t == 0 else t for t in sphere_thresholds]
        )

        # 先设置对数坐标
        ax.set_xscale("log")

        # 绘制Link-based成本曲线
        ax.plot(
            obb_x_positions,
            obb_costs,
            "o-",
            linewidth=3,
            markersize=8,
            color=LINK_COLOR,
            label="Link-based",
            alpha=0.8,
        )

        # 绘制Sphere-based成本曲线
        ax.plot(
            sphere_x_positions,
            sphere_costs,
            "s-",
            linewidth=3,
            markersize=8,
            color=SPHERE_COLOR,
            label="Sphere-based",
            alpha=0.8,
        )

        # 标注最优点
        obb_min_idx = np.argmin(obb_costs)
        sphere_min_idx = np.argmin(sphere_costs)

        ax.plot(
            obb_x_positions[obb_min_idx],
            obb_costs[obb_min_idx],
            "*",
            markersize=20,
            color="red",
            alpha=0.8,
            zorder=5,
        )
        ax.annotate(
            f"Min: S={obb_thresholds[obb_min_idx]:.3f}\nCost={obb_costs[obb_min_idx]:.1f}",
            xy=(obb_x_positions[obb_min_idx], obb_costs[obb_min_idx]),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=18,
            color="navy",
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )

        ax.plot(
            sphere_x_positions[sphere_min_idx],
            sphere_costs[sphere_min_idx],
            "*",
            markersize=20,
            color="red",
            alpha=0.8,
            zorder=5,
        )
        ax.annotate(
            f"Min: S={sphere_thresholds[sphere_min_idx]:.3f}\nCost={sphere_costs[sphere_min_idx]:.1f}",
            xy=(sphere_x_positions[sphere_min_idx], sphere_costs[sphere_min_idx]),
            xytext=(15, -30),
            textcoords="offset points",
            fontsize=18,
            color="darkgreen",
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )

        # 设置坐标轴
        ax.set_xlabel("Threshold Value (S)", fontsize=28)
        ax.set_ylabel("Prediction Cost", fontsize=FONT_SIZE_LABEL)
        ax.set_title(density_labels[density], fontsize=FONT_SIZE_TITLE)

        # 设置y轴从0开始
        ax.set_ylim(bottom=0)

        # 设置x轴范围和刻度
        ax.set_xlim(left=log_offset * 0.8, right=max(obb_thresholds) * 1.2)

        # 创建刻度位置和标签
        tick_positions = []
        tick_labels = []
        for i, t in enumerate(obb_thresholds):
            tick_positions.append(obb_x_positions[i])
            if t == 0:
                tick_labels.append("0")
            else:
                tick_labels.append(f"{t:.3f}")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK
        )

        ax.grid(alpha=0.2, linestyle="--", which="both")
        ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

    # 顶部共享图例（OBB/Sphere）
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig("figs/fig_cost_vs_threshold.png", dpi=300)
    print("✅ 图5已保存: figs/fig_cost_vs_threshold.png")
    plt.close()


def plot_cost_vs_quantbits():
    """
    图6: 分析Cost随QuantBits/QuantBits的变化情况
    OBB和Sphere分别生成独立的图表,展示不同密度下最小成本随量化位数的变化
    """
    # 读取详细结果
    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    sphere_data = pd.read_csv(
        "../result_files/sphere_hashing_cost_results.csv",
        header=0,
    )

    densities = ["dens3", "dens6", "dens9", "dens12"]
    density_labels = {
        "dens3": "Density 3",
        "dens6": "Density 6",
        "dens9": "Density 9",
        "dens12": "Density 12",
    }

    quant_bits = [3, 4, 5]
    coord_bits = [3, 4, 5]

    # 颜色映射 - 每个密度一个颜色
    density_colors = {
        "dens3": "#1f77b4",
        "dens6": "#ff7f0e",
        "dens9": "#2ca02c",
        "dens12": "#d62728",
    }
    markers = ["o", "s", "^", "D"]

    # === OBB图表 ===
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rc("font", **font)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for idx, density in enumerate(densities):
        min_costs = []
        for qb in quant_bits:
            # 找到该密度和QuantBits下成本最小的配置
            subset = obb_data[
                (obb_data["Density"] == density) & (obb_data["QuantBits"] == qb)
            ]
            if len(subset) > 0:
                min_cost = subset["PredCost"].min()
                min_costs.append(min_cost)
            else:
                min_costs.append(None)

        # 绘制曲线
        ax.plot(
            quant_bits,
            min_costs,
            marker=markers[idx],
            linestyle="-",
            linewidth=2.5,
            markersize=10,
            color=density_colors[density],
            label=density_labels[density],
            alpha=0.8,
        )

    ax.set_xlabel("QuantBits", fontsize=28)
    ax.set_ylabel("Minimum Prediction Cost", fontsize=FONT_SIZE_LABEL)
    ax.set_title("OBB: Cost vs QuantBits", fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(quant_bits)
    ax.set_xticklabels([str(qb) for qb in quant_bits], fontsize=FONT_SIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("figs/fig_cost_vs_quantbits_obb.eps", format="eps")
    print("✅ 图6-OBB已保存: figs/fig_cost_vs_quantbits_obb.eps")
    plt.close()

    # === Sphere图表 ===
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rc("font", **font)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for idx, density in enumerate(densities):
        min_costs = []
        for cb in coord_bits:
            # 找到该密度和QuantBits下成本最小的配置(只考虑RadiusBits=1)
            subset = sphere_data[
                (sphere_data["Density"] == density)
                & (sphere_data["QuantBits"] == cb)
                & (sphere_data["RadiusBits"] == 1)
            ]
            if len(subset) > 0:
                min_cost = subset["PredCost"].min()
                min_costs.append(min_cost)
            else:
                min_costs.append(None)

        # 绘制曲线
        ax.plot(
            coord_bits,
            min_costs,
            marker=markers[idx],
            linestyle="-",
            linewidth=2.5,
            markersize=10,
            color=density_colors[density],
            label=density_labels[density],
            alpha=0.8,
        )

    ax.set_xlabel("QuantBits", fontsize=28)
    ax.set_ylabel("Minimum Prediction Cost", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Sphere: Cost vs QuantBits", fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(coord_bits)
    ax.set_xticklabels([str(cb) for cb in coord_bits], fontsize=FONT_SIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("figs/fig_cost_vs_QuantBits_sphere.eps", format="eps")
    print("✅ 图6-Sphere已保存: figs/fig_cost_vs_QuantBits_sphere.eps")
    plt.close()


def plot_threshold_metrics_by_density():
    """
    绘制QuantBits=4和QuantBits=4情况下不同密度场景的性能指标
    每个密度场景生成一个图，对比OBB和Sphere在不同Threshold下的Precision、Recall和SpeedUp_Pct
    """
    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    sphere_data = pd.read_csv(
        "../result_files/sphere_hashing_cost_results.csv", header=0
    )

    densities = ["dens3", "dens6", "dens9", "dens12"]
    density_labels = {
        "dens3": "Density 3",
        "dens6": "Density 6",
        "dens9": "Density 9",
        "dens12": "Density 12",
    }

    for density in densities:
        obb_density = obb_data[
            (obb_data["Density"] == density) & (obb_data["QuantBits"] == 4)
        ].sort_values("Threshold")

        sphere_density = sphere_data[
            (sphere_data["Density"] == density)
            & (sphere_data["QuantBits"] == 4)
            & (sphere_data["RadiusBits"] == 0)
        ].sort_values("Threshold")

        if len(obb_density) == 0 or len(sphere_density) == 0:
            print(f"⚠️  {density}: Missing data for QuantBits=4 or QuantBits=4")
            continue

        # 创建1x3子图
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        plt.rc("font", **font)
        fig.patch.set_facecolor("white")
        for ax in axes:
            ax.set_facecolor("white")

        # 提取数据
        thresholds_obb = obb_density["Threshold"].values
        metrics_obb = [
            obb_density["PosePrecision"].values,
            obb_density["PoseRecall"].values,
            obb_density["SpeedUp_Pct"].values,
        ]

        thresholds_sphere = sphere_density["Threshold"].values
        metrics_sphere = [
            sphere_density["PosePrecision"].values,
            sphere_density["PoseRecall"].values,
            sphere_density["SpeedUp_Pct"].values,
        ]

        metric_names = ["Precision (%)", "Recall (%)", "SpeedUp_Pct (%)"]
        metric_titles = ["Precision", "Recall", "SpeedUp_Pct"]

        # 绘制三个指标
        for i, (ax, name, title) in enumerate(zip(axes, metric_names, metric_titles)):
            x = np.arange(len(thresholds_obb))
            width = 0.35

            ax.bar(
                x - width / 2,
                metrics_obb[i],
                width,
                label="Link-based",
                color=LINK_COLOR,
                alpha=0.8,
            )
            ax.bar(
                x + width / 2,
                metrics_sphere[i],
                width,
                label="Sphere-based",
                color=SPHERE_COLOR,
                alpha=0.8,
            )

            ax.set_ylabel(name, fontsize=FONT_SIZE_LABEL)
            ax.set_title(title, fontsize=FONT_SIZE_TITLE)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{t:.3f}" for t in thresholds_obb],
                rotation=45,
                ha="right",
                fontsize=FONT_SIZE_TICK,
            )
            ax.set_xlabel("Threshold Value (S)", fontsize=FONT_SIZE_LABEL)
            ax.grid(axis="y", alpha=0.2, linestyle="--")
            ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
            ax.set_ylim(
                [0, max(np.max(metrics_obb[i]), np.max(metrics_sphere[i])) * 1.15]  # type: ignore
            )

            # 添加数值标签
            for bars in [ax.containers[0], ax.containers[1]]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=FONT_SIZE_TEXT - 4,
                    )

        # 添加共享图例
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            frameon=False,
            fontsize=FONT_SIZE_LEGEND,
        )

        fig.suptitle(
            f"{density_labels[density]} - Performance Metrics Comparison",
            fontsize=FONT_SIZE_TITLE + 2,
            y=1.02,
        )

        plt.tight_layout(rect=(0, 0.02, 1, 0.95))
        plt.savefig(f"figs/fig_threshold_metrics_{density}_comparison.png", dpi=300)
        print(f"✅ 图已保存: figs/fig_threshold_metrics_{density}_comparison.png")
        plt.close()


def plot_precision_recall_by_density():
    """
    图8: 针对每个密度场景，使用分组直方图(bar)比较Sphere与Link在不同阈值下的Precision和Recall（每个密度单独保存一张图）

    - 对每个阈值，计算Link（OBB）在所有QuantBits下的平均Precision/Recall
    - 对每个阈值，计算Sphere在所有QuantBits（且RadiusBits==1）下的平均Precision/Recall
    - 使用分组柱状图显示两种方法的平均值，便于直观比较
    """
    # 读取数据时增加错误处理，防止文件不存在导致崩溃
    try:
        obb_data = pd.read_csv(
            "../result_files/coord_hashing_cost_results.csv", header=0
        )
    except Exception as e:
        print(f"⚠️  无法读取 coord_hashing_cost_results.csv: {e}")
        return

    try:
        sphere_data = pd.read_csv(
            "../result_files/sphere_hashing_cost_results.csv",
            header=0,
        )
    except Exception as e:
        print(f"⚠️  无法读取 sphere_hashing_cost_results.csv: {e}")
        # 继续，但 sphere_data 设为空 DataFrame，后续会处理
        sphere_data = pd.DataFrame()

    densities = ["dens3", "dens6", "dens9", "dens12"]
    density_labels = {
        "dens3": "Density 3",
        "dens6": "Density 6",
        "dens9": "Density 9",
        "dens12": "Density 12",
    }

    for density in densities:
        obb_density = obb_data[obb_data["Density"] == density]
        # 直接选择 RadiusBits == 0 的 sphere 数据（如无匹配则为空）
        if not sphere_data.empty:
            sphere_density = sphere_data[
                (sphere_data["Density"] == density) & (sphere_data["RadiusBits"] == 0)
            ]
        else:
            sphere_density = pd.DataFrame()

        if obb_density.empty and sphere_density.empty:
            print(f"⚠️  {density}: No data available for plotting")
            continue

        # 合并阈值列表（取并集并排序）
        thresholds = sorted(
            set(
                obb_density["Threshold"].unique().tolist()
                if not obb_density.empty
                else []
            )
            | set(
                sphere_density["Threshold"].unique().tolist()
                if not sphere_density.empty
                else []
            )
        )
        if len(thresholds) == 0:
            print(f"⚠️  {density}: Empty thresholds, skipping")
            continue

        x = np.arange(len(thresholds))
        # 将常见阈值转换为分数标签，未命中的使用小数字格式
        frac_map = {
            0.0: "S=0",
            0.03125: "S=1/32",
            0.125: "S=1/8",
            0.25: "S=1/4",
            0.5: "S=1/2",
            1.0: "S=1",
            2.0: "S=2",
        }
        threshold_labels = [frac_map.get(round(t, 5), f"S={t:.3f}") for t in thresholds]

        # 计算每个阈值上的平均Precision/Recall
        obb_prec = []
        obb_rec = []
        sphere_prec = []
        sphere_rec = []
        for t in thresholds:
            obb_rows = obb_density[obb_density["Threshold"] == t]
            sphere_rows = sphere_density[sphere_density["Threshold"] == t]

            obb_prec.append(
                obb_rows["PosePrecision"].mean() if not obb_rows.empty else np.nan
            )
            obb_rec.append(
                obb_rows["PoseRecall"].mean() if not obb_rows.empty else np.nan
            )

            sphere_prec.append(
                sphere_rows["PosePrecision"].mean() if not sphere_rows.empty else np.nan
            )
            sphere_rec.append(
                sphere_rows["PoseRecall"].mean() if not sphere_rows.empty else np.nan
            )

        # 绘图：分组柱状图（假定源文件数值完整，无 NaN）
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        plt.rc("font", **font)
        fig.patch.set_facecolor("white")
        for ax in axes:
            ax.set_facecolor("white")

        width = 0.35

        # 子图1: Precision（grouped bar）
        ax0 = axes[0]
        bars_link = ax0.bar(
            x - width / 2, obb_prec, width, label="Link-based (avg)", color=LINK_COLOR
        )
        bars_sphere = ax0.bar(
            x + width / 2,
            sphere_prec,
            width,
            label="Sphere-based (avg)",
            color=SPHERE_COLOR,
        )

        ax0.set_title(
            f"{density_labels[density]} - Precision", fontsize=FONT_SIZE_TITLE
        )
        ax0.set_xticks(x)
        ax0.set_xticklabels(
            threshold_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK
        )
        ax0.set_ylabel("Precision (%)", fontsize=FONT_SIZE_LABEL)

        # y 上限基于最大值
        combined_prec = [v for v in (obb_prec + sphere_prec) if not np.isnan(v)]
        max_prec = max(combined_prec) if combined_prec else 100
        ax0.set_ylim([0, 100])
        ax0.grid(axis="y", alpha=0.2, linestyle="--")
        ax0.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

        # 子图2: Recall（grouped bar）
        ax1 = axes[1]
        bars_link_r = ax1.bar(
            x - width / 2, obb_rec, width, label="Link-based (avg)", color=LINK_COLOR
        )
        bars_sphere_r = ax1.bar(
            x + width / 2,
            sphere_rec,
            width,
            label="Sphere-based (avg)",
            color=SPHERE_COLOR,
        )

        ax1.set_title(f"{density_labels[density]} - Recall", fontsize=FONT_SIZE_TITLE)
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            threshold_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK
        )
        ax1.set_ylabel("Recall (%)", fontsize=FONT_SIZE_LABEL)
        ax1.set_ylim([0, 100])
        ax1.grid(axis="y", alpha=0.2, linestyle="--")
        ax1.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

        # 共享图例（仅在 dens3 显示）
        if density == "dens3":
            handles, labels = ax0.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=2,
                frameon=False,
                fontsize=FONT_SIZE_LEGEND,
            )

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(f"figs/fig_precision_recall_{density}_hist.eps", format="eps")
        print(f"✅ 图已保存: figs/fig_precision_recall_{density}_hist.eps")
        plt.close()


def plot_link_sphere_comparison_by_density():
    """
    分别生成各个密度场景下，基于sphere的预测准确率/召回率/Computation/召回率/Computation
    使用分组直方图，横坐标为S(threshold)，每组依次为：link-准确率、link-召回率、link-Computation sphere-准确率、sphere-召回率、sphere-Computation
    """

    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    sphere_data = pd.read_csv(
        "../result_files/sphere_hashing_cost_results.csv", header=0
    )

    densities = ["dens3", "dens6", "dens9", "dens12"]
    density_labels = {
        "dens3": "Density 3",
        "dens6": "Density 6",
        "dens9": "Density 9",
        "dens12": "Density 12",
    }

    for density in densities:
        obb_density = obb_data[obb_data["Density"] == density]
        sphere_density = sphere_data[
            (sphere_data["Density"] == density) & (sphere_data["RadiusBits"] == 0)
        ]

        # 仅使用预定义的阈值集合（忽略 S=4 等非目标值）
        allowed_thresholds = [0.0, 0.03125, 0.125, 0.5, 1.0, 2.0]
        present = set(obb_density["Threshold"]).union(set(sphere_density["Threshold"]))
        thresholds = [t for t in allowed_thresholds if t in present]
        if len(thresholds) == 0:
            print(
                f"⚠️  {density}: No matching thresholds from allowed set {allowed_thresholds} found."
            )
            continue

        # 构造x轴标签
        frac_map = {
            0.0: "S=0",
            0.03125: "S=1/32",
            0.125: "S=1/8",
            0.25: "S=1/4",
            0.5: "S=1/2",
            1.0: "S=1",
            2.0: "S=2",
        }
        threshold_labels = [frac_map.get(round(t, 5), f"S={t:.3f}") for t in thresholds]

        # 计算每个阈值下的指标
        link_prec, link_rec, link_speed = [], [], []
        sphere_prec, sphere_rec, sphere_speed = [], [], []
        for t in thresholds:
            obb_rows = obb_density[obb_density["Threshold"] == t]
            sphere_rows = sphere_density[sphere_density["Threshold"] == t]
            link_prec.append(
                obb_rows["PosePrecision"].mean() if not obb_rows.empty else np.nan
            )
            link_rec.append(
                obb_rows["PoseRecall"].mean() if not obb_rows.empty else np.nan
            )
            link_speed.append(
                obb_rows["SpeedUp_Pct"].mean() if not obb_rows.empty else np.nan
            )
            sphere_prec.append(
                sphere_rows["PosePrecision"].mean() if not sphere_rows.empty else np.nan
            )
            sphere_rec.append(
                sphere_rows["PoseRecall"].mean() if not sphere_rows.empty else np.nan
            )
            sphere_speed.append(
                sphere_rows["SpeedUp_Pct"].mean() if not sphere_rows.empty else np.nan
            )

        # 定义指标颜色方案 (Colorblind-friendly)
        # Precision: Blue, Recall: Vermilion (Orange-Red), SpeedUp: Bluish Green
        METRIC_COLORS = ["#0072B2", "#D55E00", "#009E73"]

        # 画图
        n = len(thresholds)
        ind = np.arange(n)
        width = 0.12
        # Use a slightly taller figure to accommodate the legend without squashing the plot
        fig, ax = plt.subplots(figsize=(2.5 * n + 4, 5.5))
        plt.rc("font", **font)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # 按新顺序绘制柱状：
        # Link-Precision, Sphere-Precision, Link-Recall, Sphere-Recall, Link-Computation, Sphere-Computation
        bars1 = ax.bar(
            ind - 2.5 * width,
            link_prec,
            width,
            label="Link-Precision",
            facecolor="white",
            edgecolor=METRIC_COLORS[0],
            hatch="///",
            linewidth=1.5,
        )
        bars4 = ax.bar(
            ind - 1.5 * width,
            sphere_prec,
            width,
            label="Sphere-Precision",
            color=METRIC_COLORS[0],
        )
        bars2 = ax.bar(
            ind - 0.5 * width,
            link_rec,
            width,
            label="Link-Recall",
            facecolor="white",
            edgecolor=METRIC_COLORS[1],
            hatch="///",
            linewidth=1.5,
        )
        bars5 = ax.bar(
            ind + 0.5 * width,
            sphere_rec,
            width,
            label="Sphere-Recall",
            color=METRIC_COLORS[1],
        )
        bars3 = ax.bar(
            ind + 1.5 * width,
            link_speed,
            width,
            label="Link-Computation",
            facecolor="white",
            edgecolor=METRIC_COLORS[2],
            hatch="///",
            linewidth=1.5,
        )
        bars6 = ax.bar(
            ind + 2.5 * width,
            sphere_speed,
            width,
            label="Sphere-Computation",
            color=METRIC_COLORS[2],
        )

        # 注意：由于柱子按所需展示顺序创建，后续取图例时可直接使用当前 handles 顺序。

        ax.set_xticks(ind)
        ax.set_xticklabels(
            threshold_labels, rotation=0, ha="center", fontsize=FONT_SIZE_TICK
        )
        # 取消纵轴label
        # ax.set_ylabel("Metric Value (%)", fontsize=FONT_SIZE_LABEL)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
        # 纵轴加%符号
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")

        # Use a consistent layout for all densities to keep the plot area size consistent
        plt.tight_layout(rect=(0, 0.03, 1, 0.82))

        # 只在dens6显示图例
        if density == "dens6":
            handles, labels = ax.get_legend_handles_labels()
            # Reorder for 2 rows x 3 columns, filling column-major
            # Row 1: Link-P, Link-R, Link-C
            # Row 2: Sphere-P, Sphere-R, Sphere-C
            order = [0, 3, 1, 4, 2, 5]
            handles = [handles[i] for i in order]
            labels = [labels[i] for i in order]

            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=3,
                frameon=False,
                fontsize=FONT_SIZE_LEGEND,
            )

        # for bars in [bars1, bars2, bars3, bars4, bars5, bars6]:
        #     add_bar_labels(ax, bars, fontsize=FONT_SIZE_TEXT)
        outdir = "figs"
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f"fig_link_sphere_metrics_{density}.png")
        outpath_eps = outpath.replace(".png", ".eps")
        plt.savefig(outpath_eps, format="eps")
        print(f"✅ 图已保存: {outpath_eps}")
        plt.close()


def plot_metrics_at_fixed_S_U():
    """
    在每个密度下，对固定的 (S, U) 配置进行 Link / Sphere 三指标对比。

    固定组合 (Threshold S, SampleRate U):
      - S=0,   U=0.0
      - S=0.5, U=1.0
      - S=1.0, U=0.5
      - S=2.0, U=0.25
      - S=4.0, U=0.125

    结果以每个密度生成一张 1x3 子图（Precision / Recall / SpeedUp_Pct），每个阈值处展示 Link 与 Sphere 的并列柱状图。
    """
    try:
        obb_data = pd.read_csv(
            "../result_files/coord_hashing_cost_results.csv", header=0
        )
        sphere_data = pd.read_csv(
            "../result_files/sphere_hashing_cost_results.csv", header=0
        )
    except Exception as e:
        print(f"⚠️  无法读取数据文件: {e}")
        return

    densities = ["dens3", "dens6", "dens9", "dens12"]
    density_labels = {
        "dens3": "Density 3",
        "dens6": "Density 6",
        "dens9": "Density 9",
        "dens12": "Density 12",
    }

    # 固定的 (S, U) 列表
    fixed_pairs = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.5), (2.0, 0.25), (4.0, 0.125)]
    frac_map = {0.0: "S=0", 0.5: "S=1/2", 1.0: "S=1", 2.0: "S=2", 4.0: "S=4"}

    for density in densities:
        # 为稳定对比限制到 QuantBits=4 与 RadiusBits==0 的 Sphere
        obb_subset = obb_data[
            (obb_data["Density"] == density) & (obb_data["QuantBits"] == 4)
        ]
        sphere_subset = sphere_data[
            (sphere_data["Density"] == density)
            & (sphere_data["QuantBits"] == 4)
            & (sphere_data["RadiusBits"] == 0)
        ]

        if obb_subset.empty and sphere_subset.empty:
            print(f"⚠️  {density}: No data available for fixed S/U comparison")
            continue

        S_vals = [s for s, u in fixed_pairs]
        labels = [frac_map.get(s, f"S={s}") for s in S_vals]
        u_frac_map = {0.0: "0", 0.125: "1/8", 0.25: "1/4", 0.5: "1/2", 1.0: "1"}
        xu_labels = [
            f"{lbl}\nU={u_frac_map.get(round(u, 5), f'{u:.3f}')}"
            for lbl, (s, u) in zip(labels, fixed_pairs)
        ]

        # 收集指标值
        link_prec, link_rec, link_speed = [], [], []
        sphere_prec, sphere_rec, sphere_speed = [], [], []

        for s, u in fixed_pairs:
            obb_rows = obb_subset[
                np.isclose(obb_subset["Threshold"], s)
                & np.isclose(obb_subset["SampleRate"], u)
            ]
            sphere_rows = sphere_subset[
                np.isclose(sphere_subset["Threshold"], s)
                & np.isclose(sphere_subset["SampleRate"], u)
            ]

            link_prec.append(
                obb_rows["PosePrecision"].mean() if not obb_rows.empty else np.nan
            )
            link_rec.append(
                obb_rows["PoseRecall"].mean() if not obb_rows.empty else np.nan
            )
            link_speed.append(
                obb_rows["SpeedUp_Pct"].mean() if not obb_rows.empty else np.nan
            )

            sphere_prec.append(
                sphere_rows["PosePrecision"].mean() if not sphere_rows.empty else np.nan
            )
            sphere_rec.append(
                sphere_rows["PoseRecall"].mean() if not sphere_rows.empty else np.nan
            )
            sphere_speed.append(
                sphere_rows["SpeedUp_Pct"].mean() if not sphere_rows.empty else np.nan
            )

        # 横向三子图：precision/recall/computation
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        plt.rc("font", **font)
        fig.patch.set_facecolor("white")
        for ax in axes:
            ax.set_facecolor("white")

        bar_width = 0.35
        metrics = [
            (link_prec, sphere_prec, "Precision (%)", "Precision"),
            (link_rec, sphere_rec, "Recall (%)", "Recall"),
            (link_speed, sphere_speed, "Computation (%)", "Computation"),
        ]
        for idx, (link_vals, sphere_vals, ylabel, title) in enumerate(metrics):
            ax = axes[idx]
            x = np.arange(len(S_vals))
            bars1 = ax.bar(
                x - bar_width / 2,
                link_vals,
                bar_width,
                label="Link-based",
                color=LINK_COLOR,
            )
            bars2 = ax.bar(
                x + bar_width / 2,
                sphere_vals,
                bar_width,
                label="Sphere-based",
                color=SPHERE_COLOR,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                xu_labels, rotation=0, ha="center", fontsize=FONT_SIZE_TICK-2
            )
            if idx == 0:
                ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
            ax.set_title(title, fontsize=FONT_SIZE_TITLE)
            ax.grid(axis="y", alpha=0.2, linestyle="--")
            ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
            # 纵轴加%符号
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")
            ax.set_ylim((0, 100))
            # 添加数值标签（统一调用工具函数）
            # add_bar_labels(ax, bars1, fontsize=FONT_SIZE_TEXT - 2)
            # add_bar_labels(ax, bars2, fontsize=FONT_SIZE_TEXT - 2)

        # 只在第一个子图显示图例
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False,
            fontsize=FONT_SIZE_LEGEND,
        )
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        outdir = "figs"
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f"fig_fixed_S_U_{density}.png")
        outpath_eps = outpath.replace(".png", ".eps")
        plt.savefig(outpath_eps, format="eps")
        plt.savefig(outpath, format="png")
        print(f"✅ 图已保存: {outpath_eps}")
        plt.close()


def plot_update_frequency_impact():
    """
    图7: 不同SampleRate下准确率、召回率和加速比的对比 (Density 6, QuantBits 4)
    """
    try:
        obb_data = pd.read_csv(
            "../result_files/coord_hashing_cost_results.csv", header=0
        )
        sphere_data = pd.read_csv(
            "../result_files/sphere_hashing_cost_results.csv",
            header=0,
        )
    except Exception as e:
        print(f"⚠️  无法读取数据文件: {e}")
        return

    density = "dens6"
    qb = 4

    # 筛选数据
    obb_subset = obb_data[
        (obb_data["Density"] == density) & (obb_data["QuantBits"] == qb)
    ]
    sphere_subset = sphere_data[
        (sphere_data["Density"] == density)
        & (sphere_data["QuantBits"] == qb)
        & (sphere_data["RadiusBits"] == 0)
    ]

    if obb_subset.empty or sphere_subset.empty:
        print(f"⚠️  {density} Q={qb}: No data available for SampleRate plot")
        return

    # 按SampleRate分组求均值
    obb_grouped = obb_subset.groupby("SampleRate")[
        ["PosePrecision", "PoseRecall", "SpeedUp_Pct"]
    ].mean()
    sphere_grouped = sphere_subset.groupby("SampleRate")[
        ["PosePrecision", "PoseRecall", "SpeedUp_Pct"]
    ].mean()

    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.rc("font", **font)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    x_obb = obb_grouped.index.values
    x_sphere = sphere_grouped.index.values

    # 左轴: Precision / Recall
    ax1.set_xlabel("Sample Rate", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Precision / Recall (%)", fontsize=FONT_SIZE_LABEL)

    # Link Lines
    l1 = ax1.plot(
        x_obb,
        obb_grouped["PosePrecision"],
        "o-",
        color=METRIC_COLORS[0],
        label="Link-Precision",
        linewidth=2,
    )
    l2 = ax1.plot(
        x_obb,
        obb_grouped["PoseRecall"],
        "o--",
        color=METRIC_COLORS[1],
        label="Link-Recall",
        linewidth=2,
    )

    # Sphere Lines
    l3 = ax1.plot(
        x_sphere,
        sphere_grouped["PosePrecision"],
        "s-",
        color=METRIC_COLORS[0],
        label="Sphere-Precision",
        linewidth=2,
        markerfacecolor="white",
    )
    l4 = ax1.plot(
        x_sphere,
        sphere_grouped["PoseRecall"],
        "s--",
        color=METRIC_COLORS[1],
        label="Sphere-Recall",
        linewidth=2,
        markerfacecolor="white",
    )

    ax1.set_ylim((0, 105))
    ax1.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax1.tick_params(axis="x", labelsize=FONT_SIZE_TICK)

    # 右轴: SpeedUp
    ax2 = ax1.twinx()
    ax2.set_ylabel("SpeedUp (%)", fontsize=FONT_SIZE_LABEL, color=METRIC_COLORS[2])

    l5 = ax2.plot(
        x_obb,
        obb_grouped["SpeedUp_Pct"],
        "^-",
        color=METRIC_COLORS[2],
        label="Link-SpeedUp",
        linewidth=2,
    )
    l6 = ax2.plot(
        x_sphere,
        sphere_grouped["SpeedUp_Pct"],
        "^--",
        color=METRIC_COLORS[2],
        label="Sphere-SpeedUp",
        linewidth=2,
        markerfacecolor="white",
    )

    ax2.tick_params(axis="y", labelcolor=METRIC_COLORS[2], labelsize=FONT_SIZE_TICK)
    # ax2.set_ylim([0, max(obb_grouped["SpeedUp_Pct"].max(), sphere_grouped["SpeedUp_Pct"].max()) * 1.1])

    # 合并图例
    lines = l1 + l2 + l3 + l4 + l5 + l6
    labels = [l.get_label() for l in lines]
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND - 4,
    )

    plt.title(
        f"Impact of Sample Rate ({density}, Q={qb})", fontsize=FONT_SIZE_TITLE, y=1.02
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.92))

    plt.savefig("figs/fig_sample_rate_impact.eps", format="eps")
    print("✅ 图7 (SampleRate Impact) 已保存: figs/fig_sample_rate_impact.eps")
    plt.close()


# 通用函数：可绘制precision/recall/computation三种指标
def plot_metric_vs_S_multi_density(
    metric="PosePrecision",
    ylabel="Precision",
    filename="fig_precision_vs_S_multi_density.png",
    add_legend=False,
):
    """
    横向排列三个子图，分别为dens6/dens9/dens12下link和sphere的metric在各S参数下的对比。
    metric: "PosePrecision"/"PoseRecall"/"SpeedUp_Pct"
    ylabel: y轴标签
    filename: 输出文件名
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    sphere_data = pd.read_csv(
        "../result_files/sphere_hashing_cost_results.csv", header=0
    )

    densities = ["dens3", "dens6", "dens9"]
    density_labels = {"dens3": "Density 3", "dens6": "Density 6", "dens9": "Density 9"}
    quant_bits = 4
    radius_bits = 0

    # 仅保留S=0~S=2的阈值
    allowed_thresholds = [0.0, 0.03125, 0.125, 0.25, 0.5, 1.0, 2.0]
    frac_map = {
        0.0: "S=0",
        0.03125: "S=1/32",
        0.125: "S=1/8",
        0.25: "S=1/4",
        0.5: "S=1/2",
        1.0: "S=1",
        2.0: "S=2",
    }
    # 检查哪些S在数据中存在
    present_thresholds = set()
    for density in densities:
        present_thresholds.update(
            obb_data[(obb_data["Density"] == density) & (obb_data["QuantBits"] == quant_bits)]["Threshold"].unique()
        )
        present_thresholds.update(
            sphere_data[(sphere_data["Density"] == density) & (sphere_data["QuantBits"] == quant_bits) & (sphere_data["RadiusBits"] == radius_bits)]["Threshold"].unique()
        )
    # 只保留allowed且实际存在的S
    thresholds = [s for s in allowed_thresholds if any(np.isclose(s, t) for t in present_thresholds)]
    threshold_labels = [frac_map.get(round(t, 5), f"S={t:.3f}") for t in thresholds]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    plt.rc("font", **font)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    width = 0.35
    for idx, density in enumerate(densities):
        ax = axes[idx]
        # link数据
        obb_vals = []
        for t in thresholds:
            rows = obb_data[
                (obb_data["Density"] == density)
                & (obb_data["QuantBits"] == quant_bits)
                & (np.isclose(obb_data["Threshold"], t))
            ]
            obb_vals.append(rows[metric].mean() if not rows.empty else np.nan)
        # sphere数据
        sphere_vals = []
        for t in thresholds:
            rows = sphere_data[
                (sphere_data["Density"] == density)
                & (sphere_data["QuantBits"] == quant_bits)
                & (sphere_data["RadiusBits"] == radius_bits)
                & (np.isclose(sphere_data["Threshold"], t))
            ]
            sphere_vals.append(rows[metric].mean() if not rows.empty else np.nan)

        x = np.arange(len(thresholds))
        bars1 = ax.bar(
            x - width / 2, obb_vals, width, label="Link-based", color=LINK_COLOR
        )
        bars2 = ax.bar(
            x + width / 2, sphere_vals, width, label="Sphere-based", color=SPHERE_COLOR
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            threshold_labels, rotation=0, ha="center", fontsize=FONT_SIZE_TICK-2
        )
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
        ax.set_title(density_labels[density], fontsize=FONT_SIZE_TITLE)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK + 2)
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")
        ax.set_ylim([0, 100])
        # 添加数值标签
        # add_bar_labels(ax, bars1, fontsize=FONT_SIZE_TEXT - 2)
        # add_bar_labels(ax, bars2, fontsize=FONT_SIZE_TEXT - 2)
    if add_legend:
      handles, labels = axes[0].get_legend_handles_labels()
      fig.legend(
          handles,
          labels,
          loc="upper center",
          bbox_to_anchor=(0.5, 1.04),
          ncol=2,
          frameon=False,
          fontsize=FONT_SIZE_LEGEND,
      )
    plt.tight_layout(rect=(0, 0.03, 1, 0.98))
    outdir = "figs"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, filename)
    outpath_eps = outpath.replace(".png", ".eps")
    plt.savefig(outpath_eps, format="eps")
    plt.savefig(outpath, format="png")
    print(f"✅ {ylabel} 对比图已保存: {outpath_eps}")


# 便捷函数
def plot_precision_vs_S_multi_density():
    plot_metric_vs_S_multi_density(
        metric="PosePrecision",
        ylabel="Precision (%)",
        filename="fig_precision_vs_S_multi_density.png",
        add_legend=True,
    )


def plot_recall_vs_S_multi_density():
    plot_metric_vs_S_multi_density(
        metric="PoseRecall",
        ylabel="Recall (%)",
        filename="fig_recall_vs_S_multi_density.png",
    )


def plot_computation_vs_S_multi_density():
    plot_metric_vs_S_multi_density(
        metric="SpeedUp_Pct",
        ylabel="Computation (%)",
        filename="fig_computation_vs_S_multi_density.png",
    )


def main():
    """主函数"""

    # 确保figs目录存在
    os.makedirs("figs", exist_ok=True)

    print("=" * 70)
    print("OBB与Sphere碰撞预测性能对比绘图")
    print("=" * 70)

    # 图1: 准确率和召回率对比
    # print("\n生成图1: 不同密度下的精确率和召回率对比...")
    # plot_accuracy_recall_comparison()

    # # 图2: 计算成本对比
    # print("\n生成图2: 不同密度下的最小计算成本对比...")
    # plot_cost_comparison()

    # # 图3: 不同阈值下的性能对比
    # print("\n生成图3: 不同阈值下的性能对比...")
    # for density in ["dens3", "dens6", "dens9", "dens12"]:
    #     plot_threshold_comparison(density)

    # # 图3综合版
    # print("\n生成图3综合版: 所有密度下的阈值对比...")
    # plot_combined_threshold_comparison()

    # # 图4: P-R曲线
    # print("\n生成图4: 不同密度下的P-R曲线对比...")
    # plot_pr_curves()

    # # 图5: 成本vs阈值曲线
    # print("\n生成图5: 不同密度下成本随阈值变化的对比...")
    # plot_cost_vs_threshold()

    # # 图6: 成本vs量化位数分析
    # print("\n生成图6: Cost随QuantBits/QuantBits变化分析...")
    # plot_cost_vs_quantbits()

    # 图7: QuantBits=4下不同密度场景的Threshold对比
    # print("\n生成图7: QuantBits=4下不同Threshold的性能指标对比...")
    # plot_threshold_metrics_by_density()

    # 图8: 每个密度下Precision/Recall对比
    # print("\n生成图8: 每个密度下Sphere vs Link Precision/Recall对比...")
    # plot_precision_recall_by_density()
    # # 新增: 各密度下Link/Sphere三指标对比
    # print("\n生成图9: 各密度下Link/Sphere三指标对比...")
    # plot_link_sphere_comparison_by_density()
    # 图9: Sample Rate 影响
    print("\n生成图9: Sample Rate 影响分析...")
    plot_metrics_at_fixed_S_U()
    # # 图9: Sample Rate 影响
    # print("\n生成图9: Sample Rate 影响分析...")
    # plot_update_frequency_impact()
    plot_precision_vs_S_multi_density()
    plot_recall_vs_S_multi_density()
    plot_computation_vs_S_multi_density()

    print("\n" + "=" * 70)
    print("✅ 所有图表生成完成!")
    print("图表保存在: prediction_approaches/plot/figs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
