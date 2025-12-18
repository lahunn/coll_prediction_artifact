#!/usr/bin/env python3
"""
OBB与Sphere碰撞预测性能对比绘图脚本
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

font = {
    "family": "serif",
    "weight": "normal",
    "size": 28,
}

# ===== 全局配色方案 =====
LINK_COLOR = "#0072B2"      # Link-based 方法主色
SPHERE_COLOR = "#D55E00"    # Sphere-based 方法主色
# 供多曲线使用的渐变色（由深到浅）
LINK_COLOR_SET = ["#00558a", "#1f78b4", "#6baed6"]
SPHERE_COLOR_SET = ["#a34700", "#d55e00", "#f4a582"]
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
    统一取QuantBits=4(OBB)和CoordBits=4(Sphere)，对不同Threshold和SampleRate计算平均值
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

        # Sphere: 固定CoordBits=4，对所有Threshold和SampleRate求平均
        sphere_density = sphere_data[
            (sphere_data["Density"] == density) & (sphere_data["CoordBits"] == 4)
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
    ax_prec.set_ylim([0, 115])
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
    ax_rec.set_ylim([0, 115])
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
    ax.set_ylim([0, max_val * 1.15 if max_val > 0 else 100])
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
    为每个QuantBits/CoordBits生成独立的图表

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
        axes[0].set_title(f"Link-based (Q={qb}) - {density.upper()}", fontsize=FONT_SIZE_TITLE)
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
        axes[1].set_title(f"Link-based (Q={qb}) - {density.upper()}", fontsize=FONT_SIZE_TITLE)
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

    # 为每个CoordBits生成Sphere图表
    for cb in coord_bits:
        sphere_fixed = sphere_data[
            (sphere_data["Density"] == density)
            & (sphere_data["CoordBits"] == cb)
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
    为每个QuantBits/CoordBits绘制单独的曲线
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

        # 绘制不同CoordBits的Sphere曲线
        for cb_idx, cb in enumerate(coord_bits):
            sphere_subset = sphere_data[
                (sphere_data["Density"] == density)
                & (sphere_data["CoordBits"] == cb)
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

        # 绘制不同CoordBits的Sphere曲线
        for cb_idx, cb in enumerate(coord_bits):
            sphere_subset = sphere_data[
                (sphere_data["Density"] == density)
                & (sphere_data["CoordBits"] == cb)
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
    图6: 分析Cost随QuantBits/CoordBits的变化情况
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
    plt.savefig("figs/fig_cost_vs_quantbits_obb.png", dpi=300)
    print("✅ 图6-OBB已保存: figs/fig_cost_vs_quantbits_obb.png")
    plt.close()

    # === Sphere图表 ===
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rc("font", **font)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for idx, density in enumerate(densities):
        min_costs = []
        for cb in coord_bits:
            # 找到该密度和CoordBits下成本最小的配置(只考虑RadiusBits=1)
            subset = sphere_data[
                (sphere_data["Density"] == density)
                & (sphere_data["CoordBits"] == cb)
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

    ax.set_xlabel("CoordBits", fontsize=28)
    ax.set_ylabel("Minimum Prediction Cost", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Sphere: Cost vs CoordBits", fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(coord_bits)
    ax.set_xticklabels([str(cb) for cb in coord_bits], fontsize=FONT_SIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("figs/fig_cost_vs_coordbits_sphere.png", dpi=300)
    print("✅ 图6-Sphere已保存: figs/fig_cost_vs_coordbits_sphere.png")
    plt.close()


def plot_threshold_metrics_by_density():
    """
    绘制QuantBits=4和CoordBits=4情况下不同密度场景的性能指标
    每个密度场景生成一个图，对比OBB和Sphere在不同Threshold下的Precision、Recall和SpeedUp_Pct
    """
    obb_data = pd.read_csv("../result_files/coord_hashing_cost_results.csv", header=0)
    sphere_data = pd.read_csv("../result_files/sphere_hashing_cost_results.csv", header=0)
    
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
            (sphere_data["Density"] == density) & 
            (sphere_data["CoordBits"] == 4) &
            (sphere_data["RadiusBits"] == 1)
        ].sort_values("Threshold")
        
        if len(obb_density) == 0 or len(sphere_density) == 0:
            print(f"⚠️  {density}: Missing data for QuantBits=4 or CoordBits=4")
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
            obb_density["SpeedUp_Pct"].values
        ]
        
        thresholds_sphere = sphere_density["Threshold"].values
        metrics_sphere = [
            sphere_density["PosePrecision"].values,
            sphere_density["PoseRecall"].values,
            sphere_density["SpeedUp_Pct"].values
        ]
        
        metric_names = ["Precision (%)", "Recall (%)", "SpeedUp_Pct (%)"]
        metric_titles = ["Precision", "Recall", "SpeedUp_Pct"]
        
        # 绘制三个指标
        for i, (ax, name, title) in enumerate(zip(axes, metric_names, metric_titles)):
            x = np.arange(len(thresholds_obb))
            width = 0.35
            
            ax.bar(x - width/2, metrics_obb[i], width, label="Link-based", 
                   color=LINK_COLOR, alpha=0.8)
            ax.bar(x + width/2, metrics_sphere[i], width, label="Sphere-based", 
                   color=SPHERE_COLOR, alpha=0.8)
            
            ax.set_ylabel(name, fontsize=FONT_SIZE_LABEL)
            ax.set_title(title, fontsize=FONT_SIZE_TITLE)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{t:.3f}" for t in thresholds_obb], 
                              rotation=45, ha="right", fontsize=FONT_SIZE_TICK)
            ax.set_xlabel("Threshold Value (S)", fontsize=FONT_SIZE_LABEL)
            ax.grid(axis="y", alpha=0.2, linestyle="--")
            ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
            ax.set_ylim([0, max(np.max(metrics_obb[i]), np.max(metrics_sphere[i])) * 1.15]) # type: ignore
            
            # 添加数值标签
            for bars in [ax.containers[0], ax.containers[1]]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.1f}",
                           ha="center", va="bottom", fontsize=FONT_SIZE_TEXT - 4)
        
        # 添加共享图例
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
                  ncol=2, frameon=False, fontsize=FONT_SIZE_LEGEND)
        
        fig.suptitle(f"{density_labels[density]} - Performance Metrics Comparison", 
                    fontsize=FONT_SIZE_TITLE + 2, y=1.02)
        
        plt.tight_layout(rect=(0, 0.02, 1, 0.95))
        plt.savefig(f"figs/fig_threshold_metrics_{density}_comparison.png", dpi=300)
        print(f"✅ 图已保存: figs/fig_threshold_metrics_{density}_comparison.png")
        plt.close()


def plot_update_frequency_impact():
    """
    图7: 不同更新频率下准确率和召回率的对比
    注意: 当前数据只有update_prob=0.5的结果,这里演示如何绘制
    如需完整图表,需要运行不同update_prob参数的优化脚本
    """
    # 这里假设我们有不同更新频率的数据
    # 实际使用时需要先运行optimize_s_parameters.py和optimize_s_parameters_sphere.py
    # 使用不同的update_prob参数(如0.1, 0.3, 0.5, 0.7, 0.9)

    print("\n⚠️  注意: 图6需要运行不同update_prob参数的优化脚本")
    print("示例命令:")
    print("  python optimize_s_parameters.py 4 0.1")
    print("  python optimize_s_parameters.py 4 0.3")
    print("  python optimize_s_parameters.py 4 0.5")
    print("  python optimize_s_parameters.py 4 0.7")
    print("  python optimize_s_parameters.py 4 0.9")
    print("\n  同样对optimize_s_parameters_sphere.py执行相同操作")


def main():
    """主函数"""
    import os

    # 确保figs目录存在
    os.makedirs("figs", exist_ok=True)

    print("=" * 70)
    print("OBB与Sphere碰撞预测性能对比绘图")
    print("=" * 70)

    # 图1: 准确率和召回率对比
    print("\n生成图1: 不同密度下的精确率和召回率对比...")
    plot_accuracy_recall_comparison()

    # 图2: 计算成本对比
    print("\n生成图2: 不同密度下的最小计算成本对比...")
    plot_cost_comparison()

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
    # print("\n生成图6: Cost随QuantBits/CoordBits变化分析...")
    # plot_cost_vs_quantbits()

    # 图7: QuantBits=4下不同密度场景的Threshold对比
    print("\n生成图7: QuantBits=4下不同Threshold的性能指标对比...")
    plot_threshold_metrics_by_density()

    # 图8提示
    # plot_update_frequency_impact()

    print("\n" + "=" * 70)
    print("✅ 所有图表生成完成!")
    print("图表保存在: prediction_approaches/plot/figs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
