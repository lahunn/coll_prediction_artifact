#!/usr/bin/env python3
"""
分析 calculate_expected_checks 函数的结果 S 关于各参数的变化趋势
"""

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import matplotlib
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    calculate_expected_checks,
    calculate_baseline_expectation,
    find_sim_cost,
)

# --- 统一绘图风格配置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")


def analyze_vs_P():
    """分析 S 关于真实碰撞率 P 的变化"""
    print("分析 S vs P...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "期望检测次数 S 与碰撞率 P 的关系", fontsize=16, fontweight="bold"
    )

    P_values = np.linspace(0.01, 0.99, 100)

    configs = [
        {"C": 0.8, "A": 0.8, "N": 100, "label": "C=0.8, A=0.8, N=100"},
        {"C": 0.9, "A": 0.9, "N": 100, "label": "C=0.9, A=0.9, N=100"},
        {"C": 0.6, "A": 0.8, "N": 100, "label": "C=0.6, A=0.8, N=100"},
        {"C": 0.8, "A": 0.6, "N": 100, "label": "C=0.8, A=0.6, N=100"},
    ]

    for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
        S_values = []
        baseline_values = []
        for P in P_values:
            try:
                S = calculate_expected_checks(P, config["C"], config["A"], config["N"])
                S_values.append(S)
                baseline = calculate_baseline_expectation(config["N"], P)
                baseline_values.append(baseline)
            except ValueError:
                S_values.append(np.nan)
                baseline_values.append(np.nan)

        ax.plot(P_values, S_values, linewidth=2, label="使用预测器", color=colors[0])
        ax.plot(
            P_values,
            baseline_values,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="基准 (无预测器)",
            color=colors[1],
        )
        ax.set_xlabel("碰撞率 P", fontsize=11)
        ax.set_ylabel("期望检测次数 S", fontsize=11)
        ax.set_title(config["label"], fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 调整纵轴范围
        max_val = max([v for v in baseline_values if not np.isnan(v)])
        ax.set_ylim([0, max_val * 1.1])

    plt.tight_layout()
    plt.savefig("analysis_S_vs_R.eps", dpi=300, bbox_inches="tight")
    print("保存: analysis_S_vs_R.eps")
    plt.close()


def analyze_vs_C():
    """分析 S 关于覆盖率 C 的变化"""
    print("分析 S vs C...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("期望检测次数 S 与覆盖率 C 的关系", fontsize=16, fontweight="bold")

    C_values = np.linspace(0.1, 1.0, 24)

    configs = [
        {"P": 0.3, "A": 0.8, "N": 24, "label": "P=0.3, A=0.8, N=24"},
        {"P": 0.5, "A": 0.8, "N": 24, "label": "P=0.5, A=0.8, N=24"},
        {"P": 0.7, "A": 0.8, "N": 24, "label": "P=0.7, A=0.8, N=24"},
        {"P": 0.5, "A": 0.6, "N": 24, "label": "P=0.5, A=0.6, N=24"},
    ]

    for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
        S_values = []
        valid_C = []
        baseline = calculate_baseline_expectation(config["N"], config["P"])

        for C in C_values:
            try:
                S = calculate_expected_checks(config["P"], C, config["A"], config["N"])
                S_values.append(S)
                valid_C.append(C)
            except ValueError:
                pass

        ax.plot(valid_C, S_values, linewidth=2, label="使用预测器", color=colors[0])
        ax.axhline(
            y=baseline,
            color=colors[1],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"基准={baseline:.1f}",
        )
        ax.set_xlabel("覆盖率 C (召回率)", fontsize=11)
        ax.set_ylabel("期望检测次数 S", fontsize=11)
        ax.set_title(config["label"], fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if S_values:
            max_val = max(baseline, max(S_values))
            ax.set_ylim([0, max_val * 1.1])

    plt.tight_layout()
    plt.savefig("analysis_S_vs_C.eps", dpi=300, bbox_inches="tight")
    print("保存: analysis_S_vs_C.eps")
    plt.close()


def analyze_vs_A():
    """分析 S 关于准确率 A 的变化"""
    print("分析 S vs A...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("期望检测次数 S 与准确率 A 的关系", fontsize=16, fontweight="bold")

    A_values = np.linspace(0.1, 1.0, 100)

    configs = [
        {"P": 0.3, "C": 0.8, "N": 100, "label": "P=0.3, C=0.8, N=100"},
        {"P": 0.5, "C": 0.8, "N": 100, "label": "P=0.5, C=0.8, N=100"},
        {"P": 0.7, "C": 0.8, "N": 100, "label": "P=0.7, C=0.8, N=100"},
        {"P": 0.5, "C": 0.6, "N": 100, "label": "P=0.5, C=0.6, N=100"},
    ]

    for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
        S_values = []
        valid_A = []
        baseline = calculate_baseline_expectation(config["N"], config["P"])

        for A in A_values:
            try:
                S = calculate_expected_checks(config["P"], config["C"], A, config["N"])
                S_values.append(S)
                valid_A.append(A)
            except ValueError:
                pass

        ax.plot(valid_A, S_values, linewidth=2, label="使用预测器", color=colors[0])
        ax.axhline(
            y=baseline,
            color=colors[1],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"基准={baseline:.1f}",
        )
        ax.set_xlabel("准确率 A (精确率)", fontsize=11)
        ax.set_ylabel("期望检测次数 S", fontsize=11)
        ax.set_title(config["label"], fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if S_values:
            max_val = max(baseline, max(S_values))
            ax.set_ylim([0, max_val * 1.1])

    plt.tight_layout()
    plt.savefig("analysis_S_vs_A.eps", dpi=300, bbox_inches="tight")
    print("保存: analysis_S_vs_A.eps")
    plt.close()


def analyze_vs_N():
    """分析 S/N 关于任务总数 N 的变化"""
    print("分析 S/N vs N...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "归一化检测次数 S/N 与总任务数 N 的关系", fontsize=16, fontweight="bold"
    )

    N_values = np.arange(10, 501, 10)

    configs = [
        {"P": 0.3, "C": 0.8, "A": 0.8, "label": "P=0.3, C=0.8, A=0.8"},
        {"P": 0.5, "C": 0.8, "A": 0.8, "label": "P=0.5, C=0.8, A=0.8"},
        {"P": 0.7, "C": 0.8, "A": 0.8, "label": "P=0.7, C=0.8, A=0.8"},
    ]

    for idx, config in enumerate(configs):
        S_over_N = []
        baseline_over_N = []

        for N in N_values:
            try:
                S = calculate_expected_checks(config["P"], config["C"], config["A"], N)
                baseline = calculate_baseline_expectation(N, config["P"])
                S_over_N.append(S / N)
                baseline_over_N.append(baseline / N)
            except ValueError:
                S_over_N.append(np.nan)
                baseline_over_N.append(np.nan)

        ax = axes[idx]
        ax.plot(N_values, S_over_N, linewidth=2, label="使用预测器", color=colors[0])
        ax.plot(
            N_values,
            baseline_over_N,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="基准",
            color=colors[1],
        )
        ax.set_xlabel("总任务数 N", fontsize=11)
        ax.set_ylabel("S/N (归一化检测次数)", fontsize=11)
        ax.set_title(config["label"], fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        valid_baseline = [v for v in baseline_over_N if not np.isnan(v)]
        valid_S = [v for v in S_over_N if not np.isnan(v)]
        if valid_baseline and valid_S:
            max_val = max(max(valid_baseline), max(valid_S))
            ax.set_ylim([0, max_val * 1.1])

    plt.tight_layout()
    plt.savefig("analysis_S_over_N_vs_N.eps", dpi=300, bbox_inches="tight")
    print("保存: analysis_S_over_N_vs_N.eps")
    plt.close()


def analyze_S_vs_N():
    """分析 S 关于任务总数 N 的变化"""
    print("分析 S vs N...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("期望检测次数 S 与总任务数 N 的关系", fontsize=16, fontweight="bold")

    N_values = np.arange(10, 501, 10)

    configs = [
        {"P": 0.3, "C": 0.8, "A": 0.8, "label": "P=0.3, C=0.8, A=0.8"},
        {"P": 0.5, "C": 0.8, "A": 0.8, "label": "P=0.5, C=0.8, A=0.8"},
        {"P": 0.7, "C": 0.8, "A": 0.8, "label": "P=0.7, C=0.8, A=0.8"},
    ]

    for idx, config in enumerate(configs):
        S_values = []
        baseline_values = []

        for N in N_values:
            try:
                S = calculate_expected_checks(config["P"], config["C"], config["A"], N)
                S_values.append(S)
                baseline = calculate_baseline_expectation(N, config["P"])
                baseline_values.append(baseline)
            except ValueError:
                S_values.append(np.nan)
                baseline_values.append(np.nan)

        ax = axes[idx]
        ax.plot(N_values, S_values, linewidth=2, label="使用预测器", color=colors[0])
        ax.plot(
            N_values,
            baseline_values,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="基准",
            color=colors[1],
        )
        ax.set_xlabel("总任务数 N", fontsize=11)
        ax.set_ylabel("期望检测次数 S", fontsize=11)
        ax.set_title(config["label"], fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        valid_baseline = [v for v in baseline_values if not np.isnan(v)]
        valid_S = [v for v in S_values if not np.isnan(v)]
        if valid_baseline and valid_S:
            max_val = max(max(valid_baseline), max(valid_S))
            ax.set_ylim([0, max_val * 1.1])

    plt.tight_layout()
    plt.savefig("analysis_S_vs_N.eps", dpi=300, bbox_inches="tight")
    print("保存: analysis_S_vs_N.eps")
    plt.close()


def analyze_heatmap_C_A():
    """绘制 S 关于 C 和 A 的热力图"""
    print("分析 S 热力图 (C vs A)...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "热力图：期望检测次数 S (覆盖率 C vs 准确率 A)",
        fontsize=16,
        fontweight="bold",
    )

    C_values = np.linspace(0.1, 1.0, 50)
    A_values = np.linspace(0.1, 1.0, 50)

    configs = [
        {"P": 0.3, "N": 100, "title": "P=0.3, N=100"},
        {"P": 0.7, "N": 100, "title": "P=0.7, N=100"},
    ]

    for ax, config in zip(axes, configs):
        S_matrix = np.zeros((len(A_values), len(C_values)))

        for i, A in enumerate(A_values):
            for j, C in enumerate(C_values):
                try:
                    S = calculate_expected_checks(config["P"], C, A, config["N"])
                    S_matrix[i, j] = S
                except ValueError:
                    S_matrix[i, j] = np.nan

        im = ax.imshow(
            S_matrix,
            aspect="auto",
            origin="lower",
            extent=[C_values[0], C_values[-1], A_values[0], A_values[-1]],
            cmap="viridis",
        )
        ax.set_xlabel("覆盖率 C (召回率)", fontsize=12)
        ax.set_ylabel("准确率 A (精确率)", fontsize=12)
        ax.set_title(config["title"], fontsize=13)
        plt.colorbar(im, ax=ax, label="期望检测次数 S")

    plt.tight_layout()
    plt.savefig("analysis_heatmap_C_A.eps", dpi=300, bbox_inches="tight")
    print("保存: analysis_heatmap_C_A.eps")
    plt.close()


def analyze_efficiency_ratio():
    """分析预测器的效率比 (Oracle/S)"""
    print("分析效率比 (Oracle/S)...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("效率比：N/S 与参数的关系", fontsize=16, fontweight="bold")

    # N/S vs P
    ax = axes[0, 0]
    P_values = np.linspace(0.1, 0.9, 100)
    for C, A in [(0.6, 0.6), (0.8, 0.8), (0.9, 0.9)]:
        ratios = []
        for P in P_values:
            try:
                S = calculate_expected_checks(P, C, A, 100)
                ratios.append(100 / S)
            except ValueError:
                ratios.append(np.nan)
        ax.plot(P_values, ratios, linewidth=2, label=f"C={C}, A={A}")
    ax.set_xlabel("碰撞率 P", fontsize=11)
    ax.set_ylabel("效率比 (N/S)", fontsize=11)
    ax.set_title("效率 vs P", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # N/S vs C
    ax = axes[0, 1]
    C_values = np.linspace(0.1, 1.0, 100)
    for P, A in [(0.3, 0.8), (0.5, 0.8), (0.7, 0.8)]:
        ratios = []
        valid_C = []
        for C in C_values:
            try:
                S = calculate_expected_checks(P, C, A, 100)
                ratios.append(100 / S)
                valid_C.append(C)
            except ValueError:
                pass
        ax.plot(valid_C, ratios, linewidth=2, label=f"P={P}, A={A}")
    ax.set_xlabel("覆盖率 C", fontsize=11)
    ax.set_ylabel("效率比 (N/S)", fontsize=11)
    ax.set_title("效率 vs C", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # N/S vs A
    ax = axes[1, 0]
    A_values = np.linspace(0.1, 1.0, 100)
    for P, C in [(0.3, 0.8), (0.5, 0.8), (0.7, 0.8)]:
        ratios = []
        valid_A = []
        for A in A_values:
            try:
                S = calculate_expected_checks(P, C, A, 100)
                ratios.append(100 / S)
                valid_A.append(A)
            except ValueError:
                pass
        ax.plot(valid_A, ratios, linewidth=2, label=f"P={P}, C={C}")
    ax.set_xlabel("准确率 A", fontsize=11)
    ax.set_ylabel("效率比 (N/S)", fontsize=11)
    ax.set_title("效率 vs A", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # N/S vs N
    ax = axes[1, 1]
    N_values = np.arange(10, 501, 10)
    for P, C, A in [(0.3, 0.8, 0.8), (0.5, 0.8, 0.8), (0.7, 0.8, 0.8)]:
        ratios = []
        for N in N_values:
            try:
                S = calculate_expected_checks(P, C, A, N)
                ratios.append(N / S)
            except ValueError:
                ratios.append(np.nan)
        ax.plot(N_values, ratios, linewidth=2, label=f"P={P}, C={C}, A={A}")
    ax.set_xlabel("总任务数 N", fontsize=11)
    ax.set_ylabel("效率比 (N/S)", fontsize=11)
    ax.set_title("效率 vs N", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("analysis_efficiency_ratio.eps", dpi=300, bbox_inches="tight")
    print("保存: analysis_efficiency_ratio.eps")
    plt.close()


def compare_simulation_vs_formula():
    """对比蒙特卡洛模拟和精确公式的结果"""
    print("对比蒙特卡洛模拟 vs 精确公式...")

    plt.style.use("default")
    # 重新应用配置（style.use 可能会重置设置）
    plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    test_configs = [
        {
            "param": "P",
            "values": np.linspace(0.1, 0.9, 9),
            "fixed": {"C": 0.8, "A": 0.8, "N": 20},
            "xlabel": "碰撞率 (P)",
            "label": "(a)",
        },
        {
            "param": "C",
            "values": np.linspace(0.4, 1.0, 7),
            "fixed": {"P": 0.5, "A": 0.8, "N": 20},
            "xlabel": "召回率 (C)",
            "label": "(b)",
        },
        {
            "param": "A",
            "values": np.linspace(0.4, 1.0, 7),
            "fixed": {"P": 0.5, "C": 0.8, "N": 20},
            "xlabel": "精确率 (A)",
            "label": "(c)",
        },
        {
            "param": "N",
            "values": np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,150]),
            "fixed": {"P": 0.5, "C": 0.8, "A": 0.8},
            "xlabel": "总任务数 (N)",
            "label": "(d)",
        },
    ]

    all_errors = []

    for idx, (ax, config) in enumerate(zip(axes.flat, test_configs)):
        param_values = config["values"]
        simulation_results = []
        formula_results = []
        valid_params = []

        for val in param_values:
            params = config["fixed"].copy()
            params[config["param"]] = val
            if "N" in params:
                params["N"] = int(params["N"])

            try:
                if params["C"] * params["P"] > params["A"] + 1e-9:
                    continue

                formula_result = calculate_expected_checks(
                    params["P"], params["C"], params["A"], params["N"]
                )
                formula_results.append(formula_result)

                print(f"  模拟 {config['param']}={val:.2f}...", end=" ")
                simulation_result = find_sim_cost(
                    params["P"], params["C"], params["A"], params["N"]
                )
                simulation_results.append(simulation_result)
                valid_params.append(val)
                print("完成")

            except (ValueError, ZeroDivisionError):
                continue

        if valid_params:
            # 精确公式 - 使用深蓝
            ax.plot(
                valid_params,
                formula_results,
                "o-",
                linewidth=2.5,
                color=colors[0],
                markersize=8,
                label="精确公式",
                zorder=2,
            )
            # 蒙特卡洛 - 使用深红 (colors[3])
            ax.plot(
                valid_params,
                simulation_results,
                "s",
                linewidth=0,
                color=colors[3],
                markersize=10,
                markerfacecolor="none",
                markeredgewidth=2,
                label="蒙特卡洛模拟 (1万次)",
                zorder=3,
            )

            errors = [abs(s - f) / f * 100 for s, f in zip(simulation_results, formula_results)]
            avg_error = np.mean(errors)
            all_errors.append({"config": config["label"], "avg": avg_error})

            ax.set_xlabel(config["xlabel"], fontsize=14)
            if idx % 2 == 0:
                ax.set_ylabel("期望检测次数", fontsize=14)
            
            ax.tick_params(axis='both', which='major', labelsize=12)

            ax.text(
                0.1, 0.95, config["label"], transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none")
            )

            if idx == 0:
                ax.legend(loc="best", fontsize=12)

            ax.grid(True, alpha=0.2, linestyle="--", color="gray")
            ax.set_ylim([0, 4.5])
            ax.set_facecolor("white")

    error_text = "平均误差: " + ", ".join([f"{e['config']}: {e['avg']:.2f}%" for e in all_errors])
    fig.text(0.5, 0.02, error_text, ha="center", fontsize=14, style="italic")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig("analysis_simulation_vs_formula.pdf", dpi=300, bbox_inches="tight")
    print("保存: analysis_simulation_vs_formula.pdf")
    plt.close()
    sns.set_theme(style="whitegrid") # 恢复主题


def main():
    """主函数"""
    print("=" * 70)
    print("分析 calculate_expected_checks 函数")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "analysis_results")
    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)

    # analyze_vs_P()
    # analyze_vs_C()
    # analyze_vs_A()
    # analyze_vs_N()
    # analyze_S_vs_N()
    # analyze_heatmap_C_A()
    # analyze_efficiency_ratio()
    compare_simulation_vs_formula()

    print("=" * 70)
    print(f"分析完成! 所有图表已保存到 {results_dir} 目录")
    print("=" * 70)


if __name__ == "__main__":
    main()
