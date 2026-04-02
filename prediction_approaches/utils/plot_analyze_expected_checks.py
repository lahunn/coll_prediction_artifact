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
FONT_SIZE = 18

def compare_simulation_vs_formula():
    """对比蒙特卡洛模拟和本研究提出的公式的结果"""
    print("对比蒙特卡洛模拟 vs 本研究提出的公式...")

    plt.style.use("default")
    # 重新应用配置（style.use 可能会重置设置）
    plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 16

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
            # 本研究提出的公式 - 使用深蓝
            ax.plot(
                valid_params,
                formula_results,
                "o-",
                linewidth=2.5,
                color=colors[0],
                markersize=8,
                label="本研究提出的公式",
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

            ax.set_xlabel(config["xlabel"], fontsize=FONT_SIZE)
            if idx % 2 == 0:
                ax.set_ylabel("期望检测次数", fontsize=FONT_SIZE)

            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

            ax.text(
                0.1, 0.95, config["label"], transform=ax.transAxes,
                fontsize=FONT_SIZE, fontweight="bold", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none")
            )

            if idx == 0:
                ax.legend(loc="best", fontsize=FONT_SIZE)

            ax.grid(True, alpha=0.2, linestyle="--", color="gray")
            ax.set_ylim([0, 4.5])
            ax.set_facecolor("white")

    error_text = "平均误差: " + ", ".join([f"{e['config']}: {e['avg']:.2f}%" for e in all_errors])
    fig.text(0.5, 0.02, error_text, ha="center", fontsize=FONT_SIZE, style="italic")

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
