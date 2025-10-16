#!/usr/bin/env python3
"""
分析 calculate_expected_checks 函数的结果 S 关于各参数的变化趋势
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils import calculate_expected_checks, calculate_baseline_expectation

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')


def analyze_vs_R():
    """分析 S 关于真实碰撞率 R 的变化"""
    print("分析 S vs R...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Expected Checks S vs Collision Rate R', fontsize=16, fontweight='bold')
    
    R_values = np.linspace(0.01, 0.99, 100)
    
    configs = [
        {'C': 0.8, 'A': 0.8, 'N': 100, 'label': 'C=0.8, A=0.8, N=100'},
        {'C': 0.9, 'A': 0.9, 'N': 100, 'label': 'C=0.9, A=0.9, N=100'},
        {'C': 0.6, 'A': 0.8, 'N': 100, 'label': 'C=0.6, A=0.8, N=100'},
        {'C': 0.8, 'A': 0.6, 'N': 100, 'label': 'C=0.8, A=0.6, N=100'},
    ]
    
    for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
        S_values = []
        baseline_values = []
        for R in R_values:
            try:
                S = calculate_expected_checks(R, config['C'], config['A'], config['N'])
                S_values.append(S)
                baseline = calculate_baseline_expectation(config['N'], R)
                baseline_values.append(baseline)
            except ValueError:
                S_values.append(np.nan)
                baseline_values.append(np.nan)
        
        ax.plot(R_values, S_values, linewidth=2, label='With Predictor', color='blue')
        ax.plot(R_values, baseline_values, linestyle='--', linewidth=2, alpha=0.7, 
                label='Baseline (No Predictor)', color='orange')
        ax.set_xlabel('Collision Rate R', fontsize=11)
        ax.set_ylabel('Expected Checks S', fontsize=11)
        ax.set_title(config['label'], fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 调整纵轴范围，从0开始，上限为baseline最大值的1.1倍
        max_val = max([v for v in baseline_values if not np.isnan(v)])
        ax.set_ylim([0, max_val * 1.1])
    
    plt.tight_layout()
    plt.savefig('analysis_S_vs_R.png', dpi=300, bbox_inches='tight')
    print("保存: analysis_S_vs_R.png")
    plt.close()


def analyze_vs_C():
    """分析 S 关于覆盖率 C 的变化"""
    print("分析 S vs C...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Expected Checks S vs Coverage C', fontsize=16, fontweight='bold')
    
    C_values = np.linspace(0.1, 1.0, 24)
    
    configs = [
        {'R': 0.3, 'A': 0.8, 'N': 24, 'label': 'R=0.3, A=0.8, N=24'},
        {'R': 0.5, 'A': 0.8, 'N': 24, 'label': 'R=0.5, A=0.8, N=24'},
        {'R': 0.7, 'A': 0.8, 'N': 24, 'label': 'R=0.7, A=0.8, N=24'},
        {'R': 0.5, 'A': 0.6, 'N': 24, 'label': 'R=0.5, A=0.6, N=24'},
    ]
    
    for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
        S_values = []
        valid_C = []
        baseline = calculate_baseline_expectation(config['N'], config['R'])
        
        for C in C_values:
            try:
                S = calculate_expected_checks(config['R'], C, config['A'], config['N'])
                S_values.append(S)
                valid_C.append(C)
            except ValueError:
                pass
        
        ax.plot(valid_C, S_values, linewidth=2, label='With Predictor', color='blue')
        ax.axhline(y=baseline, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                   label=f'Baseline={baseline:.1f}')
        ax.set_xlabel('Coverage C (Recall)', fontsize=11)
        ax.set_ylabel('Expected Checks S', fontsize=11)
        ax.set_title(config['label'], fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 调整纵轴范围
        if S_values:
            max_val = max(baseline, max(S_values))
            ax.set_ylim([0, max_val * 1.1])
    
    plt.tight_layout()
    plt.savefig('analysis_S_vs_C.png', dpi=300, bbox_inches='tight')
    print("保存: analysis_S_vs_C.png")
    plt.close()


def analyze_vs_A():
    """分析 S 关于准确率 A 的变化"""
    print("分析 S vs A...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Expected Checks S vs Accuracy A', fontsize=16, fontweight='bold')
    
    A_values = np.linspace(0.1, 1.0, 100)
    
    configs = [
        {'R': 0.3, 'C': 0.8, 'N': 100, 'label': 'R=0.3, C=0.8, N=100'},
        {'R': 0.5, 'C': 0.8, 'N': 100, 'label': 'R=0.5, C=0.8, N=100'},
        {'R': 0.7, 'C': 0.8, 'N': 100, 'label': 'R=0.7, C=0.8, N=100'},
        {'R': 0.5, 'C': 0.6, 'N': 100, 'label': 'R=0.5, C=0.6, N=100'},
    ]
    
    for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
        S_values = []
        valid_A = []
        baseline = calculate_baseline_expectation(config['N'], config['R'])
        
        for A in A_values:
            try:
                S = calculate_expected_checks(config['R'], config['C'], A, config['N'])
                S_values.append(S)
                valid_A.append(A)
            except ValueError:
                pass
        
        ax.plot(valid_A, S_values, linewidth=2, label='With Predictor', color='blue')
        ax.axhline(y=baseline, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Baseline={baseline:.1f}')
        ax.set_xlabel('Accuracy A (Precision)', fontsize=11)
        ax.set_ylabel('Expected Checks S', fontsize=11)
        ax.set_title(config['label'], fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 调整纵轴范围
        if S_values:
            max_val = max(baseline, max(S_values))
            ax.set_ylim([0, max_val * 1.1])
    
    plt.tight_layout()
    plt.savefig('analysis_S_vs_A.png', dpi=300, bbox_inches='tight')
    print("保存: analysis_S_vs_A.png")
    plt.close()


def analyze_vs_N():
    """分析 S/N 关于任务总数 N 的变化"""
    print("分析 S/N vs N...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Normalized Checks S/N vs Total Tasks N', fontsize=16, fontweight='bold')
    
    N_values = np.arange(10, 501, 10)
    
    configs = [
        {'R': 0.3, 'C': 0.8, 'A': 0.8, 'label': 'R=0.3, C=0.8, A=0.8'},
        {'R': 0.5, 'C': 0.8, 'A': 0.8, 'label': 'R=0.5, C=0.8, A=0.8'},
        {'R': 0.7, 'C': 0.8, 'A': 0.8, 'label': 'R=0.7, C=0.8, A=0.8'},
    ]
    
    for idx, config in enumerate(configs):
        S_over_N = []
        baseline_over_N = []
        
        for N in N_values:
            try:
                S = calculate_expected_checks(config['R'], config['C'], config['A'], N)
                baseline = calculate_baseline_expectation(N, config['R'])
                S_over_N.append(S / N)
                baseline_over_N.append(baseline / N)
            except ValueError:
                S_over_N.append(np.nan)
                baseline_over_N.append(np.nan)
        
        ax = axes[idx]
        ax.plot(N_values, S_over_N, linewidth=2, label='With Predictor', color='blue')
        ax.plot(N_values, baseline_over_N, linestyle='--', linewidth=2, alpha=0.7, 
                label='Baseline', color='orange')
        ax.set_xlabel('Total Tasks N', fontsize=11)
        ax.set_ylabel('S/N (Normalized Checks)', fontsize=11)
        ax.set_title(config["label"], fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 计算有效的最大值（排除NaN）
        valid_baseline = [v for v in baseline_over_N if not np.isnan(v)]
        valid_S = [v for v in S_over_N if not np.isnan(v)]
        if valid_baseline and valid_S:
            max_val = max(max(valid_baseline), max(valid_S))
            ax.set_ylim([0, max_val * 1.1])
    
    plt.tight_layout()
    plt.savefig('analysis_S_over_N_vs_N.png', dpi=300, bbox_inches='tight')
    print("保存: analysis_S_over_N_vs_N.png")
    plt.close()


def analyze_heatmap_C_A():
    """绘制 S 关于 C 和 A 的热力图"""
    print("分析 S 热力图 (C vs A)...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Heatmap: Expected Checks S (Coverage C vs Accuracy A)', fontsize=16, fontweight='bold')
    
    C_values = np.linspace(0.1, 1.0, 50)
    A_values = np.linspace(0.1, 1.0, 50)
    
    configs = [
        {'R': 0.3, 'N': 100, 'title': 'R=0.3, N=100'},
        {'R': 0.7, 'N': 100, 'title': 'R=0.7, N=100'},
    ]
    
    for ax, config in zip(axes, configs):
        S_matrix = np.zeros((len(A_values), len(C_values)))
        
        for i, A in enumerate(A_values):
            for j, C in enumerate(C_values):
                try:
                    S = calculate_expected_checks(config['R'], C, A, config['N'])
                    S_matrix[i, j] = S
                except ValueError:
                    S_matrix[i, j] = np.nan
        
        im = ax.imshow(S_matrix, aspect='auto', origin='lower', 
                      extent=[C_values[0], C_values[-1], A_values[0], A_values[-1]],
                      cmap='viridis')
        ax.set_xlabel('Coverage C (Recall)', fontsize=12)
        ax.set_ylabel('Accuracy A (Precision)', fontsize=12)
        ax.set_title(config['title'], fontsize=13)
        plt.colorbar(im, ax=ax, label='Expected Checks S')
    
    plt.tight_layout()
    plt.savefig('analysis_heatmap_C_A.png', dpi=300, bbox_inches='tight')
    print("保存: analysis_heatmap_C_A.png")
    plt.close()


def analyze_efficiency_ratio():
    """分析预测器的效率比 (Oracle/S)"""
    print("分析效率比 (Oracle/S)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Efficiency Ratio: N/S vs Parameters', fontsize=16, fontweight='bold')
    
    # N/S vs R
    ax = axes[0, 0]
    R_values = np.linspace(0.1, 0.9, 100)
    for C, A in [(0.6, 0.6), (0.8, 0.8), (0.9, 0.9)]:
        ratios = []
        for R in R_values:
            try:
                S = calculate_expected_checks(R, C, A, 100)
                ratios.append(100 / S)
            except ValueError:
                ratios.append(np.nan)
        ax.plot(R_values, ratios, linewidth=2, label=f'C={C}, A={A}')
    ax.set_xlabel('Collision Rate R', fontsize=11)
    ax.set_ylabel('Efficiency Ratio (N/S)', fontsize=11)
    ax.set_title('Efficiency vs R', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # N/S vs C
    ax = axes[0, 1]
    C_values = np.linspace(0.1, 1.0, 100)
    for R, A in [(0.3, 0.8), (0.5, 0.8), (0.7, 0.8)]:
        ratios = []
        valid_C = []
        for C in C_values:
            try:
                S = calculate_expected_checks(R, C, A, 100)
                ratios.append(100 / S)
                valid_C.append(C)
            except ValueError:
                pass
        ax.plot(valid_C, ratios, linewidth=2, label=f'R={R}, A={A}')
    ax.set_xlabel('Coverage C', fontsize=11)
    ax.set_ylabel('Efficiency Ratio (N/S)', fontsize=11)
    ax.set_title('Efficiency vs C', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # N/S vs A
    ax = axes[1, 0]
    A_values = np.linspace(0.1, 1.0, 100)
    for R, C in [(0.3, 0.8), (0.5, 0.8), (0.7, 0.8)]:
        ratios = []
        valid_A = []
        for A in A_values:
            try:
                S = calculate_expected_checks(R, C, A, 100)
                ratios.append(100 / S)
                valid_A.append(A)
            except ValueError:
                pass
        ax.plot(valid_A, ratios, linewidth=2, label=f'R={R}, C={C}')
    ax.set_xlabel('Accuracy A', fontsize=11)
    ax.set_ylabel('Efficiency Ratio (N/S)', fontsize=11)
    ax.set_title('Efficiency vs A', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # N/S vs N
    ax = axes[1, 1]
    N_values = np.arange(10, 501, 10)
    for R, C, A in [(0.3, 0.8, 0.8), (0.5, 0.8, 0.8), (0.7, 0.8, 0.8)]:
        ratios = []
        for N in N_values:
            try:
                S = calculate_expected_checks(R, C, A, N)
                ratios.append(N / S)
            except ValueError:
                ratios.append(np.nan)
        ax.plot(N_values, ratios, linewidth=2, label=f'R={R}, C={C}, A={A}')
    ax.set_xlabel('Total Tasks N', fontsize=11)
    ax.set_ylabel('Efficiency Ratio (N/S)', fontsize=11)
    ax.set_title('Efficiency vs N', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_efficiency_ratio.png', dpi=300, bbox_inches='tight')
    print("保存: analysis_efficiency_ratio.png")
    plt.close()


def main():
    """主函数"""
    print("=" * 70)
    print("分析 calculate_expected_checks 函数")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs('prediction_approaches/utils/analysis_results', exist_ok=True)
    os.chdir('prediction_approaches/utils/analysis_results')
    
    # 执行各项分析
    analyze_vs_R()
    analyze_vs_C()
    analyze_vs_A()
    analyze_vs_N()
    # analyze_heatmap_C_A()
    analyze_efficiency_ratio()
    
    print("=" * 70)
    print("分析完成! 所有图表已保存到 prediction_approaches/utils/analysis_results/ 目录")
    print("=" * 70)
    
    # 打印一些关键观察
    print("\n关键观察:")
    print("1. S vs R: 随着碰撞率增加，期望检测次数通常减少")
    print("2. S vs C: 更高的覆盖率(召回率)通常降低期望检测次数")
    print("3. S vs A: 更高的准确率(精确率)显著降低期望检测次数")
    print("4. S/N vs N: 归一化检测次数随N趋于稳定")
    print("5. 效率比 N/S: 反映预测器相比Oracle的加速比")


if __name__ == "__main__":
    main()
