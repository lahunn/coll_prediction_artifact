#!/usr/bin/env python3
"""
Plotting OBB and Sphere Methods Comparison Bar Charts
Based on data from result_files/obb_vs_sphere.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 统一绘图风格
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
sns.set_style("white")
sns.set_palette("colorblind")
font = {
    "family": "Times New Roman",
    "weight": "normal",
    "size": 28,
}
plt.rc("font", **font)


def plot_obb_comparison():
    """绘制OBB方法的详细统计柱状图"""
    print("Plotting OBB method statistics...")

    # 读取数据
    csv_path = os.path.join(os.path.dirname(__file__), "result_files", "obb_vs_sphere.csv")
    df = pd.read_csv(csv_path)

    # 获取OBB数据
    obb_data = df[df['方法'] == 'OBB'].iloc[0]

    # 准备数据
    categories = ['Total Actual Queries', 'Predicted Final Stats', 'Oracle Final Stats']
    values = [
        obb_data['实际查询总数'],
        obb_data['预测最终统计'],
        obb_data['oracle最终统计']
    ]

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(categories, values, color=['navy', 'darkgreen', 'darkgreen'], alpha=0.8)

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置标题和标签
    ax.set_title('OBB Method Query Statistics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Queries', fontsize=12)
    ax.set_xlabel('Statistics Type', fontsize=12)

    # 添加查询减少率信息
    reduction_rate = obb_data['查询减少率']
    ax.text(0.02, 0.98, f'Query Reduction Rate: {reduction_rate}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # grid removed per project style

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('result_files/obb_statistics_comparison.png', dpi=300, bbox_inches='tight')
    print("保存: result_files/obb_statistics_comparison.png")
    plt.close()


def plot_sphere_comparison():
    """绘制Sphere方法的详细统计柱状图"""
    print("Plotting Sphere method statistics...")

    # 读取数据
    csv_path = os.path.join(os.path.dirname(__file__), "result_files", "obb_vs_sphere.csv")
    df = pd.read_csv(csv_path)

    # 获取Sphere数据
    sphere_data = df[df['方法'] == 'Sphere'].iloc[0]

    # 准备数据
    categories = ['Total Actual Queries', 'Predicted Final Stats', 'Oracle Final Stats']
    values = [
        sphere_data['实际查询总数'],
        sphere_data['预测最终统计'],
        sphere_data['oracle最终统计']
    ]

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(categories, values, color=['navy', 'darkgreen', 'darkgreen'], alpha=0.8)

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置标题和标签
    ax.set_title('Sphere Method Query Statistics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Queries', fontsize=12)
    ax.set_xlabel('Statistics Type', fontsize=12)

    # 添加查询减少率信息
    reduction_rate = sphere_data['查询减少率']
    ax.text(0.02, 0.98, f'Query Reduction Rate: {reduction_rate}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # grid removed per project style

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('sphere_statistics_comparison.png', dpi=300, bbox_inches='tight')
    print("保存: sphere_statistics_comparison.png")
    plt.close()


def plot_methods_comparison():
    """绘制OBB和Sphere方法的对比柱状图"""
    print("Plotting methods comparison...")

    # 读取数据
    csv_path = os.path.join(os.path.dirname(__file__), "result_files", "obb_vs_sphere.csv")
    df = pd.read_csv(csv_path)

    # 准备数据
    methods = df['方法'].tolist()
    actual_queries = df['实际查询总数'].tolist()
    predicted_queries = df['预测最终统计'].tolist()
    oracle_queries = df['oracle最终统计'].tolist()
    reduction_rates = df['查询减少率'].tolist()

    # 设置柱状图位置
    x = np.arange(len(methods))
    width = 0.25

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制三组柱状图
    bars1 = ax.bar(x - width, actual_queries, width, label='Total Actual Queries',
                   color='navy', alpha=0.8)
    bars2 = ax.bar(x, predicted_queries, width, label='Predicted Final Stats',
                   color='darkgreen', alpha=0.8)
    bars3 = ax.bar(x + width, oracle_queries, width, label='Oracle Final Stats',
                   color='lightgreen', alpha=0.8)

    # 添加数值标签
    for bars, values in [(bars1, actual_queries), (bars2, predicted_queries), (bars3, oracle_queries)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(actual_queries)*0.01,
                    f'{value:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 设置标题和标签
    ax.set_title('OBB vs Sphere Methods Query Statistics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Queries', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    # 添加查询减少率信息
    for i, (method, rate) in enumerate(zip(methods, reduction_rates)):
        ax.text(i, max(actual_queries) * 0.9, f'Reduction Rate: {rate}',
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 添加图例
    ax.legend()

    # grid removed per project style

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('obb_sphere_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("保存: obb_sphere_methods_comparison.png")
    plt.close()


def main():
    """主函数"""
    print("=" * 70)
    print("Plotting OBB and Sphere Methods Comparison Bar Charts")
    print("=" * 70)

    # 检查CSV文件是否存在
    csv_path = os.path.join(os.path.dirname(__file__), "result_files", "obb_vs_sphere.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found {csv_path}")
        return

    # 执行绘图
    plot_obb_comparison()
    plot_sphere_comparison()
    plot_methods_comparison()

    print("=" * 70)
    print("Plotting completed! Images saved to current directory")
    print("Generated files:")
    print("- obb_statistics_comparison.png")
    print("- sphere_statistics_comparison.png")
    print("- obb_sphere_methods_comparison.png")
    print("=" * 70)


if __name__ == "__main__":
    main()