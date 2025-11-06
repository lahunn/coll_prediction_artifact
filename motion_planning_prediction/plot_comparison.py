#!/usr/bin/env python3
"""
绘制球体和OBB碰撞检测结果对比图

读取 sphere_results.csv 和 obb_results.csv 文件
生成三个直方图：
1. Sphere的实际查询总数、预测查询总数、Oracle查询总数对比
2. OBB的实际查询总数、预测查询总数、Oracle查询总数对比
3. Sphere和OBB的预测周期总数对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持（如果需要）
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def read_csv_data(csv_file):
    """
    读取CSV文件
    
    CSV格式：threshold, sample_rate, qnoncoll_multiplier, basename, num_benchmarks, 
             robot_name, total_checks, fall_prediction, fall_oracle, fall_cycle, reduction_rate
    """
    if not os.path.exists(csv_file):
        print(f"警告: 文件 {csv_file} 不存在")
        return None
    
    # 定义列名
    columns = [
        'threshold', 'sample_rate', 'qnoncoll_multiplier', 'basename', 
        'num_benchmarks', 'robot_name', 'total_checks', 'fall_prediction', 
        'fall_oracle', 'fall_cycle', 'reduction_rate'
    ]
    
    df = pd.read_csv(csv_file, names=columns, header=None)
    return df


def plot_sphere_comparison(df_sphere, output_dir='result_files'):
    """
    绘制Sphere碰撞检测对比图
    
    Args:
        df_sphere: Sphere数据DataFrame
        output_dir: 输出目录
    """
    if df_sphere is None or df_sphere.empty:
        print("无Sphere数据，跳过绘图")
        return
    
    # 根据数据条目数量调整图表大小和柱宽
    num_items = len(df_sphere)
    fig_width = max(8, min(3 * num_items, 16))  # 宽度在8-16之间
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # 准备数据
    x_labels = [f"{row['robot_name']}\n{row['basename']}" for _, row in df_sphere.iterrows()]
    x_pos = np.arange(len(x_labels))
    # 根据条目数量调整柱宽，条目少时柱子更窄
    width = min(0.25, 0.8 / (3 * num_items)) if num_items < 5 else 0.25
    
    actual_checks = df_sphere['total_checks'].values
    prediction_checks = df_sphere['fall_prediction'].values
    oracle_checks = df_sphere['fall_oracle'].values
    
    # 绘制柱状图
    bars1 = ax.bar(x_pos - width, actual_checks, width, label='Actual Checks', 
                   color='#2E86C1', alpha=0.8)
    bars2 = ax.bar(x_pos, prediction_checks, width, label='Prediction Checks', 
                   color='#E74C3C', alpha=0.8)
    bars3 = ax.bar(x_pos + width, oracle_checks, width, label='Oracle Checks', 
                   color='#27AE60', alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel('Robot / Dataset', fontweight='bold')
    ax.set_ylabel('Number of Checks', fontweight='bold')
    ax.set_title('Sphere Collision Detection: Query Count Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱状图上方添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'sphere_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Sphere对比图已保存至: {output_file}")
    plt.close()


def plot_obb_comparison(df_obb, output_dir='result_files'):
    """
    绘制OBB碰撞检测对比图
    
    Args:
        df_obb: OBB数据DataFrame
        output_dir: 输出目录
    """
    if df_obb is None or df_obb.empty:
        print("无OBB数据，跳过绘图")
        return
    
    # 根据数据条目数量调整图表大小和柱宽
    num_items = len(df_obb)
    fig_width = max(8, min(3 * num_items, 16))  # 宽度在8-16之间
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # 准备数据
    x_labels = [f"{row['robot_name']}\n{row['basename']}" for _, row in df_obb.iterrows()]
    x_pos = np.arange(len(x_labels))
    # 根据条目数量调整柱宽，条目少时柱子更窄
    width = min(0.25, 0.8 / (3 * num_items)) if num_items < 5 else 0.25
    
    actual_checks = df_obb['total_checks'].values
    prediction_checks = df_obb['fall_prediction'].values
    oracle_checks = df_obb['fall_oracle'].values
    
    # 绘制柱状图
    bars1 = ax.bar(x_pos - width, actual_checks, width, label='Actual Checks', 
                   color='#8E44AD', alpha=0.8)
    bars2 = ax.bar(x_pos, prediction_checks, width, label='Prediction Checks', 
                   color='#F39C12', alpha=0.8)
    bars3 = ax.bar(x_pos + width, oracle_checks, width, label='Oracle Checks', 
                   color='#16A085', alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel('Robot / Dataset', fontweight='bold')
    ax.set_ylabel('Number of Checks', fontweight='bold')
    ax.set_title('OBB Collision Detection: Query Count Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱状图上方添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'obb_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"OBB对比图已保存至: {output_file}")
    plt.close()


def plot_cycle_comparison(df_sphere, df_obb, output_dir='result_files'):
    """
    绘制Sphere和OBB的预测周期总数对比图
    
    Args:
        df_sphere: Sphere数据DataFrame
        df_obb: OBB数据DataFrame
        output_dir: 输出目录
    """
    if (df_sphere is None or df_sphere.empty) and (df_obb is None or df_obb.empty):
        print("无数据，跳过周期对比绘图")
        return
    
    # 准备数据 - 使用robot_name作为x轴标签
    x_labels = []
    sphere_cycles = []
    obb_cycles = []
    
    # 合并两个数据集的robot_name
    all_robots = set()
    if df_sphere is not None and not df_sphere.empty:
        all_robots.update(df_sphere['robot_name'].unique())
    if df_obb is not None and not df_obb.empty:
        all_robots.update(df_obb['robot_name'].unique())
    
    all_robots = sorted(all_robots)
    
    for robot in all_robots:
        x_labels.append(robot)
        
        # Sphere周期
        if df_sphere is not None and not df_sphere.empty:
            sphere_data = df_sphere[df_sphere['robot_name'] == robot]
            if not sphere_data.empty:
                sphere_cycles.append(sphere_data['fall_cycle'].values[0])
            else:
                sphere_cycles.append(0)
        else:
            sphere_cycles.append(0)
        
        # OBB周期
        if df_obb is not None and not df_obb.empty:
            obb_data = df_obb[df_obb['robot_name'] == robot]
            if not obb_data.empty:
                obb_cycles.append(obb_data['fall_cycle'].values[0])
            else:
                obb_cycles.append(0)
        else:
            obb_cycles.append(0)
    
    # 根据条目数量调整图表大小和柱宽
    num_items = len(x_labels)
    fig_width = max(8, min(3 * num_items, 16))  # 宽度在8-16之间
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    x_pos = np.arange(len(x_labels))
    # 根据条目数量调整柱宽，条目少时柱子更窄
    width = min(0.35, 1.2 / (2 * num_items)) if num_items < 5 else 0.35
    
    # 绘制柱状图
    bars1 = ax.bar(x_pos - width/2, sphere_cycles, width, label='Sphere Cycles', 
                   color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, obb_cycles, width, label='OBB Cycles', 
                   color='#E67E22', alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel('Robot Name', fontweight='bold')
    ax.set_ylabel('Total Prediction Cycles', fontweight='bold')
    ax.set_title('Sphere vs OBB: Prediction Cycle Cost Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱状图上方添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # 添加加速比文本
    for i, (s_cycle, o_cycle) in enumerate(zip(sphere_cycles, obb_cycles)):
        if s_cycle > 0 and o_cycle > 0:
            ratio = o_cycle / s_cycle
            y_pos = max(s_cycle, o_cycle) * 1.05
            ax.text(x_pos[i], y_pos, f'{ratio:.2f}x',
                   ha='center', va='bottom', fontsize=9, 
                   fontweight='bold', color='red')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'cycle_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"周期对比图已保存至: {output_file}")
    plt.close()


def main():
    """主函数"""
    print("="*70)
    print("碰撞检测结果可视化")
    print("="*70)
    
    # 设置文件路径
    result_dir = 'result_files'
    sphere_csv = os.path.join(result_dir, 'sphere_results.csv')
    obb_csv = os.path.join(result_dir, 'obb_results.csv')
    
    # 读取数据
    print("\n读取数据...")
    df_sphere = read_csv_data(sphere_csv)
    df_obb = read_csv_data(obb_csv)
    
    if df_sphere is not None:
        print(f"  Sphere数据: {len(df_sphere)} 条记录")
    if df_obb is not None:
        print(f"  OBB数据: {len(df_obb)} 条记录")
    
    # 创建输出目录
    os.makedirs(result_dir, exist_ok=True)
    
    # 生成图表
    print("\n生成图表...")
    plot_sphere_comparison(df_sphere, result_dir)
    plot_obb_comparison(df_obb, result_dir)
    plot_cycle_comparison(df_sphere, df_obb, result_dir)
    
    print("\n"+"="*70)
    print("所有图表已生成完成!")
    print("="*70)


if __name__ == "__main__":
    main()
