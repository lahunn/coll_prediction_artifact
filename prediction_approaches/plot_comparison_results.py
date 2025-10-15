#!/usr/bin/env python3
"""
OBB与Sphere碰撞预测性能对比绘图脚本
"""
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font = {
    'family': 'serif',
    'weight': 'normal',
    'size': 28,
}


def plot_accuracy_recall_comparison():
    """
    图1: 不同密度场景下OBB和Sphere策略的准确率和召回率对比
    """
    # 读取OBB详细结果
    obb_data = pd.read_csv('result_files/s_params_detailed_16bins.csv', header=0)
    # 读取Sphere详细结果
    sphere_data = pd.read_csv('result_files/s_params_sphere_detailed_16coord_1radius_no_radius.csv', header=0)
    
    # 提取每个密度级别的最优结果
    densities = ['low', 'mid', 'high']
    density_labels = ['Low Density', 'Mid Density', 'High Density']
    
    obb_precision = []
    obb_recall = []
    sphere_precision = []
    sphere_recall = []
    
    for density in densities:
        # OBB: 找到该密度下成本最小的配置
        obb_density = obb_data[obb_data['密度'] == density]
        best_obb = obb_density.loc[obb_density['平均成本'].idxmin()]
        obb_precision.append(best_obb['精确率(%)'])
        obb_recall.append(best_obb['召回率(%)'])
        
        # Sphere: 找到该密度下成本最小的配置
        sphere_density = sphere_data[sphere_data['密度'] == density]
        best_sphere = sphere_density.loc[sphere_density['平均成本'].idxmin()]
        sphere_precision.append(best_sphere['精确率(%)'])
        sphere_recall.append(best_sphere['召回率(%)'])
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    plt.rc('font', **font)
    
    x = np.arange(len(densities))
    width = 0.35
    
    # 子图1: 精确率对比
    bars1 = ax1.bar(x - width/2, obb_precision, width, label='OBB', color='navy')
    bars2 = ax1.bar(x + width/2, sphere_precision, width, label='Sphere', color='darkgreen')
    
    ax1.set_ylabel('Precision (%)')
    ax1.set_title('Precision Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(density_labels)
    ax1.legend()
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=22)
    
    # 子图2: 召回率对比
    bars3 = ax2.bar(x - width/2, obb_recall, width, label='OBB', color='navy')
    bars4 = ax2.bar(x + width/2, sphere_recall, width, label='Sphere', color='darkgreen')
    
    ax2.set_ylabel('Recall (%)')
    ax2.set_title('Recall Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(density_labels)
    ax2.legend()
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=22)
    
    plt.tight_layout()
    plt.savefig('figs/fig_obb_sphere_precision_recall.pdf')
    print('✅ 图1已保存: figs/fig_obb_sphere_precision_recall.pdf')
    plt.clf()


def plot_cost_comparison():
    """
    图2: 不同密度场景下OBB和Sphere策略的最小计算成本对比
    """
    # 读取优化结果
    obb_opt = pd.read_csv('result_files/s_params_optimization_16bins.csv', header=0)
    sphere_opt = pd.read_csv('result_files/s_params_sphere_16coord_1radius_no_radius.csv', header=0)
    
    densities = ['low', 'mid', 'high']
    density_labels = ['Low Density', 'Mid Density', 'High Density']
    
    obb_costs = []
    sphere_costs = []
    
    for density in densities:
        obb_cost = obb_opt[obb_opt['密度'] == density]['平均成本'].values[0]
        sphere_cost = sphere_opt[sphere_opt['密度'] == density]['平均成本'].values[0]
        obb_costs.append(obb_cost)
        sphere_costs.append(sphere_cost)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.rc('font', **font)
    
    x = np.arange(len(densities))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, obb_costs, width, label='OBB', color='navy')
    bars2 = ax.bar(x + width/2, sphere_costs, width, label='Sphere', color='darkgreen')
    
    ax.set_ylabel('Average Cost (Checks)')
    ax.set_title('Minimum Computation Cost Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(density_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=22)
    
    plt.tight_layout()
    plt.savefig('figs/fig_obb_sphere_cost.pdf')
    print('✅ 图2已保存: figs/fig_obb_sphere_cost.pdf')
    plt.clf()


def plot_threshold_comparison(density='mid'):
    """
    图3: 不同阈值下,OBB和Sphere策略的准确率和召回率对比
    
    Args:
        density: 密度级别 ('low', 'mid', 'high')
    """
    # 读取详细结果
    obb_data = pd.read_csv('result_files/s_params_detailed_16bins.csv', header=0)
    sphere_data = pd.read_csv('result_files/s_params_sphere_detailed_16coord_1radius_no_radius.csv', header=0)
    
    # 筛选指定密度的数据
    obb_density = obb_data[obb_data['密度'] == density]
    sphere_density = sphere_data[sphere_data['密度'] == density]
    
    # 提取固定阈值策略的数据
    obb_fixed = obb_density[obb_density['策略类型'] == '固定阈值']
    sphere_fixed = sphere_density[sphere_density['策略类型'] == '固定阈值']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    plt.rc('font', **font)
    
    # 获取阈值列表(取OBB的阈值作为参考)
    thresholds = obb_fixed['参数值'].values
    threshold_labels = [f'S={t:.4f}' if t >= 0.0312 else 'S=0' for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    # 子图1: OBB精确率
    axes[0, 0].plot(x, obb_fixed['精确率(%)'].values, 'o-', linewidth=2, 
                    markersize=8, color='navy', label='OBB Precision')
    axes[0, 0].set_ylabel('Precision (%)')
    axes[0, 0].set_title(f'OBB - {density.upper()} Density')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(threshold_labels, rotation=45, ha='right')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 100])
    
    # 子图2: OBB召回率
    axes[0, 1].plot(x, obb_fixed['召回率(%)'].values, 's-', linewidth=2, 
                    markersize=8, color='cornflowerblue', label='OBB Recall')
    axes[0, 1].set_ylabel('Recall (%)')
    axes[0, 1].set_title(f'OBB - {density.upper()} Density')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(threshold_labels, rotation=45, ha='right')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 100])
    
    # 子图3: Sphere精确率
    axes[1, 0].plot(x, sphere_fixed['精确率(%)'].values, 'o-', linewidth=2, 
                    markersize=8, color='darkgreen', label='Sphere Precision')
    axes[1, 0].set_ylabel('Precision (%)')
    axes[1, 0].set_title(f'Sphere - {density.upper()} Density')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(threshold_labels, rotation=45, ha='right')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 100])
    
    # 子图4: Sphere召回率
    axes[1, 1].plot(x, sphere_fixed['召回率(%)'].values, 's-', linewidth=2, 
                    markersize=8, color='lightgreen', label='Sphere Recall')
    axes[1, 1].set_ylabel('Recall (%)')
    axes[1, 1].set_title(f'Sphere - {density.upper()} Density')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(threshold_labels, rotation=45, ha='right')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(f'figs/fig_threshold_comparison_{density}.pdf')
    print(f'✅ 图3已保存: figs/fig_threshold_comparison_{density}.pdf')
    plt.clf()


def plot_combined_threshold_comparison():
    """
    图3综合版: 在同一张图中对比OBB和Sphere在不同阈值下的表现
    """
    # 读取详细结果
    obb_data = pd.read_csv('result_files/s_params_detailed_16bins.csv', header=0)
    sphere_data = pd.read_csv('result_files/s_params_sphere_detailed_16coord_1radius_no_radius.csv', header=0)
    
    densities = ['low', 'mid', 'high']
    density_labels = {'low': 'Low', 'mid': 'Mid', 'high': 'High'}
    
    # 创建3行2列的子图
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    plt.rc('font', **font)
    
    for idx, density in enumerate(densities):
        # 筛选数据
        obb_density = obb_data[obb_data['密度'] == density]
        sphere_density = sphere_data[sphere_data['密度'] == density]
        
        obb_fixed = obb_density[obb_density['策略类型'] == '固定阈值']
        sphere_fixed = sphere_density[sphere_density['策略类型'] == '固定阈值']
        
        # 阈值标签
        thresholds = obb_fixed['参数值'].values
        threshold_labels = [f'{t:.3f}' if t > 0 else '0' for t in thresholds]
        
        x = np.arange(len(thresholds))
        
        # 左列: 精确率对比
        ax_prec = axes[idx, 0]
        ax_prec.plot(x, obb_fixed['精确率(%)'].values, 'o-', linewidth=2, 
                    markersize=6, color='navy', label='OBB')
        ax_prec.plot(x, sphere_fixed['精确率(%)'].values, 's-', linewidth=2, 
                    markersize=6, color='darkgreen', label='Sphere')
        ax_prec.set_ylabel('Precision (%)')
        ax_prec.set_title(f'{density_labels[density]} Density - Precision')
        ax_prec.set_xticks(x[::2])  # 每隔一个显示标签
        ax_prec.set_xticklabels(threshold_labels[::2], rotation=45, ha='right', fontsize=20)
        ax_prec.grid(alpha=0.3)
        ax_prec.legend(fontsize=22)
        ax_prec.set_ylim([0, 100])
        
        # 右列: 召回率对比
        ax_rec = axes[idx, 1]
        ax_rec.plot(x, obb_fixed['召回率(%)'].values, 'o-', linewidth=2, 
                   markersize=6, color='navy', label='OBB')
        ax_rec.plot(x, sphere_fixed['召回率(%)'].values, 's-', linewidth=2, 
                   markersize=6, color='darkgreen', label='Sphere')
        ax_rec.set_ylabel('Recall (%)')
        ax_rec.set_title(f'{density_labels[density]} Density - Recall')
        ax_rec.set_xticks(x[::2])
        ax_rec.set_xticklabels(threshold_labels[::2], rotation=45, ha='right', fontsize=20)
        ax_rec.grid(alpha=0.3)
        ax_rec.legend(fontsize=22)
        ax_rec.set_ylim([0, 100])
    
    # 添加x轴总标签
    fig.text(0.5, 0.02, 'Threshold Value (S)', ha='center', fontsize=32)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('figs/fig_threshold_comparison_combined.pdf')
    print('✅ 图3综合版已保存: figs/fig_threshold_comparison_combined.pdf')
    plt.clf()


def plot_update_frequency_impact():
    """
    图4: 不同更新频率下准确率和召回率的对比
    注意: 当前数据只有update_prob=0.5的结果,这里演示如何绘制
    如需完整图表,需要运行不同update_prob参数的优化脚本
    """
    # 这里假设我们有不同更新频率的数据
    # 实际使用时需要先运行optimize_s_parameters.py和optimize_s_parameters_sphere.py
    # 使用不同的update_prob参数(如0.1, 0.3, 0.5, 0.7, 0.9)
    
    print('\n⚠️  注意: 图4需要运行不同update_prob参数的优化脚本')
    print('示例命令:')
    print('  python optimize_s_parameters.py 4 0.1')
    print('  python optimize_s_parameters.py 4 0.3')
    print('  python optimize_s_parameters.py 4 0.5')
    print('  python optimize_s_parameters.py 4 0.7')
    print('  python optimize_s_parameters.py 4 0.9')
    print('\n  同样对optimize_s_parameters_sphere.py执行相同操作')
    

def main():
    """主函数"""
    import os
    
    # 确保figs目录存在
    os.makedirs('figs', exist_ok=True)
    
    print('=' * 70)
    print('OBB与Sphere碰撞预测性能对比绘图')
    print('=' * 70)
    
    # 图1: 准确率和召回率对比
    print('\n生成图1: 不同密度下的精确率和召回率对比...')
    plot_accuracy_recall_comparison()
    
    # 图2: 计算成本对比
    print('\n生成图2: 不同密度下的最小计算成本对比...')
    plot_cost_comparison()
    
    # 图3: 不同阈值下的性能对比
    print('\n生成图3: 不同阈值下的性能对比...')
    for density in ['low', 'mid', 'high']:
        plot_threshold_comparison(density)
    
    # 图3综合版
    print('\n生成图3综合版: 所有密度下的阈值对比...')
    plot_combined_threshold_comparison()
    
    # 图4提示
    plot_update_frequency_impact()
    
    print('\n' + '=' * 70)
    print('✅ 所有图表生成完成!')
    print('图表保存在: prediction_approaches/figs/')
    print('=' * 70)


if __name__ == '__main__':
    main()
