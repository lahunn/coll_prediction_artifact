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
#!/usr/bin/env python3
"""
绘制预测准确率学习曲线

专门用于可视化碰撞预测准确率随训练数据量变化的图表
"""

from pathlib import Path
import argparse

# Unified Plotting Style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")

# 确保在PDF和PS文件中正确嵌入字体
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def load_data(csv_file):
    """加载准确率数据"""
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"File {csv_file} does not exist")
    
    df = pd.read_csv(csv_file)
    return df

def plot_single_config(ax, df, config, title_suffix=""):
    """为单个配置绘制学习曲线"""
    threshold, sample_rate, qnoncoll_multiplier = config
    
    config_data = df[
        (df['threshold'] == threshold) & 
        (df['sample_rate'] == sample_rate) & 
        (df['qnoncoll_multiplier'] == qnoncoll_multiplier)
    ].sort_values('training_size')
    
    if config_data.empty:
        return False
    
    # 计算平均值和标准差
    size_groups = config_data.groupby('training_size')['accuracy']
    sizes = []
    means = []
    stds = []
    
    for size, group in size_groups:
        sizes.append(size)
        means.append(group.mean())
        stds.append(group.std())
    
    # 获取 Seaborn 颜色
    colors = sns.color_palette()
    line_color = colors[0]
    fill_color = colors[2]

    # 绘制曲线
    ax.plot(
        sizes,
        means,
        'o-',
        linewidth=2,
        markersize=4,
        color=line_color,
        label=f'Threshold={threshold}, Sample Rate={sample_rate}, Queue Multiplier={qnoncoll_multiplier}'
    )
    
    # 添加误差条
    if any(stds):
        ax.fill_between(
            sizes,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.3,
            color=fill_color
        )
    
    ax.set_xlabel('训练数据量 (历史字典大小)')
    ax.set_ylabel('准确率')
    ax.set_title(f'碰撞预测准确率学习曲线 {title_suffix}')
    # grid removed per project style
    ax.legend()
    
    return True

def plot_comparison(df, configs, save_path=None):
    """绘制多个配置的对比图"""
    n_configs = len(configs)
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plotted_count = 0
    for i, config in enumerate(configs):
        if i < len(axes):
            success = plot_single_config(axes[i], df, config, f'(Config {i+1})')
            if success:
                plotted_count += 1
    
    # 隐藏未使用的子图
    for i in range(plotted_count, len(axes)):
        axes[i].set_visible(False)
    
    sns.despine()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    plt.show()

def plot_aggregated_curve(df, save_path=None):
    """绘制聚合的学习曲线"""
    plt.figure(figsize=(10, 6))
    
    # 按训练大小聚合所有数据
    size_groups = df.groupby('training_size')['accuracy']
    
    sizes = []
    means = []
    stds = []
    counts = []
    
    for size, group in size_groups:
        sizes.append(size)
        means.append(group.mean())
        stds.append(group.std())
        counts.append(len(group))
    
    # 获取 Seaborn 颜色
    colors = sns.color_palette()
    line_color = colors[0]
    fill_color = colors[2]

    # 绘制主曲线
    plt.plot(
        sizes,
        means,
        '-',
        linewidth=3,
        marker='o',
        markersize=6,
        color=line_color,
        label='Mean Accuracy'
    )
    
    # 添加误差条
    plt.fill_between(
        sizes,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        alpha=0.3,
        color=fill_color,
        label='Standard Deviation Range'
    )
    
    # 添加数据点数量标注
    for i, (size, count) in enumerate(zip(sizes, counts)):
        if i % max(1, len(sizes)//10) == 0:  # 每隔10%标注一次
            plt.annotate(f'n={count}', (size, means[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    
    plt.xlabel('Training Data Size (History Dictionary Size)')
    plt.ylabel('Prediction Accuracy')
    plt.title('Aggregated Learning Curve of 碰撞预测 Accuracy vs Training Data Size', pad=20)
    # grid removed per project style
    plt.legend()
    
    # 添加统计信息文本框
    final_accuracy = means[-1] if means else 0
    max_accuracy = max(means) if means else 0
    total_samples = len(df)
    
    stats_text = f'Final Accuracy: {final_accuracy:.4f}\nMax Accuracy: {max_accuracy:.4f}\nTotal Samples: {total_samples}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
    
    sns.despine()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"聚合学习曲线图已保存到: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot collision prediction accuracy learning curves')
    parser.add_argument('csv_file', help='Path to accuracy data CSV file')
    parser.add_argument('--output', '-o', help='Output image file path')
    parser.add_argument('--mode', choices=['aggregated', 'comparison'], 
                       default='aggregated', help='Plotting mode')
    parser.add_argument('--configs', nargs='+', 
                       help='Specify configurations (format: threshold,sample_rate,qnoncoll_multiplier) for comparison mode')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        print(f"Loading data: {args.csv_file}")
        df = load_data(args.csv_file)
        print(f"Loaded {len(df)} records with {len(df.groupby(['threshold', 'sample_rate', 'qnoncoll_multiplier']))} configurations")
        
        if args.mode == 'aggregated':
            plot_aggregated_curve(df, args.output)
        elif args.mode == 'comparison':
            if not args.configs:
                # 自动选择前3种配置
                configs = list(df.groupby(['threshold', 'sample_rate', 'qnoncoll_multiplier']).groups.keys())[:3]
                print(f"Automatically selected first {len(configs)} configurations for comparison")
            else:
                # 解析指定的配置
                configs = []
                for config_str in args.configs:
                    try:
                        threshold, sample_rate, qnoncoll_multiplier = map(float, config_str.split(','))
                        configs.append((threshold, sample_rate, int(qnoncoll_multiplier)))
                    except ValueError:
                        print(f"Configuration format error: {config_str}, should be threshold,sample_rate,qnoncoll_multiplier")
                        continue
            
            if configs:
                plot_comparison(df, configs, args.output)
            else:
                print("No valid configurations available for comparison")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()