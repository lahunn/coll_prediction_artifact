#!/usr/bin/env python3
"""
Figure 1: Single Dual-Port CHT 访问冲突统计
绘制柱状图，展示在不同场景复杂度 (G1-G5) 下，集中式双端口存储架构产生的访问冲突总数。

数据来源:
- result_files/shared_dual_port_results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# 1. 设置绘图风格（白底、带刻度）
sns.set_style("ticks") 

# 2. 设置调色板（推荐色盲友好型）
sns.set_palette("colorblind")

# 3. 设置上下文（自动调整线条粗细和字体大小，'paper' 适合论文）
sns.set_context("paper", font_scale=1.5)

# 确保在PDF和PS文件中正确嵌入字体
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def plot_cht_conflicts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, "../result_files")
    
    # 文件路径
    data_file = os.path.join(result_dir, "shared_dual_port_results.csv")
    
    df = load_data(data_file)
    
    if df is None:
        return

    # 确保按场景排序 G1-G5
    scenes = ["G1", "G2", "G3", "G4", "G5"]
    df = df[df['Scene'].isin(scenes)].set_index('Scene').reindex(scenes)
    
    conflicts = df['Conflicts']
    
    x = np.arange(len(scenes))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 获取 Seaborn 调色板颜色 (使用红色系以警示“冲突”)
    palette = sns.color_palette()
    bar_color = palette[3] # 通常是红色/橘红色
    
    # 绘制冲突数柱状图
    bars = ax.bar(x, conflicts, width, color=bar_color, edgecolor='black', alpha=0.8)
    
    # 添加数值标注
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (conflicts.max() * 0.01),
                f'{int(height):,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 设置标签
    ax.set_ylabel('Total Memory Conflicts')
    ax.set_xlabel('Benchmark Scenario')
    ax.set_title('CHT Access Conflicts in Shared Dual-Port Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    
    # 格式化 Y 轴
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    
    # grid removed per project style
    
    # 移除顶部和右侧边框
    sns.despine()
    
    plt.tight_layout()
    
    output_path = os.path.join(base_dir, "figs/fig4_1_cht_conflicts.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Figure 1 (Conflicts) saved to {output_path}")

if __name__ == "__main__":
    plot_cht_conflicts()