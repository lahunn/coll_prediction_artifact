import math
import sys

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import os
from matplotlib.ticker import FuncFormatter


# --- 字体大小变量 ---
FONT_SIZE = 16  # 其它字体大小（如标签、标题等）
TICK_FONT_SIZE = 12  # 坐标轴刻度字体大小
LEGEND_FONT_SIZE = 12  # legend字体大小

# --- 统一绘图风格配置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE
colors = sns.color_palette("deep")
#!/usr/bin/env python3
"""
Plot Hash Strategy Comparison (StdDev)
展示不同 Hash 策略在 Bank 负载均衡上的表现（取每种策略的最佳/最小 StdDev）。

数据来源:
- motion_planning_prediction/analysis/result_files/hash_analysis_results.csv
"""

def plot_hash_strategy_comparison():
    # 2. 读取数据
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'analysis/result_files/hash_analysis_results.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 3. 数据处理：按策略分组，取最小 StdDev
    # 这一步筛选出每种策略下的"最佳配置"
    best_configs = df.loc[df.groupby("Strategy")["StdDev"].idxmin()].copy()
    
    # 按 StdDev 从小到大排序（越小越均衡）
    best_configs = best_configs.sort_values("StdDev")

    # 4. 绘图
    plt.figure(figsize=(9.8, 3.0))
    
    # 使用 Seaborn 绘制条形图
    ax = sns.barplot(
        data=best_configs,
        x="StdDev",
        y="Strategy",
        hue="Strategy",      # 使用 Strategy 着色
        palette="viridis",   # 选择美观的渐变色板
        dodge=False,         # 不需要分组偏移
        edgecolor="black",   # 添加边框
        alpha=0.85
    )
    # 设置刻度字体
    ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # 5. 添加数值标签
    for i, (idx, row) in enumerate(best_configs.iterrows()):
        # 在柱子末尾标注数值
        ax.text(row["StdDev"] * 1.02, i, 
                f"{int(row['StdDev']):,}", 
                va='center', fontsize=FONT_SIZE, fontweight='bold', color='#333333')

    # 6. 图表装饰
    plt.xlabel("标准差（越低越好）", fontsize=FONT_SIZE)
    plt.ylabel("地址划分策略", fontsize=FONT_SIZE)
    
    # 移除多余的图例（因为Y轴已经是标签了）
    if ax.legend_:
        ax.legend_.remove()
    
    # 调整X轴范围，留出空间给标签
    plt.xlim(0, best_configs["StdDev"].max() * 1.15)
    
    plt.tight_layout()

    # 7. 保存
    output_path = os.path.join(base_dir, 'plots/figs/hash_strategy_comparison.pdf')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    plot_hash_strategy_comparison()
