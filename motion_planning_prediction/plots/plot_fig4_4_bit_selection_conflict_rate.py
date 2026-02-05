#!/usr/bin/env python3
"""
Plot Bit Selection Strategy Comparison (Conflict Rate)
展示不同 Bit Selection 策略在 Multi-Bank CHT 中的冲突率表现。

数据来源:
- motion_planning_prediction/analysis/result_files/multi_bank_bit_selection_results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
from matplotlib.ticker import PercentFormatter

# 1. 统一绘图风格
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")

# 确保在PDF和PS文件中正确嵌入字体
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def plot_conflict_rate_comparison():
    # 2. 读取数据
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'analysis/result_files/multi_bank_bit_selection_results.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 3. 数据处理
    # 按冲突率从高到低排序 (绘图时从上到下即为从低到高，或者反过来取决于绘图顺序)
    # 我们希望最好的策略(冲突率最低)排在最上面
    df = df.sort_values("conflict_rate", ascending=True)

    # 4. 绘图
    plt.figure(figsize=(10, 7))
    
    # 使用 Seaborn 绘制水平条形图
    # 使用 'viridis_r' 颜色映射：数值越低(越好)颜色越亮/绿，数值越高(越差)颜色越深/紫
    # 或者使用单一颜色保持简洁
    ax = sns.barplot(
        data=df,
        x="conflict_rate",
        y="config",
        hue="conflict_rate", 
        palette="viridis", 
        edgecolor="black",
        alpha=0.9,
        legend=False
    )

    # 5. 添加数值标签
    for i, (idx, row) in enumerate(df.iterrows()):
        # 在柱子末尾标注数值 (百分比形式)
        rate_pct = row["conflict_rate"] * 100
        ax.text(
            row["conflict_rate"] * 1.02, i, 
            f"{rate_pct:.2f}%", 
            va='center', 
            fontsize=11, 
            fontweight='bold', 
            color='#333333'
        )

    # 6. 图表装饰
    plt.title("Conflict Rate Comparison by Bit Selection Strategy", pad=20, fontweight='bold', fontsize=16)
    plt.xlabel("Conflict Rate (Lower is Better)", fontweight='bold')
    plt.ylabel("Bit Selection Configuration", fontweight='bold')
    
    # X轴格式化为百分比
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    
    # 添加垂直网格线
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    
    # 调整X轴范围，留出空间给标签
    plt.xlim(0, df["conflict_rate"].max() * 1.15)
    
    # 移除顶部和右侧边框
    sns.despine()
    
    plt.tight_layout()

    # 7. 保存
    output_path = os.path.join(base_dir, 'plots/figs/fig4_4_bit_selection_conflict_rate.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    plot_conflict_rate_comparison()
