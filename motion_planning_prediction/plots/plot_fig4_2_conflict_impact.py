#!/usr/bin/env python3
"""
Figure 2: 集中式与无冲突存储架构的性能对比
绘制柱状图，展示 Shared Dual-Port SRAM 与 Conflict-Free SRAM 的总周期数对比。

数据来源:
- result_files/shared_dual_port_results.csv (Shared)
- result_files/no_conflict_results.csv (No-Conflict)
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
matplotlib.rcParams["font.family"] = "Times New Roman"

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

def plot_conflict_impact():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, "../result_files")
    
    # 文件路径
    shared_file = os.path.join(result_dir, "shared_dual_port_results.csv")
    nc_file = os.path.join(result_dir, "no_conflict_results.csv")
    
    df_shared = load_data(shared_file)
    df_nc = load_data(nc_file)
    
    if df_shared is None or df_nc is None:
        return

    # 确保按场景排序 G1-G5
    scenes = ["G1", "G2", "G3", "G4", "G5"]
    df_shared = df_shared[df_shared['Scene'].isin(scenes)].set_index('Scene').reindex(scenes)
    df_nc = df_nc[df_nc['Scene'].isin(scenes)].set_index('Scene').reindex(scenes)
    
    cycles_shared = df_shared['Total_Cycles']
    cycles_nc = df_nc['Total_Cycles']
    
    x = np.arange(len(scenes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 从 Seaborn 调色板获取颜色
    colors = sns.color_palette()
    shared_color = colors[0]
    nc_color = colors[1]
    
    # 绘制 Shared
    rects1 = ax.bar(x - width/2, cycles_shared, width, label='Shared Dual-Port SRAM', 
                    color=shared_color, edgecolor='black', alpha=0.85)
    
    # 绘制 No-Conflict
    rects2 = ax.bar(x + width/2, cycles_nc, width, label='Conflict-Free SRAM', 
                    color=nc_color, edgecolor='black', alpha=0.85)
    
    # 添加数值和差异标注
    for i in range(len(scenes)):
        shared = cycles_shared.iloc[i]
        nc = cycles_nc.iloc[i]
        
        # 计算差异百分比 (Shared比No-Conflict慢多少)
        overhead = (shared - nc) / nc * 100
        
        # 在 Shared 柱子上标注 Overhead
        if overhead > 1.0: # 忽略微小差异
            ax.text(i - width/2, shared * 1.02, f"+{overhead:.1f}%", 
                    ha='center', va='bottom', fontsize=10, 
                    color=shared_color, fontweight='bold')
            
    ax.set_ylabel('Total Simulation Cycles')
    ax.set_xlabel('Benchmark Scenario')
    ax.set_title('Performance Impact of Memory Conflicts: Shared vs Conflict-Free')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    
    # grid removed per project style
    
    # 移除顶部和右侧边框
    sns.despine()
    
    ax.legend()
    
    plt.tight_layout()
    
    output_path = os.path.join(base_dir, "figs/fig4_2_conflict_impact.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Figure 2 saved to {output_path}")

if __name__ == "__main__":
    plot_conflict_impact()
