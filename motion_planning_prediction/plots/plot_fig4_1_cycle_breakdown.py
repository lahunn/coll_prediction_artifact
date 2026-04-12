#!/usr/bin/env python3
"""
图 4.1: Sphere 场景下共享双端口 CHT 访问冲突统计
绘制柱状图，展示在不同场景复杂度 (G1-G5) 下，sphere 碰撞模型在集中式双端口存储架构产生的访问冲突总数。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

# --- 1. 统一绘图风格配置 ---
sns.set_theme(style="whitegrid")
sns.set_style("ticks") 
sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.5)

# 字体加载与配置
font_path = os.path.expanduser("~/.local/share/fonts/simsun.ttc")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

plt.rcParams.update({
    'font.sans-serif': ['SimSun', 'NSimSun', 'Arial Unicode MS', 'sans-serif'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"错误: 未找到文件: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"读取 {filepath} 时出错: {e}")
        return None

def plot_cht_conflicts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, "../result_files")
    
    # 文件路径
    data_file = os.path.join(result_dir, "shared_dual_port_sphere_results.csv")
    
    df = load_data(data_file)
    
    if df is None:
        return

    # 确保按场景排序 G1-G5
    scenes = ["G1", "G2", "G3", "G4", "G5"]
    df = df[df['Scene'].isin(scenes)].set_index('Scene').reindex(scenes)
    
    conflicts = df['Conflicts']
    
    x = np.arange(len(scenes))
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # 获取 Seaborn 调色板颜色 (使用红色系以警示“冲突”)
    palette = sns.color_palette()
    line_color = palette[3]  # 通常是红色/橘红色

    # 绘制冲突数折线图
    ax.plot(
        x,
        conflicts.values,
        color=line_color,
        linewidth=2.5,
        marker='o',
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2,
    )
    
    # 添加数值标注
    for xi, yi in zip(x, conflicts.values):
        ax.text(
            xi,
            yi + (conflicts.max() * 0.02),
            f'{int(yi):,}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
        )

    # 设置中文标签
    ax.set_ylabel('存储访问冲突总数')
    ax.set_xlabel('运动规划任务分组(按碰撞检测请求总数)')
    ax.set_title('Sphere 场景下共享双端口架构的 CHT 访问冲突统计')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    
    # 格式化 Y 轴
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    
    # 移除顶部和右侧边框
    sns.despine()
    
    plt.tight_layout()
    
    output_path = os.path.join(base_dir, "figs/fig_cht_conflicts.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"图表已保存至: {output_path}")

if __name__ == "__main__":
    plot_cht_conflicts()
