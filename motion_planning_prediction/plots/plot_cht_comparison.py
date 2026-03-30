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
import os
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")


# Unified plotting style (serif font, seaborn whitegrid, colors: navy/darkgreen)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
sns.set_style("white")
sns.set_palette("colorblind")



# 读取数据
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df_dual = pd.read_csv(os.path.join(base_dir, 'result_files/dual_port_results.csv'))
df_multi = pd.read_csv(os.path.join(base_dir, 'result_files/multi_bank_results.csv'))

# 处理百分比
for df in [df_dual, df_multi]:
    df['Utilization'] = df['Utilization'].str.rstrip('%').astype('float')

# 合并数据
df = pd.merge(df_dual, df_multi, on='Scene', suffixes=('_Dual', '_Multi'))

# 绘图
metrics = ['Total_Cycles', 'Throughput', 'Utilization', 'Conflicts']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Dual Port vs Multi Bank CHT Performance Comparison')

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    # Use consistent colors: Dual=navy, Multi=darkgreen
    df.plot(
        x='Scene',
        y=[f'{metric}_Dual', f'{metric}_Multi'],
        kind='bar',
        ax=ax,
        rot=0,
        color=[colors[0], "darkgreen"],
    )
    ax.set_title(metric)
    ax.set_ylabel('Value' if metric != 'Utilization' else 'Percentage (%)')
    # grid removed per project style

    # 在 Total_Cycles 图中标注差距百分比
    if metric == 'Total_Cycles':
        bars = ax.patches
        n = len(df)
        # pandas plot 分组柱状图，前n个是第一列数据(Dual)，后n个是第二列数据(Multi)
        for j in range(n):
            bar_dual = bars[j]
            bar_multi = bars[j + n]
            
            val_dual = bar_dual.get_height()
            val_multi = bar_multi.get_height()
            
            if val_dual > 0:
                diff_pct = (val_multi - val_dual) / val_dual * 100
                # 标注在较高的柱子上方
                height = max(val_dual, val_multi)
                # 计算两个柱子的中心位置
                x_pos = (bar_dual.get_x() + bar_multi.get_x() + bar_multi.get_width()) / 2
                
                ax.text(x_pos, height * 1.02, f'{diff_pct:+.1f}%', 
                        ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
            
            # 适当增加y轴上限，防止标注被切掉
            ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'figs/cht_comparison.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Figure saved to {output_path}")
