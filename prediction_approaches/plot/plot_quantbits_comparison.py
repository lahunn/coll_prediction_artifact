import math
import sys
import os
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib

# --- 统一绘图风格配置 ---
import matplotlib.font_manager as fm

sns.set_theme(style="whitegrid")
sns.set_style("ticks")
sns.set_palette("colorblind")

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

# 3. 加载数据
script_dir = os.path.dirname(os.path.abspath(__file__))
# 尝试多个可能的路径
possible_paths = [
    os.path.join(script_dir, "..", "result_files", "sphere_hashing_cost_results.csv"),
    os.path.join(script_dir, "result_files", "sphere_hashing_cost_results.csv"),
    "result_files/sphere_hashing_cost_results.csv"
]

csv_path = None
for p in possible_paths:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print(f"错误: 无法找到数据文件。请确保 sphere_hashing_cost_results.csv 存在于 result_files 目录中。")
    exit()

try:
    df = pd.read_csv(csv_path)
    print(f"成功加载数据: {csv_path}")
except Exception as e:
    print(f"错误: 加载数据文件 {csv_path} 时发生异常: {e}")
    exit()

# 3. 数据预处理
density_levels = ["dens6", "dens9", "dens12"]
density_labels = {
    "dens6": "障碍物密度 低",
    "dens9": "障碍物密度 中",
    "dens12": "障碍物密度 高",
}

# 按 Density 和 QuantBits 分组并取均值
agg_df = df.groupby(['Density', 'QuantBits']).mean().reset_index()

# 4. 开始绘图
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

metrics = [
    ('PosePrecision', '精确率 %'),
    ('PoseRecall', '召回率 %'),
    ('SpeedUp_Pct', '计算开销 %')
]

palette = sns.color_palette("colorblind")
markers = ['o', 's', '^']

for i, (col, title) in enumerate(metrics):
    ax = axes[i]
    for j, density in enumerate(density_levels):
        density_data = agg_df[agg_df['Density'] == density]
        ax.plot(
            density_data['QuantBits'], 
            density_data[col], 
            marker=markers[j], 
            label=density_labels[density],
            linewidth=2.5,
            markersize=10,
            color=palette[j]
        )
    
    ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel('量化位数', fontsize=20)
    if i == 0:
        ax.set_ylabel('百分比', fontsize=20)
    
    ax.set_xticks(sorted(agg_df['QuantBits'].unique()))
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # 针对成本图，计算相对于 100% 的减少量
    if col == 'SpeedUp_Pct':
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='基准')

# 5. 添加图例和整理布局
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False, fontsize=18)

plt.tight_layout()

# 6. 保存图表
output_filename = 'plot/figs/fig_quantbits_analysis.pdf'
os.makedirs('plot/figs', exist_ok=True)
plt.savefig(output_filename, bbox_inches='tight')
plt.savefig(output_filename.replace('.pdf', '.pdf'), bbox_inches='tight', dpi=300)

print(f"绘图完成！结果已保存至: {output_filename}")
