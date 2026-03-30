import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import os

# 1. 设置绘图风格与字体
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set_style("ticks")
sns.set_palette("colorblind")

font = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}
plt.rc('font', **font)

# 2. 加载数据
csv_path = "result_files/sphere_hashing_cost_results.csv"
if not os.path.exists(csv_path):
    # 尝试从 plot 目录下运行时寻找上级目录
    csv_path = "../result_files/sphere_hashing_cost_results.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"成功加载数据: {csv_path}")
except Exception as e:
    print(f"错误: 无法加载数据文件 {csv_path}。请确保文件存在。")
    exit()

# 3. 数据预处理
# 我们通常关注 RadiusBits=0 且 Threshold=1.0 的情况来观察 QuantBits 的影响
# 或者对所有 Threshold 和 SampleRate 取均值来展示一般趋势
densities = ["dens3", "dens6", "dens9", "dens12"]
density_labels = {
    "dens3": "Density 3",
    "dens6": "Density 6",
    "dens9": "Density 9",
    "dens12": "Density 12"
}

# 按 Density 和 QuantBits 分组并取均值
agg_df = df.groupby(['Density', 'QuantBits']).mean().reset_index()

# 4. 开始绘图
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

metrics = [
    ('PosePrecision', 'Precision (%)'),
    ('PoseRecall', 'Recall (%)'),
    ('SpeedUp_Pct', 'Computation Cost (%)')
]

palette = sns.color_palette("colorblind")
markers = ['o', 's', '^', 'D']

for i, (col, title) in enumerate(metrics):
    ax = axes[i]
    for j, density in enumerate(densities):
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
    ax.set_xlabel('QuantBits', fontsize=20)
    if i == 0:
        ax.set_ylabel('Percentage (%)', fontsize=20)
    
    ax.set_xticks(sorted(agg_df['QuantBits'].unique()))
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # 针对成本图，计算相对于 100% 的减少量
    if col == 'SpeedUp_Pct':
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# 5. 添加图例和整理布局
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False, fontsize=18)

plt.tight_layout()

# 6. 保存图表
output_filename = 'plot/figs/fig_quantbits_analysis.pdf'
os.makedirs('plot/figs', exist_ok=True)
plt.savefig(output_filename, bbox_inches='tight')
plt.savefig(output_filename.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)

print(f"绘图完成！结果已保存至: {output_filename}")
