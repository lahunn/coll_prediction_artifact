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
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")


# 设置绘图风格
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set_style("white")
sns.set_palette("colorblind")



# 加载实验结果
try:
    df = pd.read_csv("result_files/adaptive_evaluation.csv", header=None)
    df.columns = ['精确率', '召回率']
except FileNotFoundError:
    print("错误: 未找到 result_files/adaptive_evaluation.csv")
    print("请先运行 'bash launch_adaptive_evaluation.sh'")
    exit()

# --- 绘图 --- 
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(1,1,1)

# 为每个测试策略定义标签
labels = [
    'Fixed S: Random',
    'Fixed S: 2.0',
    'Fixed S: 1.0',
    'Fixed S: 0.5',
    'Fixed S: 0.125',
    'Fixed S: 0.031',
    'Adaptive S (0.1-1.0)',
    'Adaptive S (0.1-1.5)',
    'Adaptive S (0.05-2.0)',
]

precision = df['精确率'].tolist()
recall = df['召回率'].tolist()

x = np.arange(len(labels))  # 标签位置
width = 0.35  # 柱状图宽度

palette = sns.color_palette("colorblind")
rects1 = ax.bar(x - width/2, precision, width, label='精确率', color=palette[0])
rects2 = ax.bar(x + width/2, recall, width, label='召回率', color=palette[2])

# 添加标签、标题和坐标轴
ax.set_ylabel('Percentage (%)')
ax.set_title('Adaptive vs. Fixed S Strategies in a Changing Environment')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right") # 旋转标签以防重叠
ax.set_ylim(0, 100)
ax.legend()

# 在柱状图上显示数值
ax.bar_label(rects1, padding=3, fmt='%.1f')
ax.bar_label(rects2, padding=3, fmt='%.1f')

# 添加分界线以区分策略类型
ax.axvline(x=5.5, color=palette[7], linestyle='--', linewidth=1)
plt.text(2.5, -30, "Fixed S Strategies", ha="center", va="top", fontsize=20)
plt.text(7, -30, "Adaptive S Strategies", ha="center", va="top", fontsize=20)


fig.tight_layout()

# 保存图表
output_filename = 'adaptive_evaluation_fig.pdf'
plt.savefig(output_filename)

print(f"绘图已保存至 {output_filename}")
