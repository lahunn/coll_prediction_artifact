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


# Set plot styles
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set_style("white")
sns.set_palette("colorblind")



# Load results
try:
    df = pd.read_csv("result_files/adaptive_comparison.csv", header=None)
    df.columns = ['精确率', '召回率']
except FileNotFoundError:
    print("Error: result_files/adaptive_comparison.csv not found.")
    print("Please run 'bash launch_adaptive_comparison.sh' first.")
    exit()

# --- Plotting ---
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1,1,1)

labels = [
    'Random Baseline',
    'Fixed S (for High Density)',
    'Fixed S (for Low Density)',
    'Adaptive S (Ours)'
]

precision = df['精确率'].tolist()
recall = df['召回率'].tolist()

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

palette = sns.color_palette("colorblind")
rects1 = ax.bar(x - width/2, precision, width, label='精确率', color=palette[0])
rects2 = ax.bar(x + width/2, recall, width, label='召回率', color=palette[2])

# Add some text for labels, title and axes ticks
ax.set_ylabel('Percentage (%)')
ax.set_title('Performance in a Changing Environment (Low to High Density)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylim(0, 100)
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.1f')
ax.bar_label(rects2, padding=3, fmt='%.1f')

fig.tight_layout()

# Save the figure
output_filename = 'adaptive_comparison_fig.pdf'
plt.savefig(output_filename)

print(f"Plot saved to {output_filename}")
