import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Set plot styles
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
font = {
    'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 28,
}
plt.rc('font', **font)

# Load results
try:
    df = pd.read_csv("result_files/adaptive_comparison.csv", header=None)
    df.columns = ['Precision', 'Recall']
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

precision = df['Precision'].tolist()
recall = df['Recall'].tolist()

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax.bar(x - width/2, precision, width, label='Precision', color='navy')
rects2 = ax.bar(x + width/2, recall, width, label='Recall', color='cornflowerblue')

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
