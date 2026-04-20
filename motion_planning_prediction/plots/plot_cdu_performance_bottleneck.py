import math
import sys

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import os

# --- 统一绘图风格配置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")
#!/usr/bin/env python3
"""
Plot CDU performance bottleneck analysis.
Displays the breakdown of CDU cycles into Processing and various Idle states across scenes.
This naming convention (CDU) matches the terminology used in the thesis.
"""
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")


# 1. Set plotting style
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = 'SimSun'

def plot_cdu_performance_bottleneck():
    # 2. Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '../analysis/result_files/performance_bottleneck_analysis.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    # 3. Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 4. Data processing
    # Total capacity = cycles * number of units
    df['total_capacity'] = df['total_cycles'] * df['num_oocds']
    
    # Calculate percentages for the stacked bar
    df['Active Processing'] = (df['total_capacity'] - df['oocd_idle_cycles']) / df['total_capacity'] * 100
    df['Idle: No Tasks'] = df['oocd_idle_no_tasks'] / df['total_capacity'] * 100
    df['Idle: Queue Constraint'] = df['oocd_idle_qnoncoll_not_full'] / df['total_capacity'] * 100
    df['Idle: Startup Latency'] = df['oocd_idle_startup'] / df['total_capacity'] * 100
    
    # Sort by scene G1-G5
    if 'scene' in df.columns:
        df['scene_num'] = df['scene'].str.extract(r'(\d+)').astype(float)
        df = df.sort_values('scene_num')
    
    scenes = df['scene'].tolist()
    x = range(len(scenes))
    
    # 5. Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    
    plot_components = [
        ('Active Processing', 'Processing detected tasks'),
        ('Idle: No Tasks', 'Queue empty (No tasks generated)'),
        ('Idle: Queue Constraint', 'Waiting for queue density'),
        ('Idle: Startup Latency', 'Pipeline setup overhead')
    ]
    
    colors = sns.color_palette("colorblind", len(plot_components))
    bottom = pd.Series([0.0] * len(scenes))
    
    for i, (col, desc) in enumerate(plot_components):
        ax.bar(x, df[col], bottom=bottom, label=col, 
               color=colors[i], edgecolor='black', alpha=0.85)
        bottom += df[col].values

    # 6. Aesthetics
    ax.set_ylabel('Cycle Distribution (%)', fontweight='bold')
    ax.set_xlabel('运动规划问题分组(按任务复杂度划分)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylim(0, 100)
    
    # Legend
    ax.legend(title='CDU State', loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    
    # grid removed per project style
    sns.despine()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(base_dir, 'figs/chapter4_4/cdu_performance_bottleneck.pdf')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    plot_cdu_performance_bottleneck()
