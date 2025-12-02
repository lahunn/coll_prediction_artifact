#!/usr/bin/env python3
"""
Compare fall_cycle (total cycles) between OBB and Sphere collision models
across all difficulty levels (G1, G2, G3, G4, G5)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create figs directory if it doesn't exist
os.makedirs('figs', exist_ok=True)

# Read data
obb_df = pd.read_csv('../strategy_evaluation/result_files/obb_results.csv')
sphere_df = pd.read_csv('../strategy_evaluation/result_files/sphere_results.csv')

# Get data for all difficulty levels (assuming data is ordered G1-G5)
difficulty_levels = ['G1', 'G2', 'G3', 'G4', 'G5']
obb_cycles = obb_df['fall_cycle'].tolist()
sphere_cycles = sphere_df['fall_cycle'].tolist()

# Create grouped bar plot
x = np.arange(len(difficulty_levels))
width = 0.35

plt.figure(figsize=(12, 7))
obb_bars = plt.bar(x - width/2, obb_cycles, width, color='skyblue', alpha=0.8, label='OBB')
sphere_bars = plt.bar(x + width/2, sphere_cycles, width, color='lightcoral', alpha=0.8, label='Sphere')

# Add value labels
for bar, cycle in zip(obb_bars, obb_cycles):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
            f'{cycle:,}', ha='center', va='bottom', fontweight='bold', fontsize=8)

for bar, cycle in zip(sphere_bars, sphere_cycles):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
            f'{cycle:,}', ha='center', va='bottom', fontweight='bold', fontsize=8, color='darkred')

plt.title('Total Cycles Comparison: OBB vs Sphere Across Difficulty Levels', fontsize=14, fontweight='bold')
plt.ylabel('Total Cycles', fontsize=12)
plt.xlabel('Difficulty Level', fontsize=12)
plt.xticks(x, difficulty_levels)
plt.grid(axis='y', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('figs/cycle_comparison_obb_vs_sphere_difficulty_levels.png', dpi=300, bbox_inches='tight')
plt.close()

# Print results
print("Total Cycles Comparison:")
print("Level | OBB Cycles | Sphere Cycles | Difference (OBB - Sphere)")
print("-" * 65)
for level, obb_c, sphere_c in zip(difficulty_levels, obb_cycles, sphere_cycles):
    diff = obb_c - sphere_c
    print(f"{level:5} | {obb_c:10,} | {sphere_c:12,} | {diff:10,}")

print("\nPlot saved as: figs/cycle_comparison_obb_vs_sphere_difficulty_levels.png")