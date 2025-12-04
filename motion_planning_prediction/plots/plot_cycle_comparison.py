#!/usr/bin/env python3
"""
Compare fall_cycle (total cycles) between LINK and Sphere collision models
across all difficulty levels (G1, G2, G3, G4, G5)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create figs directory if it doesn't exist
os.makedirs("figs", exist_ok=True)

# Read data
link_df = pd.read_csv("../result_files/link_results.csv")
sphere_df = pd.read_csv("../result_files/sphere_results.csv")

# Get data for all difficulty levels (assuming data is ordered G1-G5)
difficulty_levels = ["G1", "G2", "G3", "G4", "G5"]
link_cycles = link_df["Total_Cycles"].tolist()
sphere_cycles = sphere_df["Total_Cycles"].tolist()

# Create grouped bar plot
x = np.arange(len(difficulty_levels))
width = 0.35

plt.figure(figsize=(12, 7))
link_bars = plt.bar(
    x - width / 2, link_cycles, width, color="skyblue", alpha=0.8, label="LINK"
)
sphere_bars = plt.bar(
    x + width / 2, sphere_cycles, width, color="lightcoral", alpha=0.8, label="Sphere"
)

# Add value labels
for bar, cycle in zip(link_bars, link_cycles):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10000,
        f"{cycle:,}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=8,
    )

for bar, cycle in zip(sphere_bars, sphere_cycles):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10000,
        f"{cycle:,}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=8,
        color="darkred",
    )

# Add percentage difference labels above the bars
max_height = max(max(link_cycles), max(sphere_cycles))
for i, (link_c, sphere_c) in enumerate(zip(link_cycles, sphere_cycles)):
    if link_c > 0:
        pct_diff = ((link_c - sphere_c) / link_c) * 100
        y_position = max(link_c, sphere_c) + max_height * 0.05
        
        # Color based on which is higher
        color = "green" if sphere_c < link_c else "red"
        sign = "+" if sphere_c > link_c else ""
        
        plt.text(
            i,
            y_position,
            f"{sign}{pct_diff:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.7)
        )

plt.title(
    "Total Cycles Comparison: LINK vs Sphere Across Difficulty Levels",
    fontsize=14,
    fontweight="bold",
)
plt.ylabel("Total Cycles", fontsize=12)
plt.xlabel("Difficulty Level", fontsize=12)
plt.xticks(x, difficulty_levels)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(
    "figs/cycle_comparison_link_vs_sphere.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Print results
print("Total Cycles Comparison:")
print("Level | LINK Cycles | Sphere Cycles | Difference | Percentage")
print("-" * 75)
for level, link_c, sphere_c in zip(difficulty_levels, link_cycles, sphere_cycles):
    diff = link_c - sphere_c
    pct_diff = ((link_c - sphere_c) / link_c * 100) if link_c > 0 else 0
    status = "Sphere faster" if sphere_c < link_c else "LINK faster"
    print(f"{level:5} | {link_c:10,} | {sphere_c:12,} | {diff:10,} | {pct_diff:6.1f}% ({status})")

print("\nPlot saved as: figs/cycle_comparison_link_vs_sphere.png")
