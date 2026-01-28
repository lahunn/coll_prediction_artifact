#!/usr/bin/env python3
"""
Analyze and plot collision rates at Edge, Pose, and Sphere levels across different difficulties.

Usage:
    python plot_collision_rates.py --data_folder <path_to_benchmarks> --basename <robot_basename>
    
    Example:
    python plot_collision_rates.py --data_folder ../../trace_files/scene_benchmarks/bit_collision_data --basename iiwa_7
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import simulation_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su

# --- Plotting Style ---
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.style.use("seaborn-v0_8-whitegrid")
font = {
    "family": "serif",
    "weight": "normal",
    "size": 14,
}
plt.rc("font", **font)

# Colors
EDGE_COLOR = "#D55E00"   # Vermilion
POSE_COLOR = "#0072B2"   # Blue
SPHERE_COLOR = "#009E73" # Bluish Green


def analyze_difficulty(data_folder, difficulty, basename, num_benchmarks):
    """
    Analyze collision rates for a specific difficulty level.
    """
    target_folder = os.path.join(data_folder, difficulty)
    if not os.path.exists(target_folder):
        print(f"Warning: Folder {target_folder} does not exist. Skipping.")
        return None

    total_edges = 0
    colliding_edges = 0
    
    total_poses = 0
    colliding_poses = 0
    
    total_spheres = 0
    colliding_spheres = 0

    # Iterate over benchmarks
    for benchid in tqdm(range(1, num_benchmarks + 1), desc=f"Analyzing {difficulty}", leave=False):
        try:
            # Load sphere-level data
            # We don't need coordinates (first return), only collision data (second return)
            # Use 'sphere' model type to get sphere-level collision details
            _, edge_coll_data = su.load_data(
                basename, benchid, target_folder, collision_model_type="sphere"
            )
            
            if edge_coll_data is None:
                continue

            for edge_poses in edge_coll_data:
                if not edge_poses:
                    continue
                
                total_edges += 1
                edge_has_collision = False
                
                for pose_spheres in edge_poses:
                    total_poses += 1
                    pose_has_collision = False
                    
                    # Check spheres in this pose
                    # 0 means collision, 1 means free
                    for sphere_res in pose_spheres:
                        total_spheres += 1
                        if sphere_res == 0:
                            colliding_spheres += 1
                            pose_has_collision = True
                    
                    if pose_has_collision:
                        colliding_poses += 1
                        edge_has_collision = True
                
                if edge_has_collision:
                    colliding_edges += 1

        except Exception as e:
            # print(f"Error processing {difficulty} bench {benchid}: {e}")
            continue

    # Calculate rates
    edge_rate = (colliding_edges / total_edges * 100) if total_edges > 0 else 0.0
    pose_rate = (colliding_poses / total_poses * 100) if total_poses > 0 else 0.0
    sphere_rate = (colliding_spheres / total_spheres * 100) if total_spheres > 0 else 0.0

    return {
        "Difficulty": difficulty,
        "Edge_Rate": edge_rate,
        "Pose_Rate": pose_rate,
        "Sphere_Rate": sphere_rate,
        "Total_Edges": total_edges,
        "Total_Poses": total_poses,
        "Total_Spheres": total_spheres
    }


def plot_rates(results, output_dir="figs"):
    """
    Plot the collision rates as a grouped bar chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    print("\nAnalysis Results:")
    print(df)
    
    csv_path = os.path.join(output_dir, "collision_rates_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Plotting
    difficulties = df["Difficulty"].tolist()
    edge_rates = df["Edge_Rate"].tolist()
    pose_rates = df["Pose_Rate"].tolist()
    sphere_rates = df["Sphere_Rate"].tolist()
    
    x = np.arange(len(difficulties))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, edge_rates, width, label='Edge Level', color=EDGE_COLOR)
    rects2 = ax.bar(x, pose_rates, width, label='Pose Level', color=POSE_COLOR)
    rects3 = ax.bar(x + width, sphere_rates, width, label='Sphere Level', color=SPHERE_COLOR)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Collision Rate (%)')
    ax.set_title('Collision Rates at Different Granularity Levels')
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.legend()
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "collision_rates_comparison.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Also plot a Log scale version for better visibility of Sphere rates
    fig_log, ax_log = plt.subplots(figsize=(10, 6))
    rects1_log = ax_log.bar(x - width, edge_rates, width, label='Edge Level', color=EDGE_COLOR)
    rects2_log = ax_log.bar(x, pose_rates, width, label='Pose Level', color=POSE_COLOR)
    rects3_log = ax_log.bar(x + width, sphere_rates, width, label='Sphere Level', color=SPHERE_COLOR)
    
    ax_log.set_xlabel('Difficulty Level')
    ax_log.set_ylabel('Collision Rate (%) - Log Scale')
    ax_log.set_title('Collision Rates (Log Scale)')
    ax_log.set_xticks(x)
    ax_log.set_xticklabels(difficulties)
    ax_log.legend()
    ax_log.set_yscale('log')
    ax_log.grid(axis='y', linestyle='--', alpha=0.7, which='both')
    
    plt.tight_layout()
    output_path_log = os.path.join(output_dir, "collision_rates_comparison_log.png")
    plt.savefig(output_path_log)
    print(f"Log-scale plot saved to {output_path_log}")


def main():
    parser = argparse.ArgumentParser(description="Analyze collision rates across Edge, Pose, and Sphere levels.")
    parser.add_argument("--data_folder", type=str, default="../../trace_files/scene_benchmarks/bit_collision_data",
                        help="Base folder containing difficulty subfolders (G1, G2, ...)")
    parser.add_argument("--basename", type=str, default="iiwa_7", help="Robot basename")
    parser.add_argument("--num_benchmarks", type=int, default=50, help="Number of benchmarks per difficulty")
    
    args = parser.parse_args()

    difficulties = ["G1", "G2", "G3", "G4", "G5"]
    results = []

    print(f"Analyzing collision rates for {args.basename}...")
    print(f"Data folder: {args.data_folder}")
    
    for diff in difficulties:
        res = analyze_difficulty(args.data_folder, diff, args.basename, args.num_benchmarks)
        if res:
            results.append(res)
            
    if results:
        plot_rates(results)
    else:
        print("No results found. Check data paths.")

if __name__ == "__main__":
    main()
