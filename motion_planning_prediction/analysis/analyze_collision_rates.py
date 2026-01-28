#!/usr/bin/env python3
"""
Analyze collision rates at Edge, Pose, and Sphere levels.

This script loads collision data benchmarks and calculates the collision rate
at three different granularities:
1. Edge Level: Fraction of edges that are invalid (contain at least one colliding pose).
2. Pose Level: Fraction of poses that are invalid (contain at least one colliding sphere).
3. Sphere Level: Fraction of individual sphere checks that result in a collision.

Usage:
    python analyze_collision_rates.py --basename <name> --data_folder <path> --start <N> --end <M> [--out <file.csv>]
    python analyze_collision_rates.py --basename iiwa_7 --data_folder ../../trace_files/scene_benchmarks/bit_collision_data/G5 --start 1 --end 10
"""

import argparse
import sys
import os
import csv
from tqdm import tqdm

# Add parent directory to path to import simulation_utils
# Assumes this script is located in motion_planning_prediction/analysis/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

try:
    import simulation_utils as su
except ImportError:
    # Try adding the project root if running from elsewhere
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    try:
        import motion_planning_prediction.simulation_utils as su
    except ImportError:
         print("Could not import simulation_utils. Please check python path.", file=sys.stderr)
         sys.exit(1)

def analyze_benchmarks(basename, data_folder, start_bench, end_bench, output_csv=None):
    # Global counters
    total_edges = 0
    colliding_edges = 0
    
    total_poses = 0
    colliding_poses = 0
    
    total_spheres = 0
    colliding_spheres = 0

    results = []

    bench_range = range(start_bench, end_bench + 1)
    
    print(f"Analyzing collision rates for {basename}")
    print(f"Data folder: {data_folder}")
    print(f"Benchmarks: {start_bench} to {end_bench}")
    print("-" * 60)

    for benchid in tqdm(bench_range, desc="Processing Benchmarks"):
        # Load data (using sphere model to get sphere-level details)
        # load_data returns (link_data, link_coll_data)
        # link_coll_data is [edge_index][pose_index][sphere_index] -> 0 (coll) or 1 (free)
        try:
            _, edge_coll_data = su.load_data(
                basename, 
                benchid, 
                data_folder, 
                collision_model_type='sphere'
            )
        except Exception as e:
            print(f"Error loading benchmark {benchid}: {e}", file=sys.stderr)
            continue

        if edge_coll_data is None:
            # print(f"Benchmark {benchid} data not found or empty.")
            continue

        # Per-benchmark counters
        b_edges = 0
        b_coll_edges = 0
        b_poses = 0
        b_coll_poses = 0
        b_spheres = 0
        b_coll_spheres = 0

        for edge_poses in edge_coll_data:
            b_edges += 1
            edge_is_colliding = False
            
            for pose_spheres in edge_poses:
                b_poses += 1
                pose_is_colliding = False
                
                # Check spheres in this pose
                # Convention: 0 is collision, non-zero is free (usually 1)
                # We can iterate and count
                for s_val in pose_spheres:
                    b_spheres += 1
                    if s_val == 0:
                        b_coll_spheres += 1
                        pose_is_colliding = True
                
                if pose_is_colliding:
                    b_coll_poses += 1
                    edge_is_colliding = True
            
            if edge_is_colliding:
                b_coll_edges += 1

        # Accumulate to global
        total_edges += b_edges
        colliding_edges += b_coll_edges
        total_poses += b_poses
        colliding_poses += b_coll_poses
        total_spheres += b_spheres
        colliding_spheres += b_coll_spheres

        # Record benchmark result
        results.append({
            "bench_id": benchid,
            "total_edges": b_edges,
            "coll_edges": b_coll_edges,
            "edge_coll_rate": b_coll_edges / b_edges if b_edges > 0 else 0,
            "total_poses": b_poses,
            "coll_poses": b_coll_poses,
            "pose_coll_rate": b_coll_poses / b_poses if b_poses > 0 else 0,
            "total_spheres": b_spheres,
            "coll_spheres": b_coll_spheres,
            "sphere_coll_rate": b_coll_spheres / b_spheres if b_spheres > 0 else 0
        })

    # Summary
    print("-" * 60)
    print("SUMMARY STATISTICS")
    print("-" * 60)
    
    print(f"Total Benchmarks Processed: {len(results)}")
    
    print(f"\n[Edge Level]")
    print(f"  Total Edges:      {total_edges}")
    print(f"  Colliding Edges:  {colliding_edges}")
    edge_rate = (colliding_edges / total_edges * 100) if total_edges > 0 else 0
    print(f"  Collision Rate:   {edge_rate:.4f}%")
    
    print(f"\n[Pose Level]")
    print(f"  Total Poses:      {total_poses}")
    print(f"  Colliding Poses:  {colliding_poses}")
    pose_rate = (colliding_poses / total_poses * 100) if total_poses > 0 else 0
    print(f"  Collision Rate:   {pose_rate:.4f}%")
    
    print(f"\n[Sphere Level]")
    print(f"  Total Spheres:    {total_spheres}")
    print(f"  Colliding Spheres:{colliding_spheres}")
    sphere_rate = (colliding_spheres / total_spheres * 100) if total_spheres > 0 else 0
    print(f"  Collision Rate:   {sphere_rate:.6f}%") # Higher precision for sphere level as it can be low
    print("-" * 60)

    # Write CSV if requested
    if output_csv:
        try:
            with open(output_csv, 'w', newline='') as f:
                fieldnames = [
                    "bench_id", 
                    "total_edges", "coll_edges", "edge_coll_rate",
                    "total_poses", "coll_poses", "pose_coll_rate",
                    "total_spheres", "coll_spheres", "sphere_coll_rate"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
                
                # Write a summary row at the bottom? Or maybe better to keep it clean.
                # Let's write a summary row with bench_id = 'Total'
                writer.writerow({
                    "bench_id": "Total",
                    "total_edges": total_edges,
                    "coll_edges": colliding_edges,
                    "edge_coll_rate": colliding_edges / total_edges if total_edges > 0 else 0,
                    "total_poses": total_poses,
                    "coll_poses": colliding_poses,
                    "pose_coll_rate": colliding_poses / total_poses if total_poses > 0 else 0,
                    "total_spheres": total_spheres,
                    "coll_spheres": colliding_spheres,
                    "sphere_coll_rate": colliding_spheres / total_spheres if total_spheres > 0 else 0
                })
            print(f"Detailed results written to {output_csv}")
        except Exception as e:
            print(f"Failed to write CSV: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Analyze collision rates (Edge/Pose/Sphere).")
    parser.add_argument("--basename", type=str, required=True, help="Robot basename (e.g., iiwa_7)")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to data folder containing benchmark files")
    parser.add_argument("--start", type=int, required=True, help="Start benchmark ID")
    parser.add_argument("--end", type=int, required=True, help="End benchmark ID")
    parser.add_argument("--out", type=str, help="Optional output CSV file")

    args = parser.parse_args()

    analyze_benchmarks(
        args.basename,
        args.data_folder,
        args.start,
        args.end,
        args.out
    )

if __name__ == "__main__":
    main()
