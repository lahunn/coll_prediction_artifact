#!/usr/bin/env python3
"""
Analyze collision data file and report number of poses per edge.

Usage:
  python analyze_collision_pose_counts.py --file path/to/file.pkl

Or provide dataset identifiers (uses existing loader conventions):
  python analyze_collision_pose_counts.py --basename iiwa_7 --benchid 1 --data_folder ../../trace_files/scene_benchmarks/bit_collision_data/G1 --type link

Outputs:
  - Prints per-edge pose counts and summary statistics (min, max, mean, median)
  - Optionally writes a CSV with per-edge counts via --out
"""

import argparse
import os
import pickle
import statistics
import csv
import sys

# For plotting

import matplotlib.pyplot as plt


# 兼容直接脚本运行的导入方式
try:
    from motion_planning_prediction.simulation_core.data_loader import (
        load_data,
        load_data_with_link_coords,
    )
except ImportError:
    # 尝试将项目根目录加入sys.path
    import sys as _sys
    import os as _os
    _proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../..'))
    if _proj_root not in _sys.path:
        _sys.path.insert(0, _proj_root)
    from motion_planning_prediction.simulation_core.data_loader import (
        load_data,
        load_data_with_link_coords,
    )


def analyze_from_data(link_coll_data):
    """Given link_coll_data (list of edges -> list of poses -> per-link labels),
    compute pose counts per edge and summary stats."""
    if link_coll_data is None:
        print("No data provided", file=sys.stderr)
        return None

    counts = [len(edge) for edge in link_coll_data]
    total_edges = len(counts)
    if total_edges == 0:
        print("No edges found in file.")
        return None

    stats = {
        "total_edges": total_edges,
        "min_poses": min(counts),
        "max_poses": max(counts),
        "mean_poses": statistics.mean(counts),
        "median_poses": statistics.median(counts),
    }
    return counts, stats


def load_by_file(file_path):
    # Try to infer format by filename pattern
    base = os.path.basename(file_path)
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to load pickle file '{file_path}': {e}", file=sys.stderr)
        return None, None

    # Common formats:
    # - (link_data, link_coll_data)
    # - (link_data, link_coords_data, link_coll_data) or (link_data, link_coords_data, link_coll_data, cycles)
    # - (collision_data, collision_flags)
    if isinstance(data, tuple):
        if len(data) == 2:
            # likely (collision_data, collision_flags)
            return data[1], data[0] if isinstance(data[1], list) else (data[0], data[1])
        elif len(data) == 3:
            # could be (link_data, link_coords_data, link_coll_data)
            # We assume last is coll flags
            return data[2], None
        elif len(data) >= 4:
            return data[2], None

    # If data is a dict or list assumed to be coll flags
    if isinstance(data, list):
        # assume it's link_coll_data
        return data, None

    print("Unrecognized pickle format; please pass file produced by collision data loader.", file=sys.stderr)
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Analyze collision data: poses per edge")
    parser.add_argument("--file", type=str, help="Path to a pickle collision data file")
    parser.add_argument("--basename", type=str, help="basename (iiwa_7)")
    parser.add_argument("--benchid", type=int, help="benchmark id")
    parser.add_argument("--data_folder", type=str, help="data folder path")
    parser.add_argument("--type", choices=["link", "sphere"], default="link", help="collision model type")
    parser.add_argument("--out", type=str, help="Optional output CSV path for per-edge counts")

    args = parser.parse_args()

    link_coll_data = None

    if args.file:
        link_coll_data, _ = load_by_file(args.file)
    else:
        if not (args.basename and args.benchid and args.data_folder):
            parser.error("Either --file or ( --basename --benchid --data_folder ) must be provided")
        # Use loader
        link_data, link_coll_data = load_data(
            args.basename, args.benchid, args.data_folder, collision_model_type=args.type
        )

    result = analyze_from_data(link_coll_data)
    if result is None:
        sys.exit(1)

    counts, stats = result

    print("Per-edge pose counts (edge_index: num_poses):")
    for i, c in enumerate(counts):
        print(f"{i}: {c}")

    print("\nSummary:")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Min poses: {stats['min_poses']}")
    print(f"Max poses: {stats['max_poses']}")
    print(f"Mean poses: {stats['mean_poses']:.2f}")
    print(f"Median poses: {stats['median_poses']}")

    if args.out:
        try:
            with open(args.out, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["edge_index", "num_poses"])
                for i, c in enumerate(counts):
                    writer.writerow([i, c])
            print(f"Wrote per-edge counts to {args.out}")
        except Exception as e:
            print(f"Failed to write CSV: {e}", file=sys.stderr)

    # Plotting
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(counts, bins=20, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of Pose Counts per Edge')
        ax.set_xlabel('Number of Poses per Edge')
        ax.set_ylabel('Edge Count')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Save to result_files directory
        result_dir = os.path.join(os.path.dirname(__file__), 'result_files')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        out_path = os.path.join(result_dir, 'pose_count_distribution.png')
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved histogram to {out_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Failed to plot/save histogram: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
