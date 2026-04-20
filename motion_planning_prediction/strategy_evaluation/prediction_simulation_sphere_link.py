#!/usr/bin/env python3
"""
Sphere vs Link Coordinate Prediction Simulation

Evaluates two strategies for collision prediction:
1. Sphere collision detection, using sphere coordinates for prediction.
2. Sphere collision detection, using link coordinates for prediction.

Usage:
    python prediction_simulation_sphere_link.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <start_bench> <end_bench> <robot_name> <prediction_strategy> <num_oocds> [--cht-warmstart-dir <dir>]

    prediction_strategy: "sphere_coord" or "link_coord"
    num_oocds: number of parallel OOCDs used in simulation
"""

import sys
import os
import pickle
from tqdm import tqdm

# Add parent directory to path to import simulation_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su

# Add trace_generation directory to Python path
from trace_generation.config.ana_parameters import get_robot_params
from trace_generation.core.robot.environment import RobotEnv
from trace_generation.core.collision.sphere_detector import SphereEnvGeometric
from common_simulation_utils import get_bins, print_final_statistics

# --- Simulation Settings ---
quant_bits = 4
# --- Global Statistics ---
fall_prediction = 0
fall_oracle = 0
total_checks = 0
fall_cycle = 0
theoretical_min_cycles = 0
total_pred_coll_cycles = 0
total_pred_noncoll_cycles = 0
total_oracle_coll_cycles = 0
total_oracle_noncoll_cycles = 0
total_oocd_utilization = 0.0
total_edges = 0
total_dead_cycles = 0
total_dead_ratio_sum = 0.0
total_dead_edges = 0

# --- Simulation Parameters from Command Line ---

if len(sys.argv) < 11:
    print(
        "Usage: python prediction_simulation_sphere_link.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <start_bench> <end_bench> <robot_name> <prediction_strategy> <num_oocds> [--cht-warmstart-dir <dir>]"
    )
    sys.exit(1)

threshold = float(sys.argv[1])
sample_rate = float(sys.argv[2])
qnoncoll_multiplier = int(sys.argv[3])
data_folder = sys.argv[4]
basename = sys.argv[5]
start_bench = int(sys.argv[6])
end_bench = int(sys.argv[7])
robot_name = sys.argv[8]
prediction_strategy = sys.argv[9]
num_oocds = int(sys.argv[10])

# Parse optional warm-start directory
warmstart_dir = None
if "--cht-warmstart-dir" in sys.argv:
    ws_idx = sys.argv.index("--cht-warmstart-dir")
    if ws_idx + 1 < len(sys.argv):
        warmstart_dir = sys.argv[ws_idx + 1]

if prediction_strategy not in ["sphere_coord", "link_coord"]:
    print("Error: prediction_strategy must be 'sphere_coord' or 'link_coord'")
    sys.exit(1)

# Get robot parameters
robot_params = get_robot_params(robot_name)

# Create temporary environment to get sphere-link mapping
temp_env = RobotEnv(robot_name, OBB_GUI=False, enable_self_collision=False)
temp_sphere_env = SphereEnvGeometric(robot_env=temp_env, robot_name=robot_name)
temp_sphere_env._initialize_sphere_metadata()

sphere_link_ids = temp_sphere_env.sphere_link_ids
link_to_spheres = {}
sphere_to_link = []
for idx, link_id in enumerate(sphere_link_ids):
    lid = int(link_id)
    link_to_spheres.setdefault(lid, []).append(idx)
    sphere_to_link.append(lid)
num_spheres_per_pose = len(sphere_link_ids)

# Clean up temporary environments
temp_sphere_env.close()
temp_env.close()

# Calculate bins
bins = su.calculate_bins_from_workspace(robot_name, quant_bits)

# Always use sphere parameters
num_elements = robot_params["sphere_num"]
check_cost = robot_params["sphere_cost"]
qnoncoll_len = qnoncoll_multiplier * num_oocds

print(f"=== Sphere Collision Prediction Simulation ({prediction_strategy}) ===")
print(f"Threshold: {threshold}, Sample Rate: {sample_rate}")
print(f"Queue Multiplier: {qnoncoll_multiplier}, Non-Coll Queue Len: {qnoncoll_len}")
print(f"Num OOCDs: {num_oocds}, Data Folder: {data_folder}")
if warmstart_dir:
    print(f"Warm-start directory: {warmstart_dir}")
print("=" * 50)

# --- Benchmark Range ---
benchrange = range(start_bench, end_bench + 1)

# --- Main Simulation Loop ---
for benchid in tqdm(benchrange, desc="Processing Benchmarks"):
    all_prediction = 0
    all_oracle = 0
    all_cycle = 0
    
    # Initialize colldict (Empty or Warm-start)
    colldict = {}
    if warmstart_dir:
        warmstart_filename = f"{basename}_{benchid:04d}_warmstart.pkl"
        warmstart_path = os.path.join(warmstart_dir, warmstart_filename)
        if os.path.exists(warmstart_path):
            try:
                with open(warmstart_path, "rb") as f:
                    package = pickle.load(f)
                    colldict = package.get("memory", package)
            except Exception as e:
                print(f"  [Warning] Failed to load warm-start: {e}")

    # Load collision data
    edge_link_data, edge_link_coll_data, edge_link_coords_data = (
        su.load_data_with_link_coords(basename, benchid, data_folder)
    )

    if edge_link_data is None or edge_link_coll_data is None:
        continue

    if prediction_strategy == "link_coord" and edge_link_coords_data is None:
        print(f"Error: Link coordinates not found for benchmark {benchid}.")
        continue

    if prediction_strategy == "link_coord":
        iterator = zip(edge_link_coords_data, edge_link_coll_data)
    else:
        iterator = zip(edge_link_data, edge_link_coll_data)

    for edge_coords, edge_coll in iterator:
        if not edge_coll:
            continue

        for pose_coll in edge_coll:
            try:
                first_collision_index = pose_coll.index(0)
                total_checks += first_collision_index + 1
            except ValueError:
                total_checks += len(pose_coll)

        # Oracle Calculation
        coll_found_oracle = any(link_coll == 0 for pose_coll in edge_coll for link_coll in pose_coll)
        if coll_found_oracle:
            all_oracle += 1
        else:
            all_oracle += num_elements * len(edge_coll)

        oracle_edge_cycles = su.calculate_oracle_cycles(edge_coll, num_oocds, check_cost)
        theoretical_min_cycles += oracle_edge_cycles

        if coll_found_oracle:
            total_oracle_coll_cycles += oracle_edge_cycles
        else:
            total_oracle_noncoll_cycles += oracle_edge_cycles

        # CSP Rearrangement
        linklist, linklist_coll = su.csp_rearrange(edge_coords, edge_coll, groupsize=4)

        if prediction_strategy == "link_coord":
            (edge_query_count, colldict, coll_found, cycle, oocd_utilization, deadtime_stats) = su.simulate_parallel_collision_detection_link(
                linklist, linklist_coll, colldict, threshold, sample_rate, bins,
                link_to_spheres, sphere_to_link, num_spheres_per_pose,
                qnoncoll_len=qnoncoll_len, cycle_check=check_cost, num_oocds=num_oocds, collect_deadtime=True
            )
        else:
            (edge_query_count, colldict, coll_found, cycle, oocd_utilization, deadtime_stats) = su.simulate_parallel_collision_detection_sphere(
                linklist, linklist_coll, colldict, threshold, sample_rate, bins,
                link_to_spheres, sphere_to_link, num_spheres_per_pose,
                qnoncoll_len=qnoncoll_len * 8, cycle_check=check_cost, num_oocds=num_oocds, collect_deadtime=True
            )

        total_oocd_utilization += oocd_utilization
        total_edges += 1
        total_dead_cycles += deadtime_stats["dead_cycles"]
        total_dead_ratio_sum += deadtime_stats["dead_ratio"]
        total_dead_edges += 1

        if coll_found:
            total_pred_coll_cycles += cycle
        else:
            total_pred_noncoll_cycles += cycle

        all_cycle += cycle
        all_prediction += edge_query_count

    fall_oracle += all_oracle
    fall_prediction += all_prediction
    fall_cycle += all_cycle

    if (benchid) % 10 == 0:
        print(f"[{benchid}/{end_bench}] Pred Queries: {all_prediction:.2f}, Oracle Queries: {all_oracle}")

avg_oocd_utilization = total_oocd_utilization / total_edges if total_edges > 0 else 0.0
avg_dead_cycles_per_edge = total_dead_cycles / total_dead_edges if total_dead_edges > 0 else 0.0
avg_dead_ratio_per_edge = total_dead_ratio_sum / total_dead_edges if total_dead_edges > 0 else 0.0
dead_cycle_ratio_total = (total_dead_cycles / fall_cycle) if fall_cycle > 0 else 0.0
total_naive_cycles = (total_checks * check_cost) / num_oocds

print_final_statistics(
    total_checks=total_checks, fall_prediction=fall_prediction, fall_oracle=fall_oracle,
    total_pred_coll_cycles=total_pred_coll_cycles, total_pred_noncoll_cycles=total_pred_noncoll_cycles,
    total_oracle_coll_cycles=total_oracle_coll_cycles, total_oracle_noncoll_cycles=total_oracle_noncoll_cycles,
    extra_stats={
        "Avg OOCD Utilization": f"{avg_oocd_utilization:.4f}",
        "Dead Time Total Cycles": f"{total_dead_cycles}",
        "Dead Time Avg Cycles Per Edge": f"{avg_dead_cycles_per_edge:.4f}",
        "Dead Time Ratio (Total Dead / Total Pred Cycles)": f"{dead_cycle_ratio_total * 100:.2f}%",
        "Dead Time Avg Ratio Per Edge": f"{avg_dead_ratio_per_edge * 100:.2f}%",
    },
)

print(f"\n  Total Cycles (Prediction): {fall_cycle}")
print(f"  Total Cycles (Oracle): {theoretical_min_cycles}")
print(f"  Total Cycles (Naive): {total_naive_cycles}")
if fall_cycle > 0:
    print(f"  Cycle Efficiency: {(theoretical_min_cycles / fall_cycle) * 100:.2f}%")
print(f"\n  Average OOCD Utilization: {avg_oocd_utilization * 100:.2f}%")
print("=" * 50)
