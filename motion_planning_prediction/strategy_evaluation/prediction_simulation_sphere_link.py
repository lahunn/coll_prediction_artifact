#!/usr/bin/env python3
"""
Sphere vs Link Coordinate Prediction Simulation

Evaluates two strategies for collision prediction:
1. Sphere collision detection, using sphere coordinates for prediction.
2. Sphere collision detection, using link coordinates for prediction.

Usage:
    python prediction_simulation_sphere_link.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <start_bench> <end_bench> <robot_name> <prediction_strategy> <algorithm> <num_oocds>

    prediction_strategy: "sphere_coord" or "link_coord"
    algorithm: subdirectory name under each benchmark folder (e.g. 'bit_star')
    num_oocds: number of parallel OOCDs used in simulation
"""

import sys
import os
from tqdm import tqdm

# Add parent directory to path to import simulation_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su

# Add trace_generation directory to Python path
from trace_generation.config.ana_parameters import get_robot_params
from trace_generation.core.robot.environment import RobotEnv
from trace_generation.core.collision.sphere_detector import SphereEnvGeometric


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

# --- Simulation Parameters from Command Line ---

if len(sys.argv) < 10:
    print(
        "Usage: python prediction_simulation_sphere_link.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <start_bench> <end_bench> <robot_name> <prediction_strategy> <num_oocds>"
    )
    print(
        "Example: python prediction_simulation_sphere_link.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 1 10 iiwa sphere_coord 7"
    )
    sys.exit(1)

threshold = float(sys.argv[1])
sample_rate = float(sys.argv[2])
qnoncoll_multiplier = int(sys.argv[3])
# data_folder should point to the base folder containing per-benchmark folders (e.g. .../bit_collision_data/G1)
data_folder = sys.argv[4]
basename = sys.argv[5]
start_bench = int(sys.argv[6])
end_bench = int(sys.argv[7])
robot_name = sys.argv[8]
prediction_strategy = sys.argv[9]
num_oocds = int(sys.argv[10])

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

# Always use sphere parameters since we are doing sphere collision detection
num_elements = robot_params["sphere_num"]
check_cost = robot_params["sphere_cost"]

qnoncoll_len = qnoncoll_multiplier * num_oocds


print(f"=== Sphere Collision Prediction Simulation ({prediction_strategy}) ===")
print(f"Threshold: {threshold}")
print(f"Sample Rate: {sample_rate}")
print(f"Queue Multiplier: {qnoncoll_multiplier}")
print(f"Non-Coll Queue Len: {qnoncoll_len}")
print(f"Num OOCDs: {num_oocds}")
print(f"Data Folder (base): {data_folder}")
print(f"Benchmarks: {start_bench} - {end_bench}")
print(f"Strategy: {prediction_strategy}")
print("=" * 50)

# --- Benchmark Range ---
benchrange = range(start_bench, end_bench + 1)

# --- Main Simulation Loop ---
for benchid in tqdm(benchrange, desc="Processing Benchmarks"):
    all_prediction = 0
    all_oracle = 0
    all_cycle = 0
    colldict = {}

    # Load collision data with link coords
    edge_link_data, edge_link_coll_data, edge_link_coords_data = (
        su.load_data_with_link_coords(
            basename,
            benchid,
            data_folder,
        )
    )

    if edge_link_data is None or edge_link_coll_data is None:
        continue

    if prediction_strategy == "link_coord" and edge_link_coords_data is None:
        print(
            f"Error: Link coordinates not found for benchmark {benchid}. Make sure data was generated with link coords."
        )
        continue

    # Process each edge
    # If link_coord strategy, we iterate over edge_link_coords_data as well
    # But zip only works if all iterables are same length.
    # If edge_link_coords_data is None (e.g. sphere_coord strategy and load failed to get it but we don't care), we shouldn't zip it.

    if prediction_strategy == "link_coord":
        assert edge_link_coords_data is not None
        iterator = zip(edge_link_coords_data, edge_link_coll_data)
    else:
        iterator = zip(edge_link_data, edge_link_coll_data)

    for edge_coords, edge_coll in iterator:
        if not edge_coll:
            continue

        # Accumulate total checks
        for pose_coll in edge_coll:
            try:
                first_collision_index = pose_coll.index(0)
                total_checks += first_collision_index + 1
            except ValueError:
                total_checks += len(pose_coll)

        # --- Oracle Calculation ---
        coll_found_oracle = any(
            link_coll == 0 for pose_coll in edge_coll for link_coll in pose_coll
        )
        if coll_found_oracle:
            all_oracle += 1
        else:
            all_oracle += num_elements * len(edge_coll)

        oracle_edge_cycles = su.calculate_oracle_cycles(
            edge_coll, num_oocds, check_cost
        )

        theoretical_min_cycles += oracle_edge_cycles

        if coll_found_oracle:
            total_oracle_coll_cycles += oracle_edge_cycles
        else:
            total_oracle_noncoll_cycles += oracle_edge_cycles

        # --- CSP Rearrangement ---
        # edge_coords is either sphere coords or link coords depending on strategy
        linklist, linklist_coll = su.csp_rearrange(edge_coords, edge_coll, groupsize=4)

        if prediction_strategy == "link_coord":
            edge_query_count, colldict, coll_found, cycle, oocd_utilization = (
                su.simulate_parallel_collision_detection_link(
                    linklist,
                    linklist_coll,
                    colldict,
                    threshold,
                    sample_rate,
                    bins,
                    link_to_spheres,
                    sphere_to_link,
                    num_spheres_per_pose,
                    qnoncoll_len=qnoncoll_len * 4,
                    cycle_check=check_cost,
                    num_oocds=num_oocds,
                )
            )
        else:
            edge_query_count, colldict, coll_found, cycle, oocd_utilization = (
                su.simulate_parallel_collision_detection_sphere(
                    linklist,
                    linklist_coll,
                    colldict,
                    threshold,
                    sample_rate,
                    bins,
                    link_to_spheres,
                    sphere_to_link,
                    num_spheres_per_pose,
                    qnoncoll_len=qnoncoll_len,
                    cycle_check=check_cost,
                    num_oocds=num_oocds,
                )
            )
            # edge_query_count, colldict, coll_found, cycle, oocd_utilization = (
            #     su.simulate_parallel_collision_detection(
            #         linklist,
            #         linklist_coll,
            #         colldict,
            #         threshold,
            #         sample_rate,
            #         bins,
            #         qnoncoll_len=qnoncoll_len,
            #         cycle_check=check_cost,
            #         num_oocds=num_oocds,
            #     )
            # )
        total_oocd_utilization += oocd_utilization
        total_edges += 1

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
        print(
            f"[{benchid}/{end_bench}] Pred Queries: {all_prediction:.2f}, Oracle Queries: {all_oracle}"
        )

avg_oocd_utilization = total_oocd_utilization / total_edges
print_final_statistics(
    total_checks=total_checks,
    fall_prediction=fall_prediction,
    fall_oracle=fall_oracle,
    total_pred_coll_cycles=total_pred_coll_cycles,
    total_pred_noncoll_cycles=total_pred_noncoll_cycles,
    total_oracle_coll_cycles=total_oracle_coll_cycles,
    total_oracle_noncoll_cycles=total_oracle_noncoll_cycles,
    extra_stats={"Avg OOCD Utilization": f"{avg_oocd_utilization:.4f}"}
)

print(f"\n  Total Cycles (Prediction): {fall_cycle}")
print(f"  Total Cycles (Oracle): {theoretical_min_cycles}")
if fall_cycle > 0:
    print(f"  Cycle Efficiency: {(theoretical_min_cycles / fall_cycle) * 100:.2f}%")
else:
    print("  Cycle Efficiency: N/A")

print(f"\n  Prediction Coll Edge Cycles: {total_pred_coll_cycles}")
print(f"  Prediction Non-Coll Edge Cycles: {total_pred_noncoll_cycles}")
print(f"  Oracle Coll Edge Cycles: {total_oracle_coll_cycles}")
print(f"  Oracle Non-Coll Edge Cycles: {total_oracle_noncoll_cycles}")
print(f"\n  Average OOCD Utilization: {avg_oocd_utilization * 100:.2f}%")


print("=" * 50)
