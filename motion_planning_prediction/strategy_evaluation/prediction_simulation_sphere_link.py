#!/usr/bin/env python3
"""
Sphere vs Link Coordinate Prediction Simulation

Evaluates two strategies for collision prediction:
1. Sphere collision detection, using sphere coordinates for prediction.
2. Sphere collision detection, using link coordinates for prediction.

Usage:
    python prediction_simulation_sphere_link.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <benchmarks> <robot_name> <collision_model_type> <num_oocds> <prediction_strategy>

    prediction_strategy: "sphere_coord" or "link_coord"
"""

import sys
import os
from tqdm import tqdm

# Add parent directory to path to import simulation_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su

# Add trace_generation directory to Python path
from trace_generation.core.robot.environment import RobotEnv
from trace_generation.core.collision.sphere_detector import SphereEnvGeometric
from common_simulation_utils import (
    print_final_statistics,
    setup_simulation,
    calculate_oracle_metrics,
    create_common_parser,
    parse_benchrange,
    aggregate_oracle_stats,
    initialize_statistics,
    DEFAULT_QUANT_BITS,
)

# --- Simulation Settings & Global Statistics ---
quant_bits = DEFAULT_QUANT_BITS
stats = initialize_statistics(extra_keys=["total_oocd_utilization", "total_edges"])

# --- Simulation Parameters from Command Line ---
parser = create_common_parser("Sphere vs Link Coordinate Prediction Simulation")
parser.add_argument(
    "prediction_strategy",
    type=str,
    choices=["sphere_coord", "link_coord"],
    help="Prediction strategy",
)
parser.add_argument(
    "num_dedicated_oocds",
    type=int,
    default=0,
    nargs="?",
    help="Number of dedicated OOCDs",
)

args = parser.parse_args()

threshold = args.threshold
sample_rate = args.sample_rate
qnoncoll_multiplier = args.qnoncoll_multiplier
data_folder = args.data_folder
basename = args.basename
benchmarks_arg = args.benchmarks
robot_name = args.robot_name
collision_model_type = args.collision_model_type
num_oocds = args.num_oocds
prediction_strategy = args.prediction_strategy
num_dedicated_oocds = args.num_dedicated_oocds

# Use setup_simulation to get parameters
# We pass "sphere" to get sphere parameters as this script always compares sphere collision strategies
bins, num_elements, check_cost, _, _ = setup_simulation(
    robot_name, quant_bits, "sphere", qnoncoll_multiplier
)

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

# Specific qnoncoll_len calculation for this script
qnoncoll_len = qnoncoll_multiplier * num_oocds


print(f"=== Sphere Collision Prediction Simulation ({prediction_strategy}) ===")
print(f"Threshold: {threshold}")
print(f"Sample Rate: {sample_rate}")
print(f"Queue Multiplier: {qnoncoll_multiplier}")
print(f"Non-Coll Queue Len: {qnoncoll_len}")
print(f"Num OOCDs: {num_oocds}")
print(f"Data Folder (base): {data_folder}")
print(f"基准测试: {benchmarks_arg}")
print(f"Strategy: {prediction_strategy}")
print("=" * 50)

# --- Benchmark Range ---
benchrange = parse_benchrange(benchmarks_arg)

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

    # --- Oracle Metrics Calculation ---
    oracle_stats = aggregate_oracle_stats(
        edge_link_coll_data, num_elements, num_oocds, check_cost
    )

    stats["total_checks"] += oracle_stats["total_checks"]
    all_oracle = oracle_stats["total_oracle_queries"]
    stats["theoretical_min_cycles"] += oracle_stats["total_oracle_cycles"]
    stats["total_oracle_coll_cycles"] += oracle_stats["total_oracle_coll_cycles"]
    stats["total_oracle_noncoll_cycles"] += oracle_stats["total_oracle_noncoll_cycles"]

    # Process each edge
    # If link_coord strategy, we iterate over edge_link_coords_data as well
    if prediction_strategy == "link_coord":
        assert edge_link_coords_data is not None
        iterator = zip(edge_link_coords_data, edge_link_coll_data)
    else:
        iterator = zip(edge_link_data, edge_link_coll_data)

    for edge_coords, edge_coll in iterator:
        if not edge_coll:
            continue

        # --- CSP Rearrangement ---
        # edge_coords is either sphere coords or link coords depending on strategy
        linklist, linklist_coll = su.csp_rearrange(edge_coords, edge_coll, groupsize=4)

        # Determine simulation mode: link_coord strategy uses hierarchical dispatch,
        # while sphere_coord uses batch expansion.
        sim_mode = "hierarchical" if prediction_strategy == "link_coord" else "batch"

        edge_query_count, colldict, coll_found, cycle, oocd_utilization = (
            su.simulate_parallel_collision_detection(
                linklist,
                linklist_coll,
                colldict,
                threshold,
                sample_rate,
                bins,
                qnoncoll_len=qnoncoll_len,
                cycle_check=check_cost,
                num_oocds=num_oocds,
                mode=sim_mode,
                link_to_spheres=link_to_spheres,
                sphere_to_link=sphere_to_link,
                num_spheres_per_pose=num_spheres_per_pose,
                num_dedicated_oocds=num_dedicated_oocds,
            )
        )

        stats["total_oocd_utilization"] += oocd_utilization
        stats["total_edges"] += 1

        if coll_found:
            stats["total_pred_coll_cycles"] += cycle
        else:
            stats["total_pred_noncoll_cycles"] += cycle

        all_cycle += cycle
        all_prediction += edge_query_count

    stats["fall_oracle"] += all_oracle
    stats["fall_prediction"] += all_prediction
    stats["fall_cycle"] += all_cycle

    if (benchid) % 10 == 0:
        print(
            f"[{benchid}/{benchrange[-1]}] Pred Queries: {all_prediction:.2f}, Oracle Queries: {all_oracle}"
        )

avg_oocd_utilization = stats["total_oocd_utilization"] / stats["total_edges"]
print_final_statistics(
    total_checks=stats["total_checks"],
    fall_prediction=stats["fall_prediction"],
    fall_oracle=stats["fall_oracle"],
    fall_cycle=stats["fall_cycle"],
    theoretical_min_cycles=stats["theoretical_min_cycles"],
    total_pred_coll_cycles=stats["total_pred_coll_cycles"],
    total_pred_noncoll_cycles=stats["total_pred_noncoll_cycles"],
    total_oracle_coll_cycles=stats["total_oracle_coll_cycles"],
    total_oracle_noncoll_cycles=stats["total_oracle_noncoll_cycles"],
    oocd_utilization=avg_oocd_utilization,
)

print("=" * 50)
