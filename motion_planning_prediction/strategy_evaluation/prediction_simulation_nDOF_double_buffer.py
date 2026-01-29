#!/usr/bin/env python3
"""
双缓冲架构碰撞检测预测仿真程序（nDOF机器人）

使用双缓冲架构进行碰撞检测的预测策略评估：
- Bank A 和 Bank B 两组队列
- 当前 edge 完成时切换 Bank
- 两个预测器同时工作，为当前和下一个 edge 生成任务

数据格式:
- sphere_link_data[edge][pose][sphere] = [x, y, z, radius]
- sphere_link_coll_data[edge][pose][sphere] = 1 or 0

文件命名约定: {basename}_{benchid:04d}_{collision_model_type}.pkl
其中 collision_model_type 为 'link' 或 'sphere'
"""

import sys
import os
import numpy as np

from common_simulation_utils import (
    print_final_statistics,
    setup_simulation,
    calculate_oracle_metrics,
    parse_benchrange,
    create_common_parser,
    aggregate_oracle_stats,
    initialize_statistics,
    DEFAULT_QUANT_BITS,
)
from tqdm import tqdm
from collections import Counter

# 添加上级目录到path以导入simulation_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su

# 添加 trace_generation 目录到 Python 路径
from trace_generation.core.robot.environment import RobotEnv
from trace_generation.core.collision.sphere_detector import SphereEnvGeometric

# --- Simulation Settings ---
quant_bits = DEFAULT_QUANT_BITS

# --- Simulation Parameters from Command Line ---
parser = create_common_parser("双缓冲架构碰撞检测预测仿真程序（nDOF机器人）")
parser.add_argument(
    "num_predictions", type=int, default=2, nargs="?", help="Number of predictions"
)
parser.add_argument(
    "num_dedicated_oocds",
    type=int,
    default=1,
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
num_predictions = args.num_predictions
num_dedicated_oocds = args.num_dedicated_oocds

# 使用通用工具解析benchmark范围
benchrange = parse_benchrange(benchmarks_arg)
num_benchmarks = len(benchrange)

# 使用通用工具设置仿真参数
bins, num_elements, check_cost, qnoncoll_len, print_title = setup_simulation(
    robot_name, quant_bits, collision_model_type, qnoncoll_multiplier
)

# --- Global Statistics ---
stats = initialize_statistics(
    extra_keys=[
        "total_prediction_queries",
        "total_cdu_idle_cycles",
        "total_coll_edges",
        "total_noncoll_edges",
        "total_noncoll_edge_queries",
    ]
)

# Create temporary environment to get sphere-link mapping
# Must do this before main loop to set up global mappings
print("Initializing robot environment to extract sphere-link mapping...")
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
print(f"Mapping extracted. Total spheres: {num_spheres_per_pose}")


print(print_title)
print(f"Threshold: {threshold}")
print(f"Sample Rate: {sample_rate}")
print(f"Queue Length Multiplier: {qnoncoll_multiplier}")
print(f"Non-collision Queue Length: {qnoncoll_len}")
print(f"Data Folder: {data_folder}")
print(f"基准测试: {benchmarks_arg}")
print(f"Robot: {robot_name}")
print(f"Collision Model: {collision_model_type}")
print(f"OOCD数量: {num_oocds}")
print(f"Number of Predictions: {num_predictions}")
print(f"Dedicated OOCDs: {num_dedicated_oocds}")
print("Architecture: Double Buffer (Bank A + Bank B)")
print("=" * 50)

# Collect all qcoll and qnoncoll lengths
all_qcoll_lengths = []
all_qnoncoll_lengths = []

# --- Main Simulation Loop ---
for benchid in tqdm(benchrange, desc="Processing benchmarks"):
    bench_prediction = 0
    bench_oracle = 0
    bench_cycles = 0
    bench_oracle_cycles = 0
    colldict = {}

    # 加载数据
    # 优先尝试加载包含 Sphere 和 Link 坐标的统一数据文件，以确保与对比实验的数据一致性
    sphere_coords, edge_link_coll_data, link_coords = su.load_data_with_link_coords(
        basename, benchid, data_folder
    )

    if collision_model_type == "link":
        edge_link_data = link_coords
    else:
        edge_link_data = sphere_coords

    if edge_link_data is None or edge_link_coll_data is None:
        continue

    # --- Oracle Metrics Calculation ---
    oracle_stats = aggregate_oracle_stats(
        edge_link_coll_data, num_elements, num_oocds=num_oocds, check_cost=check_cost
    )

    stats["total_checks"] += oracle_stats["total_checks"]
    bench_oracle = oracle_stats["total_oracle_queries"]
    bench_oracle_cycles = oracle_stats["total_oracle_cycles"]
    stats["total_oracle_coll_cycles"] += oracle_stats["total_oracle_coll_cycles"]
    stats["total_oracle_noncoll_cycles"] += oracle_stats["total_oracle_noncoll_cycles"]
    stats["total_coll_edges"] += oracle_stats["total_coll_edges"]
    stats["total_noncoll_edges"] += oracle_stats["total_noncoll_edges"]

    # Determine simulation mode based on collision model type
    # link model: 1 prediction per link -> 1 detection per link (simple)
    # sphere model: 1 prediction per link -> N detections per link (batch)
    sim_mode = "simple" if collision_model_type == "link" else "batch"

    # --- Run Double Buffer Simulation ---
    edge_query_count, colldict, cycle, db_stats = (
        su.simulate_parallel_collision_detection_double_buffer(
            edge_link_data,
            edge_link_coll_data,
            colldict,
            threshold,
            sample_rate,
            bins,
            link_to_spheres=link_to_spheres,
            sphere_to_link=sphere_to_link,
            num_spheres_per_pose=num_spheres_per_pose,
            qnoncoll_len=qnoncoll_len,
            cycle_check=check_cost,
            num_oocds=num_oocds,
            num_predictions=num_predictions,
            num_dedicated_oocds=num_dedicated_oocds,
            mode=sim_mode,
        )
    )

    bench_prediction += edge_query_count
    bench_cycles += cycle
    stats["total_cdu_idle_cycles"] += db_stats["cdu_idle_cycles"]
    stats["total_pred_coll_cycles"] += db_stats["total_coll_edge_cycles"]
    stats["total_pred_noncoll_cycles"] += db_stats["total_noncoll_edge_cycles"]

    # Track non-collision edge queries
    stats["total_noncoll_edge_queries"] += db_stats.get("total_noncoll_edge_queries", 0)

    # Collect lengths for statistics
    all_qcoll_lengths.extend(db_stats["qcoll_lengths_at_start"])
    all_qnoncoll_lengths.extend(db_stats["qnoncoll_lengths_at_start"])

    stats["fall_oracle"] += bench_oracle
    stats["total_prediction_queries"] += bench_prediction
    stats["fall_cycle"] += bench_cycles
    stats["theoretical_min_cycles"] += bench_oracle_cycles

    # 每处理10个benchmark打印一次
    if (benchid) % 10 == 0:
        print(
            f"[{benchid}/{num_benchmarks}] Queries: {bench_prediction:.2f}, Oracle: {bench_oracle}, "
            f"Cycles: {bench_cycles}"
        )

# 统计打印
print_final_statistics(
    total_checks=stats["total_checks"],
    fall_prediction=stats["total_prediction_queries"],
    fall_oracle=stats["fall_oracle"],
    fall_cycle=stats["fall_cycle"],
    theoretical_min_cycles=stats["theoretical_min_cycles"],
    total_pred_coll_cycles=stats["total_pred_coll_cycles"],
    total_pred_noncoll_cycles=stats["total_pred_noncoll_cycles"],
    total_oracle_coll_cycles=stats["total_oracle_coll_cycles"],
    total_oracle_noncoll_cycles=stats["total_oracle_noncoll_cycles"],
    extra_stats={
        "Total Collision Edges": stats["total_coll_edges"],
        "Total Non-Collision Edges": stats["total_noncoll_edges"],
    },
)

# Calculate average checks for collision edges
total_coll_edge_queries = (
    stats["total_prediction_queries"] - stats["total_noncoll_edge_queries"]
)
avg_coll_edge_checks = (
    total_coll_edge_queries / stats["total_coll_edges"]
    if stats["total_coll_edges"] > 0
    else 0
)
print(f"\n  Total Collision Edge Queries: {total_coll_edge_queries:.2f}")
print(f"  Total Non-Collision Edge Queries: {stats['total_noncoll_edge_queries']:.2f}")
print(f"  Average Checks per Collision Edge: {avg_coll_edge_checks:.2f}")

print(f"\n  Total CDU Idle Cycles: {stats['total_cdu_idle_cycles']}")
print(
    f"  Average CDU Utilization: {(1.0 - stats['total_cdu_idle_cycles'] / (stats['fall_cycle'] * num_oocds)) * 100:.2f}%"
)

# Calculate statistics for qcoll lengths
if all_qcoll_lengths:
    qcoll_mean = np.mean(all_qcoll_lengths)
    qcoll_median = np.median(all_qcoll_lengths)
    qcoll_mode = Counter(all_qcoll_lengths).most_common(1)[0][0]
    qcoll_max = np.max(all_qcoll_lengths)
    qcoll_min = np.min(all_qcoll_lengths)
    qcoll_var = np.var(all_qcoll_lengths)
    print(
        f"\n  QColl lengths at start - Mean: {qcoll_mean:.2f}, Median: {qcoll_median:.2f}, Mode: {qcoll_mode}, Max: {qcoll_max}, Min: {qcoll_min}, Var: {qcoll_var:.2f}"
    )

# Calculate statistics for qnoncoll lengths
if all_qnoncoll_lengths:
    qnoncoll_mean = np.mean(all_qnoncoll_lengths)
    qnoncoll_median = np.median(all_qnoncoll_lengths)
    qnoncoll_mode = Counter(all_qnoncoll_lengths).most_common(1)[0][0]
    qnoncoll_max = np.max(all_qnoncoll_lengths)
    qnoncoll_min = np.min(all_qnoncoll_lengths)
    qnoncoll_var = np.var(all_qnoncoll_lengths)
    print(
        f"  QNonColl lengths at start - Mean: {qnoncoll_mean:.2f}, Median: {qnoncoll_median:.2f}, Mode: {qnoncoll_mode}, Max: {qnoncoll_max}, Min: {qnoncoll_min}, Var: {qnoncoll_var:.2f}"
    )

# 验证检查项
print("\n" + "=" * 50)
print("Validation Checks:")
print(f"  ✓ Total Benchmarks Processed: {len(list(benchrange))}")
print("=" * 50)
