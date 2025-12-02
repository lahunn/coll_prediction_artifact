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

脚本接受七个命令行参数：
1. threshold: 预测阈值 (float)
2. sample_rate: 采样率 (float)
3. qnoncoll_multiplier: 用于计算非碰撞队列长度的乘数 (int)
4. data_folder: 数据文件夹路径
5. basename: 数据文件基础名称（如 iiwa_7）
6. num_benchmarks: 基准测试数量 或 单个benchid 或 范围如 "2-10"
7. robot_name: 机器人名称

使用示例:
    python prediction_simulation_nDOF_double_buffer.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 10 iiwa link
    python prediction_simulation_nDOF_double_buffer.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 5 iiwa link
    python prediction_simulation_nDOF_double_buffer.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 2-10 iiwa link
"""

import sys
import os
import numpy as np
from tqdm import tqdm
from collections import Counter

# 添加上级目录到path以导入simulation_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su
import csv

# 添加 trace_generation 目录到 Python 路径
from trace_generation.config.ana_parameters import get_robot_params

# --- Simulation Settings ---
quant_bits = 4  # 4 bits per dimension (16 bins)
bins = su.calculate_bins_from_workspace("iiwa", quant_bits)

# --- Simulation Parameters from Command Line ---
if len(sys.argv) < 9:
    print(
        "Usage: python prediction_simulation_nDOF_double_buffer.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <benchmarks> <robot_name> <num_predictions> [collision_model_type]"
    )
    print(
        "  <benchmarks> can be: a single number (5), a range (2-10), or total count (10)"
    )
    print(
        "Example 1: python prediction_simulation_nDOF_double_buffer.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 10 iiwa 2 link"
    )
    print(
        "Example 2: python prediction_simulation_nDOF_double_buffer.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 5 iiwa 3 link"
    )
    print(
        "Example 3: python prediction_simulation_nDOF_double_buffer.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 2-10 iiwa 4 link"
    )
    sys.exit(1)

threshold = float(sys.argv[1])
sample_rate = float(sys.argv[2])
qnoncoll_multiplier = int(sys.argv[3])
data_folder = sys.argv[4]
basename = sys.argv[5]
benchmarks_arg = sys.argv[6]
robot_name = sys.argv[7]

# Parse optional arguments
if len(sys.argv) > 8:
    try:
        num_predictions = int(sys.argv[8])
        collision_model_type = sys.argv[9] if len(sys.argv) > 9 else "link"
    except ValueError:
        # If sys.argv[8] is not a number, treat it as collision_model_type
        collision_model_type = sys.argv[8]
        num_predictions = 2  # default
else:
    num_predictions = 2  # default
    collision_model_type = "link"

# Parse benchmarks argument to create benchrange
if "-" in benchmarks_arg:
    start_bench, end_bench = map(int, benchmarks_arg.split("-"))
    num_benchmarks = end_bench - start_bench + 1
    benchrange = range(start_bench, end_bench + 1)
else:
    benchid = int(benchmarks_arg)
    num_benchmarks = 1
    benchrange = range(benchid, benchid + 1)

# 获取机器人参数
robot_params = get_robot_params(robot_name)

if collision_model_type == "sphere":
    num_elements = robot_params["sphere_num"]
    check_cost = robot_params["sphere_cost"]
    csv_file = "../result_files/sphere_results_double_buffer.csv"
    print_title = "=== Sphere Collision Detection - Double Buffer Architecture ==="
else:
    num_elements = robot_params["obb_num"]
    check_cost = robot_params["obb_cost"]
    csv_file = "../result_files/obb_results_double_buffer.csv"
    print_title = "=== OBB Collision Detection - Double Buffer Architecture ==="

qnoncoll_len = num_elements * qnoncoll_multiplier

print(print_title)
print(f"Threshold: {threshold}")
print(f"Sample Rate: {sample_rate}")
print(f"Queue Length Multiplier: {qnoncoll_multiplier}")
print(f"Non-collision Queue Length: {qnoncoll_len}")
print(f"Data Folder: {data_folder}")
print(f"Number of Benchmarks: {num_benchmarks}")
print(f"Robot: {robot_name}")
print(f"Collision Model: {collision_model_type}")
print(f"Number of Predictions: {num_predictions}")
print("Architecture: Double Buffer (Bank A + Bank B)")
print("=" * 50)

# --- Statistics ---
total_checks = 0
total_oracle_queries = 0
total_prediction_queries = 0
total_cycles = 0
total_oracle_cycles = 0
total_cdu_idle_cycles = 0
total_coll_edge_cycles = 0
total_noncoll_edge_cycles = 0
total_oracle_coll_edge_cycles = 0
total_oracle_noncoll_edge_cycles = 0

# Collect all qcoll and qnoncoll lengths
all_qcoll_lengths = []
all_qnoncoll_lengths = []

# Edge counts
total_coll_edges = 0
total_noncoll_edges = 0

# Track queries for collision and non-collision edges
total_noncoll_edge_queries = 0

# --- Main Simulation Loop ---
for benchid in tqdm(benchrange, desc="Processing benchmarks"):
    bench_prediction = 0
    bench_oracle = 0
    bench_cycles = 0
    bench_oracle_cycles = 0
    colldict = {}

    # 加载数据
    edge_link_data, edge_link_coll_data = su.load_data(
        basename, benchid, data_folder, collision_model_type=collision_model_type
    )

    if edge_link_data is None or edge_link_coll_data is None:
        continue

    # 累计实际查询总数（理想的顺序Oracle）
    for edge_coll in edge_link_coll_data:
        for pose_coll in edge_coll:
            try:
                first_collision_index = pose_coll.index(0)
                total_checks += first_collision_index + 1
            except ValueError:
                total_checks += len(pose_coll)

    # 计算 Oracle 理论周期数
    bench_oracle_cycles = su.calculate_oracle_cycles_for_edges(
        edge_link_coll_data, num_oocds=7, cycle_check=check_cost
    )
    total_oracle_cycles += bench_oracle_cycles

    # 逐边计算 Oracle 查询数
    for edge, edge_coll in zip(edge_link_data, edge_link_coll_data):
        if not edge_coll:
            continue

        coll_found_oracle = any(
            link_coll == 0 for pose_coll in edge_coll for link_coll in pose_coll
        )
        if coll_found_oracle:
            bench_oracle += 1
        else:
            bench_oracle += num_elements * len(edge_coll)

        # 计算Oracle的edge周期数
        oracle_edge_cycles = su.calculate_oracle_cycles(
            edge_coll, num_oocds=7, cycle_check=check_cost
        )
        if coll_found_oracle:
            total_oracle_coll_edge_cycles += oracle_edge_cycles
            total_coll_edges += 1
        else:
            total_oracle_noncoll_edge_cycles += oracle_edge_cycles
            total_noncoll_edges += 1

    # --- Run Double Buffer Simulation ---
    edge_query_count, colldict, cycle, stats = (
        su.simulate_parallel_collision_detection_double_buffer(
            edge_link_data,
            edge_link_coll_data,
            colldict,
            threshold,
            sample_rate,
            bins,
            qnoncoll_len=qnoncoll_len,
            cycle_check=check_cost,
            num_oocds=7,
            num_predictions=num_predictions,
        )
    )

    bench_prediction += edge_query_count
    bench_cycles += cycle
    total_cdu_idle_cycles += stats["cdu_idle_cycles"]
    total_coll_edge_cycles += stats["total_coll_edge_cycles"]
    total_noncoll_edge_cycles += stats["total_noncoll_edge_cycles"]

    # Track non-collision edge queries
    total_noncoll_edge_queries += stats.get("total_noncoll_edge_queries", 0)

    # Collect lengths for statistics
    all_qcoll_lengths.extend(stats["qcoll_lengths_at_start"])
    all_qnoncoll_lengths.extend(stats["qnoncoll_lengths_at_start"])

    total_oracle_queries += bench_oracle
    total_prediction_queries += bench_prediction
    total_cycles += bench_cycles

    # 每处理10个benchmark打印一次
    if (benchid) % 10 == 0:
        print(
            f"[{benchid}/{num_benchmarks}] Queries: {bench_prediction:.2f}, Oracle: {bench_oracle}, "
            f"Cycles: {bench_cycles}"
        )

print("\n" + "=" * 50)
print("Final Statistics:")
print(f"  Total Actual Checks: {total_checks}")
print(f"  Total Prediction Queries: {total_prediction_queries:.2f}")
print(f"  Total Oracle Queries: {total_oracle_queries}")
print(
    f"  Query Reduction Rate: {(1 - total_prediction_queries / total_checks) * 100:.2f}%"
)
print(f"\n  Total Cycles (Prediction): {total_cycles}")
print(f"  Total Cycles (Oracle): {total_oracle_cycles}")
print(f"  Cycle Efficiency: {(total_oracle_cycles / total_cycles) * 100:.2f}%")
print(f"\n  Prediction Coll Edge Cycles: {total_coll_edge_cycles}")
print(f"  Prediction Non-Coll Edge Cycles: {total_noncoll_edge_cycles}")
print(f"  Oracle Coll Edge Cycles: {total_oracle_coll_edge_cycles}")
print(f"  Oracle Non-Coll Edge Cycles: {total_oracle_noncoll_edge_cycles}")
print(f"\n  Total Collision Edges: {total_coll_edges}")
print(f"  Total Non-Collision Edges: {total_noncoll_edges}")

# Calculate average checks for collision edges
total_coll_edge_queries = total_prediction_queries - total_noncoll_edge_queries
avg_coll_edge_checks = (
    total_coll_edge_queries / total_coll_edges if total_coll_edges > 0 else 0
)
print(f"\n  Total Collision Edge Queries: {total_coll_edge_queries:.2f}")
print(f"  Total Non-Collision Edge Queries: {total_noncoll_edge_queries:.2f}")
print(f"  Average Checks per Collision Edge: {avg_coll_edge_checks:.2f}")

print(f"\n  Total CDU Idle Cycles: {total_cdu_idle_cycles}")
print(
    f"  Average CDU Utilization: {(1.0 - total_cdu_idle_cycles / (total_cycles * 7)) * 100:.2f}%"
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

# 输出到CSV
reduction_rate = (
    (1 - total_prediction_queries / total_checks) * 100 if total_checks > 0 else 0
)
cycle_efficiency = (total_oracle_cycles / total_cycles) * 100 if total_cycles > 0 else 0
cdu_utilization = (
    (1.0 - total_cdu_idle_cycles / (total_cycles * 7)) * 100 if total_cycles > 0 else 0
)

# with open(csv_file, "a", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(
#         [
#             threshold,
#             sample_rate,
#             qnoncoll_multiplier,
#             basename,
#             num_benchmarks,
#             robot_name,
#             total_checks,
#             total_prediction_queries,
#             total_oracle_queries,
#             total_cycles,
#             total_oracle_cycles,
#             total_coll_edge_cycles,
#             total_noncoll_edge_cycles,
#             total_oracle_coll_edge_cycles,
#             total_oracle_noncoll_edge_cycles,
#             reduction_rate,
#             cycle_efficiency,
#             cdu_utilization,
#         ]
#     )
