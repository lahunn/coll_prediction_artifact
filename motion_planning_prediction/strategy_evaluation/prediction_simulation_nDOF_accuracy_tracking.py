#!/usr/bin/env python3
"""
球体碰撞检测预测仿真程序（nDOF机器人）- 准确率跟踪版本

使用球体近似进行碰撞检测的预测策略评估，并跟踪预测准确率随训练数据量的变化

数据格式:
- sphere_link_data[edge][pose][sphere] = [x, y, z, radius]
- sphere_link_coll_data[edge][pose][sphere] = 1 or 0

标准参数顺序: threshold, sample_rate, qnoncoll_multiplier, data_folder, basename, benchmarks, robot_name, collision_model_type
"""

import sys
import os
import numpy as np
import simulation_utils as su
from tqdm import tqdm

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
parser = create_common_parser("球体碰撞检测预测仿真程序（nDOF机器人）- 专用CDU版本")
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

# 使用通用工具设置仿真参数
bins, num_elements, check_cost, qnoncoll_len, print_title = setup_simulation(
    robot_name, quant_bits, collision_model_type, qnoncoll_multiplier
)

if collision_model_type == "sphere":
    csv_file = "result_files/sphere_results.csv"
    accuracy_csv_file = "result_files/sphere_accuracy_curve.csv"
    print_title = "=== Sphere Collision Detection Prediction Simulation (Accuracy Tracking Version) ==="
else:
    csv_file = "result_files/obb_results.csv"
    accuracy_csv_file = "result_files/obb_accuracy_curve.csv"
    print_title = "=== OBB Collision Detection Prediction Simulation (Accuracy Tracking Version) ==="

print(print_title)
print(f"Threshold: {threshold}")
print(f"Sample Rate: {sample_rate}")
print(f"Queue Length Multiplier: {qnoncoll_multiplier}")
print(f"Non-collision Queue Length: {qnoncoll_len}")
print(f"Data Folder: {data_folder}")
print(f"基准测试: {benchmarks_arg}")
print(f"Collision Model: {collision_model_type}")
print(f"OOCD数量: {num_oocds}")
print("=" * 50)

# --- Global Statistics ---
stats = initialize_statistics()

# --- Accuracy Tracking ---
accuracy_stages = []  # Accuracy at each stage
training_sizes = []  # Corresponding training data sizes
stage_size = 50  # Calculate accuracy every 50 edges
current_predictions = []
current_actuals = []

# --- Benchmark Range ---
benchrange = parse_benchrange(benchmarks_arg)

# --- Main Simulation Loop ---
for benchid in tqdm(benchrange, desc="处理基准测试"):
    all_prediction = 0
    all_oracle = 0
    all_cycle = 0
    colldict = {}

    # 加载数据
    edge_link_data, edge_link_coll_data = su.load_data(
        basename, benchid, data_folder, collision_model_type=collision_model_type
    )

    if edge_link_data is None or edge_link_coll_data is None:
        continue

    # --- Oracle Metrics Calculation ---
    oracle_stats = aggregate_oracle_stats(
        edge_link_coll_data, num_elements, num_oocds=num_oocds, check_cost=check_cost
    )

    stats["total_checks"] += oracle_stats["total_checks"]
    all_oracle = oracle_stats["total_oracle_queries"]

    # 处理每条边执行仿真
    for edge_idx, (edge, edge_coll) in enumerate(
        zip(edge_link_data, edge_link_coll_data)
    ):
        if not edge_coll:
            continue

        # --- CSP Rearrangement ---
        # 将edge数据重排为适合CSP策略的顺序
        linklist, linklist_coll = su.csp_rearrange(edge, edge_coll, groupsize=4)

        # --- Run Centralized Simulation with Accuracy Tracking ---
        edge_query_count, colldict, _, cycle, edge_predictions, edge_actuals = (
            su.simulate_parallel_collision_detection_with_tracking(
                linklist,
                linklist_coll,
                colldict,
                threshold,
                sample_rate,
                bins,
                qnoncoll_len=qnoncoll_len,
                cycle_check=check_cost,
                num_oocds=num_oocds,
            )
        )

        all_prediction += edge_query_count
        all_cycle += cycle

        # 收集预测结果用于准确率计算
        current_predictions.extend(edge_predictions)
        current_actuals.extend(edge_actuals)

        # 阶段性计算准确率
        total_processed_edges = sum(
            len(edge_link_data[:benchid]) for benchid in range(1, benchid)
        ) + (edge_idx + 1)
        if total_processed_edges % stage_size == 0 and current_predictions:
            accuracy = su.calculate_accuracy(current_predictions, current_actuals)
            accuracy_stages.append(accuracy)
            training_sizes.append(len(colldict))  # 当前训练数据量（历史字典大小）
            current_predictions = []
            current_actuals = []

    stats["fall_oracle"] += all_oracle
    stats["fall_prediction"] += all_prediction
    stats["fall_cycle"] += all_cycle

    # 每处理10个benchmark打印一次
    if (benchid) % 10 == 0:
        print(
            f"[{benchid}/{benchrange[-1]}] 预测查询: {all_prediction:.2f}, Oracle查询: {all_oracle}"
        )


print_final_statistics(
    total_checks=stats["total_checks"],
    fall_prediction=stats["fall_prediction"],
    fall_oracle=stats["fall_oracle"],
    fall_cycle=stats["fall_cycle"],
)

# 输出准确率曲线数据到单独的CSV文件
if accuracy_stages:
    print(f"准确率曲线数据已保存到 {accuracy_csv_file}")
    print(f"记录了 {len(accuracy_stages)} 个准确率阶段")
