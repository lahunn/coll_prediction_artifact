#!/usr/bin/env python3
"""
球体碰撞检测预测仿真程序（nDOF机器人）- 预先调度版本

使用球体近似进行碰撞检测的预测策略评估，采用预先调度策略（COLL任务可以抢占NONCOLL任务）

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
    create_common_parser,
    parse_benchrange,
    aggregate_oracle_stats,
    initialize_statistics,
    DEFAULT_QUANT_BITS
)
from tqdm import tqdm
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# 添加 trace_generation 目录到 Python 路径
import simulation_utils as su

# --- Simulation Settings & Global Statistics ---
quant_bits = DEFAULT_QUANT_BITS
stats = initialize_statistics(extra_keys=["fall_preemption"])

# --- Simulation Parameters from Command Line ---
parser = create_common_parser("球体碰撞检测预测仿真程序（nDOF机器人）- 预先调度版本")
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

# --- Benchmark Range ---
benchrange = parse_benchrange(benchmarks_arg)
num_benchmarks = len(benchrange)

# --- Main Simulation Loop ---
for benchid in tqdm(benchrange, desc="处理基准测试"):
    all_prediction = 0
    all_oracle = 0
    all_cycle = 0
    all_preemption = 0
    colldict = {}

    # 加载数据
    edge_link_data, edge_link_coll_data = su.load_data(
        basename, benchid, data_folder, collision_model_type=collision_model_type
    )

    if edge_link_data is None or edge_link_coll_data is None:
        continue

    # --- Oracle Metrics Calculation ---
    oracle_stats = aggregate_oracle_stats(edge_link_coll_data, num_elements, num_oocds=num_oocds, check_cost=check_cost)
    
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

        # --- Run Preemptive Simulation ---
        edge_query_count, colldict, _, cycle, preemption_count = (
            su.simulate_parallel_collision_detection_preemptive(
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
        all_preemption += preemption_count

    stats["fall_oracle"] += all_oracle
    stats["fall_prediction"] += all_prediction
    stats["fall_cycle"] += all_cycle
    stats["fall_preemption"] += all_preemption

    # 每处理10个benchmark打印一次
    if (benchid + 1) % 10 == 0:
        print(
            f"[{benchid + 1}/{num_benchmarks}] 预测查询: {all_prediction:.2f}, Oracle查询: {all_oracle}"
        )


print_final_statistics(
    total_checks=stats["total_checks"],
    fall_prediction=stats["fall_prediction"],
    fall_oracle=stats["fall_oracle"],
    fall_cycle=stats["fall_cycle"],
    extra_stats={"抢占事件总数": stats["fall_preemption"]}
)