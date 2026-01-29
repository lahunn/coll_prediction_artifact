#!/usr/bin/env python3
"""
球体碰撞检测预测仿真程序（nDOF机器人）- 专用CDU版本

使用球体近似进行碰撞检测的预测策略评估，采用专用CDU策略：
- num_dedicated_oocds: 1 个CDU专门用于QCOLL任务（除非QNONCOLL满）
- 剩余6个CDU为共享CDU，无优先级地处理QCOLL和QNONCOLL任务

数据格式:
- sphere_link_data[edge][pose][sphere] = [x, y, z, radius]
- sphere_link_coll_data[edge][pose][sphere] = 1 or 0

文件命名约定: {basename}_{benchid:04d}_{collision_model_type}.pkl
其中 collision_model_type 为 'link' 或 'sphere'

脚本接受八个命令行参数：
1. threshold: 预测阈值 (float)
2. sample_rate: 采样率 (float)
3. qnoncoll_multiplier: 用于计算非碰撞队列长度的乘数 (int)
4. data_folder: 数据文件夹路径
5. basename: 数据文件基础名称（如 iiwa_7）
6. num_benchmarks: 基准测试数量
7. robot_name: 机器人名称
8. num_dedicated_oocds: 专用OOCD数量 (int, 可选, 默认为1)

使用示例:
    python prediction_simulation_nDOF_dedicated.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 10 iiwa 1 link
"""

import sys
import os
import numpy as np
from tqdm import tqdm
import simulation_utils as su
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
stats = initialize_statistics()

# --- Simulation Parameters from Command Line ---
parser = create_common_parser("球体碰撞检测预测仿真程序（nDOF机器人）- 专用CDU版本")
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
num_dedicated_oocds = args.num_dedicated_oocds

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
print(f"Robot: {robot_name}")
print(f"Collision Model: {collision_model_type}")
print(f"OOCD数量: {num_oocds}")
print(f"Dedicated CDUs: {num_dedicated_oocds}")
print(f"Shared CDUs: {num_oocds - num_dedicated_oocds}")
print("=" * 50)

# --- Global Statistics ---
stats = initialize_statistics()

# --- Benchmark Range ---
benchrange = parse_benchrange(benchmarks_arg)
num_benchmarks = len(benchrange)

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

    # 处理每条边
    for edge, edge_coll in zip(edge_link_data, edge_link_coll_data):
        if not edge_coll:
            continue

        # --- CSP Rearrangement ---
        # 将edge数据重排为适合CSP策略的顺序
        linklist, linklist_coll = su.csp_rearrange(edge, edge_coll, groupsize=4)

        # --- Run Dedicated CDU Simulation ---
        edge_query_count, colldict, _, cycle, _ = (
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
                mode='simple',
                num_dedicated_oocds=num_dedicated_oocds,
            )
        )

        all_prediction += edge_query_count
        all_cycle += cycle

    stats["fall_oracle"] += all_oracle
    stats["fall_prediction"] += all_prediction
    stats["fall_cycle"] += all_cycle

    # 每处理10个benchmark打印一次
    if (benchid) % 10 == 0:
        print(
            f"[{benchid}/{num_benchmarks}] 预测查询: {all_prediction:.2f}, Oracle查询: {all_oracle}"
        )


print_final_statistics(
    total_checks=stats["total_checks"],
    fall_prediction=stats["fall_prediction"],
    fall_oracle=stats["fall_oracle"],
    fall_cycle=stats["fall_cycle"],
)

# 输出到CSV
reduction_rate = (
    (1 - stats["fall_prediction"] / stats["total_checks"]) * 100
    if stats["total_checks"] > 0
    else 0
)
