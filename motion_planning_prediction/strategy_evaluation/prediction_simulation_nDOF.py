#!/usr/bin/env python3
"""
Link级碰撞检测预测仿真程序（nDOF机器人）

使用Link级碰撞检测进行碰撞检测的预测策略评估

数据格式:
- link_data[edge][pose][link] = [x, y, z, qx, qy, qz, qw]
- link_coll_data[edge][pose][link] = 1 or 0

文件命名约定: {basename}_{benchid:04d}_{collision_model_type}.pkl
其中 collision_model_type 为 'link' 或 'sphere'

脚本接受七个命令行参数：
1. threshold: 预测阈值 (float)
2. sample_rate: 采样率 (float)
3. qnoncoll_multiplier: 用于计算非碰撞队列长度的乘数 (int)
4. data_folder: 数据文件夹路径
5. basename: 数据文件基础名称（如 iiwa_7）
6. num_benchmarks: 基准测试数量
7. robot_name: 机器人名称
8. collision_model_type: 碰撞模型类型 ("link" 或 "sphere", 可选)

使用示例:
    python prediction_simulation_nDOF.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 10 iiwa link
"""

import sys
import os
from tqdm import tqdm
# import csv

# 添加上级目录到path以导入simulation_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
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
parser = create_common_parser("Link级碰撞检测预测仿真程序（nDOF机器人）")
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
print(f"阈值: {threshold}")
print(f"采样率: {sample_rate}")
print(f"队列长度倍数: {qnoncoll_multiplier}")
print(f"非碰撞队列长度: {qnoncoll_len}")
print(f"OOCD数量: {num_oocds}")
print(f"数据文件夹: {data_folder}")
print(f"基准测试: {benchmarks_arg}")
print("=" * 50)

# --- Benchmark Range ---
benchrange = parse_benchrange(benchmarks_arg)
num_benchmarks = len(benchrange)

# --- Main Simulation Loop ---
for benchid in tqdm(benchrange, desc="处理基准测试"):
    all_prediction = 0
    all_oracle = 0
    all_cycle = 0
    colldict = {}

    # 加载碰撞数据
    edge_link_data, edge_link_coll_data = su.load_data(  # type: ignore
        basename,
        benchid,
        data_folder,
        collision_model_type=collision_model_type,
    )

    if edge_link_data is None or edge_link_coll_data is None:
        continue

    # --- Oracle Metrics Calculation ---
    # 使用通用函数一次性累计Oracle理论指标
    oracle_stats = aggregate_oracle_stats(
        edge_link_coll_data, num_elements, num_oocds, check_cost
    )

    stats["total_checks"] += oracle_stats["total_checks"]
    all_oracle = oracle_stats["total_oracle_queries"]
    stats["theoretical_min_cycles"] += oracle_stats["total_oracle_cycles"]
    stats["total_oracle_coll_cycles"] += oracle_stats["total_oracle_coll_cycles"]
    stats["total_oracle_noncoll_cycles"] += oracle_stats["total_oracle_noncoll_cycles"]

    # 处理每条边执行仿真
    for edge, edge_coll in zip(edge_link_data, edge_link_coll_data):
        if not edge_coll:
            continue

        # --- CSP Rearrangement ---
        # 将edge数据重排为适合CSP策略的顺序
        linklist, linklist_coll = su.csp_rearrange(edge, edge_coll, groupsize=4)

        # --- Run Centralized Simulation ---
        edge_query_count, colldict, coll_found, cycle, _ = (
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
            )
        )

        if coll_found:
            stats["total_pred_coll_cycles"] += cycle
        else:
            stats["total_pred_noncoll_cycles"] += cycle

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
    theoretical_min_cycles=stats["theoretical_min_cycles"],
    total_pred_coll_cycles=stats["total_pred_coll_cycles"],
    total_pred_noncoll_cycles=stats["total_pred_noncoll_cycles"],
    total_oracle_coll_cycles=stats["total_oracle_coll_cycles"],
    total_oracle_noncoll_cycles=stats["total_oracle_noncoll_cycles"],
)

# 输出到CSV (保留原有注释代码)
reduction_rate = (
    (1 - stats["fall_prediction"] / stats["total_checks"]) * 100
    if stats["total_checks"] > 0
    else 0
)
