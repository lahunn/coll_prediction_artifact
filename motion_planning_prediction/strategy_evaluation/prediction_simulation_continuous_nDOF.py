#!/usr/bin/env python3
"""
连续变化障碍物碰撞检测预测仿真程序（nDOF机器人）

用于分析常规策略与CHT继承策略在连续变化障碍物场景下的区别

数据格式:
- 连续数据集: List[Tuple[obstacles, start, goal, path]]
- link_data[edge][pose][link] = [x, y, z, qx, qy, qz, qw]
- link_coll_data[edge][pose][link] = 1 or 0

标准参数顺序: threshold, sample_rate, qnoncoll_multiplier, data_folder, basename, benchmarks, robot_name
特定参数: strategy, decay_factor
"""

import sys
import os
import glob
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
parser = create_common_parser("连续变化障碍物碰撞检测预测仿真程序（nDOF机器人）")

parser.add_argument(
    "strategy", type=str, choices=["conventional", "inheritance"], help="Strategy type"
)

parser.add_argument(
    "decay_factor",
    type=float,
    default=1.0,
    nargs="?",
    help="Decay factor for inheritance",
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
strategy = args.strategy
decay_factor = args.decay_factor
num_oocds = args.num_oocds

# 使用通用工具设置仿真参数
bins, num_elements, check_cost, qnoncoll_len, print_title = setup_simulation(
    robot_name, quant_bits, collision_model_type, qnoncoll_multiplier
)

print(print_title)
print(f"策略: {strategy}")
print(f"阈值: {threshold}")
print(f"采样率: {sample_rate}")
print(f"队列长度倍数: {qnoncoll_multiplier}")
print(f"非碰撞队列长度: {qnoncoll_len}")
print(f"OOCD数量: {num_oocds}")
print(f"数据文件夹: {data_folder}")
print(f"基准测试: {benchmarks_arg}")
if strategy == "inheritance":
    print(f"衰减因子: {decay_factor}")
print("=" * 50)


# --- Benchmark Range ---


benchrange = parse_benchrange(benchmarks_arg)
num_benchmarks = len(benchrange)


# 自动检测问题数量


pattern = os.path.join(data_folder, f"{basename}_*_ctn_{collision_model_type}.pkl")
matching_files = glob.glob(pattern)
num_problems = len(matching_files)
print(
    f"检测到 {num_problems} 个问题文件，将处理 {min(num_problems, num_benchmarks)} 个"
)


# 初始化CHT


if strategy == "inheritance":
    colldict = su.initialize_cht()


else:
    colldict = None  # 对于conventional，每个problem重置


# --- Main Simulation Loop ---


for problem_idx, benchid in enumerate(
    tqdm(benchrange[: min(num_problems, num_benchmarks)], desc="处理连续问题")
):
    all_prediction = 0
    all_oracle = 0
    all_cycle = 0
    # 如果是conventional策略，每个problem重置CHT
    if strategy == "conventional":
        colldict = su.initialize_cht()
    elif strategy == "inheritance" and problem_idx > 0:
        # 继承上一个problem的CHT，应用衰减
        colldict = su.inherit_cht(colldict, decay_factor)
    # 加载当前问题的碰撞数据
    edge_link_data, edge_link_coll_data = su.load_data(
        basename,
        benchid,
        data_folder,
        collision_model_type=collision_model_type,
        analysis_type="continue",
    )
    if edge_link_data is None or edge_link_coll_data is None:
        print(f"警告: 无法加载问题 {benchid} 的数据")
        continue
    # --- Oracle Metrics Calculation ---
    oracle_stats = aggregate_oracle_stats(
        edge_link_coll_data, num_elements, num_oocds=num_oocds, check_cost=check_cost
    )
    stats["total_checks"] += oracle_stats["total_checks"]
    all_oracle = oracle_stats["total_oracle_queries"]
    stats["theoretical_min_cycles"] += oracle_stats["total_oracle_cycles"]
    stats["total_oracle_coll_cycles"] += oracle_stats["total_oracle_coll_cycles"]
    stats["total_oracle_noncoll_cycles"] += oracle_stats["total_oracle_noncoll_cycles"]
    # 处理每条边
    for edge, edge_coll in zip(edge_link_data, edge_link_coll_data):
        if not edge_coll:
            continue
        # --- CSP Rearrangement ---
        linklist, linklist_coll = su.csp_rearrange(edge, edge_coll, groupsize=4)
        # --- Run Centralized Simulation ---
        edge_query_count, colldict, coll_found, cycle = (
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
    # 每处理10个problem打印一次
    if (problem_idx + 1) % 10 == 0:
        print(
            f"[{problem_idx + 1}/{num_problems}] 预测查询: {all_prediction:.2f}, Oracle查询: {all_oracle}"
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
    extra_stats={"Strategy": strategy},
)
