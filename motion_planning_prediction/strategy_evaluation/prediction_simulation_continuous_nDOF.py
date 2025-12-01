#!/usr/bin/env python3
"""
连续变化障碍物碰撞检测预测仿真程序（nDOF机器人）

用于分析常规策略与CHT继承策略在连续变化障碍物场景下的区别

数据格式:
- 连续数据集: List[Tuple[obstacles, start, goal, path]]
- link_data[edge][pose][link] = [x, y, z, qx, qy, qz, qw]
- link_coll_data[edge][pose][link] = 1 or 0

脚本接受命令行参数：
1. threshold: 预测阈值 (float)
2. sample_rate: 采样率 (float)
3. qnoncoll_multiplier: 用于计算非碰撞队列长度的乘数 (int)
4. data_folder: 碰撞数据文件夹路径
5. basename: 数据文件基础名称（如 iiwa_7）
6. robot_name: 机器人名称
7. strategy: 策略类型 ('conventional' 或 'inheritance')
8. num_benchmarks: 处理的基准测试数量 (int)
9. decay_factor: 继承衰减因子 (float, 默认1.0)
10. collision_model_type: 碰撞模型类型 ('link' 或 'sphere', 默认'link')

使用示例:
    python prediction_simulation_continuous_nDOF.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 iiwa inheritance 50 0.9 link
"""

import sys
import os
import glob
from tqdm import tqdm
import csv

# 添加上级目录到path以导入simulation_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su

# 添加 trace_generation 目录到 Python 路径
from trace_generation.config.ana_parameters import get_robot_params

# --- Simulation Settings ---
num_oocds = 7
quant_bits = 4

# --- Simulation Parameters from Command Line ---
if len(sys.argv) < 9:
    print(
        "Usage: python prediction_simulation_continuous_nDOF.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <robot_name> <strategy> <num_benchmarks> [decay_factor] [collision_model_type]"
    )
    print(
        "Example: python prediction_simulation_continuous_nDOF.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 iiwa inheritance 50 0.9 link"
    )
    sys.exit(1)

threshold = float(sys.argv[1])
sample_rate = float(sys.argv[2])
qnoncoll_multiplier = int(sys.argv[3])
data_folder = sys.argv[4]
basename = sys.argv[5]
robot_name = sys.argv[6]
strategy = sys.argv[7]
num_benchmarks = int(sys.argv[8])
decay_factor = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
collision_model_type = sys.argv[10] if len(sys.argv) > 10 else "link"

# 获取机器人参数
robot_params = get_robot_params(robot_name)

# 使用workspace信息计算bins
bins = su.calculate_bins_from_workspace(robot_name, quant_bits)

if collision_model_type == "sphere":
    num_elements = robot_params["sphere_num"]
    check_cost = robot_params["sphere_cost"]
    csv_file = "result_files/continuous_sphere_results.csv"
    print_title = "=== 连续场景球体碰撞检测预测仿真 ==="
else:
    num_elements = robot_params["obb_num"]
    check_cost = robot_params["obb_cost"]
    csv_file = "result_files/continuous_obb_results.csv"
    print_title = "=== 连续场景OBB碰撞检测预测仿真 ==="

qnoncoll_len = num_elements * qnoncoll_multiplier

print(print_title)
print(f"策略: {strategy}")
print(f"阈值: {threshold}")
print(f"采样率: {sample_rate}")
print(f"队列长度倍数: {qnoncoll_multiplier}")
print(f"非碰撞队列长度: {qnoncoll_len}")
print(f"数据文件夹: {data_folder}")
if strategy == "inheritance":
    print(f"衰减因子: {decay_factor}")
print("=" * 50)

# 自动检测问题数量
pattern = os.path.join(data_folder, f"{basename}_*_ctn_{collision_model_type}.pkl")
matching_files = glob.glob(pattern)
num_problems = len(matching_files)
print(
    f"检测到 {num_problems} 个问题文件，将处理 {min(num_problems, num_benchmarks)} 个"
)

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

# 初始化CHT
if strategy == "inheritance":
    colldict = su.initialize_cht()
else:
    colldict = None  # 对于conventional，每个problem重置

# --- Main Simulation Loop ---
for problem_idx in tqdm(range(min(num_problems, num_benchmarks)), desc="处理连续问题"):
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
    benchid = problem_idx + 1  # 文件编号从1开始
    edge_link_data, edge_link_coll_data = su.load_data(
        basename,
        benchid,
        data_folder,
        collision_model_type=collision_model_type,
        analysis_type="continue",
    )

    if edge_link_data is None or edge_link_coll_data is None:
        print(f"警告: 无法加载问题 {problem_idx + 1} 的数据")
        continue

    # 累计理论查询总数
    for edge_coll in edge_link_coll_data:
        for pose_coll in edge_coll:
            try:
                first_collision_index = pose_coll.index(0)
                total_checks += first_collision_index + 1
            except ValueError:
                total_checks += len(pose_coll)

    # 计算理论最小周期数消耗
    theoretical_min_cycles += su.calculate_oracle_cycles_for_edges(
        edge_link_coll_data, num_oocds=num_oocds, cycle_check=check_cost
    )

    # 处理每条边
    for edge, edge_coll in zip(edge_link_data, edge_link_coll_data):
        if not edge_coll:
            continue

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
        if coll_found_oracle:
            total_oracle_coll_cycles += oracle_edge_cycles
        else:
            total_oracle_noncoll_cycles += oracle_edge_cycles

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
            total_pred_coll_cycles += cycle
        else:
            total_pred_noncoll_cycles += cycle

        all_prediction += edge_query_count
        all_cycle += cycle

    fall_oracle += all_oracle
    fall_prediction += all_prediction
    fall_cycle += all_cycle

    # 每处理10个problem打印一次
    if (problem_idx + 1) % 10 == 0:
        print(
            f"[{problem_idx + 1}/{num_problems}] 预测查询: {all_prediction:.2f}, Oracle查询: {all_oracle}"
        )

print("\n" + "=" * 50)
print("Final Statistics:")
print(f"  Strategy: {strategy}")
print(f"  Total Actual Checks: {total_checks}")
print(f"  Total Prediction Queries: {fall_prediction:.2f}")
print(f"  Total Oracle Queries: {fall_oracle}")
print(f"  Query Reduction Rate: {(1 - fall_prediction / total_checks) * 100:.2f}%")
print(
    f"  Query Difference (Prediction - Oracle): {(fall_prediction - fall_oracle) / fall_oracle * 100:.2f}%"
)
print(f"\n  Total Cycles (Prediction): {fall_cycle}")
print(f"  Total Cycles (Oracle): {theoretical_min_cycles}")
print(f"  Cycle Efficiency: {(theoretical_min_cycles / fall_cycle) * 100:.2f}%")
print(f"\n  Prediction Coll Edge Cycles: {total_pred_coll_cycles}")
print(f"  Prediction Non-Coll Edge Cycles: {total_pred_noncoll_cycles}")
print(f"  Oracle Coll Edge Cycles: {total_oracle_coll_cycles}")
print(f"  Oracle Non-Coll Edge Cycles: {total_oracle_noncoll_cycles}")
print("=" * 50)

# 输出到CSV
reduction_rate = (1 - fall_prediction / total_checks) * 100 if total_checks > 0 else 0
with open(csv_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # 如果文件不存在，添加header
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        writer.writerow(
            [
                "strategy",
                "threshold",
                "sample_rate",
                "qnoncoll_multiplier",
                "basename",
                "num_problems",
                "robot_name",
                "decay_factor",
                "total_checks",
                "fall_prediction",
                "fall_oracle",
                "fall_cycle",
                "theoretical_min_cycles",
                "reduction_rate",
                "total_pred_coll_cycles",
                "total_pred_noncoll_cycles",
                "total_oracle_coll_cycles",
                "total_oracle_noncoll_cycles",
                "query_difference_percent",
            ]
        )
    writer.writerow(
        [
            strategy,
            threshold,
            sample_rate,
            qnoncoll_multiplier,
            basename,
            min(num_problems, num_benchmarks),
            robot_name,
            decay_factor if strategy == "inheritance" else 0.0,
            total_checks,
            fall_prediction,
            fall_oracle,
            fall_cycle,
            theoretical_min_cycles,
            reduction_rate,
            total_pred_coll_cycles,
            total_pred_noncoll_cycles,
            total_oracle_coll_cycles,
            total_oracle_noncoll_cycles,
            (fall_prediction - fall_oracle) / fall_oracle * 100
            if fall_oracle > 0
            else 0,  # Query Difference (%)
        ]
    )
