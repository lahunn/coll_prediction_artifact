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

# 添加 trace_generation 目录到 Python 路径
from trace_generation.config.ana_parameters import get_robot_params

# --- Simulation Settings ---
num_oocds = 7
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

# --- Simulation Parameters from Command Line ---
if len(sys.argv) < 8:
    print(
        "Usage: python prediction_simulation_nDOF.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <num_benchmarks> <robot_name> [collision_model_type]"
    )
    print(
        "Example: python prediction_simulation_nDOF.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 100 iiwa link"
    )
    sys.exit(1)

threshold = float(sys.argv[1])
sample_rate = float(sys.argv[2])
qnoncoll_multiplier = int(sys.argv[3])
data_folder = sys.argv[4]
basename = sys.argv[5]
num_benchmarks = int(sys.argv[6])
robot_name = sys.argv[7]
collision_model_type = sys.argv[8] if len(sys.argv) > 8 else "link"

# 获取机器人参数
robot_params = get_robot_params(robot_name)

# 使用workspace信息计算bins
bins = su.calculate_bins_from_workspace(robot_name, quant_bits)

if collision_model_type == "sphere":
    num_elements = robot_params["sphere_num"]
    check_cost = robot_params["sphere_cost"]
    csv_file = "../result_files/sphere_results.csv"
    print_title = "=== 球体碰撞检测预测仿真 ==="
else:
    num_elements = robot_params["obb_num"]
    check_cost = robot_params["obb_cost"]
    csv_file = "../result_files/obb_results.csv"
    print_title = "=== OBB碰撞检测预测仿真 ==="

qnoncoll_len = 56

print(print_title)
print(f"阈值: {threshold}")
print(f"采样率: {sample_rate}")
print(f"队列长度倍数: {qnoncoll_multiplier}")
print(f"非碰撞队列长度: {qnoncoll_len}")
print(f"数据文件夹: {data_folder}")
print(f"基准测试数量: {num_benchmarks}")
print("=" * 50)

# --- Benchmark Range ---
benchrange = range(1, num_benchmarks + 1)

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

    # 累计理论查询总数
    for edge_coll in edge_link_coll_data:
        for pose_coll in edge_coll:
            # 理想的顺序检查器：检查直到发现第一个碰撞，或者检查完所有link都没有碰撞。
            try:
                # 找到第一个碰撞(值为0)的索引
                first_collision_index = pose_coll.index(0)
                # 加上找到它所需的检查次数 (索引从0开始，所以+1)
                total_checks += first_collision_index + 1
            except ValueError:
                # 如果 pose_coll 中没有0 (即当前姿态无碰撞)，则需要检查该姿态下的所有link
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
        # Oracle: 检测到碰撞就停止，否则检查所有link
        coll_found_oracle = any(
            link_coll == 0 for pose_coll in edge_coll for link_coll in pose_coll
        )
        if coll_found_oracle:
            all_oracle += 1
        else:
            # 如果没有碰撞，需要检查所有姿态的所有link
            all_oracle += num_elements * len(edge_coll)

        # 计算oracle周期数
        oracle_edge_cycles = su.calculate_oracle_cycles(
            edge_coll, num_oocds, check_cost
        )
        if coll_found_oracle:
            total_oracle_coll_cycles += oracle_edge_cycles
        else:
            total_oracle_noncoll_cycles += oracle_edge_cycles

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
            total_pred_coll_cycles += cycle
        else:
            total_pred_noncoll_cycles += cycle

        all_prediction += edge_query_count
        all_cycle += cycle

    fall_oracle += all_oracle
    fall_prediction += all_prediction
    fall_cycle += all_cycle

    # 每处理10个benchmark打印一次
    if (benchid) % 10 == 0:
        print(
            f"[{benchid}/{num_benchmarks}] 预测查询: {all_prediction:.2f}, Oracle查询: {all_oracle}"
        )

print("\n" + "=" * 50)
print("Final Statistics:")
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
# with open(csv_file, "a", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     # 如果文件不存在，添加header
#     if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
#         writer.writerow(
#             [
#                 "threshold",
#                 "sample_rate",
#                 "qnoncoll_multiplier",
#                 "basename",
#                 "num_benchmarks",
#                 "robot_name",
#                 "total_checks",
#                 "fall_prediction",
#                 "fall_oracle",
#                 "fall_cycle",
#                 "theoretical_min_cycles",
#                 "reduction_rate",
#                 "total_pred_coll_cycles",
#                 "total_pred_noncoll_cycles",
#                 "total_oracle_coll_cycles",
#                 "total_oracle_noncoll_cycles",
#                 "query_difference_percent",
#             ]
#         )
#     writer.writerow(
#         [
#             threshold,
#             sample_rate,
#             qnoncoll_multiplier,
#             basename,
#             num_benchmarks,
#             robot_name,
#             total_checks,
#             fall_prediction,
#             fall_oracle,
#             fall_cycle,
#             theoretical_min_cycles,
#             reduction_rate,
#             total_pred_coll_cycles,
#             total_pred_noncoll_cycles,
#             total_oracle_coll_cycles,
#             total_oracle_noncoll_cycles,
#             (fall_prediction - fall_oracle) / fall_oracle * 100
#             if fall_oracle > 0
#             else 0,  # Query Difference (%)
#         ]
#     )
