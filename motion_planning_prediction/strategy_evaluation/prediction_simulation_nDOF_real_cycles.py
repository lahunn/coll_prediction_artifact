#!/usr/bin/env python3
"""
球体碰撞检测预测仿真程序（使用真实周期数）

与 prediction_simulation_nDOF.py 的区别：
- 使用真实的周期数据而不是固定的 cycle_check 值
- 每个OOCD根据实际的周期数来计算完成时间
- 需要周期数据文件格式: {basename}_{benchid:04d}_sphere.pkl

数据格式说明:
- sphere_link_data[edge][pose][sphere] = [x, y, z, radius]
- sphere_link_coll_data[edge][pose][sphere] = 1 or 0
- sphere_link_coll_cycles[edge][pose][sphere] = 周期数

文件命名约定: {basename}_{benchid:04d}_{collision_model_type}.pkl
其中 collision_model_type 为 'link' 或 'sphere'
"""

import sys
import numpy as np
from tqdm import tqdm
import simulation_utils as su
import csv

# 添加 trace_generation 目录到 Python 路径
from trace_generation.config.ana_parameters import get_robot_params

# --- Simulation Settings ---
quant_bits = 4  # 4 bits per dimension (16 bins)
bins = su.calculate_bins_from_workspace("iiwa", quant_bits)

# --- Global Statistics ---
fall_prediction = 0
fall_oracle = 0
total_checks = 0
fall_cycle = 0

# --- Simulation Parameters from Command Line ---
if len(sys.argv) < 8:
    print(
        "Usage: python prediction_simulation_nDOF_real_cycles.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <num_benchmarks> <robot_name> [collision_model_type]"
    )
    print(
        "Example: python prediction_simulation_nDOF_real_cycles.py 0.5 0.1 8 ../trace_files/scene_benchmarks/bit_collision_data iiwa_7 100 iiwa link"
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

if collision_model_type == "sphere":
    num_elements = robot_params["sphere_num"]
    csv_file = "result_files/sphere_results_real_cycles.csv"
    print_title = (
        "=== Sphere Collision Detection Prediction Simulation (Real Cycles) ==="
    )
else:
    num_elements = robot_params["obb_num"]
    csv_file = "result_files/obb_results_real_cycles.csv"
    print_title = "=== OBB Collision Detection Prediction Simulation (Real Cycles) ==="

qnoncoll_len = num_elements * qnoncoll_multiplier

print(print_title)
print(f"Threshold: {threshold}")
print(f"Sample Rate: {sample_rate}")
print(f"Queue Length Multiplier: {qnoncoll_multiplier}")
print(f"Non-collision Queue Length: {qnoncoll_len}")
print(f"Data Folder: {data_folder}")
print(f"Number of Benchmarks: {num_benchmarks}")
print(f"Collision Model: {collision_model_type}")
print("=" * 50)

# --- Benchmark Range ---
benchrange = range(1, num_benchmarks + 1)

# 统计有周期数据的benchmark数量
benchmarks_with_cycles = 0
benchmarks_without_cycles = 0

# --- Main Simulation Loop ---
for benchid in tqdm(benchrange, desc="处理基准测试"):
    all_prediction = 0
    all_oracle = 0
    all_cycle = 0
    colldict = {}

    # 加载数据（必须包含周期数据）
    edge_link_data, edge_link_coll_data, edge_link_coll_cycles = (
        su.load_data_with_cycles(
            basename, benchid, data_folder, collision_model_type=collision_model_type
        )
    )

    if edge_link_data is None or edge_link_coll_data is None:
        continue

    # 检查是否有周期数据
    if edge_link_coll_cycles is None:
        benchmarks_without_cycles += 1
        print(f"\n警告: Benchmark {benchid} 没有周期数据，跳过")
        continue

    benchmarks_with_cycles += 1

    # 累计理论查询总数 (模拟理想的顺序Oracle)
    for edge_coll in edge_link_coll_data:
        for pose_coll in edge_coll:
            try:
                first_collision_index = pose_coll.index(0)
                total_checks += first_collision_index + 1
            except ValueError:
                total_checks += len(pose_coll)

    # 处理每条边
    for edge_idx, (edge, edge_coll) in enumerate(
        zip(edge_link_data, edge_link_coll_data)
    ):
        if not edge_coll:
            continue

        # 检查是否有对应的周期数据
        if edge_idx >= len(edge_link_coll_cycles):
            continue

        edge_cycles = edge_link_coll_cycles[edge_idx]

        # --- Oracle Calculation ---
        coll_found_oracle = any(
            link_coll == 0 for pose_coll in edge_coll for link_coll in pose_coll
        )
        if coll_found_oracle:
            all_oracle += 1
        else:
            all_oracle += num_elements * len(edge_coll)

        # --- CSP Rearrangement（包括周期数据）---
        linklist, linklist_coll, linklist_cycles = su.csp_rearrange_with_cycles(
            edge, edge_coll, edge_cycles, groupsize=4
        )

        # --- Run Centralized Simulation with Real Cycles ---
        edge_query_count, colldict, _, cycle = (
            su.simulate_parallel_collision_detection_real_cycles(
                linklist,
                linklist_coll,
                linklist_cycles,
                colldict,
                threshold,
                sample_rate,
                bins,
                qnoncoll_len=qnoncoll_len,
                num_oocds=7,
            )
        )

        all_prediction += edge_query_count
        all_cycle += cycle

    fall_oracle += all_oracle
    fall_prediction += all_prediction
    fall_cycle += all_cycle

    # 每处理10个benchmark打印一次
    if (benchid + 1) % 10 == 0:
        print(
            f"[{benchid + 1}/{num_benchmarks}] 预测查询: {all_prediction:.2f}, Oracle查询: {all_oracle}"
        )

print("\n" + "=" * 50)
print("最终统计:")
print(f"  有周期数据的Benchmark: {benchmarks_with_cycles}/{num_benchmarks}")
print(f"  无周期数据的Benchmark: {benchmarks_without_cycles}/{num_benchmarks}")
print(f"  实际查询总数: {total_checks}")
print(f"  预测查询总数: {fall_prediction:.2f}")
print(f"  Oracle查询总数: {fall_oracle}")
print(f"  预测周期总数 (真实周期): {fall_cycle}")
if total_checks > 0:
    print(f"  查询减少率: {(1 - fall_prediction / total_checks) * 100:.2f}%")
else:
    print("  查询减少率: N/A")
print("=" * 50)

# 输出到CSV
reduction_rate = (1 - fall_prediction / total_checks) * 100 if total_checks > 0 else 0

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
#             benchmarks_with_cycles,
#             benchmarks_without_cycles,
#             total_checks,
#             fall_prediction,
#             fall_oracle,
#             fall_cycle,
#             reduction_rate,
#         ]
#     )

print(f"\n结果已保存到: {csv_file}")
