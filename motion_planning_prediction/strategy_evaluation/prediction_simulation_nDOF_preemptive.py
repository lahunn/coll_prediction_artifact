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

from common_simulation_utils import get_bins, print_final_statistics
from tqdm import tqdm
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# 添加 trace_generation 目录到 Python 路径
from trace_generation.config.ana_parameters import get_robot_params
import simulation_utils as su

# --- Simulation Settings ---
quant_bits = 4  # 4 bits per dimension (16 bins)
bins = su.calculate_bins_from_workspace("iiwa", quant_bits)

# --- Global Statistics ---
fall_prediction = 0
fall_oracle = 0
total_checks = 0
fall_cycle = 0
fall_preemption = 0

# --- Simulation Parameters from Command Line ---
if len(sys.argv) < 8:
    print(
        "Usage: python prediction_simulation_nDOF_preemptive.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <num_benchmarks> <robot_name> [collision_model_type]"
    )
    print(
        "Example: python prediction_simulation_nDOF_preemptive.py 0.5 0.1 8 ../trace_files/scene_benchmarks/bit_collision_data iiwa_7 10 iiwa link"
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
    check_cost = robot_params["sphere_cost"]
    csv_file = "result_files/sphere_results_preemptive.csv"
    print_title = (
        "=== Sphere Collision Detection Prediction Simulation (Preemptive Version) ==="
    )
else:
    num_elements = robot_params["obb_num"]
    check_cost = robot_params["obb_cost"]
    csv_file = "result_files/obb_results_preemptive.csv"
    print_title = (
        "=== OBB Collision Detection Prediction Simulation (Preemptive Version) ==="
    )

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

    # 累计理论查询总数 (模拟理想的顺序Oracle)
    for edge_coll in edge_link_coll_data:
        for pose_coll in edge_coll:
            # 理想的顺序检查器：检查直到发现第一个碰撞，或者检查完所有元素都没有碰撞。
            try:
                # 找到第一个碰撞(值为0)的索引
                first_collision_index = pose_coll.index(0)
                # 加上找到它所需的检查次数 (索引从0开始，所以+1)
                total_checks += first_collision_index + 1
            except ValueError:
                # 如果 pose_coll 中没有0 (即当前姿态无碰撞)，则需要检查该姿态下的所有元素
                total_checks += len(pose_coll)

    # 处理每条边
    for edge_idx, (edge, edge_coll) in enumerate(
        zip(edge_link_data, edge_link_coll_data)
    ):
        if not edge_coll:
            continue

        # --- Oracle Calculation ---
        # Oracle: 检测到碰撞就停止，否则检查所有元素
        coll_found_oracle = any(
            link_coll == 0 for pose_coll in edge_coll for link_coll in pose_coll
        )
        if coll_found_oracle:
            all_oracle += 1
        else:
            # 如果没有碰撞，需要检查所有姿态的所有元素
            all_oracle += num_elements * len(edge_coll)

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
                num_oocds=7,
            )
        )

        all_prediction += edge_query_count
        all_cycle += cycle
        all_preemption += preemption_count

    fall_oracle += all_oracle
    fall_prediction += all_prediction
    fall_cycle += all_cycle
    fall_preemption += all_preemption

    # 每处理10个benchmark打印一次
    if (benchid + 1) % 10 == 0:
        print(
            f"[{benchid + 1}/{num_benchmarks}] 预测查询: {all_prediction:.2f}, Oracle查询: {all_oracle}"
        )


print_final_statistics(
    total_checks=total_checks,
    fall_prediction=fall_prediction,
    fall_oracle=fall_oracle,
    fall_cycle=fall_cycle,
    extra_stats={"抢占事件总数": fall_preemption}
)

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
#             total_checks,
#             fall_prediction,
#             fall_oracle,
#             fall_cycle,
#             fall_preemption,
#             reduction_rate,
#         ]
#     )
