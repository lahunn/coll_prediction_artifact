#!/usr/bin/env python3
"""
球体碰撞检测预测仿真程序（nDOF机器人）- 无碰撞边筛选版本

使用球体近似进行碰撞检测的预测策略评估
只对无碰撞边数据进行仿真模拟

数据格式:
- sphere_link_data[edge][pose][sphere] = [x, y, z, radius]
- sphere_link_coll_data[edge][pose][sphere] = 1 or 0

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
binnumber = 16
intervalsize = 2 / binnumber
bins = np.zeros(binnumber)
start = -1
for i in range(binnumber):
    bins[i] = start
    start += intervalsize

# --- Global Statistics ---
fall_prediction = 0
fall_cycle = 0

# --- Simulation Parameters from Command Line ---
if len(sys.argv) < 8:
    print(
        "Usage: python prediction_simulation_nDOF_no_collision.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <num_benchmarks> <robot_name> [collision_model_type] [num_oocds_list]"
    )
    print(
        'Example: python prediction_simulation_nDOF_no_collision.py 0.5 0.1 8 ../trace_files/scene_benchmarks/bit_collision_data franka_14 100 franka link "1,2,4,7,14,28"'
    )
    sys.exit(1)

threshold = float(sys.argv[1])
sample_rate = float(sys.argv[2])
qnoncoll_multiplier = int(sys.argv[3])
data_folder = sys.argv[4]
basename = sys.argv[5]
num_benchmarks = int(sys.argv[6])
robot_name = sys.argv[7]

# Handle optional arguments for collision_model_type and num_oocds_list
collision_model_type = "link"
num_oocds_list = [7]

if len(sys.argv) > 8:
    arg8 = sys.argv[8]
    # Check if arg8 is the list (backward compatibility or omitted type)
    if "," in arg8 or arg8.replace(" ", "").isdigit():
        num_oocds_list = [int(x.strip()) for x in arg8.split(",")]
    else:
        collision_model_type = arg8
        if len(sys.argv) > 9:
            num_oocds_str = sys.argv[9]
            num_oocds_list = [int(x.strip()) for x in num_oocds_str.split(",")]

# 获取机器人参数
robot_params = get_robot_params(robot_name)

if collision_model_type == "sphere":
    num_elements = robot_params["sphere_num"]
    element_cost = 48  # sphere_cost
    csv_file = "result_files/sphere_oocds_analysis.csv"
    print_title = "=== Sphere Collision Detection Prediction Simulation (No Collision) ==="
else:
    num_elements = robot_params["obb_num"]
    element_cost = robot_params["obb_cost"]
    csv_file = "result_files/obb_oocds_analysis.csv"
    print_title = "=== OBB Collision Detection Prediction Simulation (No Collision) ==="

qnoncoll_len = num_elements * qnoncoll_multiplier

print(print_title)
print(f"Threshold: {threshold}")
print(f"Sample Rate: {sample_rate}")
print(f"Queue Length Multiplier: {qnoncoll_multiplier}")
print(f"Non-collision Queue Length: {qnoncoll_len}")
print(f"Data Folder: {data_folder}")
print(f"Number of Benchmarks: {num_benchmarks}")
print(f"Collision Model: {collision_model_type}")
print(f"OOCD数量列表: {num_oocds_list}")
print("=" * 50)

# --- Benchmark Range ---
benchrange = range(1, num_benchmarks + 1)

# --- 分析不同OOCD数量下的性能 ---
results = []

for num_oocds in num_oocds_list:
    print(f"\n--- 测试 OOCD 数量: {num_oocds} ---")

    # --- Global Statistics ---
    fall_prediction = 0
    fall_cycle = 0
    fall_theoretical_cycles = 0

    # --- Main Simulation Loop ---
    for benchid in tqdm(benchrange, desc=f"OOCD={num_oocds} 处理基准测试"):
        all_prediction = 0
        all_cycle = 0
        all_theoretical_cycles = 0
        colldict = {}

        # 加载数据（支持新的3元组格式）
        edge_link_data, edge_link_coll_data = (
            su.load_data(basename, benchid, data_folder, collision_model_type=collision_model_type)
        )

        if edge_link_data is None or edge_link_coll_data is None:
            continue

        # 处理每条边
        for edge_idx, (edge, edge_coll) in enumerate(
            zip(edge_link_data, edge_link_coll_data)
        ):
            if not edge_coll:
                continue

            # --- 检查是否为无碰撞边 ---
            # 如果边中包含任何碰撞（值为0），则跳过此边
            has_collision = any(
                link_coll == 0 for pose_coll in edge_coll for link_coll in pose_coll
            )
            if has_collision:
                continue  # 跳过有碰撞的边，只处理无碰撞边

            # 计算理论最小周期数
            all_theoretical_cycles += su.calculate_oracle_cycles(
                edge_coll, num_oocds=num_oocds, cycle_check=element_cost
            )

            # --- CSP Rearrangement ---
            # 将edge数据重排为适合CSP策略的顺序
            linklist, linklist_coll = su.csp_rearrange(edge, edge_coll, groupsize=4)

            # --- Run Centralized Simulation ---
            edge_query_count, colldict, _, cycle = (
                su.simulate_parallel_collision_detection(
                    linklist,
                    linklist_coll,
                    colldict,
                    threshold,
                    sample_rate,
                    bins,
                    qnoncoll_len=qnoncoll_len,
                    cycle_check=element_cost,
                    num_oocds=num_oocds,
                )
            )

            all_prediction += edge_query_count
            all_cycle += cycle

        fall_prediction += all_prediction
        fall_cycle += all_cycle
        fall_theoretical_cycles += all_theoretical_cycles

        # 每处理10个benchmark打印一次
        if (benchid + 1) % 10 == 0:
            print(f"[{benchid + 1}/{num_benchmarks}] 预测查询: {all_prediction:.2f}")

    # 存储结果
    results.append(
        {
            "num_oocds": num_oocds,
            "fall_prediction": fall_prediction,
            "fall_cycle": fall_cycle,
            "fall_theoretical_cycles": fall_theoretical_cycles,
        }
    )

    print(f"OOCD={num_oocds} 结果:")
    print(f"  预测查询总数: {fall_prediction:.2f}")
    print(f"  预测周期总数 (成本): {fall_cycle}")
    print(f"  理论最小周期数: {fall_theoretical_cycles}")

print("\n" + "=" * 60)
print("OOCD数量与总周期数关系分析:")
print("=" * 60)

# 分析关系
baseline_oocds = num_oocds_list[0]
baseline_result = next(r for r in results if r["num_oocds"] == baseline_oocds)

print(f"基准 OOCD 数量: {baseline_oocds}")
print(f"基准总周期数: {baseline_result['fall_cycle']}")
print()

for result in results:
    num_oocds = result["num_oocds"]
    cycle_count = result["fall_cycle"]
    prediction_count = result["fall_prediction"]

    # 计算与基准的比率
    cycle_ratio = (
        cycle_count / baseline_result["fall_cycle"]
        if baseline_result["fall_cycle"] > 0
        else 0
    )
    expected_ratio = baseline_oocds / num_oocds  # 理论上的线性关系

    print(
        f"OOCD={num_oocds:2d}: 周期数={cycle_count:8d}, 查询数={prediction_count:.0f}, "
        f"实际比率={cycle_ratio:.3f}, 期望比率={expected_ratio:.3f}"
    )

print("\n" + "=" * 60)

# 输出到CSV
with open(csv_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头（如果文件为空）
    if csvfile.tell() == 0:
        writer.writerow(
            [
                "threshold",
                "sample_rate",
                "qnoncoll_multiplier",
                "basename",
                "num_benchmarks",
                "robot_name",
                "num_oocds",
                "fall_prediction",
                "fall_cycle",
                "fall_theoretical_cycles",
            ]
        )

    # 写入结果
    for result in results:
        writer.writerow(
            [
                threshold,
                sample_rate,
                qnoncoll_multiplier,
                basename,
                num_benchmarks,
                robot_name,
                result["num_oocds"],
                result["fall_prediction"],
                result["fall_cycle"],
                result["fall_theoretical_cycles"],
            ]
        )
