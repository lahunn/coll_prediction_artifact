#!/usr/bin/env python3
"""
球体碰撞检测预测仿真程序（nDOF机器人）

使用球体近似进行碰撞检测的预测策略评估
数据格式: sphere_link_data[edge][pose][sphere] = [x, y, z, radius]
         sphere_link_coll_data[edge][pose][sphere] = 1 or 0
"""

import sys
import numpy as np
from tqdm import tqdm
import simulation_utils as su

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
fall_oracle = 0
total_sphere_checks = 0

# --- Simulation Parameters from Command Line ---
if len(sys.argv) < 6:
    print(
        "Usage: python prediction_simulation_nDOF_sphere.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <num_benchmarks>"
    )
    print(
        "Example: python prediction_simulation_nDOF_sphere.py 0.5 0.1 8 ../trace_files/scene_benchmarks/bit_collision_data franka_14 100"
    )
    sys.exit(1)

threshold = float(sys.argv[1])
sample_rate = float(sys.argv[2])
qnoncoll_multiplier = int(sys.argv[3])
data_folder = sys.argv[4]
basename = sys.argv[5]
num_benchmarks = int(sys.argv[6])

# 从示例数据推断球体数量（假设Kuka有15个球体）
# 如果需要自适应，可以从第一个文件读取
num_spheres = 61  # 可以根据实际机器人调整
qnoncoll_len = num_spheres * qnoncoll_multiplier

print("=== 球体碰撞检测预测仿真 ===")
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
    colldict = {}

    # 加载球体数据
    sphere_link_data, sphere_link_coll_data = su.load_sphere_data(
        basename, benchid, data_folder
    )

    if sphere_link_data is None or sphere_link_coll_data is None:
        continue

    # 累计理论查询总数 (模拟理想的顺序Oracle)
    for edge_coll in sphere_link_coll_data:
        for pose_coll in edge_coll:
            # 理想的顺序检查器：检查直到发现第一个碰撞，或者检查完所有球体都没有碰撞。
            try:
                # 找到第一个碰撞(值为0)的索引
                first_collision_index = pose_coll.index(0)
                # 加上找到它所需的检查次数 (索引从0开始，所以+1)
                total_sphere_checks += first_collision_index + 1
            except ValueError:
                # 如果 pose_coll 中没有0 (即当前姿态无碰撞)，则需要检查该姿态下的所有球体
                total_sphere_checks += len(pose_coll)

    # 处理每条边
    for edge, edge_coll in zip(sphere_link_data, sphere_link_coll_data):
        if not edge_coll:
            continue
        # --- Oracle Calculation ---
        # Oracle: 检测到碰撞就停止，否则检查所有球体
        coll_found_oracle = any(
            sphere_coll == 0 for pose_coll in edge_coll for sphere_coll in pose_coll
        )
        if coll_found_oracle:
            all_oracle += 1
        else:
            # 如果没有碰撞，需要检查所有姿态的所有球体
            all_oracle += num_spheres * len(edge_coll)

        # --- CSP Rearrangement ---
        # 将edge数据重排为适合CSP策略的顺序
        linklist, linklist_coll = su.csp_rearrange(edge, edge_coll, groupsize=4)

        # --- Run Centralized Simulation ---
        edge_query_count, colldict, _ = su.simulate_parallel_collision_detection(
            linklist,
            linklist_coll,
            colldict,
            threshold,
            sample_rate,
            bins,
            qnoncoll_len=qnoncoll_len,
        )

        all_prediction += edge_query_count

    fall_oracle += all_oracle
    fall_prediction += all_prediction

    # 每处理10个benchmark打印一次
    if (benchid + 1) % 10 == 0:
        print(
            f"[{benchid + 1}/{num_benchmarks}] 预测查询: {all_prediction:.2f}, Oracle查询: {all_oracle}"
        )

print("\n" + "=" * 50)
print("最终统计:")
print(f"  实际查询总数 (球体数): {total_sphere_checks}")
print(f"  预测查询总数: {fall_prediction:.2f}")
print(f"  Oracle查询总数: {fall_oracle}")
print(f"  查询减少率: {(1 - fall_prediction / total_sphere_checks) * 100:.2f}%")
print("=" * 50)
