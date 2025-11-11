#!/usr/bin/env python3
"""
统计球体碰撞检测消耗的平均cycle数

直接从数据文件中读取cycle信息，计算指定文件范围中球体碰撞检测的平均周期数统计
数据格式: sphere_link_coll_cycles[edge][pose][sphere] = cycle_count
"""

import sys
import numpy as np
from tqdm import tqdm

# --- Simulation Parameters from Command Line ---
if len(sys.argv) < 4:
    print(
        "Usage: python test_average_sphere_cycles.py <data_folder> <basename> <num_benchmarks>"
    )
    print(
        "Example: python test_average_sphere_cycles.py ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 50"
    )
    sys.exit(1)

data_folder = sys.argv[1]
basename = sys.argv[2]
num_benchmarks = int(sys.argv[3])

print("=== 球体碰撞检测Cycle分布统计 ===")
print(f"数据文件夹: {data_folder}")
print(f"基准测试数量: {num_benchmarks}")
print("=" * 50)

# --- Benchmark Range ---
benchrange = range(1, num_benchmarks + 1)

# --- Statistics Collection ---
all_cycles = []
valid_benchmarks = 0

# --- Main Analysis Loop ---
for benchid in tqdm(benchrange, desc="处理基准测试"):
    # 加载带有cycles的数据
    filename = f"{data_folder}/{basename}_{benchid:04d}_sphere_geometric_cycles.pkl"
    try:
        import pickle
        with open(filename, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 3:
                sphere_link_data, sphere_link_coll_data, sphere_link_coll_cycles = data
            else:
                continue
    except FileNotFoundError:
        continue
    except Exception as e:
        print(f"加载文件 {filename} 时出错: {e}")
        continue

    if sphere_link_coll_cycles is None or not isinstance(sphere_link_coll_cycles, list):
        continue

    valid_benchmarks += 1
    bench_cycles = []

    # 提取所有cycle值
    for edge_cycles in sphere_link_coll_cycles:
        for pose_cycles in edge_cycles:
            for sphere_cycle in pose_cycles:
                if isinstance(sphere_cycle, (int, float)):
                    bench_cycles.append(sphere_cycle)
                    all_cycles.append(sphere_cycle)

    if bench_cycles:
        bench_avg_cycle = np.mean(bench_cycles)
        bench_std_cycle = np.std(bench_cycles)
        bench_min_cycle = np.min(bench_cycles)
        bench_max_cycle = np.max(bench_cycles)

        # 每处理10个benchmark打印一次
        if valid_benchmarks % 10 == 0:
            print(
                f"[{valid_benchmarks}/{num_benchmarks}] 平均cycle: {bench_avg_cycle:.2f}, 范围: [{bench_min_cycle:.2f}, {bench_max_cycle:.2f}]"
            )

# --- Final Statistics ---
if all_cycles:
    overall_avg_cycle = np.mean(all_cycles)
    overall_std_cycle = np.std(all_cycles)
    overall_min_cycle = np.min(all_cycles)
    overall_max_cycle = np.max(all_cycles)
    overall_median_cycle = np.median(all_cycles)

    # 计算分位数
    q25 = np.percentile(all_cycles, 25)
    q75 = np.percentile(all_cycles, 75)
    q95 = np.percentile(all_cycles, 95)

    print("\n" + "=" * 50)
    print("Cycle分布统计结果:")
    print(f"  有效基准测试数: {valid_benchmarks}")
    print(f"  总cycle样本数: {len(all_cycles)}")
    print(f"  平均cycle数: {overall_avg_cycle:.2f}")
    print(f"  标准差: {overall_std_cycle:.2f}")
    print(f"  中位数: {overall_median_cycle:.2f}")
    print(f"  最小cycle数: {overall_min_cycle:.2f}")
    print(f"  最大cycle数: {overall_max_cycle:.2f}")
    print(f"  25%分位数: {q25:.2f}")
    print(f"  75%分位数: {q75:.2f}")
    print(f"  95%分位数: {q95:.2f}")
    print("=" * 50)
else:
    print("未找到有效的数据文件进行统计")