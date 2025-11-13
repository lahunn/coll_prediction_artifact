#!/usr/bin/env python3
"""
球体碰撞检测周期数统计分析程序

分析几何碰撞检测器返回的周期数据，统计：
1. 碰撞边和无碰撞边的总周期数、平均周期数
2. 单个球体的周期数分布（碰撞球体 vs 无碰撞球体）
"""

import sys
from collections import Counter
from tqdm import tqdm
import simulation_utils as su

# --- Command Line Parameters ---
if len(sys.argv) < 4:
    print(
        "Usage: python analyze_sphere_cycles.py <data_folder> <basename> <num_benchmarks>"
    )
    print(
        "Example: python analyze_sphere_cycles.py ../trace_files/scene_benchmarks/bit_collision_data iiwa_7 100"
    )
    sys.exit(1)

data_folder = sys.argv[1]
basename = sys.argv[2]
num_benchmarks = int(sys.argv[3])

print("=" * 60)
print("球体碰撞检测周期数统计分析")
print("=" * 60)
print(f"数据文件夹: {data_folder}")
print(f"基准名称: {basename}")
print(f"基准数量: {num_benchmarks}")
print("=" * 60)

# --- Statistics Variables ---
total_collision_edge_cycles = 0
total_free_edge_cycles = 0
collision_edge_count = 0
free_edge_count = 0

# 单个球体周期数分布
collision_sphere_cycles_distribution = Counter()
free_sphere_cycles_distribution = Counter()

# --- Main Analysis Loop ---
benchrange = range(1, num_benchmarks + 1)
has_cycle_data = False

for benchid in tqdm(benchrange, desc="分析周期数据"):
    # 加载球体数据
    sphere_link_data, sphere_link_coll_data, sphere_link_coll_cycles = (
        su.load_data_with_cycles(
            basename, benchid, data_folder, collision_model_type="sphere"
        )
    )

    if sphere_link_data is None or sphere_link_coll_data is None:
        continue

    if sphere_link_coll_cycles is None:
        continue

    has_cycle_data = True

    # 处理每条边
    for edge_idx, (edge, edge_coll) in enumerate(
        zip(sphere_link_data, sphere_link_coll_data)
    ):
        if not edge_coll:
            continue

        if edge_idx >= len(sphere_link_coll_cycles):
            continue

        edge_cycles_data = sphere_link_coll_cycles[edge_idx]

        # 检查该edge是否有碰撞
        edge_has_collision = any(
            sphere_coll == 0 for pose_coll in edge_coll for sphere_coll in pose_coll
        )

        # 遍历该edge的所有pose，统计每个球体的周期数
        for pose_idx, pose_cycles in enumerate(edge_cycles_data):
            if isinstance(pose_cycles, list) and pose_idx < len(edge_coll):
                pose_coll = edge_coll[pose_idx]

                # 遍历每个球体
                for sphere_idx, sphere_cycle in enumerate(pose_cycles):
                    if sphere_idx < len(pose_coll):
                        sphere_has_collision = pose_coll[sphere_idx] == 0

                        # 根据球体是否有碰撞，统计到对应的分布中
                        if sphere_has_collision:
                            collision_sphere_cycles_distribution[sphere_cycle] += 1
                        else:
                            free_sphere_cycles_distribution[sphere_cycle] += 1

        # 计算该edge所有pose的总周期数
        edge_total_cycles = 0
        for pose_cycles in edge_cycles_data:
            if isinstance(pose_cycles, list):
                edge_total_cycles += sum(pose_cycles)

        if edge_has_collision:
            total_collision_edge_cycles += edge_total_cycles
            collision_edge_count += 1
        else:
            total_free_edge_cycles += edge_total_cycles
            free_edge_count += 1

# --- Print Results ---
if not has_cycle_data:
    print("\n错误: 未找到周期数据！")
    print("请确保数据文件包含周期信息 (*_geometric_cycles.pkl)")
    sys.exit(1)

print("\n" + "=" * 60)
print("边级别统计 (Edge-Level Statistics):")
print("=" * 60)
print(f"碰撞边总数: {collision_edge_count}")
print(f"无碰撞边总数: {free_edge_count}")
print(f"碰撞边总周期数: {total_collision_edge_cycles}")
print(f"无碰撞边总周期数: {total_free_edge_cycles}")

avg_collision_cycles = (
    total_collision_edge_cycles / collision_edge_count
    if collision_edge_count > 0
    else 0
)
avg_free_cycles = total_free_edge_cycles / free_edge_count if free_edge_count > 0 else 0
print(f"碰撞边平均周期数: {avg_collision_cycles:.2f}")
print(f"无碰撞边平均周期数: {avg_free_cycles:.2f}")

# 球体级别统计
total_collision_spheres = sum(collision_sphere_cycles_distribution.values())
total_free_spheres = sum(free_sphere_cycles_distribution.values())
total_spheres = total_collision_spheres + total_free_spheres

print("\n" + "=" * 60)
print("球体级别统计 (Sphere-Level Statistics):")
print("=" * 60)
print(f"总球体数: {total_spheres}")
print(
    f"碰撞球体数: {total_collision_spheres} ({total_collision_spheres / total_spheres * 100:.2f}%)"
)
print(
    f"无碰撞球体数: {total_free_spheres} ({total_free_spheres / total_spheres * 100:.2f}%)"
)

if collision_sphere_cycles_distribution:
    total_coll_cycles = sum(
        cycle * count for cycle, count in collision_sphere_cycles_distribution.items()
    )
    avg_coll_sphere_cycles = total_coll_cycles / total_collision_spheres
    print(f"碰撞球体平均周期数: {avg_coll_sphere_cycles:.2f}")

if free_sphere_cycles_distribution:
    total_free_cycles_sum = sum(
        cycle * count for cycle, count in free_sphere_cycles_distribution.items()
    )
    avg_free_sphere_cycles = total_free_cycles_sum / total_free_spheres
    print(f"无碰撞球体平均周期数: {avg_free_sphere_cycles:.2f}")

print("\n" + "=" * 60)
print("周期数分布 (Cycle Distribution):")
print("=" * 60)

if collision_sphere_cycles_distribution:
    print(f"\n碰撞球体周期数分布 (共 {total_collision_spheres} 个球体):")
    print("-" * 60)
    print(f"{'周期数':<10} {'频数':<15} {'百分比':<15} {'累积百分比':<15}")
    print("-" * 60)

    cumulative = 0
    for cycle_count, frequency in collision_sphere_cycles_distribution.most_common(30):
        percentage = frequency / total_collision_spheres * 100
        cumulative += percentage
        print(
            f"{cycle_count:<10} {frequency:<15} {percentage:>6.2f}%{'':<8} {cumulative:>6.2f}%"
        )

    if len(collision_sphere_cycles_distribution) > 30:
        print(
            f"... (显示前30个，共 {len(collision_sphere_cycles_distribution)} 种不同周期数)"
        )

if free_sphere_cycles_distribution:
    print(f"\n无碰撞球体周期数分布 (共 {total_free_spheres} 个球体):")
    print("-" * 60)
    print(f"{'周期数':<10} {'频数':<15} {'百分比':<15} {'累积百分比':<15}")
    print("-" * 60)

    cumulative = 0
    for cycle_count, frequency in free_sphere_cycles_distribution.most_common(30):
        percentage = frequency / total_free_spheres * 100
        cumulative += percentage
        print(
            f"{cycle_count:<10} {frequency:<15} {percentage:>6.2f}%{'':<8} {cumulative:>6.2f}%"
        )

    if len(free_sphere_cycles_distribution) > 30:
        print(
            f"... (显示前30个，共 {len(free_sphere_cycles_distribution)} 种不同周期数)"
        )

# 统计信息
print("\n" + "=" * 60)
print("分布统计 (Distribution Statistics):")
print("=" * 60)

if collision_sphere_cycles_distribution:
    min_coll = min(collision_sphere_cycles_distribution.keys())
    max_coll = max(collision_sphere_cycles_distribution.keys())
    print(f"碰撞球体周期数范围: [{min_coll}, {max_coll}]")
    print(f"碰撞球体周期数种类: {len(collision_sphere_cycles_distribution)}")

if free_sphere_cycles_distribution:
    min_free = min(free_sphere_cycles_distribution.keys())
    max_free = max(free_sphere_cycles_distribution.keys())
    print(f"无碰撞球体周期数范围: [{min_free}, {max_free}]")
    print(f"无碰撞球体周期数种类: {len(free_sphere_cycles_distribution)}")

print("=" * 60)
print("分析完成！")
print("=" * 60)
