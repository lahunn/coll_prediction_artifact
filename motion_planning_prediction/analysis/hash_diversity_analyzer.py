#!/usr/bin/env python3
"""
Hash编码多样性分析工具

用于分析单个edge及多个benchmark中所有unit的hash编码多样性统计：
- 计算每个edge中不同hash编码的数量
- 统计hash碰撞率和分布熵
- 支持单个benchmark和多benchmark的平均统计
"""

import numpy as np
from collections import defaultdict
import math

def analyze_hash_diversity(edge_coords, bins):
    """
    分析单个edge中所有unit的不同hash编码数量和多样性统计

    Args:
        edge_coords: edge的坐标数据 List[pose_index][link_index][link_coords]
        bins: 量化bin边界数组

    Returns:
        diversity_stats: 包含hash多样性分析的字典，包括：
            - total_units: 总unit数量 (pose数 * 每个pose的link数)
            - num_poses: pose总数
            - num_links_per_pose: 每个pose中的link数
            - unique_hashes: 不同hash编码的数量
            - collision_rate: hash碰撞率（重复hash数 / 总hash数）
            - entropy: 香农熵，衡量hash分布的均匀性 (0-1)
            - hash_frequency: hash值与出现频率的映射字典
            - per_link_stats: 每条link在所有pose中的hash统计
    """
    num_poses = len(edge_coords)
    num_links_per_pose = len(edge_coords[0]) if num_poses > 0 else 0
    total_units = num_poses * num_links_per_pose

    # 收集所有hash编码
    hash_frequency = defaultdict(int)
    per_link_stats = {}

    for link_idx in range(num_links_per_pose):
        per_link_stats[link_idx] = {"hashes": set(), "frequency": defaultdict(int)}

    for pose_idx in range(num_poses):
        for link_idx in range(num_links_per_pose):
            link_coords = edge_coords[pose_idx][link_idx]

            # 计算该unit的hash编码
            code = np.digitize(link_coords[0:3], bins, right=True)
            code_tuple = tuple(code)

            hash_frequency[code_tuple] += 1
            per_link_stats[link_idx]["hashes"].add(code_tuple)
            per_link_stats[link_idx]["frequency"][code_tuple] += 1

    # 计算统计指标
    unique_hashes = len(hash_frequency)
    collision_rate = 1.0 - (unique_hashes / total_units) if total_units > 0 else 0.0

    # 计算香农熵（衡量hash分布均匀性）
    entropy = 0.0
    if total_units > 0:
        for count in hash_frequency.values():
            p = count / total_units
            if p > 0:
                entropy -= p * math.log2(p)
        # 归一化到 [0, 1]
        max_entropy = (
            math.log2(min(unique_hashes, total_units)) if unique_hashes > 1 else 1.0
        )
        entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # 计算每条link的多样性
    per_link_diversity = {}
    for link_idx in range(num_links_per_pose):
        link_unique_hashes = len(per_link_stats[link_idx]["hashes"])
        link_collision_rate = (
            1.0 - (link_unique_hashes / num_poses) if num_poses > 0 else 0.0
        )
        per_link_diversity[link_idx] = {
            "unique_hashes": link_unique_hashes,
            "total_instances": num_poses,
            "collision_rate": link_collision_rate,
            "hash_distribution": dict(per_link_stats[link_idx]["frequency"]),
        }

    return {
        "total_units": total_units,
        "num_poses": num_poses,
        "num_links_per_pose": num_links_per_pose,
        "unique_hashes": unique_hashes,
        "collision_rate": collision_rate,
        "entropy": entropy,
        "hash_frequency": dict(hash_frequency),
        "per_link_stats": per_link_diversity,
    }


def print_hash_diversity_report(diversity_stats):
    """打印单个edge的hash多样性分析报告"""
    print("\n" + "=" * 70)
    print("Hash多样性分析报告")
    print("=" * 70)

    print("\n总体统计:")
    print(f"  总Unit数量: {diversity_stats['total_units']}")
    print(f"  Pose数量: {diversity_stats['num_poses']}")
    print(f"  每个Pose的Link数: {diversity_stats['num_links_per_pose']}")
    print(f"  不同Hash编码数: {diversity_stats['unique_hashes']}")
    print(
        f"  Hash碰撞率: {diversity_stats['collision_rate']:.4f} ({diversity_stats['collision_rate'] * 100:.2f}%)"
    )
    print(f"  Hash分布熵: {diversity_stats['entropy']:.4f} (0=集中, 1=分散)")

    print("\nLink级别多样性统计:")
    print(f"{'Link':>6} {'不同Hash':>12} {'总实例':>10} {'碰撞率':>12}")
    print("-" * 42)

    for link_idx in sorted(diversity_stats["per_link_stats"].keys()):
        link_div = diversity_stats["per_link_stats"][link_idx]
        print(
            f"{link_idx:6d} {link_div['unique_hashes']:12d} "
            f"{link_div['total_instances']:10d} {link_div['collision_rate']:11.4f}"
        )

    # 打印最常见的hash编码（前10个）
    sorted_hashes = sorted(
        diversity_stats["hash_frequency"].items(), key=lambda x: x[1], reverse=True
    )

    print("\n最常见的Hash编码 (前10):")
    print(f"{'Hash编码':>20} {'出现次数':>10} {'占比':>10}")
    print("-" * 42)

    for code_tuple, count in sorted_hashes[:10]:
        percentage = (
            (count / diversity_stats["total_units"] * 100)
            if diversity_stats["total_units"] > 0
            else 0
        )
        print(f"{str(code_tuple):>20} {count:10d} {percentage:9.2f}%")


def analyze_hash_diversity_per_benchmark(all_results_for_benchmark, bins):
    """
    分析单个benchmark中所有edge的hash编码多样性统计（计算平均）

    Args:
        all_results_for_benchmark: 单个benchmark的所有edge分析结果列表
        bins: 量化bin边界数组

    Returns:
        benchmark_diversity_stats: 包含该benchmark的hash多样性统计字典
    """
    if not all_results_for_benchmark:
        return None

    edge_diversity_list = []

    for result in all_results_for_benchmark:
        edge_coords = result.get("edge_coords")
        if edge_coords is not None:
            diversity = analyze_hash_diversity(edge_coords, bins)
            edge_diversity_list.append(diversity)

    if not edge_diversity_list:
        return None

    # 计算所有edge的平均值
    avg_total_units = sum(d["total_units"] for d in edge_diversity_list) / len(
        edge_diversity_list
    )
    avg_unique_hashes = sum(d["unique_hashes"] for d in edge_diversity_list) / len(
        edge_diversity_list
    )
    avg_collision_rate = sum(
        d["collision_rate"] for d in edge_diversity_list
    ) / len(edge_diversity_list)
    avg_entropy = sum(d["entropy"] for d in edge_diversity_list) / len(
        edge_diversity_list
    )

    return {
        "edge_count": len(edge_diversity_list),
        "avg_total_units": avg_total_units,
        "avg_unique_hashes": avg_unique_hashes,
        "avg_collision_rate": avg_collision_rate,
        "avg_entropy": avg_entropy,
        "edge_diversity_list": edge_diversity_list,
    }


def analyze_hash_diversity_multi_benchmark(all_analysis_results, bins):
    """
    分析多个benchmark的hash编码多样性统计（计算平均）

    Args:
        all_analysis_results: 所有benchmark的所有edge分析结果列表
        bins: 量化bin边界数组

    Returns:
        multi_benchmark_diversity_stats: 多个benchmark的hash多样性统计字典
    """
    if not all_analysis_results:
        return None

    # 按benchmark分组
    benchmark_groups = defaultdict(list)
    for result in all_analysis_results:
        benchid = result["benchid"]
        benchmark_groups[benchid].append(result)

    benchmark_diversity_stats = {}
    all_unique_hashes = []
    all_collision_rates = []
    all_entropies = []

    for benchid in sorted(benchmark_groups.keys()):
        group_results = benchmark_groups[benchid]
        edge_diversity_list = []

        for result in group_results:
            edge_coords = result.get("edge_coords")
            if edge_coords is not None:
                diversity = analyze_hash_diversity(edge_coords, bins)
                edge_diversity_list.append(diversity)
                all_unique_hashes.append(diversity["unique_hashes"])
                all_collision_rates.append(diversity["collision_rate"])
                all_entropies.append(diversity["entropy"])

        if edge_diversity_list:
            avg_unique_hashes = sum(
                d["unique_hashes"] for d in edge_diversity_list
            ) / len(edge_diversity_list)
            avg_collision_rate = sum(
                d["collision_rate"] for d in edge_diversity_list
            ) / len(edge_diversity_list)
            avg_entropy = sum(d["entropy"] for d in edge_diversity_list) / len(
                edge_diversity_list
            )

            benchmark_diversity_stats[benchid] = {
                "edge_count": len(edge_diversity_list),
                "avg_unique_hashes": avg_unique_hashes,
                "avg_collision_rate": avg_collision_rate,
                "avg_entropy": avg_entropy,
            }

    # 计算全局平均值
    global_stats = {
        "total_benchmarks": len(benchmark_diversity_stats),
        "total_edges": sum(s["edge_count"] for s in benchmark_diversity_stats.values()),
        "global_avg_unique_hashes": sum(
            all_unique_hashes
        ) / len(all_unique_hashes)
        if all_unique_hashes
        else 0.0,
        "global_avg_collision_rate": sum(all_collision_rates)
        / len(all_collision_rates)
        if all_collision_rates
        else 0.0,
        "global_avg_entropy": sum(all_entropies) / len(all_entropies)
        if all_entropies
        else 0.0,
    }

    return {
        "per_benchmark": benchmark_diversity_stats,
        "global_stats": global_stats,
    }


def print_multi_benchmark_diversity_report(diversity_report):
    """
    打印多benchmark的hash多样性分析报告
    """
    if diversity_report is None:
        return

    global_stats = diversity_report["global_stats"]
    per_benchmark = diversity_report["per_benchmark"]

    print("\n" + "=" * 70)
    print("Hash多样性综合统计 (多Benchmark平均)")
    print("=" * 70)

    print("\n全局统计:")
    print(f"  总Benchmark数: {global_stats['total_benchmarks']}")
    print(f"  总Edge数: {global_stats['total_edges']}")
    print(f"  全局平均不同Hash编码数: {global_stats['global_avg_unique_hashes']:.2f}")
    print(
        f"  全局平均碰撞率: {global_stats['global_avg_collision_rate']:.4f} ({global_stats['global_avg_collision_rate']*100:.2f}%)"
    )
    print(f"  全局平均Hash分布熵: {global_stats['global_avg_entropy']:.4f}")

    print("\n按Benchmark统计:")
    print(
        f"{'Benchmark':>12} {'Edge数':>8} {'平均Hash数':>15} {'平均碰撞率':>15} {'平均熵':>12}"
    )
    print("-" * 65)

    for benchid in sorted(per_benchmark.keys()):
        stats = per_benchmark[benchid]
        print(
            f"{benchid:12d} {stats['edge_count']:8d} "
            f"{stats['avg_unique_hashes']:15.2f} "
            f"{stats['avg_collision_rate']:15.4f} "
            f"{stats['avg_entropy']:12.4f}"
        )


def analyze_pose_hash_bit_differences(pose_coords, bins):
    """
    分析单个pose中各个unit的hash编码bit位差异

    Args:
        pose_coords: 单个pose的坐标数据 List[link_index][link_coords]
        bins: 量化bin边界数组

    Returns:
        bit_stats: 包含bit位差异统计的字典：
            - total_units: 该pose中的总unit数（link数）
            - unique_hashes: 不同hash编码数
            - bit_differences: {(dim, bit_pos): count} bit位差异统计
            - per_unit_codes: 每个unit的hash编码
    """
    num_units = len(pose_coords)
    hash_codes = []
    
    # 计算所有unit的hash编码
    for link_coords in pose_coords:
        code = np.digitize(link_coords[0:3], bins, right=True)
        hash_codes.append(tuple(code))
    
    unique_hashes = len(set(hash_codes))
    
    # 统计各bit位的差异
    bit_differences = defaultdict(int)
    for i in range(num_units):
        for j in range(i + 1, num_units):
            for dim in range(len(hash_codes[i])):
                if hash_codes[i][dim] != hash_codes[j][dim]:
                    xor_val = hash_codes[i][dim] ^ hash_codes[j][dim]
                    bit_pos = 0
                    while xor_val > 0:
                        if xor_val & 1:
                            bit_differences[(dim, bit_pos)] += 1
                        xor_val >>= 1
                        bit_pos += 1
    
    return {
        "total_units": num_units,
        "unique_hashes": unique_hashes,
        "bit_differences": dict(bit_differences),
        "per_unit_codes": hash_codes,
    }


def analyze_pose_hash_bit_differences_multi_benchmark(all_analysis_results, bins):
    """
    分析多个benchid中所有edge的pose级hash编码bit位差异

    Args:
        all_analysis_results: 所有benchmark的所有edge分析结果列表
        bins: 量化bin边界数组

    Returns:
        pose_bit_stats: 包含所有pose的bit位差异统计字典：
            - total_benchmarks: 总benchmark数
            - total_edges: 总edge数
            - total_poses: 总pose数
            - total_units: 总unit数
            - avg_unique_hashes: 平均不同hash编码数
            - unique_hashes_distribution: 不同hash编码数的分布列表
            - bit_differences_aggregated: {(dim, bit_pos): count} 聚合的bit位差异
            - per_benchmark_stats: 按benchmark统计的pose级bit差异
    """
    if not all_analysis_results:
        return None

    # 按benchmark分组
    benchmark_groups = defaultdict(list)
    for result in all_analysis_results:
        benchid = result["benchid"]
        benchmark_groups[benchid].append(result)

    total_poses = 0
    total_edges = 0
    total_units_all = 0
    bit_differences_aggregated = defaultdict(int)
    unique_hashes_list = []
    per_benchmark_stats = {}

    for benchid in sorted(benchmark_groups.keys()):
        group_results = benchmark_groups[benchid]
        benchmark_bit_diffs = defaultdict(int)
        benchmark_poses = 0
        benchmark_unique_hashes = []

        for result in group_results:
            edge_coords = result.get("edge_coords")
            if edge_coords is not None:
                total_edges += 1
                num_poses = len(edge_coords)
                total_poses += num_poses
                total_units_all += num_poses * len(edge_coords[0]) if num_poses > 0 else 0
                benchmark_poses += num_poses

                # 分析每个pose
                for pose_idx, pose_coords in enumerate(edge_coords):
                    bit_stats = analyze_pose_hash_bit_differences(pose_coords, bins)
                    benchmark_unique_hashes.append(bit_stats["unique_hashes"])
                    unique_hashes_list.append(bit_stats["unique_hashes"])

                    # 聚合bit位差异
                    for (dim, bit_pos), count in bit_stats["bit_differences"].items():
                        bit_differences_aggregated[(dim, bit_pos)] += count
                        benchmark_bit_diffs[(dim, bit_pos)] += count

        if benchmark_unique_hashes:
            per_benchmark_stats[benchid] = {
                "edge_count": len(group_results),
                "pose_count": benchmark_poses,
                "avg_unique_hashes": sum(benchmark_unique_hashes) / len(benchmark_unique_hashes),
                "bit_differences": dict(benchmark_bit_diffs),
            }

    return {
        "total_benchmarks": len(benchmark_groups),
        "total_edges": total_edges,
        "total_poses": total_poses,
        "total_units": total_units_all,
        "avg_unique_hashes": sum(unique_hashes_list) / len(unique_hashes_list) if unique_hashes_list else 0.0,
        "unique_hashes_distribution": unique_hashes_list,
        "bit_differences_aggregated": dict(bit_differences_aggregated),
        "per_benchmark_stats": per_benchmark_stats,
    }


def print_pose_hash_bit_report(pose_bit_stats):
    """
    打印pose级hash编码bit位差异的汇总报告

    Args:
        pose_bit_stats: 来自 analyze_pose_hash_bit_differences_multi_benchmark 的结果
    """
    if pose_bit_stats is None:
        return

    print("\n" + "=" * 70)
    print("Pose级Hash编码Bit位差异分析汇总")
    print("=" * 70)

    print("\n全局统计:")
    print(f"  总Benchmark数: {pose_bit_stats['total_benchmarks']}")
    print(f"  总Edge数: {pose_bit_stats['total_edges']}")
    print(f"  总Pose数: {pose_bit_stats['total_poses']}")
    print(f"  总Unit数: {pose_bit_stats['total_units']}")
    print(f"  平均不同Hash编码数: {pose_bit_stats['avg_unique_hashes']:.2f}")

    # 打印bit位差异统计（前20个）
    bit_diffs = pose_bit_stats["bit_differences_aggregated"]
    if bit_diffs:
        print("\nBit位差异统计 (按差异数降序, 前20):")
        print(f"{'维度':>6} {'Bit位':>8} {'差异对数':>12}")
        print("-" * 30)

        sorted_bits = sorted(bit_diffs.items(), key=lambda x: x[1], reverse=True)
        for (dim, bit_pos), count in sorted_bits[:20]:
            print(f"{dim:6d} {bit_pos:8d} {count:12d}")
    else:
        print("\n无bit位差异")

    # 打印按Benchmark统计
    per_benchmark = pose_bit_stats["per_benchmark_stats"]
    if per_benchmark:
        print("\n按Benchmark统计:")
        print(f"{'Benchmark':>12} {'Edge数':>8} {'Pose数':>8} {'平均Hash数':>15}")
        print("-" * 50)

        for benchid in sorted(per_benchmark.keys()):
            stats = per_benchmark[benchid]
            print(
                f"{benchid:12d} {stats['edge_count']:8d} "
                f"{stats['pose_count']:8d} {stats['avg_unique_hashes']:15.2f}"
            )
