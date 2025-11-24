#!/usr/bin/env python3
"""
Hash编码差异分析工具

分析重排后相邻pose的unit coord的hash编码差异：
- 对于重排序列中相邻的两个pose对，比较它们对应的每个unit的coord hash编码
- 统计bit位级别的差异分布
- 用于评估重排策略对hash冲突率的影响
- 支持多种碰撞模型: sphere, link 等
"""
# python analyze_hash_patterns.py iiwa_7 1-20 ../../trace_files/scene_benchmarks/bit_collision_data iiwa --collision-model link

import sys
import os
import numpy as np
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su
from hash_diversity_analyzer import (
    analyze_hash_diversity_multi_benchmark,
    print_multi_benchmark_diversity_report,
    analyze_pose_hash_bit_differences_multi_benchmark,
    print_pose_hash_bit_report,
)

# 全局参数配置
QUANT_MIN = -1.5  # 量化最小值
QUANT_MAX = 1.5  # 量化最大值
QUANT_BITS = 4  # 量化位数
BINS = su.calculate_bins(QUANT_MIN, QUANT_MAX, QUANT_BITS)

# 全局变量
collision_model_type = "link"  # 碰撞模型类型: "sphere" 或 "link"


def count_bit_differences(code1, code2):
    """
    比较两个hash编码的bit位差异

    Args:
        code1, code2: 两个量化编码数组

    Returns:
        diff_bits: 差异的bit位位置列表 (dim, bit_pos)
        diff_count: 差异的bit总数
        diff_dimensions: 差异涉及的维度列表
        bit_level_diffs: 按bit位统计的差异数字典 {(dim, bit_pos): count}
    """
    diff_bits = []
    diff_dimensions = []
    diff_count = 0
    bit_level_diffs = defaultdict(int)

    # 按维度比较
    for dim in range(len(code1)):
        if code1[dim] != code2[dim]:
            diff_dimensions.append(dim)
            # 比较该维度的bit位
            xor_val = code1[dim] ^ code2[dim]
            bit_pos = 0
            while xor_val > 0:
                if xor_val & 1:
                    diff_bits.append((dim, bit_pos))
                    bit_level_diffs[(dim, bit_pos)] += 1
                    diff_count += 1
                xor_val >>= 1
                bit_pos += 1

    return diff_bits, diff_count, diff_dimensions, bit_level_diffs


def analyze_pose_pair_hashes(edge_coords, reorder_sequence):
    """
    分析重排后相邻pose对的hash差异

    Args:
        edge_coords: edge的坐标数据 List[List[link_coords]]
        reorder_sequence: 重排后的pose索引序列

    Returns:
        analysis_results: 包含详细分析数据的字典
    """
    num_poses = len(edge_coords)
    num_links_per_pose = len(edge_coords[0]) if num_poses > 0 else 0

    # 初始化统计数据结构
    analysis_results = {
        "num_poses": num_poses,
        "num_links_per_pose": num_links_per_pose,
        "pose_pairs": [],  # 相邻pose对的详细信息
        "link_level_stats": defaultdict(
            lambda: {
                "total_comparisons": 0,
                "same_hash_count": 0,
                "diff_hash_count": 0,
                "avg_bit_diff": 0.0,
                "bit_diff_distribution": defaultdict(int),  # {diff_count: frequency}
            }
        ),
        "dimension_level_stats": defaultdict(
            lambda: {
                "total_diffs": 0,
                "dimension_diff_count": defaultdict(int),  # {dim: count}
            }
        ),
        "bit_level_stats": defaultdict(int),  # {(dim, bit_pos): count}
        "overall_stats": {
            "total_pose_pairs": len(reorder_sequence) - 1,
            "total_link_comparisons": 0,
            "total_same_hash": 0,
            "total_diff_hash": 0,
            "avg_bit_diffs_per_link": 0.0,
        },
    }

    # 遍历相邻pose对
    for pair_idx in range(len(reorder_sequence) - 1):
        pose_idx1 = reorder_sequence[pair_idx]
        pose_idx2 = reorder_sequence[pair_idx + 1]

        pose_pair_info = {"pose_indices": (pose_idx1, pose_idx2), "links": []}

        # 比较每条link的hash编码
        for link_idx in range(num_links_per_pose):
            link_coords1 = edge_coords[pose_idx1][link_idx]
            link_coords2 = edge_coords[pose_idx2][link_idx]

            # 计算hash编码
            code1 = np.digitize(link_coords1[0:3], BINS, right=True)
            code2 = np.digitize(link_coords2[0:3], BINS, right=True)

            # 比较差异
            diff_bits, diff_count, diff_dims, bit_level_diffs = count_bit_differences(
                code1, code2
            )

            is_same_hash = diff_count == 0

            link_info = {
                "link_idx": link_idx,
                "code1": code1.tolist(),
                "code2": code2.tolist(),
                "is_same_hash": is_same_hash,
                "diff_bit_count": diff_count,
                "diff_dimensions": diff_dims,
                "diff_bits": diff_bits,
            }
            pose_pair_info["links"].append(link_info)

            # 更新统计
            analysis_results["link_level_stats"][link_idx]["total_comparisons"] += 1
            analysis_results["overall_stats"]["total_link_comparisons"] += 1

            if is_same_hash:
                analysis_results["link_level_stats"][link_idx]["same_hash_count"] += 1
                analysis_results["overall_stats"]["total_same_hash"] += 1
            else:
                analysis_results["link_level_stats"][link_idx]["diff_hash_count"] += 1
                analysis_results["overall_stats"]["total_diff_hash"] += 1
                analysis_results["link_level_stats"][link_idx]["bit_diff_distribution"][
                    diff_count
                ] += 1

            # 统计维度级别差异
            for dim in diff_dims:
                analysis_results["dimension_level_stats"][link_idx][
                    "dimension_diff_count"
                ][dim] += 1
                analysis_results["dimension_level_stats"][link_idx]["total_diffs"] += 1

            # 统计bit位级别差异
            for (dim, bit_pos), count in bit_level_diffs.items():
                analysis_results["bit_level_stats"][(dim, bit_pos)] += count

        analysis_results["pose_pairs"].append(pose_pair_info)

    # 计算平均值
    if analysis_results["overall_stats"]["total_link_comparisons"] > 0:
        total_bit_diffs = sum(
            diff_count * freq
            for link_stats in analysis_results["link_level_stats"].values()
            for diff_count, freq in link_stats["bit_diff_distribution"].items()
        )
        analysis_results["overall_stats"]["avg_bit_diffs_per_link"] = (
            total_bit_diffs
            / analysis_results["overall_stats"]["total_link_comparisons"]
        )

    # 计算每条link的平均bit差异
    for link_idx, link_stats in analysis_results["link_level_stats"].items():
        if link_stats["total_comparisons"] > 0:
            total_bits = sum(
                diff_count * freq
                for diff_count, freq in link_stats["bit_diff_distribution"].items()
            )
            link_stats["avg_bit_diff"] = total_bits / link_stats["total_comparisons"]

    return analysis_results


def print_analysis_report(analysis_results, verbose=False):
    """打印分析报告"""

    print("\n" + "=" * 70)
    print("Hash编码差异分析报告")
    print("=" * 70)

    overall = analysis_results["overall_stats"]
    print("\n总体统计:")
    print(f"  Pose对数: {overall['total_pose_pairs']}")
    print(f"  总Link比较数: {overall['total_link_comparisons']}")
    print(
        f"  Hash相同的Link数: {overall['total_same_hash']} ({overall['total_same_hash'] / max(1, overall['total_link_comparisons']) * 100:.1f}%)"
    )
    print(
        f"  Hash不同的Link数: {overall['total_diff_hash']} ({overall['total_diff_hash'] / max(1, overall['total_link_comparisons']) * 100:.1f}%)"
    )
    print(f"  平均bit差异数: {overall['avg_bit_diffs_per_link']:.2f}")

    print("\nLink级别统计:")
    print(
        f"{'Link':>6} {'总比较':>8} {'Hash相同':>10} {'Hash不同':>10} {'平均Bit差':>12} {'Bit差分布':>20}"
    )
    print("-" * 70)

    for link_idx in sorted(analysis_results["link_level_stats"].keys()):
        link_stats = analysis_results["link_level_stats"][link_idx]

        # 构造bit差分布字符串
        bit_dist_str = "{"
        for bit_diff in sorted(link_stats["bit_diff_distribution"].keys()):
            freq = link_stats["bit_diff_distribution"][bit_diff]
            bit_dist_str += f"{bit_diff}:{freq},"
        bit_dist_str = bit_dist_str.rstrip(",") + "}"

        print(
            f"{link_idx:6d} {link_stats['total_comparisons']:8d} "
            f"{link_stats['same_hash_count']:10d} {link_stats['diff_hash_count']:10d} "
            f"{link_stats['avg_bit_diff']:12.2f} {bit_dist_str:>20}"
        )

    if verbose:
        print("\n维度级别差异统计:")
        for link_idx in sorted(analysis_results["dimension_level_stats"].keys()):
            dim_stats = analysis_results["dimension_level_stats"][link_idx]
            print(f"  Link {link_idx}: 总差异数={dim_stats['total_diffs']}")
            for dim in sorted(dim_stats["dimension_diff_count"].keys()):
                count = dim_stats["dimension_diff_count"][dim]
                print(f"    维度{dim}: {count}次差异")

        print("\n详细Pose对信息 (前10对):")
        for pair_idx, pose_pair in enumerate(analysis_results["pose_pairs"][:10]):
            pose_indices = pose_pair["pose_indices"]
            print(f"  Pose对 {pair_idx}: ({pose_indices[0]} -> {pose_indices[1]})")

            same_count = sum(1 for link in pose_pair["links"] if link["is_same_hash"])
            print(f"    相同hash的link数: {same_count}/{len(pose_pair['links'])}")

            for link in pose_pair["links"][:3]:  # 只显示前3条link
                status = (
                    "相同"
                    if link["is_same_hash"]
                    else f"差异{link['diff_bit_count']}bit"
                )
                print(f"      Link{link['link_idx']}: {status}")


def print_bit_statistics(bit_diffs):
    """打印Bit位差异统计"""
    sorted_bit_diffs = sorted(bit_diffs.items(), key=lambda x: x[1], reverse=True)
    if sorted_bit_diffs:
        print("\nBit位差异统计 (按差异数降序):")
        print(f"{'维度':>6} {'Bit位':>8} {'差异数':>10}")
        print("-" * 30)
        for (dim, bit_pos), count in sorted_bit_diffs:
            print(f"{dim:6d} {bit_pos:8d} {count:10d}")


def print_benchmark_statistics(all_analysis_results):
    """打印按Benchmark统计"""
    print("\n" + "=" * 70)
    print("按Benchmark统计")
    print("=" * 70)
    print(f"{'Bench':>6} {'Edge数':>8} {'Hash相同':>12} {'比例':>10} {'平均bit差':>12}")
    print("-" * 60)

    benchmark_stats = defaultdict(
        lambda: {
            "edge_count": 0,
            "same_hash": 0,
            "total_comparisons": 0,
            "bit_diffs_sum": 0,
        }
    )

    for r in all_analysis_results:
        benchid = r["benchid"]
        stats = r["results"]["overall_stats"]
        benchmark_stats[benchid]["edge_count"] += 1
        benchmark_stats[benchid]["same_hash"] += stats["total_same_hash"]
        benchmark_stats[benchid]["total_comparisons"] += stats["total_link_comparisons"]
        benchmark_stats[benchid]["bit_diffs_sum"] += (
            stats["avg_bit_diffs_per_link"] * stats["total_link_comparisons"]
        )

    for benchid in sorted(benchmark_stats.keys()):
        stats = benchmark_stats[benchid]
        same_ratio = stats["same_hash"] / max(1, stats["total_comparisons"]) * 100
        avg_bit = stats["bit_diffs_sum"] / max(1, stats["total_comparisons"])
        print(
            f"{benchid:6d} {stats['edge_count']:8d} {stats['same_hash']:12d} "
            f"{same_ratio:9.1f}% {avg_bit:12.2f}"
        )


def process_edge(edge, edge_coll, benchid):
    """处理单条边的hash差异分析"""
    if not edge_coll:
        return None

    num_poses = len(edge)
    reorder_sequence = su.generate_recursive_reorder(num_poses, step_size=8)

    analysis_results = analyze_pose_pair_hashes(edge, reorder_sequence)
    return {"benchid": benchid, "edge_idx": 0, "results": analysis_results}


def process_benchmark(
    basename, benchid, data_folder, collision_model_type, load_with_cycles
):
    """处理单个benchmark的所有边"""
    # 加载数据
    if load_with_cycles:
        unit_link_data, unit_link_coll_data, _ = su.load_data_with_cycles(
            basename, benchid, data_folder, collision_model_type=collision_model_type
        )
    else:
        unit_link_data, unit_link_coll_data = su.load_data(
            basename, benchid, data_folder, collision_model_type=collision_model_type
        )

    if unit_link_data is None or unit_link_coll_data is None:
        print(f"  警告: 无法加载benchmark {benchid}的数据，跳过")
        return []

    print(f"  成功加载数据，共{len(unit_link_data)}条边")

    results = []
    for edge_idx, (edge, edge_coll) in enumerate(
        zip(unit_link_data, unit_link_coll_data)
    ):
        if not edge_coll:
            continue

        reorder_sequence = su.generate_recursive_reorder(len(edge), step_size=8)
        analysis_results = analyze_pose_pair_hashes(edge, reorder_sequence)
        results.append(
            {
                "benchid": benchid,
                "edge_idx": edge_idx,
                "results": analysis_results,
                "edge_coords": edge,
            }
        )

    return results


def parse_arguments():
    """解析命令行参数"""
    global collision_model_type

    if len(sys.argv) < 5:
        print(
            "Usage: python analyze_hash_patterns.py <basename> <benchid_range> <data_folder> "
            "<robot_name> [--collision-model TYPE] [--with-cycles]"
        )
        print(
            "Example: python analyze_hash_patterns.py iiwa_7 1-20 ../../trace_files/scene_benchmarks/bit_collision_data iiwa --collision-model link"
        )
        sys.exit(1)

    basename = sys.argv[1]
    benchid_arg = sys.argv[2]
    data_folder = sys.argv[3]
    robot_name = sys.argv[4]

    # 解析benchid范围
    if "-" in benchid_arg:
        start_id, end_id = map(int, benchid_arg.split("-"))
        benchid_list = list(range(start_id, end_id + 1))
    else:
        benchid_list = [int(benchid_arg)]

    # 解析可选参数
    if "--collision-model" in sys.argv:
        idx = sys.argv.index("--collision-model")
        if idx + 1 < len(sys.argv):
            collision_model_type = sys.argv[idx + 1]
    load_with_cycles = "--with-cycles" in sys.argv

    return basename, benchid_list, data_folder, robot_name, load_with_cycles


def main():
    """主函数"""
    global collision_model_type

    basename, benchid_list, data_folder, robot_name, load_with_cycles = (
        parse_arguments()
    )

    print("Hash编码差异分析")
    print(
        f"Basename: {basename}, Benchmark IDs: {benchid_list[0]}-{benchid_list[-1]} ({len(benchid_list)}个)"
    )
    print(f"数据文件夹: {data_folder}")
    print(f"机器人: {robot_name}")
    print(f"碰撞模型: {collision_model_type}")
    print(f"加载带cycles数据: {load_with_cycles}")
    print("=" * 70)

    # 处理所有benchid
    all_analysis_results = []
    for benchid in benchid_list:
        print(f"\n正在处理 Benchmark {benchid}...")
        results = process_benchmark(
            basename, benchid, data_folder, collision_model_type, load_with_cycles
        )
        all_analysis_results.extend(results)

    # 汇总统计
    if not all_analysis_results:
        print("错误: 未能处理任何数据")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    total_same_hash = sum(
        r["results"]["overall_stats"]["total_same_hash"] for r in all_analysis_results
    )
    total_comparisons = sum(
        r["results"]["overall_stats"]["total_link_comparisons"]
        for r in all_analysis_results
    )
    total_pose_pairs = sum(
        r["results"]["overall_stats"]["total_pose_pairs"] for r in all_analysis_results
    )
    avg_bit_diffs = sum(
        r["results"]["overall_stats"]["avg_bit_diffs_per_link"]
        * r["results"]["overall_stats"]["total_link_comparisons"]
        for r in all_analysis_results
    ) / max(1, total_comparisons)

    print(f"分析的Benchmark数量: {len(benchid_list)}")
    print(f"分析的Edge数量: {len(all_analysis_results)}")
    print(f"总Pose对数: {total_pose_pairs}")
    print(f"总Unit比较数: {total_comparisons}")
    print(f"Hash相同比例: {total_same_hash / max(1, total_comparisons) * 100:.1f}%")
    print(f"平均bit差异数: {avg_bit_diffs:.2f}")

    # 统计bit位差异
    bit_diffs = defaultdict(int)
    for r in all_analysis_results:
        for (dim, bit_pos), count in r["results"]["bit_level_stats"].items():
            bit_diffs[(dim, bit_pos)] += count

    print_bit_statistics(bit_diffs)
    print_benchmark_statistics(all_analysis_results)

    # Pose级Hash编码Bit位差异分析汇总（多benchmark）
    print("\n" + "=" * 70)
    print("Pose级Hash编码Bit位差异综合分析")
    print("=" * 70)
    pose_bit_stats = analyze_pose_hash_bit_differences_multi_benchmark(
        all_analysis_results, BINS
    )
    print_pose_hash_bit_report(pose_bit_stats)

    # # Hash多样性分析
    # print("\n" + "=" * 70)
    # print("Hash编码多样性分析")
    # print("=" * 70)
    # diversity_report = analyze_hash_diversity_multi_benchmark(all_analysis_results, BINS)
    # print_multi_benchmark_diversity_report(diversity_report)


if __name__ == "__main__":
    main()
