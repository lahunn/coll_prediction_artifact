#!/usr/bin/env python3
"""
OBB哈希编码变化规律分析脚本

分析多COPU场景下不同COPU中相同位置OBB的哈希编码差异，
确定差异主要由三维坐标哪些位引起，指导CHT的分块策略。

用法:
    python analyze_obb_hash_patterns.py <basename> <benchid> <data_folder>
    例如: python analyze_obb_hash_patterns.py iiwa_7 46 ../../trace_files/scene_benchmarks/bit_collision_data
"""

import sys
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import simulation_utils as su

# ============================================================================
# 配置参数
# ============================================================================
QUANT_MIN = -1.5
QUANT_MAX = 1.5
QUANT_BITS = 3
NUM_BINS = 2**QUANT_BITS

# ============================================================================
# 核心函数
# ============================================================================


def create_bins(quant_bits=QUANT_BITS):
    """生成量化bins"""
    num_bins = 2**quant_bits
    return np.linspace(QUANT_MIN, QUANT_MAX, num_bins + 1)


def quantize_coord(coord, bins):
    """将单个坐标量化为bin索引"""
    return np.digitize(coord, bins, right=True)


def encode_hash(code):
    """将量化码转换为哈希键字符串"""
    keyy = ""
    for val in code:
        if val < 10:
            keyy += "0"
        keyy += str(val)
    return keyy


def allocate_poses_to_copus(num_poses, num_copus):
    """
    根据COPU数量分配pose索引
    返回list of (start_idx, end_idx) tuples for each COPU
    """
    poses_per_copu = num_poses // num_copus
    remainder = num_poses % num_copus
    allocations = []

    for copu_id in range(num_copus):
        if copu_id < remainder:
            start = copu_id * (poses_per_copu + 1)
            end = start + poses_per_copu + 1
        else:
            start = (
                remainder * (poses_per_copu + 1)
                + (copu_id - remainder) * poses_per_copu
            )
            end = start + poses_per_copu
        allocations.append((start, end))

    return allocations


def extract_copu_data(edge_coords, edge_colls, copu_id, num_copus):
    """
    提取分配给特定COPU的数据

    Returns:
        list of coords, list of collision flags
    """
    allocations = allocate_poses_to_copus(len(edge_coords), num_copus)
    start_pose, end_pose = allocations[copu_id]

    copu_coords = []
    copu_colls = []

    for pose_idx in range(start_pose, end_pose):
        pose_coords = edge_coords[pose_idx]
        pose_colls = edge_colls[pose_idx]
        copu_coords.extend(pose_coords)
        copu_colls.extend(pose_colls)

    return copu_coords, copu_colls


def analyze_single_edge(edge_coords, edge_colls, num_copus, bins):
    """
    分析单条edge中不同COPU分配的OBB的哈希编码差异

    Returns:
        dict with analysis results
    """
    results = {
        "num_poses": len(edge_coords),
        "num_copus": num_copus,
        "obbs": [],  # list of OBB analysis
        "copus": {},  # copu_id -> {obb_idx -> hash_key}
    }

    # 为每个COPU提取数据并编码
    for copu_id in range(num_copus):
        copu_coords, _ = extract_copu_data(edge_coords, edge_colls, copu_id, num_copus)
        results["copus"][copu_id] = {}

        for obb_idx, coord in enumerate(copu_coords):
            code = quantize_coord(coord, bins)
            hash_key = encode_hash(code)
            results["copus"][copu_id][obb_idx] = {
                "coord": coord,
                "code": code,
                "hash_key": hash_key,
            }

    # 分析OBB差异
    # 找出同一位置OBB在不同COPU中的差异
    max_obbs = max(len(results["copus"][cid]) for cid in range(num_copus))

    for obb_idx in range(max_obbs):
        obb_data = {"obb_idx": obb_idx, "copus": {}}

        for copu_id in range(num_copus):
            if obb_idx in results["copus"][copu_id]:
                obb_data["copus"][copu_id] = results["copus"][copu_id][obb_idx]

        if len(obb_data["copus"]) > 1:
            # 计算差异
            obb_data["has_differences"] = check_obb_differences(obb_data["copus"])
            results["obbs"].append(obb_data)

    return results


def check_obb_differences(copu_obbs):
    """
    检查同一OBB在不同COPU中的差异

    Returns:
        dict with difference analysis
    """
    if len(copu_obbs) <= 1:
        return None

    copu_ids = sorted(copu_obbs.keys())
    reference_data = copu_obbs[copu_ids[0]]

    diff_info = {
        "reference_copu": copu_ids[0],
        "reference_hash": reference_data["hash_key"],
        "reference_code": reference_data["code"].copy(),
        "reference_coord": reference_data["coord"],
        "differences": {},
    }

    for copu_id in copu_ids[1:]:
        other_data = copu_obbs[copu_id]
        diff = {
            "hash_matches": reference_data["hash_key"] == other_data["hash_key"],
            "coord_diff": np.array(other_data["coord"])
            - np.array(reference_data["coord"]),
            "code_diff": other_data["code"] - reference_data["code"],
            "other_hash": other_data["hash_key"],
            "other_code": other_data["code"].copy(),
            "other_coord": other_data["coord"],
        }
        diff_info["differences"][copu_id] = diff

    return diff_info


def aggregate_analysis(all_edges_results):
    """
    聚合多条edge的分析结果

    Returns:
        dict with aggregated statistics
    """
    stats = {
        "total_edges": len(all_edges_results),
        "hash_match_rate": 0.0,
        "dimension_diff_freq": [0, 0, 0],  # x, y, z维度的差异频率
        "bit_diff_freq": defaultdict(int),  # {(dim, bit_pos): count}
        "difference_types": defaultdict(int),  # 差异类型计数
    }

    total_obbs_with_diff = 0
    total_hash_matches = 0
    total_comparisons = 0

    for edge_result in all_edges_results:
        for obb_data in edge_result["obbs"]:
            if not obb_data.get("has_differences"):
                continue

            diff_info = obb_data["has_differences"]
            for copu_id, diff in diff_info["differences"].items():
                total_comparisons += 1
                if diff["hash_matches"]:
                    total_hash_matches += 1
                else:
                    total_obbs_with_diff += 1

                # 分析维度差异
                for dim in range(3):
                    if diff["code_diff"][dim] != 0:
                        stats["dimension_diff_freq"][dim] += 1

                    # 分析bit位差异
                    for bit_pos in range(8):  # 8个bin = 3bit + 1
                        if (diff["code_diff"][dim] >> bit_pos) & 1:
                            stats["bit_diff_freq"][(dim, bit_pos)] += 1

                # 分类差异类型
                diff_type = classify_difference(diff)
                stats["difference_types"][diff_type] += 1

    if total_comparisons > 0:
        stats["hash_match_rate"] = total_hash_matches / total_comparisons
        stats["total_with_diff"] = total_obbs_with_diff
        stats["total_comparisons"] = total_comparisons

    return stats


def classify_difference(diff):
    """分类差异类型"""
    code_diff = diff["code_diff"]
    num_diff_dims = np.count_nonzero(code_diff)
    max_diff = np.max(np.abs(code_diff))

    if num_diff_dims == 0:
        return "no_diff"
    elif num_diff_dims == 1:
        if max_diff == 1:
            return "single_dim_lsb"
        else:
            return "single_dim_large"
    elif num_diff_dims == 2:
        return "two_dims"
    else:
        return "all_dims"


def generate_report(stats, output_dir):
    """生成分析报告"""
    report_path = os.path.join(output_dir, "hash_analysis_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("OBB哈希编码变化规律分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("【汇总统计】\n")
        f.write(f"  总edges数: {stats['total_edges']}\n")
        f.write(f"  总比较次数: {stats.get('total_comparisons', 0)}\n")
        f.write(f"  哈希值匹配率: {stats['hash_match_rate']:.2%}\n")
        f.write(f"  存在差异的OBB: {stats.get('total_with_diff', 0)}\n\n")

        f.write("【维度差异频率】\n")
        dims = ["X", "Y", "Z"]
        total_dim_diffs = sum(stats["dimension_diff_freq"])
        for dim_idx, dim_name in enumerate(dims):
            freq = stats["dimension_diff_freq"][dim_idx]
            rate = freq / total_dim_diffs * 100 if total_dim_diffs > 0 else 0
            f.write(f"  {dim_name}维: {freq:6d} ({rate:6.2f}%)\n")

        f.write("\n【Bit位差异分布】\n")
        f.write("  维度  Bit0  Bit1  Bit2  Bit3\n")
        for dim_idx, dim_name in enumerate(["X", "Y", "Z"]):
            f.write(f"  {dim_name}:    ")
            for bit_pos in range(4):
                count = stats["bit_diff_freq"].get((dim_idx, bit_pos), 0)
                f.write(f"{count:5d} ")
            f.write("\n")

        f.write("\n【差异类型分布】\n")
        total_types = sum(stats["difference_types"].values())
        for diff_type, count in sorted(
            stats["difference_types"].items(), key=lambda x: x[1], reverse=True
        ):
            rate = count / total_types * 100 if total_types > 0 else 0
            f.write(f"  {diff_type:20s}: {count:6d} ({rate:6.2f}%)\n")

        f.write("\n【CHT分块策略建议】\n")
        if stats["hash_match_rate"] > 0.95:
            f.write("  ✓ 建议：全局共享CHT（95%以上的哈希值匹配）\n")
            f.write("    - 多COPU间的OBB编码高度一致\n")
            f.write("    - 分块策略成本效益不显著\n")
        elif stats["dimension_diff_freq"][0] > stats["dimension_diff_freq"][1]:
            f.write("  ✓ 建议：按X维分块CHT（X维差异最频繁）\n")
            f.write("    - X维是主要变化来源\n")
            f.write("    - 考虑按X维bin索引分块存储\n")
        else:
            f.write("  ✓ 建议：维度独立分块（三维差异均衡）\n")
            f.write("    - 各维差异均衡分布\n")
            f.write("    - 考虑维度级分块或bit级分块\n")

        f.write("\n")

    print(f"✓ 报告已保存: {report_path}")
    return report_path


def generate_visualizations(stats, output_dir):
    """生成可视化图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 维度差异频率柱状图
    fig, ax = plt.subplots(figsize=(8, 6))
    dims = ["X", "Y", "Z"]
    freqs = stats["dimension_diff_freq"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    bars = ax.bar(
        dims, freqs, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )
    ax.set_ylabel("Difference Frequency", fontsize=12, fontweight="bold")
    ax.set_title(
        "OBB Hash Code Differences by Dimension", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)

    # 添加数值标签
    for bar, freq in zip(bars, freqs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(freq)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "dimension_diff_frequency.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("✓ 维度差异频率图已保存")

    # 2. Bit位差异热力图
    fig, ax = plt.subplots(figsize=(10, 6))
    dims = ["X", "Y", "Z"]
    bits = ["Bit0 (LSB)", "Bit1", "Bit2", "Bit3 (MSB)"]

    data_matrix = np.zeros((3, 4))
    for dim_idx in range(3):
        for bit_pos in range(4):
            data_matrix[dim_idx, bit_pos] = stats["bit_diff_freq"].get(
                (dim_idx, bit_pos), 0
            )

    im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_yticks(range(3))
    ax.set_xticklabels(bits)
    ax.set_yticklabels(dims)
    ax.set_xlabel("Bit Position", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dimension", fontsize=12, fontweight="bold")
    ax.set_title(
        "Bit-level Difference Distribution Heatmap", fontsize=14, fontweight="bold"
    )

    # 添加数值标签
    for i in range(3):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{int(data_matrix[i, j])}",
                ha="center",
                va="center",
                color="black",
                fontsize=11,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, label="Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "bit_diff_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Bit位差异热力图已保存")

    # 3. 差异类型分布饼图
    fig, ax = plt.subplots(figsize=(10, 8))
    types = sorted(stats["difference_types"].items(), key=lambda x: x[1], reverse=True)
    labels = [t[0] for t in types]
    sizes = [t[1] for t in types]
    colors_pie = plt.get_cmap("Set3")(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors_pie, startangle=90
    )
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)

    ax.set_title("Distribution of Difference Types", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        output_dir / "difference_types_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("✓ 差异类型分布图已保存")


def save_statistics_csv(stats, output_dir):
    """保存统计数据到CSV"""
    csv_path = os.path.join(output_dir, "hash_statistics.csv")

    data = {
        "Metric": [
            "Total Edges",
            "Total Comparisons",
            "Hash Match Rate (%)",
            "OBBs with Differences",
            "X Dimension Differences",
            "Y Dimension Differences",
            "Z Dimension Differences",
        ],
        "Value": [
            stats["total_edges"],
            stats.get("total_comparisons", 0),
            f"{stats['hash_match_rate'] * 100:.2f}",
            stats.get("total_with_diff", 0),
            stats["dimension_diff_freq"][0],
            stats["dimension_diff_freq"][1],
            stats["dimension_diff_freq"][2],
        ],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"✓ 统计数据已保存: {csv_path}")


def main():
    """主程序"""
    if len(sys.argv) < 4:
        print(
            "用法: python analyze_obb_hash_patterns.py <basename> <benchid> <data_folder>"
        )
        print(
            "例如: python analyze_obb_hash_patterns.py iiwa_7 46 ../../trace_files/scene_benchmarks/bit_collision_data"
        )
        sys.exit(1)

    basename = sys.argv[1]
    benchid = int(sys.argv[2])
    data_folder = sys.argv[3]

    # 创建输出目录
    output_dir = Path(__file__).parent / "result_files" / "obb_hash_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OBB哈希编码变化规律分析")
    print("=" * 80)
    print("\n【输入参数】")
    print(f"  数据集: {basename}")
    print(f"  Benchmark: {benchid}")
    print(f"  数据文件夹: {data_folder}")
    print(f"  量化位数: {QUANT_BITS}")
    print(f"  输出目录: {output_dir}")

    # 加载数据
    print("\n【步骤1】加载数据")
    all_data, all_coll = su.load_data(
        basename, benchid, data_folder, collision_model_type="link"
    )

    if all_data is None:
        print(f"✗ 无法加载数据: {basename}_{benchid:04d}")
        sys.exit(1)

    print("✓ 成功加载数据")
    print(f"  总edges: {len(all_data)}")

    # 生成bins
    bins = create_bins(QUANT_BITS)
    print(f"✓ 生成量化bins (共{len(bins) - 1}个)")

    # 分析具有足够pose的edge
    print("\n【步骤2】寻找可分析的edge")
    valid_edges = []
    for edge_idx, edge_coords in enumerate(all_data):
        if len(edge_coords) >= 2:  # 至少需要2个pose以支持多COPU分配
            valid_edges.append((edge_idx, edge_coords, all_coll[edge_idx]))  # type: ignore

    print(f"  找到 {len(valid_edges)} 条可分析的edge (poses >= 2)")

    if not valid_edges:
        print("✗ 没有足够poses的edge，无法进行多COPU分析")
        sys.exit(1)

    # 分析前5条有效edge（或全部，如果少于5条）
    max_edges_to_analyze = min(5, len(valid_edges))
    all_results = {}

    for edge_idx, edge_coords, edge_colls in valid_edges[:max_edges_to_analyze]:
        print(f"\n  分析 Edge {edge_idx} (poses={len(edge_coords)})")

        for num_copus in [1, 2, 4, 8]:
            if num_copus > len(edge_coords):
                continue  # 跳过COPU数 > poses的情况

            if num_copus not in all_results:
                all_results[num_copus] = []

            result = analyze_single_edge(edge_coords, edge_colls, num_copus, bins)
            all_results[num_copus].append(result)

    # 聚合分析
    print(f"\n【步骤3】聚合分析 (分析了 {max_edges_to_analyze} 条edge)")
    # 合并所有配置下的edge结果
    combined_results = []
    for num_copus in sorted(all_results.keys()):
        combined_results.extend(all_results[num_copus])

    stats = aggregate_analysis(combined_results)

    print("  ✓ 汇总统计完成")
    print(f"    - 哈希值匹配率: {stats['hash_match_rate']:.2%}")
    print(
        f"    - 维度差异: X={stats['dimension_diff_freq'][0]}, Y={stats['dimension_diff_freq'][1]}, Z={stats['dimension_diff_freq'][2]}"
    )

    # 生成输出
    print("\n【步骤4】生成输出")
    generate_report(stats, output_dir)
    generate_visualizations(stats, output_dir)
    save_statistics_csv(stats, output_dir)

    # 总结
    print("\n【分析完成】")
    print(f"  所有结果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
