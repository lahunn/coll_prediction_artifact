#!/usr/bin/env python3
"""
Hash编码分布与Bank划分方案分析工具

功能：
1. 加载实际机器人运动轨迹数据
2. 生成所有Link的Hash Key
3. 分析Hash Key的分布情况
4. 评估不同的Bank划分方案（即选择哪些Bit作为Bank ID），寻找负载最均衡的方案
"""

import sys
import os
import argparse
import numpy as np
import csv
# import itertools
# from collections import Counter
# import matplotlib.pyplot as plt

# 添加父目录到sys.path以导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import simulation_utils as su
from simulation_core import hash_utils


def load_all_hash_keys(
    basename, benchid_range, data_folder, quant_bits, collision_model_type="link"
):
    """
    加载数据并计算所有Hash Keys
    """
    print(
        f"正在加载数据 {basename} (模型: {collision_model_type}, Benchmarks: {benchid_range})..."
    )

    # 获取Workspace Bins
    robot_name = basename.split("_")[0]  # 假设命名规则为 name_dof
    try:
        bins = hash_utils.calculate_bins_from_workspace(robot_name, quant_bits)
    except Exception as e:
        print(f"Error calculating bins: {e}")
        return []

    all_hash_keys = []

    # 解析benchid范围
    if "-" in benchid_range:
        start, end = map(int, benchid_range.split("-"))
        bench_ids = range(start, end + 1)
    else:
        bench_ids = [int(benchid_range)]

    total_edges = 0

    for bid in bench_ids:
        # 加载单个Benchmark数据
        # 注意：这里只加载数据，不需要碰撞检测结果
        data, _ = su.load_data(
            basename, bid, data_folder, collision_model_type=collision_model_type
        )

        if data is None:
            continue

        # data结构: [edge1, edge2, ...]
        # edge结构: [pose1, pose2, ...]
        # 对于 link 模型，element 结构为 [x, y, z, qx, qy, qz, qw]
        # 对于 sphere 模型，element 结构为 [x, y, z, radius]

        for edge in data:
            total_edges += 1
            for pose in edge:
                for link_coords in pose:
                    # 计算Hash Key
                    key = hash_utils.compute_hash_keyy(link_coords, bins)
                    all_hash_keys.append(key)

    print(f"数据加载完成: 处理了 {len(bench_ids)} 个Benchmarks, {total_edges} 个Edges")
    print(f"生成的Hash Key总数: {len(all_hash_keys)}")

    return all_hash_keys


def evaluate_bank_schemes(all_hash_keys, num_banks, quant_bits):
    """
    评估多种Bank Bit选择方案
    """
    if not all_hash_keys:
        return []

    key_len = len(all_hash_keys[0])
    num_select_bits = int(np.log2(num_banks))

    print(f"\n=== Bank方案评估 (KeyLen={key_len}, SelectBits={num_select_bits}) ===")

    # 定义10种评估方案 (假设 num_select_bits=3, 对应 num_banks=8)
    # 如果 num_banks != 8, 这些硬编码方案可能需要调整，这里做个简单适配

    # 基础方案 (Low Bits)
    schemes = [
        tuple(range(num_select_bits)),  # (0, 1, 2)
    ]

    # 生成其他9种方案
    if num_select_bits == 3:
        schemes.extend(
            [
                (3, 4, 5),  # Level 1
                (6, 7, 8),  # Level 2
                (9, 10, 11),  # Level 3 (High)
                (0, 3, 6),  # Strided X
                (1, 4, 7),  # Strided Y
                (2, 5, 8),  # Strided Z
                (0, 6, 11),  # Mixed
                (0, 4, 8),  # Spread
                (1, 5, 9),  # Sparse
            ]
        )
    else:
        # 对于非8 Bank的情况，简单生成一些跨步方案
        import random

        np.random.seed(42)  # 固定随机数以保持一致性
        for _ in range(9):
            # 随机选择不重复的位
            scheme = tuple(
                sorted(np.random.choice(key_len, num_select_bits, replace=False))
            )
            if scheme not in schemes:
                schemes.append(scheme)

    # 评估所有方案
    results = []

    collected_metrics = []

    for idx, combo in enumerate(schemes):
        # 检查 combo 中的索引是否超出 key_len
        if any(c >= key_len for c in combo):
            continue

        bank_counts = np.zeros(num_banks, dtype=int)

        for key_str in all_hash_keys:
            bank_id = 0
            for i, bit_idx in enumerate(combo):
                # bit_idx 是字符串中的索引
                if key_str[bit_idx] == "1":
                    bank_id |= 1 << i
            bank_counts[bank_id] += 1

        # 计算统计指标
        std_dev = np.std(bank_counts)
        max_load = np.max(bank_counts)
        min_load = np.min(bank_counts)
        range_ratio = max_load / min_load if min_load > 0 else float("inf")

        desc = get_bit_meaning_short(combo, quant_bits)

        results.append(
            {
                "id": idx,
                "combo": combo,
                "std": std_dev,
                "ratio": range_ratio,
                "counts": bank_counts,
                "desc": desc,
            }
        )

        collected_metrics.append(
            {
                "Strategy": "Bit Selection",
                "Configuration": str(combo),
                "StdDev": std_dev,
                "MaxMinRatio": range_ratio,
                "Counts": str(bank_counts.tolist()),
            }
        )

    # 按标准差排序输出
    results.sort(key=lambda x: x["std"])

    print(f"{'ID':<4} | {'Bits':<15} | {'StdDev':<8} | {'Ratio':<6} | {'Description'}")
    print("-" * 60)

    for res in results:
        print(
            f"{res['id']:<4} | {str(res['combo']):<15} | {res['std']:<8.2f} | {res['ratio']:<6.2f} | {res['desc']}"
        )

    print("-" * 60)
    print(f"推荐最佳方案: Bits {results[0]['combo']} (Std={results[0]['std']:.2f})")
    print("-" * 60)

    return collected_metrics


def get_bit_meaning_short(combo, quant_bits):
    """简短解释位的含义"""
    dims = ["X", "Y", "Z"]
    num_dims = 3
    desc = []
    for bit_idx in combo:
        bit_level = bit_idx // num_dims
        dim_index = bit_idx % num_dims
        desc.append(f"{dims[dim_index]}{bit_level}")
    return ",".join(desc)


def evaluate_strong_mix_schemes(all_hash_keys, num_banks):
    """
    评估强混淆Hash方案 (Murmur3 Finalizer / Wang Hash)
    原理: 使用一系列位运算(XOR, Shift, Mult)彻底打散输入位的相关性。
    """
    if not all_hash_keys:
        return []

    print("\n=== 强混淆Hash方案评估 (Strong Mixing) ===")

    # 预处理
    int_keys = [int(k, 2) for k in all_hash_keys]

    # 1. Murmur3 32-bit Finalizer
    # 这是一个非常优秀的Avalanche Mixer
    def murmur3_mix(k):
        k ^= k >> 16
        k = (k * 0x85EBCA6B) & 0xFFFFFFFF
        k ^= k >> 13
        k = (k * 0xC2B2AE35) & 0xFFFFFFFF
        k ^= k >> 16
        return k & 0xFFFFFFFF

    # 2. Thomas Wang's 32-bit Integer Hash
    def wang_hash(key):
        key = (~key) + (key << 21)
        key = key & 0xFFFFFFFF
        key = key ^ (key >> 24)
        key = (key + (key << 3)) + (key << 8)
        key = key & 0xFFFFFFFF
        key = key ^ (key >> 14)
        key = (key + (key << 2)) + (key << 4)
        key = key & 0xFFFFFFFF
        key = key ^ (key >> 28)
        key = (key + (key << 31)) & 0xFFFFFFFF
        return key

    schemes = [("Murmur3 Mixer", murmur3_mix), ("Wang Hash", wang_hash)]

    collected_metrics = []

    for name, func in schemes:
        bank_counts = np.zeros(num_banks, dtype=int)

        # Mask for bank mapping
        # 如果num_banks是2的幂，直接与掩码
        # 否则取模
        is_power_of_2 = (num_banks & (num_banks - 1) == 0) and num_banks > 0
        mask = num_banks - 1

        for val in int_keys:
            hashed = func(val)
            if is_power_of_2:
                bank_id = hashed & mask
            else:
                bank_id = hashed % num_banks
            bank_counts[bank_id] += 1

        # 计算统计指标
        std_dev = np.std(bank_counts)
        max_load = np.max(bank_counts)
        min_load = np.min(bank_counts)
        range_ratio = max_load / min_load if min_load > 0 else float("inf")

        print(f"Scheme: {name}")
        print(f"  Std Dev: {std_dev:.2f}")
        print(f"  Max/Min Ratio: {range_ratio:.2f}")
        print(f"  Counts: {bank_counts}")
        print("-" * 20)

        collected_metrics.append(
            {
                "Strategy": "Strong Mixing",
                "Configuration": name,
                "StdDev": std_dev,
                "MaxMinRatio": range_ratio,
                "Counts": str(bank_counts.tolist()),
            }
        )

    return collected_metrics


def evaluate_h3_hash_schemes(all_hash_keys, num_banks):
    """
    评估H3 Hash方案 (Universal Hashing)
    原理: 每一个输入位对应一个随机数。如果输入位为1，则将对应的随机数异或到结果中。
    这是硬件CHT/Cache中常用的消除冲突的方法。
    """
    if not all_hash_keys:
        return []

    print("\n=== H3 Hash方案评估 (Universal Hashing) ===")

    key_len = len(all_hash_keys[0])
    int(np.log2(num_banks))

    # 预处理：将所有key转换为int列表
    int_keys = [int(k, 2) for k in all_hash_keys]

    # 尝试不同的随机种子
    seeds = [42, 123, 999, 2024, 7]

    best_ratio = float("inf")
    best_seed = -1

    collected_metrics = []

    for seed in seeds:
        np.random.seed(seed)
        # 为每一位生成一个随机的Mask (范围 0 到 num_banks-1)
        # H3矩阵: key_len 行, num_select_bits 列
        # 这里直接用整数表示每一行
        h3_matrix = np.random.randint(0, num_banks, size=key_len)

        bank_counts = np.zeros(num_banks, dtype=int)

        for val in int_keys:
            bank_id = 0
            temp_val = val
            bit_idx = 0

            # 遍历每一位
            # 注意：这里假设key_len足够覆盖val的所有位
            # 由于val是int_keys中的值，其位数由key_len决定
            # 但为了效率，我们直接位移
            while temp_val > 0:
                if temp_val & 1:
                    bank_id ^= h3_matrix[bit_idx]
                temp_val >>= 1
                bit_idx += 1

            bank_counts[bank_id] += 1

        # 计算统计指标
        std_dev = np.std(bank_counts)
        max_load = np.max(bank_counts)
        min_load = np.min(bank_counts)
        range_ratio = max_load / min_load if min_load > 0 else float("inf")

        print(f"Seed {seed}:")
        print(f"  Std Dev: {std_dev:.2f}")
        print(f"  Max/Min Ratio: {range_ratio:.2f}")
        print(f"  Counts: {bank_counts}")

        if range_ratio < best_ratio:
            best_ratio = range_ratio
            best_seed = seed

        collected_metrics.append(
            {
                "Strategy": "H3 Hash",
                "Configuration": f"Seed {seed}",
                "StdDev": std_dev,
                "MaxMinRatio": range_ratio,
                "Counts": str(bank_counts.tolist()),
            }
        )

    print("-" * 20)
    print(f"H3 最佳 Seed: {best_seed}, Ratio: {best_ratio:.2f}")

    return collected_metrics


def evaluate_xor_schemes(all_hash_keys, num_banks):
    """
    评估XOR Hash方案 (Folding)
    """
    if not all_hash_keys:
        return []

    key_len = len(all_hash_keys[0])
    num_select_bits = int(np.log2(num_banks))

    print("\n=== XOR Hash方案评估 (Folding) ===")
    print(f"原理: 将 {key_len} bits 的 Key 分割成 {num_select_bits} bits 的段进行异或")

    # 预处理：将所有key转换为int列表
    int_keys = [int(k, 2) for k in all_hash_keys]

    bank_counts = np.zeros(num_banks, dtype=int)

    # 掩码，用于提取 num_select_bits
    mask = (1 << num_select_bits) - 1

    for val in int_keys:
        bank_id = 0
        temp_val = val

        # Folding XOR
        while temp_val > 0:
            bank_id ^= temp_val & mask
            temp_val >>= num_select_bits

        bank_counts[bank_id] += 1

    # 计算统计指标
    std_dev = np.std(bank_counts)
    max_load = np.max(bank_counts)
    min_load = np.min(bank_counts)
    range_ratio = max_load / min_load if min_load > 0 else float("inf")

    print("XOR Folding 结果:")
    print(f"  Std Dev: {std_dev:.2f}")
    print(f"  Max/Min Ratio: {range_ratio:.2f}")
    print(f"  Counts: {bank_counts}")
    print("-" * 40)

    return [
        {
            "Strategy": "XOR Folding",
            "Configuration": f"Fold {key_len}->{num_select_bits}",
            "StdDev": std_dev,
            "MaxMinRatio": range_ratio,
            "Counts": str(bank_counts.tolist()),
        }
    ]


def evaluate_prime_hash_schemes(all_hash_keys, num_banks):
    """
    评估乘法Hash方案 (Multiplicative Hash)
    原理: (Key * Prime) >> Shift
    这是最常用的打散规律性数据的低成本Hash方法
    """
    if not all_hash_keys:
        return []

    print("\n=== 乘法Hash方案评估 (Multiplicative) ===")

    # 预处理：将所有key转换为int列表
    int_keys = [int(k, 2) for k in all_hash_keys]

    # 尝试几个著名的素数
    primes = [
        2654435761,  # Knuth's Multiplicative Hash (Golden Ratio for 32bit)
        3632628803,  # Another good prime
        16777619,  # FNV prime
    ]

    num_select_bits = int(np.log2(num_banks))
    # 我们在一个32位的空间内做乘法，然后取高位
    # Shift amount to get the top N bits from a 32-bit result
    shift_amount = 32 - num_select_bits

    collected_metrics = []

    for prime in primes:
        bank_counts = np.zeros(num_banks, dtype=int)

        for val in int_keys:
            # 模拟32位无符号整数溢出截断
            hashed_val = (val * prime) & 0xFFFFFFFF
            # 取高位作为Bank ID
            bank_id = hashed_val >> shift_amount
            bank_counts[bank_id] += 1

        # 计算统计指标
        std_dev = np.std(bank_counts)
        max_load = np.max(bank_counts)
        min_load = np.min(bank_counts)
        range_ratio = max_load / min_load if min_load > 0 else float("inf")

        print(f"Prime {prime}:")
        print(f"  Std Dev: {std_dev:.2f}")
        print(f"  Max/Min Ratio: {range_ratio:.2f}")
        print(f"  Counts: {bank_counts}")
        print("-" * 20)

        collected_metrics.append(
            {
                "Strategy": "Multiplicative Hash",
                "Configuration": f"Prime {prime}",
                "StdDev": std_dev,
                "MaxMinRatio": range_ratio,
                "Counts": str(bank_counts.tolist()),
            }
        )

    return collected_metrics


def main():
    parser = argparse.ArgumentParser(description="Hash Pattern Analysis")
    parser.add_argument("basename", help="Dataset basename (e.g., iiwa_7)")
    parser.add_argument("benchid", help="Benchmark ID range (e.g., 1-10)")
    parser.add_argument("data_folder", help="Path to data folder")
    parser.add_argument("--quant-bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--num-banks", type=int, default=8, help="Number of banks")
    parser.add_argument(
        "--collision-model-type",
        type=str,
        default="sphere",
        choices=["link", "sphere"],
        help="Collision model type (link or sphere)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result_files/hash_analysis_results.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    # 1. 加载并生成Hash
    keys = load_all_hash_keys(
        args.basename,
        args.benchid,
        args.data_folder,
        args.quant_bits,
        collision_model_type=args.collision_model_type,
    )

    all_results = []

    # 2. 评估普通方案
    all_results.extend(evaluate_bank_schemes(keys, args.num_banks, args.quant_bits))

    # 3. 评估XOR方案
    all_results.extend(evaluate_xor_schemes(keys, args.num_banks))

    # 4. 评估乘法Hash方案
    all_results.extend(evaluate_prime_hash_schemes(keys, args.num_banks))

    # 5. 评估H3 Hash方案 (Universal Hashing)
    all_results.extend(evaluate_h3_hash_schemes(keys, args.num_banks))

    # 6. 评估强混淆Hash方案 (Murmur3/Wang)
    all_results.extend(evaluate_strong_mix_schemes(keys, args.num_banks))

    # 7. 保存结果到 CSV
    if all_results:
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(args.output, "w", newline="") as csvfile:
                fieldnames = [
                    "Strategy",
                    "Configuration",
                    "StdDev",
                    "MaxMinRatio",
                    "Counts",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row in all_results:
                    writer.writerow(row)
            print(f"\nAnalysis results saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()
