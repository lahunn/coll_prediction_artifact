#!/usr/bin/env python3
"""
多Bank CHT策略分析脚本

功能：
- 仅支持multi_bank CHT
- 统计bank访问分布、冲突率、负载均衡等
- 支持单个或范围benchmark分析

用法:
  python analyze_multi_bank_cht.py <basename> <benchid|start-end> <data_folder> <num_copus> <threshold>
                                   [num_oocds] [sample_rate] [max_cycles]
                                   [--num-banks N]
                                   [--real-cycles]

示例:
  python analyze_multi_bank_cht.py iiwa_7 1-10 ../../trace_files/scene_benchmarks/bit_collision_data 16 1.0 8 1.0 100000 --num-banks 8
"""

import sys
import os
import argparse
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from simulation_core.multi_copu_scheduler import MultiCOPU_Scheduler
import simulation_utils as su

QUANT_MIN = -1.5
QUANT_MAX = 1.5


def print_bank_stats(cht_stats):
    print("\n【Bank访问统计】")
    print(f"  Bank数量: {len(cht_stats['bank_access_counts'])}")
    print(f"  每Bank访问数: {cht_stats['bank_access_counts']}")
    print(f"  负载均衡Std: {cht_stats['load_balance_std']:.2f}")
    print(f"  总冲突数: {cht_stats['total_conflicts']}")
    print(f"  冲突率: {cht_stats['conflict_rate']:.4f}")
    print(f"  总条目数: {cht_stats['entries_used']}")

    # 输出bank编号与编码对应关系
    if "bank_config" in cht_stats:
        print("\n【Bank编码对应关系】")
        bank_config = cht_stats["bank_config"]
        num_banks = len(cht_stats["bank_access_counts"])
        for bank_id in range(num_banks):
            # 将bank_id转换为二进制，长度为len(bank_config)
            bin_str = format(bank_id, f"0{len(bank_config)}b")
            mapping = []
            for i, bit_pos in enumerate(bank_config):
                bit_val = int(bin_str[i])
                mapping.append(f"HashKey第{bit_pos}位:{bit_val}")
            print(f"  bank{bank_id} 对应 {' '.join(mapping)}")


def run_simulation(
    basename,
    benchid_arg,
    data_folder,
    num_copus,
    threshold,
    num_oocds=7,
    sample_rate=1.0,
    max_cycles=100000,
    num_banks=8,
    quant_bits=3,
    use_real_cycles=False,
    collision_model_type="link",
):
    is_range_mode = "-" in benchid_arg
    cht_type = "multi_bank"
    cht_kwargs = {"num_banks": num_banks}
    # 从basename提取机器人名称（例如从"iiwa_7"提取"iiwa"）
    robot_name = basename.split("_")[0]
    bins = su.calculate_bins_from_workspace(robot_name, quant_bits)

    def simulate_one(benchid):
        all_cycles = None
        if use_real_cycles:
            all_data, all_coll, all_cycles = su.load_data_with_cycles(
                basename, benchid, data_folder, collision_model_type=collision_model_type
            )
        else:
            all_data, all_coll = su.load_data(
                basename, benchid, data_folder, collision_model_type=collision_model_type
            )
        if all_data is None or all_coll is None:
            return None

        # 按edge粒度循环，聚合bank统计
        all_edge_bank_stats = []
        for edge_idx, edge_coords in enumerate(all_data):
            edge_colls = all_coll[edge_idx]
            edge_cycles = all_cycles[edge_idx] if all_cycles is not None else None
            scheduler = MultiCOPU_Scheduler(
                num_copus=num_copus,
                num_oocds=num_oocds,
                cht_size=4096,
                enable_conflict_check=True,
                cht_type=cht_type,
                **cht_kwargs,
            )
            copus_coords, copus_colls, copus_cycles = su.allocate_edge_data_to_copus(
                edge_coords, edge_colls, edge_cycles, num_copus
            )
            for copu_id in range(num_copus):
                scheduler.copus[copu_id].load_data(
                    copus_coords[copu_id], copus_colls[copu_id], copus_cycles[copu_id]
                )
            result = scheduler.simulate(
                bins,
                threshold=threshold,
                sample_rate=sample_rate,
            )
            # 只聚合bank相关统计
            edge_bank_stats = result["cht_stats"]
            edge_bank_stats["num_edges"] = 1
            all_edge_bank_stats.append(edge_bank_stats)
        # 汇总所有edge的bank统计
        # 这里返回结构与原simulate_one一致，方便后续处理
        # 合并bank访问数、冲突数等
        if not all_edge_bank_stats:
            return None
        # 合并bank访问数
        num_banks = len(all_edge_bank_stats[0]["bank_access_counts"])
        total_bank_access = [0] * num_banks
        total_conflicts = 0
        total_entries = 0
        stds = []
        conflict_rates = []
        for stats in all_edge_bank_stats:
            for i in range(num_banks):
                total_bank_access[i] += stats["bank_access_counts"][i]
            total_conflicts += stats["total_conflicts"]
            total_entries += stats["entries_used"]
            stds.append(stats["load_balance_std"])
            conflict_rates.append(stats["conflict_rate"])
        # 重新构造聚合结果
        agg_stats = {
            "bank_access_counts": total_bank_access,
            "load_balance_std": sum(stds) / len(stds) if stds else 0.0,
            "total_conflicts": total_conflicts,
            "conflict_rate": sum(conflict_rates) / len(conflict_rates)
            if conflict_rates
            else 0.0,
            "entries_used": total_entries,
            "num_edges": len(all_edge_bank_stats),
            "bank_config": all_edge_bank_stats[-1]["bank_config"],
        }
        return {"cht_stats": agg_stats, "num_edges": len(all_edge_bank_stats)}

    if is_range_mode:
        benchid_start, benchid_end = map(int, benchid_arg.split("-"))
    else:
        benchid_start = benchid_end = int(benchid_arg)

    # 定义要评估的Bank Configurations
    # 假设quant_bits=3 (key_len约21), num_banks=8 (select_bits=3)
    # 这些配置对应不同的Bit Selection策略
    bank_configs = [
        (0, 1, 2),  # Low Bits
        (3, 4, 5),  # Level 1
        (6, 7, 8),  # Level 2
        (9, 10, 11),  # Level 3
        (0, 3, 6),  # Strided X
        (1, 4, 7),  # Strided Y
        (2, 5, 8),  # Strided Z
        (0, 6, 11),  # Mixed
        (0, 4, 8),  # Spread
        (1, 5, 9),  # Sparse
    ]

    print(f"\n分析范围: {benchid_start}-{benchid_end}")
    print(f"将评估 {len(bank_configs)} 种 Bank 配置...")

    all_config_results = []

    for config in bank_configs:
        print(f"\n>>> 评估配置: {config}")

        # 更新 CHT 参数
        cht_kwargs["bank_config"] = config

        total_benchmarks = 0
        all_bank_stats = []

        for benchid in range(benchid_start, benchid_end + 1):
            # print(f"  Benchmark {benchid} ...", end=" ")
            result = simulate_one(benchid)
            if result is None:
                # print("✗ 加载失败")
                continue
            # print(f"✓ edges={result['num_edges']}")
            all_bank_stats.append(result["cht_stats"])
            total_benchmarks += 1

        if all_bank_stats:
            # 汇总当前配置的所有benchmark统计
            num_banks = len(all_bank_stats[0]["bank_access_counts"])
            total_conflicts = sum(s["total_conflicts"] for s in all_bank_stats)
            avg_conflict_rate = sum(s["conflict_rate"] for s in all_bank_stats) / len(
                all_bank_stats
            )
            avg_std = sum(s["load_balance_std"] for s in all_bank_stats) / len(
                all_bank_stats
            )

            print(
                f"   [结果] 总冲突数: {total_conflicts}, 平均冲突率: {avg_conflict_rate:.4f}, 负载均衡Std: {avg_std:.2f}"
            )

            all_config_results.append(
                {
                    "config": config,
                    "total_conflicts": total_conflicts,
                    "conflict_rate": avg_conflict_rate,
                    "std": avg_std,
                }
            )

    # 输出最终对比表
    print("\n" + "=" * 60)
    print(
        f"{'Configuration':<20} | {'Total Conflicts':<15} | {'Conflict Rate':<15} | {'Std Dev':<10}"
    )
    print("-" * 60)
    # 按总冲突数排序
    all_config_results.sort(key=lambda x: x["total_conflicts"])

    for res in all_config_results:
        print(
            f"{str(res['config']):<20} | {res['total_conflicts']:<15} | {res['conflict_rate']:<15.4f} | {res['std']:<10.2f}"
        )
    print("=" * 60)

    # 保存结果到 CSV
    if all_config_results:
        output_dir = os.path.join(current_dir, "result_files")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "multi_bank_bit_selection_results.csv")

        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["config", "total_conflicts", "conflict_rate", "std"]
                )
                writer.writeheader()
                for res in all_config_results:
                    writer.writerow(
                        {
                            "config": str(res["config"]),
                            "total_conflicts": res["total_conflicts"],
                            "conflict_rate": round(res["conflict_rate"], 6),
                            "std": round(res["std"], 2),
                        }
                    )
            print(f"\n[结果保存] 已保存到: {csv_path}")
        except Exception as e:
            print(f"\n[错误] 无法保存 CSV 文件: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi_bank CHT strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("basename", help="dataset basename, e.g. iiwa_7")
    parser.add_argument("benchid", help="benchmark id or range, e.g. 46 or 1-10")
    parser.add_argument("data_folder", help="path to data folder")
    parser.add_argument("num_copus", type=int, help="number of COPUs")
    parser.add_argument("threshold", type=float, help="collision threshold")
    parser.add_argument(
        "num_oocds", type=int, nargs="?", default=6, help="number of OOCDs per COPU"
    )
    parser.add_argument(
        "sample_rate",
        type=float,
        nargs="?",
        default=1.0,
        help="sampling rate for free samples",
    )
    parser.add_argument(
        "max_cycles", type=int, nargs="?", default=100000, help="max simulation cycles"
    )
    parser.add_argument(
        "--num-banks", type=int, default=8, help="number of banks for multi_bank CHT"
    )
    parser.add_argument("--quant-bits", type=int, default=3, help="quantization bits")
    parser.add_argument(
        "--real-cycles", action="store_true", help="use real cycles from dataset"
    )
    parser.add_argument(
        "--collision-model-type",
        type=str,
        default="link",
        choices=["link", "sphere"],
        help="collision model type: link or sphere",
    )
    args = parser.parse_args()
    run_simulation(
        args.basename,
        args.benchid,
        args.data_folder,
        args.num_copus,
        args.threshold,
        num_oocds=args.num_oocds,
        sample_rate=args.sample_rate,
        max_cycles=args.max_cycles,
        num_banks=args.num_banks,
        quant_bits=args.quant_bits,
        use_real_cycles=args.real_cycles,
        collision_model_type=args.collision_model_type,
    )


if __name__ == "__main__":
    main()
