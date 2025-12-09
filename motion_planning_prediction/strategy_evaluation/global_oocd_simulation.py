#!/usr/bin/env python3
"""
使用全局OOCD池的多prediction并行仿真

与原 MultiCOPU_Scheduler 的区别：
- 移除 COPU 层级限制
- 任意 prediction 可申请任意空闲 OOCD（受 per-pred 配额限制）
- 支持动态 edge 加载与轮转分配
- 简化派发逻辑（qcoll优先、qnoncoll兜底）
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from simulation_core.global_oocd_scheduler import GlobalOOCDScheduler
import simulation_utils as su
from simulation_core.constants import DEFAULT_CYCLE_CHECK


def run_global_oocd_simulation(
    all_data,
    all_coll,
    num_oocds=7,
    num_predictions=4,
    max_oocd_per_pred=2,
    quant_bits=4,
    threshold=1.0,
    sample_rate=1.0,
    enable_conflict_check=True,
    cht_type="dual_port",
    robot_name="iiwa",
    **cht_kwargs,
):
    """
    执行全局OOCD池仿真

    Args:
        all_data: 所有edge的数据
        all_coll: 所有edge的碰撞标志
        num_oocds: OOCD总数
        num_predictions: prediction缓冲数
        max_oocd_per_pred: 每个prediction的最大OOCD配额（None表示自动计算）
        quant_bits: 量化位数
        threshold: 碰撞预测阈值
        sample_rate: 采样率
        enable_conflict_check: CHT冲突检查
        cht_type: CHT类型
        robot_name: 机器人名称（用于读workspace）
        **cht_kwargs: CHT额外参数

    Returns:
        dict: 仿真结果
    """
    # 计算bins
    bins = su.calculate_bins_from_workspace(robot_name, quant_bits)

    # 创建调度器
    scheduler = GlobalOOCDScheduler(
        num_oocds=num_oocds,
        num_predictions=num_predictions,
        max_oocd_per_pred=max_oocd_per_pred,
        cht_size=4096,
        enable_conflict_check=enable_conflict_check,
        cht_type=cht_type,
        qcoll_size=8,
        qnoncoll_size=56,
        cycle_check=DEFAULT_CYCLE_CHECK,
        **cht_kwargs,
    )

    # 设置数据
    scheduler.set_benchmark_data(all_data, all_coll)

    # 执行仿真
    results = scheduler.simulate(bins, threshold, sample_rate)

    return results


def simulate_single_benchmark(
    basename,
    benchid,
    data_folder,
    num_oocds,
    num_predictions=4,
    quant_bits=4,
    threshold=1.0,
    sample_rate=1.0,
    enable_conflict_check=True,
    cht_type="dual_port",
    max_oocd_per_pred=10,
    **cht_kwargs,
):
    """
    执行单个benchmark的仿真（加载数据 -> 调用 run_global_oocd_simulation）
    """
    all_data, all_coll = su.load_data(
        basename, benchid, data_folder, collision_model_type="link"
    )

    if all_data is None:
        return None

    result = run_global_oocd_simulation(
        all_data,
        all_coll,
        num_oocds=num_oocds,
        num_predictions=num_predictions,
        max_oocd_per_pred=max_oocd_per_pred,
        threshold=threshold,
        sample_rate=sample_rate,
        enable_conflict_check=enable_conflict_check,
        cht_type=cht_type,
        **cht_kwargs,
    )

    return result


def run_benchmark_range_simulation(
    basename,
    benchid_start,
    benchid_end,
    data_folder,
    num_oocds,
    num_predictions=4,
    quant_bits=4,
    threshold=1.0,
    sample_rate=1.0,
    enable_conflict_check=True,
    cht_type="dual_port",
    max_oocd_per_pred=10,
    **cht_kwargs,
):
    """
    对benchid范围内的每个benchmark依次仿真并聚合结果
    """
    total_cycles_all = 0
    total_queries_all = 0
    total_cht_conflicts_all = 0
    total_collisions_all = 0
    total_safe_all = 0
    all_oocd_utils = []
    num_benchmarks_processed = 0

    print(f"\n开始批量仿真 (benchmark {benchid_start}-{benchid_end})...")

    for benchid in range(benchid_start, benchid_end + 1):
        print(f"  处理 benchmark {benchid}...", end=" ")
        result = simulate_single_benchmark(
            basename,
            benchid,
            data_folder,
            num_oocds,
            num_predictions=num_predictions,
            quant_bits=quant_bits,
            threshold=threshold,
            sample_rate=sample_rate,
            enable_conflict_check=enable_conflict_check,
            cht_type=cht_type,
            max_oocd_per_pred=max_oocd_per_pred,
            **cht_kwargs,
        )

        if result is None:
            print("✗ 加载失败")
            continue

        print(f"✓ ({result.get('num_edges', 'N/A')} edges)")

        total_cycles_all += result.get("total_cycles", 0)
        total_queries_all += result.get("total_queries", 0)
        total_cht_conflicts_all += result.get("cht_stats", {}).get("conflicts", 0)

        edge_results = result.get("edge_results", {})
        total_collisions_all += sum(
            1 for r in edge_results.values() if r == "collision"
        )
        total_safe_all += sum(1 for r in edge_results.values() if r == "safe")

        if "oocd_utilization" in result:
            all_oocd_utils.append(result["oocd_utilization"])

        num_benchmarks_processed += 1

    avg_oocd_util = sum(all_oocd_utils) / len(all_oocd_utils) if all_oocd_utils else 0.0

    batch_result = {
        "basename": basename,
        "benchid_start": benchid_start,
        "benchid_end": benchid_end,
        "num_benchmarks": num_benchmarks_processed,
        "total_cycles": total_cycles_all,
        "total_queries": total_queries_all,
        "avg_oocd_utilization": avg_oocd_util,
        "total_cht_conflicts": total_cht_conflicts_all,
        "num_collisions": total_collisions_all,
        "num_safe": total_safe_all,
    }

    return batch_result


def print_results(results, is_range=False):
    title = "批量仿真结果汇总" if is_range else "仿真结果汇总"
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("\n【指标汇总】")
    if is_range:
        print(f"  数据集: {results.get('basename', 'N/A')}")
        print(
            f"  Benchmark范围: {results.get('benchid_start')} - {results.get('benchid_end')}"
        )
        print(f"  处理Benchmark数: {results.get('num_benchmarks', 0)}")
    else:
        print(f"  总Edge数: {results.get('num_edges', 'N/A')}")

    print(f"  总周期: {results.get('total_cycles', 0)}")
    print(f"  总查询数: {results.get('total_queries', 0):.0f}")

    if results.get("total_cycles", 0) > 0:
        print(
            f"  系统吞吐量: {results.get('total_queries', 0) / max(1, results.get('total_cycles', 1)):.4f} queries/cycle"
        )

    print(f"  平均OOCD占用率: {results.get('avg_oocd_utilization', 0.0):.2%}")
    print(f"  CHT冲突数: {results.get('total_cht_conflicts', 0)}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用全局OOCD池执行多prediction并行仿真",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("basename", help="数据集基名（例如 iiwa_7）")
    parser.add_argument("benchid", help="Benchmark ID 或 范围 (start-end)")
    parser.add_argument("data_folder", help="数据文件夹路径")
    parser.add_argument(
        "num_oocds", nargs="?", type=int, default=7, help="OOCD数量（默认7）"
    )
    parser.add_argument(
        "threshold", nargs="?", type=float, default=1.0, help="碰撞预测阈值（默认1.0）"
    )
    parser.add_argument(
        "sample_rate", nargs="?", type=float, default=1.0, help="采样率（默认1.0）"
    )
    parser.add_argument(
        "num_predictions",
        nargs="?",
        type=int,
        default=4,
        help="Prediction缓冲数（默认4）",
    )

    parser.add_argument(
        "--max-oocd-per-pred",
        type=int,
        default=None,
        help="每个prediction最多占用的OOCD数（默认自动计算为 num_oocds / num_predictions）",
    )
    parser.add_argument(
        "--no-cht-conflict", action="store_true", help="禁用CHT冲突检查"
    )
    parser.add_argument(
        "--cht-type",
        choices=["dual_port", "multi_bank"],
        default="dual_port",
        help="CHT类型（默认dual_port）",
    )
    parser.add_argument(
        "--num-banks", type=int, default=8, help="Multi-bank CHT的bank数（默认8）"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # 归一化参数
    basename = args.basename
    benchid_arg = args.benchid
    data_folder = args.data_folder
    num_oocds = args.num_oocds
    threshold = args.threshold
    sample_rate = args.sample_rate
    num_predictions = args.num_predictions
    enable_conflict_check = not args.no_cht_conflict
    cht_type = args.cht_type

    is_range_mode = "-" in benchid_arg

    # CHT 额外参数
    cht_kwargs = {"num_banks": args.num_banks} if cht_type == "multi_bank" else {}

    # 计算实际的max_oocd_per_pred（向上取整）
    if args.max_oocd_per_pred is None:
        actual_max_oocd_per_pred = (
            args.num_oocds + args.num_predictions - 1
        ) // args.num_predictions
    else:
        actual_max_oocd_per_pred = args.max_oocd_per_pred

    print("=" * 80)
    print("全局OOCD仿真")
    print("=" * 80)

    print("\n【输入参数】")
    print(f"  数据集: {basename}")
    if is_range_mode:
        benchid_start, benchid_end = map(int, benchid_arg.split("-"))
        print(f"  Benchmark范围: {benchid_start} - {benchid_end}")
    else:
        benchid = int(benchid_arg)
        print(f"  Benchmark: {benchid}")
    print(f"  数据文件夹: {data_folder}")
    print(f"  OOCD数量: {num_oocds}")
    print(f"  碰撞阈值: {threshold}")
    print(f"  采样率: {sample_rate}")
    print(f"  Prediction缓冲数: {num_predictions}")
    print(f"  Max OOCD per pred: {actual_max_oocd_per_pred}")
    print(f"  CHT类型: {cht_type}")

    print("\n【步骤1】执行仿真")
    if is_range_mode:
        benchid_start, benchid_end = map(int, benchid_arg.split("-"))
        results = run_benchmark_range_simulation(
            basename,
            benchid_start,
            benchid_end,
            data_folder,
            num_oocds,
            num_predictions=num_predictions,
            quant_bits=4,
            threshold=threshold,
            sample_rate=sample_rate,
            enable_conflict_check=enable_conflict_check,
            cht_type=cht_type,
            max_oocd_per_pred=actual_max_oocd_per_pred,
            **cht_kwargs,
        )
        print_results(results, is_range=True)
    else:
        benchid = int(benchid_arg)
        result = simulate_single_benchmark(
            basename,
            benchid,
            data_folder,
            num_oocds,
            num_predictions=num_predictions,
            quant_bits=4,
            threshold=threshold,
            sample_rate=sample_rate,
            enable_conflict_check=enable_conflict_check,
            cht_type=cht_type,
            max_oocd_per_pred=actual_max_oocd_per_pred,
            **cht_kwargs,
        )

        if result is None:
            print(f"✗ 无法加载数据: {basename}_{benchid}")
            return 1

        # 输出单个结果（保持原有格式）
        print("\n仿真结果:")
        print(f"总周期: {result['total_cycles']}")
        print(f"总查询数: {result['total_queries']:.1f}")
        print(
            f"系统吞吐量: {result['total_queries'] / max(1, result['total_cycles']):.4f} queries/cycle"
        )
        print(f"平均OOCD占用率: {result.get('oocd_utilization', 0.0):.4f}")
        print(f"碰撞发现: {result.get('collision_found', False)}")

        cht_stats = result.get("cht_stats", {})
        print(f"CHT冲突数: {cht_stats.get('conflicts', 0)}")
        print(f"CHT读次数: {cht_stats.get('total_reads', 0)}")
        print(f"CHT写次数: {cht_stats.get('total_writes', 0)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
