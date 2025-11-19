#!/usr/bin/env python3
"""
多COPU系统实际数据仿真

功能：
1. 读入实际数据文件
2. 按pose维度将任务分派到各个COPU
3. 执行多COPU协同仿真
4. 输出仿真结果（总周期、吞吐量、利用率、CHT冲突率）

数据分派策略：
- 将原始配置按pose维度均匀分配给各COPU
- 每个COPU处理其分配的pose对应的所有link检测任务
"""
# 用法: python multi_copu_real_data_simulation.py <basename> <benchid> <data_folder> <num_copus> <threshold> [sample_rate] [max_cycles] [--real-cycles]
# 示例: python multi_copu_real_data_simulation.py iiwa_7 1 ../trace_files/scene_benchmarks/bit_collision_data 8 1.0 1.0 100000

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_copu_simulation import MultiCOPU_Scheduler, _allocate_edge_data_to_copus
import simulation_utils as su

# ============================================================================
# 全局参数配置
# ============================================================================
QUANT_MIN = -1.5  # 量化最小值
QUANT_MAX = 1.5  # 量化最大值
DEFAULT_CHECK_CYCLE = 45


def simulate_single_edge(
    edge_coords,
    edge_colls,
    edge_cycles,
    num_copus,
    num_oocds,
    bins,
    threshold,
    sample_rate,
    max_cycles,
    enable_conflict_check=True,
):
    """
    执行单条edge的仿真

    Args:
        edge_coords: 该edge的pose坐标数据 List[List[coords]]
        edge_colls: 该edge的碰撞标志 List[List[flags]]
        edge_cycles: 该edge的周期数据 List[List[cycles]] (可为None)
        num_copus: COPU数量
        num_oocds: OOCD数量（CDU数量）
        bins: 量化bin
        threshold: 碰撞预测阈值
        sample_rate: 采样率
        max_cycles: 最大仿真周期
        enable_conflict_check: 是否启用CHT冲突检测 (默认True)

    Returns:
        dict: 该edge的仿真结果
    """
    # 创建调度器
    scheduler = MultiCOPU_Scheduler(
        num_copus=num_copus, num_oocds=num_oocds, cht_size=4096, enable_conflict_check=enable_conflict_check
    )

    # 使用通用数据分配函数一次性为所有COPU分配数据
    copus_coords, copus_colls, copus_cycles = _allocate_edge_data_to_copus(
        edge_coords,
        edge_colls,
        edge_cycles,
        num_copus,
    )

    # 为每个COPU加载其分配的数据
    for copu_id in range(num_copus):
        scheduler.copus[copu_id].load_data(
            copus_coords[copu_id], copus_colls[copu_id], copus_cycles[copu_id]
        )

    # 执行仿真
    result = scheduler.simulate(
        bins, threshold=threshold, sample_rate=sample_rate, max_cycles=max_cycles
    )

    return result


def aggregate_edge_results(result):
    """
    聚合单条edge的仿真结果

    Args:
        result: 单条edge的仿真结果

    Returns:
        dict: 聚合后的结果
    """
    total_queries = sum(c["total_queries"] for c in result["copus"])
    copu_utilizations = [c["oocd_utilization"] for c in result["copus"]]
    cht_conflicts = result["cht_stats"].get("total_conflicts", 0)

    return {
        "cycles": result["total_cycles"],
        "queries": total_queries,
        "copu_utilizations": copu_utilizations,
        "cht_conflicts": cht_conflicts,
    }


def run_multi_copu_simulation(
    all_data,
    all_coll,
    all_cycles,
    num_copus,
    num_oocds=7,
    quant_bits=3,
    threshold=1.0,
    sample_rate=1.0,
    max_cycles=100000,
    enable_conflict_check=True,
):
    """
    执行多COPU协同仿真（以edge为单位）

    Args:
        all_data: 所有edge的pose数据 List[List[List[coords]]]
        all_coll: 所有edge的碰撞标志 List[List[List[flags]]]
        all_cycles: 所有edge的周期数据 List[List[List[cycles]]] (可为None)
        num_copus: COPU数量
        num_oocds: OOCD数量（CDU数量），默认7
        quant_bits: 量化位数，分桶数 = 2^quant_bits（默认3，即8个桶）
        threshold: 碰撞预测阈值
        sample_rate: 采样率
        max_cycles: 最大仿真周期

    Returns:
        dict: 聚合的仿真结果
    """
    # 根据量化位数计算分桶数
    num_bins = 2**quant_bits

    # 生成均匀分布的bin（使用全局参数QUANT_MIN和QUANT_MAX）
    bins = np.linspace(QUANT_MIN, QUANT_MAX, num_bins + 1)

    # 统计聚合变量
    total_cycles = 0
    total_queries = 0
    total_cht_conflicts = 0
    all_edge_results = []

    print(f"\n开始按edge仿真 ({len(all_data)} edges, {num_copus} COPU)...")

    # 按edge进行仿真
    for edge_idx, (edge_coords, edge_colls) in enumerate(zip(all_data, all_coll)):
        # 获取该edge的周期数据（如果有）
        edge_cycles = all_cycles[edge_idx] if all_cycles is not None else None

        # 执行单条edge的仿真
        result = simulate_single_edge(
            edge_coords,
            edge_colls,
            edge_cycles,
            num_copus,
            num_oocds,
            bins,
            threshold,
            sample_rate,
            max_cycles,
            enable_conflict_check=enable_conflict_check,
        )

        # 聚合该edge的结果
        edge_agg = aggregate_edge_results(result)

        # 累计到总体指标
        total_cycles += edge_agg["cycles"]
        total_queries += edge_agg["queries"]
        total_cht_conflicts += edge_agg["cht_conflicts"]

        # 保存到结果列表
        edge_agg["edge_idx"] = edge_idx
        all_edge_results.append(edge_agg)

    # 计算平均COPU占用率（跨所有edge）
    all_copu_utils = []
    for edge_result in all_edge_results:
        all_copu_utils.extend(edge_result["copu_utilizations"])

    avg_copu_utilization = (
        sum(all_copu_utils) / len(all_copu_utils) if all_copu_utils else 0.0
    )

    # 构建聚合结果
    aggregated_result = {
        "total_cycles": total_cycles,
        "total_queries": total_queries,
        "num_edges": len(all_data),
        "edge_results": all_edge_results,
        "avg_copu_utilization": avg_copu_utilization,
        "total_cht_conflicts": total_cht_conflicts,
    }

    return aggregated_result


def simulate_single_benchmark(
    basename,
    benchid,
    data_folder,
    num_copus,
    num_oocds=7,
    quant_bits=3,
    threshold=1.0,
    sample_rate=1.0,
    max_cycles=100000,
    use_real_cycles=False,
    enable_conflict_check=True,
):
    """
    执行单个benchmark的完整仿真流程（加载数据 → 执行仿真 → 返回结果）

    Args:
        basename: 数据集基名（如 "iiwa_7"）
        benchid: benchmark编号
        data_folder: 数据文件夹路径
        num_copus: COPU数量
        num_oocds: OOCD数量（CDU数量），默认7
        quant_bits: 量化位数
        threshold: 碰撞预测阈值
        sample_rate: 采样率
        max_cycles: 最大仿真周期
        use_real_cycles: 是否使用真实周期数据

    Returns:
        dict: 该benchmark的仿真结果，若失败返回None
    """
    # 加载数据
    all_cycles = None
    if use_real_cycles:
        all_data, all_coll, all_cycles = su.load_data_with_cycles(
            basename, benchid, data_folder, collision_model_type="link"
        )
    else:
        all_data, all_coll = su.load_data(
            basename, benchid, data_folder, collision_model_type="link"
        )

    if all_data is None:
        return None

    # 执行仿真
    result = run_multi_copu_simulation(
        all_data,
        all_coll,
        all_cycles,
        num_copus,
        num_oocds=num_oocds,
        quant_bits=quant_bits,
        threshold=threshold,
        sample_rate=sample_rate,
        max_cycles=max_cycles,
        enable_conflict_check=enable_conflict_check,
    )

    return result


def run_benchmark_range_simulation(
    basename,
    benchid_start,
    benchid_end,
    data_folder,
    num_copus,
    num_oocds=7,
    quant_bits=3,
    threshold=1.0,
    sample_rate=1.0,
    max_cycles=100000,
    use_real_cycles=False,
    enable_conflict_check=True,
):
    """
    对指定范围内的benchid进行批量仿真

    Args:
        basename: 数据集基名（如 "iiwa_7"）
        benchid_start: 起始benchmark编号（包含）
        benchid_end: 结束benchmark编号（包含）
        data_folder: 数据文件夹路径
        num_copus: COPU数量
        num_oocds: OOCD数量（CDU数量），默认7
        quant_bits: 量化位数
        threshold: 碰撞预测阈值
        sample_rate: 采样率
        max_cycles: 最大仿真周期
        use_real_cycles: 是否使用真实周期数据

    Returns:
        dict: 聚合的批量仿真结果
    """
    total_cycles_all = 0
    total_queries_all = 0
    total_cht_conflicts_all = 0
    all_copu_utils_all = []
    num_benchmarks_processed = 0

    print(
        f"\n开始批量仿真 (benchmark {benchid_start}-{benchid_end}, {num_copus} COPU)..."
    )

    for benchid in range(benchid_start, benchid_end + 1):
        print(f"  处理 benchmark {benchid}...", end=" ")

        # 执行该benchmark的仿真
        result = simulate_single_benchmark(
            basename,
            benchid,
            data_folder,
            num_copus,
            num_oocds=num_oocds,
            quant_bits=quant_bits,
            threshold=threshold,
            sample_rate=sample_rate,
            max_cycles=max_cycles,
            use_real_cycles=use_real_cycles,
            enable_conflict_check=enable_conflict_check,
        )

        if result is None:
            print("✗ 加载失败")
            continue

        print(f"✓ ({result['num_edges']} edges)")

        # 累计到全局指标
        total_cycles_all += result["total_cycles"]
        total_queries_all += result["total_queries"]
        total_cht_conflicts_all += result.get("total_cht_conflicts", 0)

        # 累计COPU占用率样本
        for edge_result in result["edge_results"]:
            all_copu_utils_all.extend(edge_result["copu_utilizations"])

        num_benchmarks_processed += 1

    # 计算全局平均COPU占用率
    avg_copu_utilization_all = (
        sum(all_copu_utils_all) / len(all_copu_utils_all) if all_copu_utils_all else 0.0
    )

    # 构建批量仿真结果
    batch_result = {
        "basename": basename,
        "benchid_start": benchid_start,
        "benchid_end": benchid_end,
        "num_benchmarks": num_benchmarks_processed,
        "total_cycles": total_cycles_all,
        "total_queries": total_queries_all,
        "avg_copu_utilization": avg_copu_utilization_all,
        "total_cht_conflicts": total_cht_conflicts_all,
    }

    return batch_result


def print_results(results, is_range=False):
    """
    统一的结果输出函数

    Args:
        results: 仿真结果字典
        is_range: 是否为范围模式
    """
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

    print(f"  总周期: {results['total_cycles']}")
    print(f"  总查询数: {results['total_queries']:.0f}")

    if results["total_cycles"] > 0:
        print(
            f"  系统吞吐量: {results['total_queries'] / results['total_cycles']:.4f} queries/cycle"
        )

    print(f"  平均COPU占用率: {results.get('avg_copu_utilization', 0.0):.2%}")
    print(f"  CHT冲突数: {results.get('total_cht_conflicts', 0)}")
    print("\n" + "=" * 80)


def main():
    """主程序"""
    # 命令行参数
    if len(sys.argv) < 6:
        print(
            "用法 (单个): python multi_copu_real_data_simulation.py <basename> <benchid> <data_folder> <num_copus> <threshold> [num_oocds] [sample_rate] [max_cycles] [--real-cycles] [--no-cht-conflict]"
        )
        print(
            "用法 (范围): python multi_copu_real_data_simulation.py <basename> <benchid_start>-<benchid_end> <data_folder> <num_copus> <threshold> [num_oocds] [sample_rate] [max_cycles] [--real-cycles] [--no-cht-conflict]"
        )
        print(
            "示例 (单个): python multi_copu_real_data_simulation.py iiwa_7 46 ../trace_files/scene_benchmarks/bit_collision_data 4 1.0 7 0.1 10000"
        )
        print(
            "示例 (范围): python multi_copu_real_data_simulation.py iiwa_7 1-10 ../trace_files/scene_benchmarks/bit_collision_data 4 1.0 7 0.1 10000 --no-cht-conflict"
        )
        sys.exit(1)

    is_range_mode = "-" in sys.argv[2]
    num_oocds = 7
    sample_rate = 1.0
    max_cycles = 100000
    use_real_cycles = "--real-cycles" in sys.argv
    enable_conflict_check = "--no-cht-conflict" not in sys.argv

    # 解析可选数值参数 (顺序: num_oocds, sample_rate, max_cycles)
    optional_values = []
    for arg in sys.argv[6:]:
        if arg == "--real-cycles":
            continue
        if arg.replace(".", "", 1).isdigit():
            optional_values.append(float(arg) if "." in arg else int(arg))

    if len(optional_values) >= 1:
        num_oocds = int(optional_values[0])
    if len(optional_values) >= 2:
        sample_rate = float(optional_values[1])
    if len(optional_values) >= 3:
        max_cycles = int(optional_values[2])

    basename = sys.argv[1]
    benchid_arg = sys.argv[2]
    data_folder = sys.argv[3]
    num_copus = int(sys.argv[4])
    threshold = float(sys.argv[5])

    print("=" * 80)
    print("多COPU实际数据仿真")
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
    print(f"  COPU数量: {num_copus}")
    print(f"  碰撞阈值: {threshold}")
    print(f"  CDU数量(OOCD): {num_oocds}")
    print(f"  采样率: {sample_rate}")
    print(f"  最大周期: {max_cycles}")
    print(f"  使用真实周期: {use_real_cycles}")
    print(f"  CHT冲突检测: {enable_conflict_check}")

    print("\n【步骤1】执行仿真")
    if is_range_mode:
        benchid_start, benchid_end = map(int, benchid_arg.split("-"))
        results = run_benchmark_range_simulation(
            basename,
            benchid_start,
            benchid_end,
            data_folder,
            num_copus,
            num_oocds=num_oocds,
            quant_bits=3,
            threshold=threshold,
            sample_rate=sample_rate,
            max_cycles=max_cycles,
            use_real_cycles=use_real_cycles,
            enable_conflict_check=enable_conflict_check,
        )
    else:
        benchid = int(benchid_arg)
        results = simulate_single_benchmark(
            basename,
            benchid,
            data_folder,
            num_copus,
            num_oocds=num_oocds,
            quant_bits=3,
            threshold=threshold,
            sample_rate=sample_rate,
            max_cycles=max_cycles,
            use_real_cycles=use_real_cycles,
            enable_conflict_check=enable_conflict_check,
        )

        if results is None:
            print(f"✗ 无法加载数据: {basename}_{benchid:04d}")
            sys.exit(1)

    print("\n【步骤2】输出结果")
    print_results(results, is_range=is_range_mode)


if __name__ == "__main__":
    main()
