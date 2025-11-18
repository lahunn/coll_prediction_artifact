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
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_copu_simulation import MultiCOPU_Scheduler, analyze_multi_copu_performance
import simulation_utils as su


def load_and_partition_data(
    basename, benchid, data_folder, num_copus, use_real_cycles=False
):
    """
    加载实际数据文件并按pose维度分派给各COPU

    Args:
        basename: 数据集基名（如 "iiwa_7"）
        benchid: benchmark编号
        data_folder: 数据文件夹路径
        num_copus: COPU数量
        use_real_cycles: 是否使用数据文件中的真实周期数

    Returns:
        (data_list, coll_list, cycles_list) 或 (None, None, None)
        其中每个list的长度为num_copus，对应各COPU分配的数据
    """
    # 使用 simulation_utils 加载数据
    if use_real_cycles:
        all_data, all_coll, all_cycles = su.load_data_with_cycles(
            basename, benchid, data_folder, collision_model_type="link"
        )
        if all_data is None:
            print(f"✗ 无法加载数据: {basename}_{benchid:04d}")
            return None, None, None
    else:
        all_data, all_coll = su.load_data(
            basename, benchid, data_folder, collision_model_type="link"
        )
        if all_data is None:
            print(f"✗ 无法加载数据: {basename}_{benchid:04d}")
            return None, None, None
        all_cycles = None

    print("✓ 成功加载数据")
    print(f"  Edge数量: {len(all_data)}")
    print(f"  首个Edge的Pose数: {len(all_data[0]) if all_data else 0}")
    if all_data and all_data[0]:
        print(f"  首个Pose的Link数: {len(all_data[0][0]) if all_data[0][0] else 0}")

    # 按pose维度分派任务
    all_poses = []
    all_pose_colls = []
    all_pose_cycles = [] if use_real_cycles else None

    for edge_idx, edge in enumerate(all_data):
        for pose_idx, pose in enumerate(edge):
            all_poses.append(pose)
            all_pose_colls.append(all_coll[edge_idx][pose_idx])  # pyright: ignore[reportOptionalSubscript]
            if use_real_cycles:
                all_pose_cycles.append(all_cycles[edge_idx][pose_idx])  # type: ignore

    total_poses = len(all_poses)
    print(f"  总Pose数: {total_poses}")

    # 分派策略：均匀分配
    poses_per_copu = total_poses // num_copus
    remainder = total_poses % num_copus

    data_list = []
    coll_list = []
    cycles_list = []

    for copu_id in range(num_copus):
        if copu_id < remainder:
            start_idx = copu_id * (poses_per_copu + 1)
            end_idx = start_idx + poses_per_copu + 1
        else:
            start_idx = (
                remainder * (poses_per_copu + 1)
                + (copu_id - remainder) * poses_per_copu
            )
            end_idx = start_idx + poses_per_copu

        copu_poses = all_poses[start_idx:end_idx]
        copu_pose_colls = all_pose_colls[start_idx:end_idx]

        copu_linklist = []
        copu_linklist_coll = []
        copu_cycles = []

        for pose_idx, (pose, pose_coll) in enumerate(zip(copu_poses, copu_pose_colls)):
            copu_linklist.extend(pose)
            copu_linklist_coll.extend(pose_coll)

            if use_real_cycles and all_pose_cycles is not None:
                pose_global_idx = start_idx + pose_idx
                copu_cycles.extend(all_pose_cycles[pose_global_idx])
            else:
                copu_cycles.extend([40 for _ in range(len(pose))])

        data_list.append(copu_linklist)
        coll_list.append(copu_linklist_coll)
        cycles_list.append(copu_cycles)

        print(
            f"  COPU[{copu_id}]: {end_idx - start_idx} poses, {len(copu_linklist)} links"
        )

    return data_list, coll_list, cycles_list


def run_multi_copu_simulation(
    data_list,
    coll_list,
    cycles_list,
    num_copus,
    bins=None,
    threshold=1.0,
    sample_rate=1.0,
    max_cycles=100000,
):
    """
    执行多COPU协同仿真

    Args:
        data_list: 各COPU的数据列表
        coll_list: 各COPU的碰撞标志列表
        cycles_list: 各COPU的周期列表
        num_copus: COPU数量
        bins: 量化bin（默认为均匀分布）
        threshold: 碰撞预测阈值
        sample_rate: 采样率
        max_cycles: 最大仿真周期

    Returns:
        results dict containing global metrics and per-COPU stats
    """
    if bins is None:
        bins = np.linspace(0, 100, 10)

    # 创建调度器
    scheduler = MultiCOPU_Scheduler(num_copus=num_copus, num_oocds=7, cht_size=4096)

    # 加载数据
    scheduler.load_data_for_all_copus(data_list, coll_list, cycles_list)

    print(f"\n开始仿真 ({num_copus} COPU)...")
    start_time = time.time()

    # 执行仿真
    results = scheduler.simulate(
        bins, threshold=threshold, sample_rate=sample_rate, max_cycles=max_cycles
    )

    elapsed_time = time.time() - start_time

    return results, elapsed_time


def print_simulation_results(results, elapsed_time, num_copus):
    """
    打印仿真结果

    Args:
        results: 仿真结果字典
        elapsed_time: 执行时间
        num_copus: COPU数量
    """
    # 分析性能指标
    perf = analyze_multi_copu_performance(results)

    print("\n" + "=" * 80)
    print("仿真结果汇总")
    print("=" * 80)

    print("\n【整体指标】")
    print(f"  总周期: {results['total_cycles']}")
    print(f"  总查询数: {perf['total_queries']:.0f}")
    print(f"  系统吞吐量: {perf['system_throughput']:.4f} queries/cycle")
    print(f"  碰撞发现: {'是' if results['collision_found'] else '否'}")
    print(f"  周期限制达到: {'是' if results['cycle_limit_reached'] else '否'}")

    print("\n【利用率分析】")
    print(
        f"  平均COPU利用率: {perf['avg_copu_utilization']:.4f} ({perf['avg_copu_utilization'] * 100:.2f}%)"
    )

    copu_stats = results["copus"]
    for copu_stat in copu_stats:
        print(
            f"    COPU[{copu_stat['copu_id']}]: {copu_stat['oocd_utilization']:.4f} "
            f"({copu_stat['oocd_utilization'] * 100:.2f}%), 查询={copu_stat['total_queries']}"
        )

    print("\n【负载均衡】")
    print(f"  负载均衡系数: {perf['load_balance_variance']:.4f}")
    per_copu_queries = perf["per_copu_queries"]
    print(f"  各COPU查询数: {[int(q) for q in per_copu_queries]}")
    if per_copu_queries:
        print(
            f"  最小/最大查询数: {min(per_copu_queries):.0f} / {max(per_copu_queries):.0f}"
        )

    print("\n【CHT性能】")
    cht_stats = results["cht_stats"]
    print(f"  总读操作: {cht_stats['total_reads']}")
    print(f"  总写操作: {cht_stats['total_writes']}")
    print(f"  总冲突数: {cht_stats['total_conflicts']}")
    print(
        f"  CHT冲突率: {cht_stats['conflict_rate']:.4f} ({cht_stats['conflict_rate'] * 100:.2f}%)"
    )
    print(f"  CHT条目使用数: {cht_stats['entries_used']}")

    print("\n【执行时间】")
    print(f"  仿真耗时: {elapsed_time:.3f}s")

    print("\n" + "=" * 80)

    return perf


def main():
    """主程序"""
    # 命令行参数
    if len(sys.argv) < 6:
        print(
            "用法: python multi_copu_real_data_simulation.py <basename> <benchid> <data_folder> <num_copus> <threshold> [sample_rate] [max_cycles] [--real-cycles]"
        )
        print(
            "示例: python multi_copu_real_data_simulation.py iiwa_7 1 ../trace_files/scene_benchmarks/bit_collision_data 8 1.0 1.0 100000"
        )
        sys.exit(1)

    basename = sys.argv[1]
    benchid = int(sys.argv[2])
    data_folder = sys.argv[3]
    num_copus = int(sys.argv[4])
    threshold = float(sys.argv[5])

    # 可选参数
    sample_rate = 1.0
    max_cycles = 100000
    use_real_cycles = False

    for arg in sys.argv[6:]:
        if arg == "--real-cycles":
            use_real_cycles = True
        elif arg.replace(".", "", 1).isdigit():
            # 检查是否为第6或第7个位置的参数
            if len(sys.argv) > 6 and sys.argv[6].replace(".", "", 1).isdigit():
                sample_rate = float(sys.argv[6])
            if len(sys.argv) > 7 and sys.argv[7].replace(".", "", 1).isdigit():
                max_cycles = int(sys.argv[7])

    print("=" * 80)
    print("多COPU实际数据仿真")
    print("=" * 80)

    print("\n【输入参数】")
    print(f"  数据集: {basename}_{benchid:04d}")
    print(f"  数据文件夹: {data_folder}")
    print(f"  COPU数量: {num_copus}")
    print(f"  碰撞阈值: {threshold}")
    print(f"  采样率: {sample_rate}")
    print(f"  最大周期: {max_cycles}")
    print(f"  使用真实周期: {use_real_cycles}")

    # 步骤1: 加载并分派数据
    print("\n【步骤1】加载和分派数据")
    data_list, coll_list, cycles_list = load_and_partition_data(
        basename, benchid, data_folder, num_copus, use_real_cycles=use_real_cycles
    )

    if data_list is None:
        print("✗ 数据加载失败，程序退出")
        sys.exit(1)

    # 步骤2: 执行仿真
    print("\n【步骤2】执行仿真")
    results, elapsed_time = run_multi_copu_simulation(
        data_list,
        coll_list,
        cycles_list,
        num_copus,
        bins=np.linspace(0, 100, 10),
        threshold=threshold,
        sample_rate=sample_rate,
        max_cycles=max_cycles,
    )

    # 步骤3: 输出结果
    print("\n【步骤3】输出结果")
    perf = print_simulation_results(results, elapsed_time, num_copus)

    # 返回性能指标用于脚本链接
    return perf


if __name__ == "__main__":
    main()
