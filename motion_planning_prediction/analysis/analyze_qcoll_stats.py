#!/usr/bin/env python3
"""
分析QCOLL统计数据的脚本

功能：
1. 分析两个被预测为qcoll的任务之间的平均间隔（以仿真周期/任务数为单位）
2. 对于实际碰撞的edge，分析平均需要执行几次qcoll任务才能得出碰撞结论

使用方法：
    python analyze_qcoll_stats.py <threshold> <sample_rate> <data_folder> <basename> <num_benchmarks>
    python analyze_qcoll_stats.py 0.5 0.1 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 10
"""

import sys
import os
import numpy as np
from tqdm import tqdm
from collections import deque, namedtuple

# 添加上级目录到path以导入simulation_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su
from simulation_core.constants import DEFAULT_CYCLE_CHECK

# Constants
NUM_OOCDS = 7
MAX_COLLISION_COUNT = 15
DEFAULT_QNONCOLL_LEN = 56
DEFAULT_QCOLL_LEN = 8
DEFAULT_CYCLE_CHECK_VALUE = DEFAULT_CYCLE_CHECK

# 扩展OOCDState以包含任务来源
OOCDStateAnalysis = namedtuple(
    "OOCDStateAnalysis", ["hash_key", "result", "busy", "free_cycle", "source"]
)


def simulate_collision_analysis(
    linklist,
    linklist_coll,
    colldict,
    threshold,
    sample_rate,
    bins,
    qnoncoll_len=DEFAULT_QNONCOLL_LEN,
    qcoll_len=DEFAULT_QCOLL_LEN,
    cycle_check=DEFAULT_CYCLE_CHECK,
    num_oocds=NUM_OOCDS,
):
    """
    修改版的仿真函数，用于收集统计数据
    """
    # 初始化硬件碰撞检测器 (OOCD)
    oocds = [
        OOCDStateAnalysis(hash_key=0, result=0, busy=0, free_cycle=0, source=None)
        for _ in range(num_oocds)
    ]

    qcoll = deque(maxlen=qcoll_len)
    qnoncoll = deque(maxlen=qnoncoll_len)
    cycle = 0
    first_two_running = 0
    first_two_checked = 0
    coll_found = 0
    links_remaining = len(linklist)
    everything_free = 0

    # 统计数据
    qcoll_add_times = []  # 记录加入qcoll的时间点（cycle）
    qcoll_tasks_started = 0  # 记录已经开始执行的qcoll任务数
    checks_at_collision = None  # 发现碰撞时已执行的qcoll任务数
    collision_found_source = None  # 发现碰撞的任务来源
    total_tasks_predicted_coll = 0  # 预测为碰撞的总任务数
    total_tasks = 0  # 总任务数
    first_collision_position = None  # 首个实际碰撞任务的位置（1-based）

    # 主循环
    while not coll_found and not everything_free:
        # --- 步骤1: 处理硬件检测器 (OOCD) 的状态 ---
        dequeued_this_cycle = False
        for oocd_id in range(len(oocds)):
            oocd = oocds[oocd_id]

            # 检查任务完成
            if oocd.busy == 1 and oocd.free_cycle <= cycle:
                if oocd.result == 0:  # 发现碰撞
                    coll_found = 1
                    if checks_at_collision is None:
                        checks_at_collision = qcoll_tasks_started
                        collision_found_source = oocd.source

                # 更新历史表
                colldict = su.update_collision_dict(
                    colldict, oocd.hash_key, oocd.result, sample_rate
                )

            # 分配新任务
            if oocd.free_cycle <= cycle and not dequeued_this_cycle:
                # 优先从 qcoll 取任务
                if len(qcoll) > 0 and first_two_checked < cycle:
                    first_two_running += 1
                    if first_two_running == 1:
                        first_two_checked = cycle + cycle_check

                    oocds[oocd_id] = OOCDStateAnalysis(
                        hash_key=qcoll[0][0],
                        result=qcoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                        source="qcoll",
                    )
                    qcoll.popleft()
                    dequeued_this_cycle = True
                    qcoll_tasks_started += 1  # 计数增加

                # 其次从 qnoncoll 取任务
                elif (
                    len(qnoncoll) == qnoncoll_len
                    or (links_remaining == 0 and len(qnoncoll) > 0)
                    and first_two_checked < cycle
                ):
                    oocds[oocd_id] = OOCDStateAnalysis(
                        hash_key=qnoncoll[0][0],
                        result=qnoncoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                        source="qnoncoll",
                    )
                    qnoncoll.popleft()
                    dequeued_this_cycle = True
                else:
                    oocds[oocd_id] = OOCDStateAnalysis(
                        hash_key=0, result=0, busy=0, free_cycle=0, source=None
                    )

        # --- 步骤2: 预测下一个配置 ---
        if len(linklist) > 0:
            link, linkcoll = linklist[0], linklist_coll[0]
            total_tasks += 1

            # 记录首个实际碰撞任务的位置
            if first_collision_position is None and linkcoll == 0:
                first_collision_position = total_tasks

            code_quant = np.digitize(link, bins, right=True)
            quant_bits = (len(bins) - 1).bit_length()
            keyy = su.return_keyy(code_quant, quant_bits)

            is_collision_predicted = su.predict_collision(colldict, keyy, threshold)

            if is_collision_predicted:
                total_tasks_predicted_coll += 1
                if len(qcoll) < qcoll_len:
                    qcoll.append([keyy, linkcoll])
                    qcoll_add_times.append(cycle)  # 记录加入时间
                    del linklist[0]
                    del linklist_coll[0]
            else:
                if len(qnoncoll) < qnoncoll_len:
                    qnoncoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]

        # --- 步骤3: 检查结束条件 ---
        links_remaining = len(linklist)
        if (
            links_remaining == 0
            and not any(oocd.free_cycle > cycle for oocd in oocds)
            and not qnoncoll
            and not qcoll
        ):
            everything_free = 1

        cycle += 1

    return {
        "qcoll_add_times": qcoll_add_times,
        "checks_at_collision": checks_at_collision,
        "collision_found_source": collision_found_source,
        "total_tasks": total_tasks,
        "total_tasks_predicted_coll": total_tasks_predicted_coll,
        "first_collision_position": first_collision_position,
        "total_cycles": cycle,
    }


def main():
    if len(sys.argv) < 6:
        print(
            "Usage: python analyze_qcoll_stats.py <threshold> <sample_rate> <data_folder> <basename> <num_benchmarks>"
        )
        sys.exit(1)

    threshold = float(sys.argv[1])
    sample_rate = float(sys.argv[2])
    data_folder = sys.argv[3]
    basename = sys.argv[4]
    num_benchmarks = int(sys.argv[5])

    # 初始化分箱
    binnumber = 16
    intervalsize = 2 / binnumber
    bins = np.zeros(binnumber)
    start = -1
    for i in range(binnumber):
        bins[i] = start
        start += intervalsize

    # 统计变量
    all_intervals = []
    checks_for_collisions = []
    total_collision_edges = 0
    first_collision_positions = []  # 记录所有碰撞edge的首个碰撞位置
    collision_edge_cycles = []  # 记录所有碰撞edge消耗的cycle数

    # Per-benchmark stats
    benchmark_stats = []

    print(f"开始分析: Threshold={threshold}, SampleRate={sample_rate}")
    print(f"Benchmarks: 1-{num_benchmarks}")

    for benchid in tqdm(range(1, num_benchmarks + 1), desc="Processing"):
        colldict = {}

        bench_intervals = []
        bench_checks = []
        bench_coll_edges = 0
        bench_total_edges = 0
        bench_pred_coll_tasks = 0
        bench_total_tasks = 0
        bench_found_in_qcoll = 0

        # 加载数据
        edge_link_data, edge_link_coll_data = su.load_data(
            basename, benchid, data_folder, collision_model_type="link"
        )

        if edge_link_data is None or edge_link_coll_data is None:
            continue

        bench_total_edges = len(edge_link_data)

        for edge, edge_coll in zip(edge_link_data, edge_link_coll_data):
            if not edge_coll:
                continue

            # 检查该edge是否包含真实碰撞
            has_real_collision = any(
                link_coll == 0 for pose_coll in edge_coll for link_coll in pose_coll
            )

            # CSP重排
            linklist, linklist_coll = su.csp_rearrange(edge, edge_coll, groupsize=4)

            # 运行仿真
            stats = simulate_collision_analysis(
                linklist, linklist_coll, colldict, threshold, sample_rate, bins
            )

            intervals = stats["qcoll_add_times"]
            checks = stats["checks_at_collision"]
            source = stats["collision_found_source"]
            first_pos = stats["first_collision_position"]
            total_cycles = stats["total_cycles"]

            bench_total_tasks += stats["total_tasks"]
            bench_pred_coll_tasks += stats["total_tasks_predicted_coll"]

            # 收集间隔数据
            if len(intervals) > 1:
                # 计算相邻时间点的差值
                diffs = [
                    intervals[i + 1] - intervals[i] for i in range(len(intervals) - 1)
                ]
                all_intervals.extend(diffs)
                bench_intervals.extend(diffs)

            # 收集碰撞检查次数数据
            if has_real_collision:
                total_collision_edges += 1
                bench_coll_edges += 1
                if checks is not None:
                    checks_for_collisions.append(checks)
                    bench_checks.append(checks)

                if source == "qcoll":
                    bench_found_in_qcoll += 1
                
                # 收集首个碰撞位置
                if first_pos is not None:
                    first_collision_positions.append(first_pos)
                
                # 收集cycle消耗
                collision_edge_cycles.append(total_cycles)

        # 计算该benchmark的统计数据
        avg_int = np.mean(bench_intervals) if bench_intervals else 0.0
        avg_chk = np.mean(bench_checks) if bench_checks else 0.0
        pred_rate = (
            (bench_pred_coll_tasks / bench_total_tasks * 100)
            if bench_total_tasks > 0
            else 0.0
        )
        recall = (
            (bench_found_in_qcoll / bench_coll_edges * 100)
            if bench_coll_edges > 0
            else 0.0
        )

        benchmark_stats.append(
            {
                "id": benchid,
                "edges": bench_total_edges,
                "coll_edges": bench_coll_edges,
                "interval": avg_int,
                "checks": avg_chk,
                "pred_rate": pred_rate,
                "recall": recall,
            }
        )

    # 输出Per-Benchmark统计表
    print("\n" + "-" * 90)
    print("Per-Benchmark Statistics")
    print("-" * 90)
    print(
        f" {'BenchID':>8} {'Edges':>8} {'Coll':>6} {'Interval':>10} {'CDetect':>10} {'PredRate%':>10} {'Recall%':>10}"
    )
    print("-" * 90)

    for s in benchmark_stats:
        print(
            f" {s['id']:8d} {s['edges']:8d} {s['coll_edges']:6d} {s['interval']:10.2f} {s['checks']:10.2f} {s['pred_rate']:10.2f} {s['recall']:10.2f}"
        )
    print("=" * 90)

    # 输出结果
    print("\n" + "=" * 50)
    print("分析结果汇总")
    print("=" * 50)

    if all_intervals:
        avg_interval = np.mean(all_intervals)
        median_interval = np.median(all_intervals)
        print("1. 两个被预测qcoll的任务之间的平均间隔:")
        print(f"   平均值: {avg_interval:.2f} cycles")
        print(f"   中位数: {median_interval:.2f} cycles")
        print(f"   样本数: {len(all_intervals)}")
    else:
        print("1. 无qcoll任务间隔数据 (预测器未预测出碰撞)")

    if checks_for_collisions:
        avg_checks = np.mean(checks_for_collisions)
        median_checks = np.median(checks_for_collisions)
        print("\n2. 实际碰撞Edge发现碰撞所需的qcoll任务数:")
        print(f"   平均值: {avg_checks:.2f} tasks")
        print(f"   中位数: {median_checks:.2f} tasks")
        print(f"   统计Edge数: {len(checks_for_collisions)} / {total_collision_edges}")
    else:
        print("\n2. 无碰撞Edge统计数据")

    if first_collision_positions:
        avg_position = np.mean(first_collision_positions)
        median_position = np.median(first_collision_positions)
        min_position = np.min(first_collision_positions)
        max_position = np.max(first_collision_positions)
        print("\n3. 重排后首个实际碰撞任务的位置统计:")
        print(f"   平均值: {avg_position:.2f}")
        print(f"   中位数: {median_position:.2f}")
        print(f"   最小值: {min_position}")
        print(f"   最大值: {max_position}")
        print(f"   统计Edge数: {len(first_collision_positions)} / {total_collision_edges}")
    else:
        print("\n3. 无首个碰撞位置统计数据")

    if collision_edge_cycles:
        avg_cycles = np.mean(collision_edge_cycles)
        median_cycles = np.median(collision_edge_cycles)
        min_cycles = np.min(collision_edge_cycles)
        max_cycles = np.max(collision_edge_cycles)
        print("\n4. 碰撞Edge的碰撞检测平均消耗cycle数:")
        print(f"   平均值: {avg_cycles:.2f} cycles")
        print(f"   中位数: {median_cycles:.2f} cycles")
        print(f"   最小值: {min_cycles} cycles")
        print(f"   最大值: {max_cycles} cycles")
        print(f"   统计Edge数: {len(collision_edge_cycles)} / {total_collision_edges}")
    else:
        print("\n4. 无碰撞Edge cycle消耗统计数据")

    print("=" * 50)


if __name__ == "__main__":
    main()
