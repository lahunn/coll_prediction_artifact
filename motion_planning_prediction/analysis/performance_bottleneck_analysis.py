#!/usr/bin/env python3
"""
Cycle数与OOCD数量关系分析程序

专注于分析并行碰撞检测仿真中cycle数随OOCD数量变化的规律和限制因素
"""

import sys
import os
import numpy as np
from tqdm import tqdm
import csv
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su

# 添加 trace_generation 目录到 Python 路径
from trace_generation.config.ana_parameters import get_robot_params

# --- Simulation Settings ---
quant_bits = 4  # 4 bits per dimension (16 bins)
bins = su.calculate_bins_from_workspace("iiwa", quant_bits)

# --- Simulation Parameters from Command Line ---
if len(sys.argv) < 8:
    print(
        "Usage: python performance_bottleneck_analysis.py <threshold> <sample_rate> <qnoncoll_multiplier> <data_folder> <basename> <num_benchmarks> <robot_name> <num_oocds> [--with-cycles]"
    )
    print(
        "Example: python performance_bottleneck_analysis.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 10 iiwa 7"
    )
    print(
        "Example with cycles: python performance_bottleneck_analysis.py 0.5 0.1 8 ../../trace_files/scene_benchmarks/bit_collision_data iiwa_7 10 iiwa 7 --with-cycles"
    )
    sys.exit(1)

threshold = float(sys.argv[1])
sample_rate = float(sys.argv[2])
qnoncoll_multiplier = int(sys.argv[3])
data_folder = sys.argv[4]
basename = sys.argv[5]
num_benchmarks = int(sys.argv[6])
robot_name = sys.argv[7]
num_oocds = int(sys.argv[8])

# 检查是否指定加载带cycles的数据
load_with_cycles = "--with-cycles" in sys.argv

# 获取机器人参数
robot_params = get_robot_params(robot_name)
sphere_num = robot_params["sphere_num"]
sphere_cost = 45  # 假设每个球体的碰撞检测消耗的周期数为30

num_spheres = 8
qnoncoll_len = 56

print("=== Cycle数与OOCD数量关系分析 ===")
print(f"阈值: {threshold}")
print(f"采样率: {sample_rate}")
print(f"队列长度倍数: {qnoncoll_multiplier}")
print(f"非碰撞队列长度: {qnoncoll_len}")
print(f"OOCD数量: {num_oocds}")
print(f"数据文件夹: {data_folder}")
print(f"基准测试数量: {num_benchmarks}")
print(f"加载带cycles数据: {load_with_cycles}")
print("=" * 50)

# --- Benchmark Range ---
benchrange = range(1, num_benchmarks + 1)

# --- 核心性能分析数据结构（专注于cycle数和OOCD扩展性） ---
performance_stats = {
    "total_cycles": 0,  # 总cycle数 - 核心指标
    "total_queries": 0.0,  # 总查询数
    "total_edges_processed": 0,  # 处理的边数
    "total_spheres_processed": 0,  # 处理的球体数
    "queue_full_events": 0,  # 队列满事件数 - 反映队列瓶颈
    "oocd_idle_cycles": 0,  # OOCD空闲周期数 - 用于计算利用率
    "total_tasks_processed": 0,  # 总任务处理数
    "simulation_iterations": 0,  # 仿真迭代次数
    # 空闲原因细分统计
    "oocd_idle_no_tasks": 0,  # 因队列为空而空闲的周期数
    "oocd_idle_waiting_first_two": 0,  # 因等待前两个任务而空闲的周期数
    "oocd_idle_qnoncoll_not_full": 0,  # 因qnoncoll未满而空闲的周期数
    # 队列统计
    "qcoll_lengths_sum": 0,  # qcoll队列长度总和
    "qnoncoll_lengths_sum": 0,  # qnoncoll队列长度总和
    "qcoll_max_length": 0,  # qcoll最大长度
    "qnoncoll_max_length": 0,  # qnoncoll最大长度
    "active_oocds_sum": 0,  # 活跃OOCD数量总和
    "qnoncoll_added_count": 0,
    "qnoncoll_consumed_count": 0,
}


# --- 详细分析函数（专注于cycle数和OOCD扩展性） ---
def analyze_simulation_bottlenecks(
    linklist,
    linklist_coll,
    colldict,
    threshold,
    sample_rate,
    bins,
    qnoncoll_len,
    cycle_check,
    num_oocds,
    linklist_cycles=None,
):
    """
    分析仿真中cycle数与OOCD数量关系的限制因素
    基于实际碰撞检测结果进行分析
    """
    local_stats = {
        "queue_full_events": 0,  # 队列满事件数
        "oocd_idle_cycles": 0,  # OOCD空闲周期数
        "total_tasks_processed": 0,  # 总任务处理数
        "simulation_iterations": 0,  # 仿真迭代次数
        # 空闲原因细分统计
        "oocd_idle_no_tasks": 0,  # 因队列为空而空闲的周期数
        "oocd_idle_waiting_first_two": 0,  # 因等待前两个任务而空闲的周期数
        "oocd_idle_qnoncoll_not_full": 0,  # 因qnoncoll未满而空闲的周期数
        # 队列统计
        "qcoll_lengths_sum": 0,  # qcoll队列长度总和
        "qnoncoll_lengths_sum": 0,  # qnoncoll队列长度总和
        "qcoll_max_length": 0,  # qcoll最大长度
        "qnoncoll_max_length": 0,  # qnoncoll最大长度
        "active_oocds_sum": 0,  # 活跃OOCD数量总和
        "qnoncoll_added_count": 0,  # 添加到qnoncoll的任务总数
        "qnoncoll_consumed_count": 0,  # 从qnoncoll消耗的任务总数
    }

    # 初始化硬件碰撞检测器 (OOCD)
    oocds = [
        su.OOCDState(hash_key=0, result=0, busy=0, free_cycle=0)
        for _ in range(num_oocds)
    ]

    # 使用deque替代list，提高队列操作效率
    qcoll = deque(maxlen=8)  # 预测碰撞任务队列 [keyy, linkcoll, cycle]
    qnoncoll = deque(maxlen=qnoncoll_len)  # 预测无碰撞任务队列 [keyy, linkcoll, cycle]

    cycle = 0  # 仿真周期计数器
    first_two_running = 0  # 当前正在运行的前两个任务计数
    first_two_checked = 0  # 前两个任务开始处理的周期标记
    coll_found = 0  # 是否发现真实碰撞的标志
    links_remaining = len(linklist)  # 剩余待处理的配置数量
    everything_free = 0  # 所有任务是否完成的标志
    query_count = 0.0  # 实际执行的硬件查询总数

    # 主循环：直到发现碰撞或所有任务完成
    while not coll_found and not everything_free:
        local_stats["simulation_iterations"] += 1

        # 记录本周期的队列长度
        qcoll_len = len(qcoll)
        qnoncoll_len_current = len(qnoncoll)
        local_stats["qcoll_lengths_sum"] += qcoll_len
        local_stats["qnoncoll_lengths_sum"] += qnoncoll_len_current
        local_stats["qcoll_max_length"] = max(
            local_stats["qcoll_max_length"], qcoll_len
        )
        local_stats["qnoncoll_max_length"] = max(
            local_stats["qnoncoll_max_length"], qnoncoll_len_current
        )

        # 计算本周期活跃的OOCD数量
        active_oocds = sum(
            1 for oocd in oocds if oocd.busy == 1 and oocd.free_cycle > cycle
        )
        local_stats["active_oocds_sum"] += active_oocds

        # 引入标志位，确保每个周期最多只有一个出队操作
        dequeued_this_cycle = False
        idle_oocds = 0
        for oocd_id in range(len(oocds)):
            oocd = oocds[oocd_id]
            # 如果一个检测器任务已完成 (繁忙状态且到达完成周期)
            if oocd.busy == 1 and oocd.free_cycle <= cycle:
                local_stats["total_tasks_processed"] += 1
                query_count += 1  # 增加硬件查询计数
                if oocd.result == 0:  # 实际检测到碰撞
                    coll_found = 1
                # 根据真实的检测结果，更新碰撞历史表
                colldict = su.update_collision_dict(
                    colldict, oocd.hash_key, oocd.result, sample_rate
                )

            # 如果一个检测器现在空闲 (到达完成周期) 并且本周期还未分配过任务
            if oocd.free_cycle <= cycle and not dequeued_this_cycle:
                # 优先从"预测碰撞"队列 (qcoll) 中取任务
                if len(qcoll) > 0 and first_two_checked < cycle:
                    first_two_running += 1
                    if first_two_running == 1:
                        first_two_checked = cycle + qcoll[0][2]  # 使用真实周期数
                    # 分配新任务给这个OOCD，使用真实周期数
                    oocds[oocd_id] = su.OOCDState(
                        hash_key=qcoll[0][0],
                        result=qcoll[0][1],
                        busy=1,
                        free_cycle=cycle + qcoll[0][2],
                    )
                    qcoll.popleft()
                    dequeued_this_cycle = True  # 标记本周期已出队
                # 如果qcoll为空，则从"预测不碰撞"队列 (qnoncoll) 中取任务
                elif (
                    len(qnoncoll) == qnoncoll_len
                    or (links_remaining == 0 and len(qnoncoll) > 0)
                    and first_two_checked < cycle
                ):
                    oocds[oocd_id] = su.OOCDState(
                        hash_key=qnoncoll[0][0],
                        result=qnoncoll[0][1],
                        busy=1,
                        free_cycle=cycle + qnoncoll[0][2],  # 使用真实周期数
                    )
                    qnoncoll.popleft()
                    local_stats["qnoncoll_consumed_count"] += 1
                    dequeued_this_cycle = True  # 标记本周期已出队
                else:
                    # 如果两个队列都没有任务，则OOCD变为空闲状态
                    oocds[oocd_id] = su.OOCDState(
                        hash_key=0, result=0, busy=0, free_cycle=0
                    )
                    idle_oocds += 1

                    # 记录空闲原因
                    if first_two_checked >= cycle:
                        # 因为等待前两个任务完成而空闲
                        local_stats["oocd_idle_waiting_first_two"] += 1
                    elif (
                        len(qcoll) == 0
                        and len(qnoncoll) < qnoncoll_len
                        and len(qnoncoll) > 0
                    ):
                        # qcoll为空，qnoncoll未满（且有任务）而空闲
                        local_stats["oocd_idle_qnoncoll_not_full"] += 1
                    else:
                        # 两个队列都为空而空闲
                        local_stats["oocd_idle_no_tasks"] += 1
            elif oocd.free_cycle <= cycle:
                # OOCD空闲，但本周期已出队，因此该OOCD也计为空闲
                idle_oocds += 1
                local_stats["oocd_idle_no_tasks"] += 1

        local_stats["oocd_idle_cycles"] += idle_oocds

        # --- 步骤2: 预测下一个配置并放入相应队列 ---
        if len(linklist) > 0:
            link, linkcoll = linklist[0], linklist_coll[0]
            # 获取周期数：如果有真实周期数则使用，否则使用固定值
            if linklist_cycles is not None:
                link_cycle = linklist_cycles[0]
            else:
                link_cycle = cycle_check

            # 将配置数据"量化"以生成用于查询历史表的键 (key)
            code_quant = np.digitize(link, bins, right=True)
            keyy = su.return_keyy(code_quant, quant_bits=quant_bits)

            # 使用历史表进行碰撞预测
            is_collision_predicted = su.predict_collision(colldict, keyy, threshold)

            # 根据预测结果，将配置放入不同的队列（包含周期数）
            if is_collision_predicted:
                if len(qcoll) < 8:  # 如果队列未满
                    qcoll.append([keyy, linkcoll, link_cycle])
                    del linklist[0]
                    del linklist_coll[0]
                    if linklist_cycles is not None:
                        del linklist_cycles[0]
            else:
                if len(qnoncoll) < qnoncoll_len:  # 如果队列未满
                    qnoncoll.append([keyy, linkcoll, link_cycle])
                    del linklist[0]
                    del linklist_coll[0]
                    if linklist_cycles is not None:
                        del linklist_cycles[0]
                    local_stats["qnoncoll_added_count"] += 1

        # --- 步骤3: 检查仿真是否结束 ---
        links_remaining = len(linklist)

        # 如果所有输入配置都已处理，所有检测器都空闲，且所有队列都为空
        if (
            links_remaining == 0
            and not any(oocd.free_cycle > cycle for oocd in oocds)
            and not qnoncoll
            and not qcoll
        ):
            everything_free = 1  # 设置结束标志

        cycle += 1  # 时间周期前进

    # --- 步骤4: 计算仿真结束时仍在运行的任务 ---
    # 对于未完成的检查，按其已执行的比例计入查询总数
    for oocd in oocds:
        if oocd.free_cycle > cycle:
            query_count += (cycle_check - oocd.free_cycle + cycle) / cycle_check

    return query_count, colldict, coll_found, cycle, local_stats


# --- Main Analysis Loop ---
for benchid in tqdm(benchrange, desc="性能分析"):
    # 根据全局参数决定是否加载带cycles的数据
    if load_with_cycles:
        sphere_link_data, sphere_link_coll_data, sphere_link_coll_cycles = (
            su.load_data_with_cycles(
                basename, benchid, data_folder, collision_model_type="sphere"
            )
        )
    else:
        sphere_link_data, sphere_link_coll_data = su.load_data(
            basename, benchid, data_folder, collision_model_type="link"
        )
        sphere_link_coll_cycles = None

    if sphere_link_data is None or sphere_link_coll_data is None:
        continue

    # 处理每条边
    if load_with_cycles and sphere_link_coll_cycles is not None:
        # 有cycles数据的情况
        for edge_idx, (edge, edge_coll, edge_cycles) in enumerate(
            zip(sphere_link_data, sphere_link_coll_data, sphere_link_coll_cycles)
        ):
            if not edge_coll:
                continue

            # --- 检查是否为无碰撞边 ---
            # 如果边中包含任何碰撞（值为0），则跳过此边
            has_collision = any(
                sphere_coll == 0 for pose_coll in edge_coll for sphere_coll in pose_coll
            )
            # if has_collision:
            #     continue  # 跳过有碰撞的边，只处理无碰撞边

            performance_stats["total_edges_processed"] += 1
            performance_stats["total_spheres_processed"] += len(edge) * len(
                edge_coll[0]
            )

            # --- CSP Rearrangement ---
            linklist, linklist_coll, linklist_cycles = su.csp_rearrange_with_cycles(
                edge, edge_coll, edge_cycles, groupsize=8
            )

            # --- Run Detailed Simulation Analysis ---
            edge_query_count, colldict, _, cycle, local_stats = (
                analyze_simulation_bottlenecks(
                    linklist,
                    linklist_coll,
                    {},
                    threshold,
                    sample_rate,
                    bins,
                    qnoncoll_len=qnoncoll_len,
                    cycle_check=sphere_cost,
                    num_oocds=num_oocds,
                    linklist_cycles=linklist_cycles,
                )
            )

            # 累积统计数据
            performance_stats["total_queries"] += edge_query_count
            performance_stats["total_cycles"] += cycle
            performance_stats["queue_full_events"] += local_stats["queue_full_events"]
            performance_stats["oocd_idle_cycles"] += local_stats["oocd_idle_cycles"]
            performance_stats["total_tasks_processed"] += local_stats[
                "total_tasks_processed"
            ]
            performance_stats["simulation_iterations"] += local_stats[
                "simulation_iterations"
            ]
            # 累积空闲原因统计
            performance_stats["oocd_idle_no_tasks"] += local_stats["oocd_idle_no_tasks"]
            performance_stats["oocd_idle_waiting_first_two"] += local_stats[
                "oocd_idle_waiting_first_two"
            ]
            performance_stats["oocd_idle_qnoncoll_not_full"] += local_stats[
                "oocd_idle_qnoncoll_not_full"
            ]
            # 累积队列统计
            performance_stats["qcoll_lengths_sum"] += local_stats["qcoll_lengths_sum"]
            performance_stats["qnoncoll_lengths_sum"] += local_stats[
                "qnoncoll_lengths_sum"
            ]
            performance_stats["qcoll_max_length"] = max(
                performance_stats["qcoll_max_length"], local_stats["qcoll_max_length"]
            )
            performance_stats["qnoncoll_max_length"] = max(
                performance_stats["qnoncoll_max_length"],
                local_stats["qnoncoll_max_length"],
            )
            performance_stats["active_oocds_sum"] += local_stats["active_oocds_sum"]
            performance_stats["qnoncoll_added_count"] += local_stats[
                "qnoncoll_added_count"
            ]
            performance_stats["qnoncoll_consumed_count"] += local_stats[
                "qnoncoll_consumed_count"
            ]
    else:
        # 没有cycles数据的情况
        for edge_idx, (edge, edge_coll) in enumerate(
            zip(sphere_link_data, sphere_link_coll_data)
        ):
            if not edge_coll:
                continue

            # --- 检查是否为无碰撞边 ---
            # 如果边中包含任何碰撞（值为0），则跳过此边
            has_collision = any(
                sphere_coll == 0 for pose_coll in edge_coll for sphere_coll in pose_coll
            )
            # if has_collision:
            #     continue  # 跳过有碰撞的边，只处理无碰撞边

            performance_stats["total_edges_processed"] += 1
            performance_stats["total_spheres_processed"] += len(edge) * len(
                edge_coll[0]
            )

            # --- CSP Rearrangement （不带cycles）---
            linklist, linklist_coll = su.csp_rearrange(edge, edge_coll, groupsize=8)
            linklist_cycles = None

            # --- Run Detailed Simulation Analysis ---
            edge_query_count, colldict, _, cycle, local_stats = (
                analyze_simulation_bottlenecks(
                    linklist,
                    linklist_coll,
                    {},
                    threshold,
                    sample_rate,
                    bins,
                    qnoncoll_len=qnoncoll_len,
                    cycle_check=sphere_cost,
                    num_oocds=num_oocds,
                    linklist_cycles=None,
                )
            )

            # 累积统计数据
            performance_stats["total_queries"] += edge_query_count
            performance_stats["total_cycles"] += cycle
            performance_stats["queue_full_events"] += local_stats["queue_full_events"]
            performance_stats["oocd_idle_cycles"] += local_stats["oocd_idle_cycles"]
            performance_stats["total_tasks_processed"] += local_stats[
                "total_tasks_processed"
            ]
            performance_stats["simulation_iterations"] += local_stats[
                "simulation_iterations"
            ]
            # 累积空闲原因统计
            performance_stats["oocd_idle_no_tasks"] += local_stats["oocd_idle_no_tasks"]
            performance_stats["oocd_idle_waiting_first_two"] += local_stats[
                "oocd_idle_waiting_first_two"
            ]
            performance_stats["oocd_idle_qnoncoll_not_full"] += local_stats[
                "oocd_idle_qnoncoll_not_full"
            ]
            # 累积队列统计
            performance_stats["qcoll_lengths_sum"] += local_stats["qcoll_lengths_sum"]
            performance_stats["qnoncoll_lengths_sum"] += local_stats[
                "qnoncoll_lengths_sum"
            ]
            performance_stats["qcoll_max_length"] = max(
                performance_stats["qcoll_max_length"], local_stats["qcoll_max_length"]
            )
            performance_stats["qnoncoll_max_length"] = max(
                performance_stats["qnoncoll_max_length"],
                local_stats["qnoncoll_max_length"],
            )
            performance_stats["active_oocds_sum"] += local_stats["active_oocds_sum"]
            performance_stats["qnoncoll_added_count"] += local_stats[
                "qnoncoll_added_count"
            ]
            performance_stats["qnoncoll_consumed_count"] += local_stats[
                "qnoncoll_consumed_count"
            ]

# --- 性能分析报告 ---
print("\n" + "=" * 60)
print("性能瓶颈分析报告")
print("=" * 60)

print("\n仿真统计:")
print(f"  处理边数: {performance_stats['total_edges_processed']}")
print(f"  处理球体数: {performance_stats['total_spheres_processed']}")
print(f"  总查询数: {performance_stats['total_queries']:.0f}")
print(f"  总周期数: {performance_stats['total_cycles']}")
print(
    f"  平均每边周期数: {performance_stats['total_cycles'] / max(1, performance_stats['total_edges_processed']):.1f}"
)
print(f"  总任务处理数: {performance_stats['total_tasks_processed']}")
print(f"  仿真迭代次数: {performance_stats['simulation_iterations']}")

print("\n瓶颈指标:")
print(f"  队列满事件: {performance_stats['queue_full_events']}")
print(f"  OOCD空闲周期: {performance_stats['oocd_idle_cycles']}")

# 计算效率指标
oocd_utilization = 0.0

if performance_stats["total_cycles"] > 0:
    oocd_utilization = 1 - (
        performance_stats["oocd_idle_cycles"]
        / (performance_stats["total_cycles"] * num_oocds)
    )
    print(f"  OOCD利用率: {oocd_utilization:.3f}")

# OOCD空闲原因分析
if performance_stats["oocd_idle_cycles"] > 0:
    print("\nOOCD空闲原因分析:")
    idle_no_tasks_pct = (
        performance_stats["oocd_idle_no_tasks"]
        / performance_stats["oocd_idle_cycles"]
        * 100
    )
    idle_waiting_pct = (
        performance_stats["oocd_idle_waiting_first_two"]
        / performance_stats["oocd_idle_cycles"]
        * 100
    )
    idle_qnoncoll_pct = (
        performance_stats["oocd_idle_qnoncoll_not_full"]
        / performance_stats["oocd_idle_cycles"]
        * 100
    )
    print(
        f"  因队列为空: {idle_no_tasks_pct:.1f}% ({performance_stats['oocd_idle_no_tasks']} 周期)"
    )
    print(
        f"  因等待前两个任务: {idle_waiting_pct:.1f}% ({performance_stats['oocd_idle_waiting_first_two']} 周期)"
    )
    print(
        f"  因qnoncoll未满: {idle_qnoncoll_pct:.1f}% ({performance_stats['oocd_idle_qnoncoll_not_full']} 周期)"
    )

    # 计算因qnoncoll未满导致的周期浪费比例
    # 周期浪费 = 因qnoncoll未满而空闲的周期数
    # 总周期数 = 边的总周期 * CDU数
    qnoncoll_wasted_cycles = performance_stats["oocd_idle_qnoncoll_not_full"]
    total_cycles_with_copus = performance_stats["total_cycles"] * num_oocds
    if total_cycles_with_copus > 0:
        qnoncoll_waste_ratio = qnoncoll_wasted_cycles / total_cycles_with_copus * 100
        print(
            f"\n  ⚠️ 因qnoncoll未满导致的周期浪费: {qnoncoll_wasted_cycles} / {total_cycles_with_copus} = {qnoncoll_waste_ratio:.2f}%"
        )

# 队列利用率分析
if performance_stats["simulation_iterations"] > 0:
    print("\n队列利用率:")
    avg_qcoll_len = (
        performance_stats["qcoll_lengths_sum"]
        / performance_stats["simulation_iterations"]
    )
    avg_qnoncoll_len = (
        performance_stats["qnoncoll_lengths_sum"]
        / performance_stats["simulation_iterations"]
    )
    print(f"  qcoll平均长度: {avg_qcoll_len:.1f} / 8 ({avg_qcoll_len / 8 * 100:.1f}%)")
    print(f"  qcoll最大长度: {performance_stats['qcoll_max_length']}")
    print(
        f"  qnoncoll平均长度: {avg_qnoncoll_len:.1f} / {qnoncoll_len} ({avg_qnoncoll_len / qnoncoll_len * 100:.1f}%)"
    )
    print(f"  qnoncoll最大长度: {performance_stats['qnoncoll_max_length']}")

# OOCD并行度分析
if performance_stats["simulation_iterations"] > 0:
    print("\nOOCD并行度:")
    avg_active_oocds = (
        performance_stats["active_oocds_sum"]
        / performance_stats["simulation_iterations"]
    )
    print(
        f"  平均同时工作的OOCD: {avg_active_oocds:.2f} / {num_oocds} ({avg_active_oocds / num_oocds * 100:.1f}%)"
    )

print("\n性能限制分析:")
if performance_stats["queue_full_events"] > 0:
    print(f"  ⚠️  队列限制: 发生 {performance_stats['queue_full_events']} 次队列满事件")
    print("     建议: 增加队列长度以减少等待时间")

if oocd_utilization < 0.8:
    print(f"  ⚠️  OOCD利用不足: 利用率仅为 {oocd_utilization:.3f}")
    print("     原因: 任务调度不均或队列瓶颈")

    # 根据空闲原因给出针对性建议
    if performance_stats["oocd_idle_cycles"] > 0:
        idle_waiting_pct_check = (
            performance_stats["oocd_idle_waiting_first_two"]
            / performance_stats["oocd_idle_cycles"]
            * 100
        )
        idle_qnoncoll_pct_check = (
            performance_stats["oocd_idle_qnoncoll_not_full"]
            / performance_stats["oocd_idle_cycles"]
            * 100
        )
        idle_no_tasks_pct_check = (
            performance_stats["oocd_idle_no_tasks"]
            / performance_stats["oocd_idle_cycles"]
            * 100
        )
        if idle_waiting_pct_check > 30:
            print("     主要瓶颈: 等待前两个任务完成 - 考虑优化first_two_checked逻辑")
        if idle_qnoncoll_pct_check > 30:
            print("     主要瓶颈: qnoncoll未满条件限制 - 考虑放宽qnoncoll调度条件")
        if idle_no_tasks_pct_check > 30:
            print("     主要瓶颈: 队列为空 - 考虑增加预处理或调整CSP策略")

print("\n优化建议:")
if (
    performance_stats["queue_full_events"]
    > performance_stats["total_edges_processed"] * 0.1
):
    print("  1. 增加队列长度 (qnoncoll_len 和 qcoll_len)")
    print("  2. 优化任务调度策略")

if oocd_utilization < 0.7:
    print("  1. 检查任务分配是否均匀")
    print("  2. 考虑减少OOCD数量或增加队列大小")

print("  1. 考虑使用更高效的数据结构")
print("  2. 优化内存访问模式")
print("  3. 并行化预测计算")

# 输出到CSV
csv_file = "result_files/performance_bottleneck_analysis.csv"
with open(csv_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头（如果文件为空）
    if csvfile.tell() == 0:
        writer.writerow(
            [
                "threshold",
                "sample_rate",
                "qnoncoll_multiplier",
                "basename",
                "num_benchmarks",
                "robot_name",
                "num_oocds",
                "total_queries",
                "total_cycles",
                "total_edges_processed",
                "total_spheres_processed",
                "queue_full_events",
                "oocd_idle_cycles",
                "total_tasks_processed",
                "simulation_iterations",
                "oocd_idle_no_tasks",
                "oocd_idle_waiting_first_two",
                "oocd_idle_qnoncoll_not_full",
                "avg_qcoll_length",
                "avg_qnoncoll_length",
                "qcoll_max_length",
                "qnoncoll_max_length",
                "avg_active_oocds",
                "oocd_utilization",
            ]
        )

    # 计算平均值
    avg_qcoll_len = 0.0
    avg_qnoncoll_len = 0.0
    avg_active_oocds = 0.0
    if performance_stats["simulation_iterations"] > 0:
        avg_qcoll_len = (
            performance_stats["qcoll_lengths_sum"]
            / performance_stats["simulation_iterations"]
        )
        avg_qnoncoll_len = (
            performance_stats["qnoncoll_lengths_sum"]
            / performance_stats["simulation_iterations"]
        )
        avg_active_oocds = (
            performance_stats["active_oocds_sum"]
            / performance_stats["simulation_iterations"]
        )

    writer.writerow(
        [
            threshold,
            sample_rate,
            qnoncoll_multiplier,
            basename,
            num_benchmarks,
            robot_name,
            num_oocds,
            performance_stats["total_queries"],
            performance_stats["total_cycles"],
            performance_stats["total_edges_processed"],
            performance_stats["total_spheres_processed"],
            performance_stats["queue_full_events"],
            performance_stats["oocd_idle_cycles"],
            performance_stats["total_tasks_processed"],
            performance_stats["simulation_iterations"],
            performance_stats["oocd_idle_no_tasks"],
            performance_stats["oocd_idle_waiting_first_two"],
            performance_stats["oocd_idle_qnoncoll_not_full"],
            avg_qcoll_len,
            avg_qnoncoll_len,
            performance_stats["qcoll_max_length"],
            performance_stats["qnoncoll_max_length"],
            avg_active_oocds,
            oocd_utilization,
        ]
    )

print(f"\n详细数据已保存到: {csv_file}")
