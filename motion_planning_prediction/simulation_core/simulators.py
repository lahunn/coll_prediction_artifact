"""
Simulation functions for parallel collision detection.
"""

from collections import deque
from .constants import (
    NUM_OOCDS,
    DEFAULT_QNONCOLL_LEN,
    DEFAULT_QCOLL_LEN,
    DEFAULT_CYCLE_CHECK,
)
from .data_structures import OOCDState, OOCDStatePreemptive, Prediction
from .collision_prediction import (
    enqueue_predictions,
    enqueue_link_predictions,
    enqueue_predictions_by_link,
)
from .oocd_processor import (
    process_oocds,
    process_oocd_states_preemptive,
    handle_preemption,
    process_oocd_states_dedicated,
    process_oocds_link,
)
from .data_preprocessing import csp_rearrange


def simulate_parallel_collision_detection(
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
    模拟并行的碰撞检测过程，该过程结合了硬件检测器 (OOCD) 和基于历史的碰撞预测。
    """
    oocds = [OOCDState() for _ in range(num_oocds)]
    pred = Prediction(qcoll_len, qnoncoll_len)
    pred.linklist = linklist
    pred.linklist_coll = linklist_coll
    cycle = 0
    coll_found = 0
    links_remaining = len(linklist)
    everything_free = 0
    query_count = 0.0
    total_idle_cycles = 0

    while not coll_found and not everything_free:
        query_count, coll_found, cdu_idle_this_cycle = process_oocds(
            oocds,
            pred.qcoll,
            pred.qnoncoll,
            pred.linklist,
            cycle,
            query_count,
            coll_found,
            cycle_check,
            colldict,
            sample_rate,
            num_oocds,
            qnoncoll_len,
        )
        total_idle_cycles += cdu_idle_this_cycle
        # 执行预测
        enqueue_predictions(
            pred.linklist,
            pred.linklist_coll,
            pred.qcoll,
            pred.qnoncoll,
            colldict,
            threshold,
            bins,
            qcoll_len,
            qnoncoll_len,
        )

        links_remaining = len(pred.linklist)
        if (
            links_remaining == 0
            and not any(oocd.free_cycle > cycle for oocd in oocds)
            and not pred.qnoncoll
            and not pred.qcoll
        ):
            everything_free = 1

        cycle += 1

    for oocd in oocds:
        if oocd.free_cycle > cycle:
            query_count += (cycle_check - oocd.free_cycle + cycle) / cycle_check
    oocd_utilization = (
        1.0 - (total_idle_cycles / (cycle * num_oocds)) if cycle > 0 else 0.0
    )
    return query_count, colldict, coll_found, cycle, oocd_utilization


def simulate_parallel_collision_detection_sphere(
    linklist,
    linklist_coll,
    colldict,
    threshold,
    sample_rate,
    bins,
    link_to_spheres,
    sphere_to_link,
    num_spheres_per_pose,
    qnoncoll_len=DEFAULT_QNONCOLL_LEN,
    qcoll_len=DEFAULT_QCOLL_LEN,
    cycle_check=DEFAULT_CYCLE_CHECK,
    num_oocds=NUM_OOCDS,
    collect_deadtime=False,
):
    """
    模拟并行的碰撞检测过程，每次预测时对属于同一link的所有sphere都进行预测。

    与simulate_parallel_collision_detection的区别：
    - process_oocds保持不变
    - enqueue阶段使用enqueue_predictions_by_link，对同一link的所有sphere一起预测
    """
    oocds = [OOCDState() for _ in range(num_oocds)]
    pred = Prediction(qcoll_len, qnoncoll_len)
    pred.linklist = linklist
    pred.linklist_coll = linklist_coll

    pose_cursor = [0]
    cycle = 0
    coll_found = 0
    everything_free = 0
    query_count = 0.0
    total_idle_cycles = 0
    first_dispatch_cycle = None

    while not coll_found and not everything_free:
        query_count, coll_found, cdu_idle_this_cycle = process_oocds(
            oocds,
            pred.qcoll,
            pred.qnoncoll,
            pred.linklist,
            cycle,
            query_count,
            coll_found,
            cycle_check,
            colldict,
            sample_rate,
            num_oocds,
            qnoncoll_len,
        )
        total_idle_cycles += cdu_idle_this_cycle

        # Dead-time metric: first cycle when any CDU starts processing this edge.
        if first_dispatch_cycle is None and any(
            oocd.busy == 1 and oocd.free_cycle > cycle for oocd in oocds
        ):
            first_dispatch_cycle = cycle

        # 使用enqueue_predictions_by_link：对同一link的所有sphere一起预测
        enqueue_predictions_by_link(
            pred.linklist,
            pred.linklist_coll,
            pred.qcoll,
            pred.qnoncoll,
            colldict,
            threshold,
            bins,
            qcoll_len,
            qnoncoll_len,
            link_to_spheres,
            sphere_to_link,
            num_spheres_per_pose,
            pose_cursor,
        )

        links_remaining = len(pred.linklist)
        if (
            links_remaining == 0
            and not any(oocd.free_cycle > cycle for oocd in oocds)
            and not pred.qnoncoll
            and not pred.qcoll
        ):
            everything_free = 1

        cycle += 1

    for oocd in oocds:
        if oocd.free_cycle > cycle:
            query_count += (cycle_check - oocd.free_cycle + cycle) / cycle_check

    oocd_utilization = (
        1.0 - (total_idle_cycles / (cycle * num_oocds)) if cycle > 0 else 0.0
    )

    dead_cycles = first_dispatch_cycle if first_dispatch_cycle is not None else cycle
    dead_ratio = (dead_cycles / cycle) if cycle > 0 else 0.0

    if collect_deadtime:
        deadtime_stats = {
            "issue_cycle": 0,
            "first_dispatch_cycle": first_dispatch_cycle,
            "dead_cycles": dead_cycles,
            "dead_ratio": dead_ratio,
        }
        return query_count, colldict, coll_found, cycle, oocd_utilization, deadtime_stats

    return query_count, colldict, coll_found, cycle, oocd_utilization


def simulate_parallel_collision_detection_link(
    linklist,
    linklist_coll,
    colldict,
    threshold,
    sample_rate,
    bins,
    link_to_spheres,
    sphere_to_link,
    num_spheres_per_pose,
    qnoncoll_len=DEFAULT_QNONCOLL_LEN,
    qcoll_len=DEFAULT_QCOLL_LEN,
    cycle_check=DEFAULT_CYCLE_CHECK,
    num_oocds=NUM_OOCDS,
    collect_deadtime=False,
):
    """Parallel collision simulation with link-level prediction enqueue and per-sphere dispatch."""
    oocds = [OOCDState() for _ in range(num_oocds)]
    pred = Prediction(qcoll_len, qnoncoll_len)
    pred.linklist = linklist
    pred.linklist_coll = linklist_coll

    pending_spheres = deque()
    pose_cursor = [0]
    cycle = 0
    coll_found = 0
    everything_free = 0
    query_count = 0.0
    total_idle_cycles = 0
    first_dispatch_cycle = None

    while not coll_found and not everything_free:
        query_count, coll_found, cdu_idle_this_cycle = process_oocds_link(
            oocds,
            pred.qcoll,
            pred.qnoncoll,
            pending_spheres,
            pred.linklist,
            cycle,
            query_count,
            coll_found,
            cycle_check,
            colldict,
            sample_rate,
            num_oocds,
            qnoncoll_len,
        )
        total_idle_cycles += cdu_idle_this_cycle

        # Dead-time metric: first cycle when any CDU starts processing this edge.
        if first_dispatch_cycle is None and any(
            oocd.busy == 1 and oocd.free_cycle > cycle for oocd in oocds
        ):
            first_dispatch_cycle = cycle

        enqueue_link_predictions(
            pred.linklist,
            pred.linklist_coll,
            pred.qcoll,
            pred.qnoncoll,
            colldict,
            threshold,
            bins,
            qcoll_len,
            qnoncoll_len,
            link_to_spheres,
            sphere_to_link,
            num_spheres_per_pose,
            pose_cursor,
        )

        links_remaining = len(pred.linklist)
        if (
            links_remaining == 0
            and not any(oocd.free_cycle > cycle for oocd in oocds)
            and not pred.qnoncoll
            and not pred.qcoll
            and not pending_spheres
        ):
            everything_free = 1

        cycle += 1

    for oocd in oocds:
        if oocd.free_cycle > cycle:
            query_count += (cycle_check - oocd.free_cycle + cycle) / cycle_check

    oocd_utilization = (
        1.0 - (total_idle_cycles / (cycle * num_oocds)) if cycle > 0 else 0.0
    )

    dead_cycles = first_dispatch_cycle if first_dispatch_cycle is not None else cycle
    dead_ratio = (dead_cycles / cycle) if cycle > 0 else 0.0

    if collect_deadtime:
        deadtime_stats = {
            "issue_cycle": 0,
            "first_dispatch_cycle": first_dispatch_cycle,
            "dead_cycles": dead_cycles,
            "dead_ratio": dead_ratio,
        }
        return query_count, colldict, coll_found, cycle, oocd_utilization, deadtime_stats

    return query_count, colldict, coll_found, cycle, oocd_utilization


def simulate_parallel_collision_detection_real_cycles(
    linklist,
    linklist_coll,
    linklist_cycles,
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
    使用真实周期数的并行碰撞检测仿真。
    """
    oocds = [OOCDState() for _ in range(num_oocds)]
    pred = Prediction(qcoll_len, qnoncoll_len)
    pred.linklist = linklist
    pred.linklist_coll = linklist_coll
    cycle = 0
    coll_found = 0
    links_remaining = len(linklist)
    everything_free = 0
    query_count = 0.0
    total_idle_cycles = 0

    while not coll_found and not everything_free:
        query_count, coll_found, cdu_idle_this_cycle = process_oocds(
            oocds,
            pred.qcoll,
            pred.qnoncoll,
            pred.linklist,
            cycle,
            query_count,
            coll_found,
            cycle_check,
            colldict,
            sample_rate,
            num_oocds,
            qnoncoll_len,
        )
        total_idle_cycles += cdu_idle_this_cycle
        # 执行预测
        enqueue_predictions(
            pred.linklist,
            pred.linklist_coll,
            pred.qcoll,
            pred.qnoncoll,
            colldict,
            threshold,
            bins,
            qcoll_len,
            qnoncoll_len,
            linklist_cycles,
        )

        links_remaining = len(pred.linklist)
        if (
            links_remaining == 0
            and not any(oocd.free_cycle > cycle for oocd in oocds)
            and not pred.qnoncoll
            and not pred.qcoll
        ):
            everything_free = 1

        cycle += 1

    for oocd in oocds:
        if oocd.free_cycle > cycle:
            query_count += 0.5

    oocd_utilization = (
        1.0 - (total_idle_cycles / (cycle * num_oocds)) if cycle > 0 else 0.0
    )

    return query_count, colldict, coll_found, cycle, oocd_utilization


def simulate_parallel_collision_detection_with_tracking(
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
    带预测跟踪的并行碰撞检测仿真，返回预测结果用于准确率计算。
    """
    oocds = [OOCDState() for _ in range(num_oocds)]
    pred = Prediction(qcoll_len, qnoncoll_len)
    pred.linklist = linklist
    pred.linklist_coll = linklist_coll
    cycle = 0
    coll_found = 0
    links_remaining = len(linklist)
    everything_free = 0
    query_count = 0.0

    predictions = []
    actuals = []

    while not coll_found and not everything_free:
        query_count, coll_found, _ = process_oocds(
            oocds,
            pred.qcoll,
            pred.qnoncoll,
            pred.linklist,
            cycle,
            query_count,
            coll_found,
            cycle_check,
            colldict,
            sample_rate,
            num_oocds,
            qnoncoll_len,
        )

        enqueue_predictions(
            pred.linklist,
            pred.linklist_coll,
            pred.qcoll,
            pred.qnoncoll,
            colldict,
            threshold,
            bins,
            qcoll_len,
            qnoncoll_len,
            predictions=predictions,
            actuals=actuals,
        )

        links_remaining = len(pred.linklist)
        if (
            links_remaining == 0
            and not any(oocd.free_cycle > cycle for oocd in oocds)
            and not pred.qnoncoll
            and not pred.qcoll
        ):
            everything_free = 1

        cycle += 1

    for oocd in oocds:
        if oocd.free_cycle > cycle:
            query_count += 0.5

    return query_count, colldict, coll_found, cycle, predictions, actuals


def simulate_parallel_collision_detection_preemptive(
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
    预先调度并行碰撞检测仿真，COLL任务可以抢占NONCOLL任务。
    """
    oocds = [
        OOCDStatePreemptive(hash_key=0, result=0, busy=0, free_cycle=0, task_type=None)
        for _ in range(num_oocds)
    ]
    qcoll = deque(maxlen=qcoll_len)
    qnoncoll = deque(maxlen=qnoncoll_len)
    current_time = 0
    coll_found = False
    everything_free = False
    query_count = 0.0
    preemption_count = 0

    while not coll_found and not everything_free:
        oocds, query_count, coll_found, colldict = process_oocd_states_preemptive(
            oocds,
            qcoll,
            qnoncoll,
            current_time,
            cycle_check,
            query_count,
            coll_found,
            colldict,
            sample_rate,
        )

        enqueue_predictions(
            linklist,
            linklist_coll,
            qcoll,
            qnoncoll,
            colldict,
            threshold,
            bins,
            qcoll_len,
            qnoncoll_len,
        )

        oocds, preemption_count = handle_preemption(
            oocds, qcoll, qnoncoll, current_time, cycle_check, preemption_count
        )

        links_remaining = len(linklist)
        if (
            links_remaining == 0
            and not any(oocd.free_cycle > current_time for oocd in oocds)
            and not qnoncoll
            and not qcoll
        ):
            everything_free = True

        current_time += 1

    for oocd in oocds:
        if oocd.free_cycle > current_time and oocd.busy == 1:
            executed_cycles = current_time - (oocd.free_cycle - cycle_check)
            query_count += executed_cycles / cycle_check

    return query_count, colldict, coll_found, current_time, preemption_count


def simulate_parallel_collision_detection_dedicated(
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
    num_dedicated_oocds=1,
):
    """
    模拟并行的碰撞检测过程，支持专用CDU策略。
    """
    oocds = [OOCDState() for _ in range(num_oocds)]
    qcoll = deque(maxlen=qcoll_len)
    qnoncoll = deque(maxlen=qnoncoll_len)
    cycle = 0
    coll_found = 0
    links_remaining = len(linklist)
    everything_free = 0
    query_count = 0.0

    while not coll_found and not everything_free:
        (
            oocds,
            query_count,
            coll_found,
            colldict,
        ) = process_oocd_states_dedicated(
            oocds,
            qcoll,
            qnoncoll,
            cycle,
            cycle_check,
            query_count,
            coll_found,
            colldict,
            sample_rate,
            num_dedicated_oocds,
            qnoncoll_len,
            linklist,
        )

        enqueue_predictions(
            linklist,
            linklist_coll,
            qcoll,
            qnoncoll,
            colldict,
            threshold,
            bins,
            qcoll_len,
            qnoncoll_len,
        )

        links_remaining = len(linklist)
        if (
            links_remaining == 0
            and not any(oocd.free_cycle > cycle for oocd in oocds)
            and not qnoncoll
            and not qcoll
        ):
            everything_free = 1

        cycle += 1

    for oocd in oocds:
        if oocd.free_cycle > cycle:
            query_count += (cycle_check - oocd.free_cycle + cycle) / cycle_check

    return query_count, colldict, coll_found, cycle


def simulate_edge_double_buffer(
    edge_idx,
    cycle,
    predictions,
    oocds,
    total_query_count,
    colldict,
    cycle_check,
    sample_rate,
    num_dedicated_oocds,
    qnoncoll_len,
    qcoll_len,
    threshold,
    bins,
    num_predictions,
    link_to_spheres,
    sphere_to_link,
    num_spheres_per_pose,
):
    """
    Simulate collision detection for a single edge in double buffer architecture.
    """
    active_index = edge_idx % num_predictions
    active_pred = predictions[active_index]

    qcoll_len_start = len(active_pred.qcoll)
    qnoncoll_len_start = len(active_pred.qnoncoll)

    edge_start_cycle = cycle
    edge_completed = False
    coll_found = 0

    while not edge_completed:
        (
            oocds,
            total_query_count,
            coll_found,
            colldict,
        ) = process_oocd_states_dedicated(
            oocds,
            active_pred.qcoll,
            active_pred.qnoncoll,
            cycle,
            cycle_check,
            total_query_count,
            coll_found,
            colldict,
            sample_rate,
            num_dedicated_oocds,
            qnoncoll_len,
            active_pred.linklist,
        )

        for pred in predictions:
            # 使用 enqueue_predictions_by_link
            enqueue_predictions_by_link(
                pred.linklist,
                pred.linklist_coll,
                pred.qcoll,
                pred.qnoncoll,
                colldict,
                threshold,
                bins,
                qcoll_len,
                qnoncoll_len,
                link_to_spheres,
                sphere_to_link,
                num_spheres_per_pose,
                pred.pose_cursor,
            )

        everything_free = (
            len(active_pred.linklist) == 0
            and not any(oocd.free_cycle > cycle for oocd in oocds)
            and len(active_pred.qnoncoll) == 0
            and len(active_pred.qcoll) == 0
        )
        if coll_found or everything_free:
            edge_completed = True
        cycle += 1

    edge_cycles = cycle - edge_start_cycle

    return (
        cycle,
        total_query_count,
        coll_found,
        edge_cycles,
        qcoll_len_start,
        qnoncoll_len_start,
    )


def simulate_parallel_collision_detection_double_buffer(
    edges_data,
    edges_coll,
    colldict,
    threshold,
    sample_rate,
    bins,
    link_to_spheres,
    sphere_to_link,
    num_spheres_per_pose,
    qnoncoll_len=DEFAULT_QNONCOLL_LEN,
    qcoll_len=DEFAULT_QCOLL_LEN,
    cycle_check=DEFAULT_CYCLE_CHECK,
    num_oocds=NUM_OOCDS,
    num_predictions=2,
    num_dedicated_oocds=1,
):
    """
    双缓冲架构的并行碰撞检测仿真。
    """
    oocds = [OOCDState() for _ in range(num_oocds)]

    predictions = [Prediction(qcoll_len, qnoncoll_len) for _ in range(num_predictions)]
    # 为每个Prediction对象初始化pose_cursor
    for pred in predictions:
        pred.pose_cursor = [0]

    active_index = 0
    next_load_edge_idx = 0
    cycle = 0
    total_query_count = 0.0
    cdu_idle_cycles = 0
    total_coll_edge_cycles = 0
    total_noncoll_edge_cycles = 0

    qcoll_lengths_at_start = []
    qnoncoll_lengths_at_start = []

    for i in range(min(num_predictions, len(edges_data))):
        edge_flat, edge_coll_flat = csp_rearrange(
            edges_data[i], edges_coll[i], groupsize=8
        )
        predictions[i].linklist = edge_flat
        predictions[i].linklist_coll = edge_coll_flat
        next_load_edge_idx += 1

    for edge_idx in range(len(edges_data)):
        (
            cycle,
            total_query_count,
            coll_found,
            edge_cycles,
            qcoll_len_start,
            qnoncoll_len_start,
        ) = simulate_edge_double_buffer(
            edge_idx,
            cycle,
            predictions,
            oocds,
            total_query_count,
            colldict,
            cycle_check,
            sample_rate,
            num_dedicated_oocds,
            qnoncoll_len,
            qcoll_len,
            threshold,
            bins,
            num_predictions,
            link_to_spheres,
            sphere_to_link,
            num_spheres_per_pose,
        )

        active_index = edge_idx % num_predictions
        active_pred = predictions[active_index]

        for oocd in oocds:
            oocd.reset()
        active_pred.qcoll.clear()
        active_pred.qnoncoll.clear()
        # 重置pose_cursor
        active_pred.pose_cursor[0] = 0

        if next_load_edge_idx < len(edges_data):
            edge_flat, edge_coll_flat = csp_rearrange(
                edges_data[next_load_edge_idx],
                edges_coll[next_load_edge_idx],
                groupsize=8,
            )
            active_pred.linklist = edge_flat
            active_pred.linklist_coll = edge_coll_flat
            next_load_edge_idx += 1

        qcoll_lengths_at_start.append(qcoll_len_start)
        qnoncoll_lengths_at_start.append(qnoncoll_len_start)

        if coll_found > 0:
            total_coll_edge_cycles += edge_cycles
        else:
            total_noncoll_edge_cycles += edge_cycles

    for oocd in oocds:
        if oocd.free_cycle > cycle:
            total_query_count += (cycle_check - oocd.free_cycle + cycle) / cycle_check

    stats = {
        "cdu_idle_cycles": cdu_idle_cycles,
        "cdu_utilization": 1.0 - (cdu_idle_cycles / (cycle * num_oocds))
        if cycle > 0
        else 0.0,
        "total_coll_edge_cycles": total_coll_edge_cycles,
        "total_noncoll_edge_cycles": total_noncoll_edge_cycles,
        "qcoll_lengths_at_start": qcoll_lengths_at_start,
        "qnoncoll_lengths_at_start": qnoncoll_lengths_at_start,
    }

    return total_query_count, colldict, cycle, stats
