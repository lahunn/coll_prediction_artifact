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
from .collision_prediction import enqueue_predictions
from .oocd_processor import (
    process_oocds,
    process_oocd_states_preemptive,
    handle_preemption,
    process_oocd_states_dedicated,
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
    oocds = [
        OOCDState(hash_key=0, result=0, busy=0, free_cycle=0) for _ in range(num_oocds)
    ]
    pred = Prediction(qcoll_len, qnoncoll_len)
    pred.linklist = linklist
    pred.linklist_coll = linklist_coll
    cycle = 0
    first_two_running = 0
    first_two_checked = 0
    coll_found = 0
    links_remaining = len(linklist)
    everything_free = 0
    query_count = 0.0

    while not coll_found and not everything_free:
        query_count, coll_found, first_two_running, first_two_checked, _ = (
            process_oocds(
                oocds,
                pred.qcoll,
                pred.qnoncoll,
                pred.linklist,
                cycle,
                query_count,
                coll_found,
                first_two_running,
                first_two_checked,
                cycle_check,
                colldict,
                sample_rate,
                num_oocds,
                qnoncoll_len,
            )
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

    return query_count, colldict, coll_found, cycle


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
    oocds = [
        OOCDState(hash_key=0, result=0, busy=0, free_cycle=0) for _ in range(num_oocds)
    ]
    pred = Prediction(qcoll_len, qnoncoll_len)
    pred.linklist = linklist
    pred.linklist_coll = linklist_coll
    cycle = 0
    first_two_running = 0
    first_two_checked = 0
    coll_found = 0
    links_remaining = len(linklist)
    everything_free = 0
    query_count = 0.0

    while not coll_found and not everything_free:
        query_count, coll_found, first_two_running, first_two_checked, _ = (
            process_oocds(
                oocds,
                pred.qcoll,
                pred.qnoncoll,
                pred.linklist,
                cycle,
                query_count,
                coll_found,
                first_two_running,
                first_two_checked,
                cycle_check,
                colldict,
                sample_rate,
                num_oocds,
                qnoncoll_len,
            )
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

    return query_count, colldict, coll_found, cycle


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
    oocds = [
        OOCDState(hash_key=0, result=0, busy=0, free_cycle=0) for _ in range(num_oocds)
    ]
    pred = Prediction(qcoll_len, qnoncoll_len)
    pred.linklist = linklist
    pred.linklist_coll = linklist_coll
    cycle = 0
    first_two_running = 0
    first_two_checked = 0
    coll_found = 0
    links_remaining = len(linklist)
    everything_free = 0
    query_count = 0.0

    predictions = []
    actuals = []

    while not coll_found and not everything_free:
        query_count, coll_found, first_two_running, first_two_checked, _ = (
            process_oocds(
                oocds,
                pred.qcoll,
                pred.qnoncoll,
                pred.linklist,
                cycle,
                query_count,
                coll_found,
                first_two_running,
                first_two_checked,
                cycle_check,
                colldict,
                sample_rate,
                num_oocds,
                qnoncoll_len,
            )
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
    oocds = [
        OOCDState(hash_key=0, result=0, busy=0, free_cycle=0) for _ in range(num_oocds)
    ]
    qcoll = deque(maxlen=qcoll_len)
    qnoncoll = deque(maxlen=qnoncoll_len)
    cycle = 0
    first_two_running = 0
    first_two_checked = 0
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
            first_two_running,
            first_two_checked,
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
            first_two_running,
            first_two_checked,
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


def simulate_parallel_collision_detection_double_buffer(
    edges_data,
    edges_coll,
    colldict,
    threshold,
    sample_rate,
    bins,
    qnoncoll_len=DEFAULT_QNONCOLL_LEN,
    qcoll_len=DEFAULT_QCOLL_LEN,
    cycle_check=DEFAULT_CYCLE_CHECK,
    num_oocds=NUM_OOCDS,
    num_predictions=2,
):
    """
    双缓冲架构的并行碰撞检测仿真。
    """
    oocds = [
        OOCDState(hash_key=0, result=0, busy=0, free_cycle=0) for _ in range(num_oocds)
    ]

    predictions = [Prediction(qcoll_len, qnoncoll_len) for _ in range(num_predictions)]

    active_index = 0
    next_load_edge_idx = 0
    cycle = 0
    total_query_count = 0.0
    cdu_idle_cycles = 0
    first_two_running = 0
    first_two_checked = 0
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
        edge_start_cycle = cycle
        active_index = edge_idx % num_predictions
        active_pred = predictions[active_index]

        qcoll_lengths_at_start.append(len(active_pred.qcoll))
        qnoncoll_lengths_at_start.append(len(active_pred.qnoncoll))

        edge_completed = False
        coll_found = 0
        while not edge_completed:
            (
                total_query_count,
                coll_found,
                first_two_running,
                first_two_checked,
                cdu_idle_this_cycle,
            ) = process_oocds(
                oocds,
                active_pred.qcoll,
                active_pred.qnoncoll,
                active_pred.linklist,
                cycle,
                total_query_count,
                coll_found,
                first_two_running,
                first_two_checked,
                cycle_check,
                colldict,
                sample_rate,
                num_oocds,
                qnoncoll_len,
            )
            cdu_idle_cycles += cdu_idle_this_cycle

            for pred in predictions:
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
        if coll_found > 0:
            total_coll_edge_cycles += edge_cycles
        else:
            total_noncoll_edge_cycles += edge_cycles

        for oocd_id in range(num_oocds):
            oocds[oocd_id] = OOCDState(hash_key=0, result=1, busy=0, free_cycle=0)
        active_pred.qcoll.clear()
        active_pred.qnoncoll.clear()

        if next_load_edge_idx < len(edges_data):
            edge_flat, edge_coll_flat = csp_rearrange(
                edges_data[next_load_edge_idx],
                edges_coll[next_load_edge_idx],
                groupsize=8,
            )
            active_pred.linklist = edge_flat
            active_pred.linklist_coll = edge_coll_flat
            next_load_edge_idx += 1

        first_two_running = 0
        first_two_checked = 0

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
