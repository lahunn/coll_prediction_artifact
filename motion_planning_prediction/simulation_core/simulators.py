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
    enqueue_predictions_with_mode,
)
from .oocd_processor import (
    process_oocds,
    process_oocd_states_preemptive,
    handle_preemption,
    process_oocd_states_dedicated,
    process_oocds_link,
    process_oocds_with_mode,
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
    mode="simple",
    num_dedicated_oocds=0,
    link_to_spheres=None,
    sphere_to_link=None,
    num_spheres_per_pose=None,
):
    """
    Unified simulation function for parallel collision detection.
    Supports 'simple', 'batch', and 'hierarchical' modes.
    """

    oocds = [OOCDState() for _ in range(num_oocds)]
    pred = Prediction(qcoll_len, qnoncoll_len)
    pred.linklist = linklist
    pred.linklist_coll = linklist_coll

    # Initialize mode-specific structures
    if mode in ["batch", "hierarchical"]:
        pred.pose_cursor = [0]
    if mode == "hierarchical":
        pred.pending_spheres = deque()

    cycle = 0
    coll_found = 0
    everything_free = 0
    query_count = 0.0
    total_idle_cycles = 0

    while not coll_found and not everything_free:
        result = process_oocds_with_mode(
            mode=mode,
            oocds=oocds,
            qcoll=pred.qcoll,
            qnoncoll=pred.qnoncoll,
            linklist=pred.linklist,
            cycle=cycle,
            total_query_count=query_count,
            coll_found=coll_found,
            cycle_check=cycle_check,
            colldict=colldict,
            sample_rate=sample_rate,
            num_oocds=num_oocds,
            qnoncoll_len=qnoncoll_len,
            num_dedicated_oocds=num_dedicated_oocds,
            pending_spheres=getattr(pred, "pending_spheres", None),
        )

        query_count = result["total_query_count"]
        coll_found = result["coll_found"]
        cdu_idle_this_cycle = result.get("cdu_idle_cycles", 0)
        colldict = result["colldict"]

        # If 'oocds' is returned (e.g. from dedicated processor), update local reference if needed
        # (Though list is mutable, so in-place updates in sub-function work fine usually)

        # For dedicated mode, idle cycles calculation might be different or implicit in process_oocds logic
        # process_oocd_states_dedicated currently doesn't return idle cycles directly in dict
        # We can calculate it here if missing
        if "cdu_idle_cycles" not in result:
            cdu_idle_this_cycle = sum(
                1 for oocd in oocds if oocd.free_cycle <= cycle and not oocd.busy
            )

        total_idle_cycles += cdu_idle_this_cycle

        enqueue_predictions_with_mode(
            mode=mode,
            linklist=pred.linklist,
            linklist_coll=pred.linklist_coll,
            qcoll=pred.qcoll,
            qnoncoll=pred.qnoncoll,
            colldict=colldict,
            threshold=threshold,
            bins=bins,
            qcoll_len=qcoll_len,
            qnoncoll_len=qnoncoll_len,
            link_to_spheres=link_to_spheres,
            sphere_to_link=sphere_to_link,
            num_spheres_per_pose=num_spheres_per_pose,
            pose_cursor=getattr(pred, "pose_cursor", None),
        )

        links_remaining = len(pred.linklist)

        is_buffers_empty = links_remaining == 0 and not pred.qnoncoll and not pred.qcoll

        if mode == "hierarchical":
            is_buffers_empty = is_buffers_empty and not pred.pending_spheres

        if is_buffers_empty and not any(oocd.free_cycle > cycle for oocd in oocds):
            everything_free = 1

        cycle += 1

    for oocd in oocds:
        if oocd.free_cycle > cycle:
            query_count += (cycle_check - oocd.free_cycle + cycle) / cycle_check

    oocd_utilization = (
        1.0 - (total_idle_cycles / (cycle * num_oocds)) if cycle > 0 else 0.0
    )
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
        result = process_oocds(
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
        query_count = result["total_query_count"]
        coll_found = result["coll_found"]
        cdu_idle_this_cycle = result["cdu_idle_cycles"]
        colldict = result["colldict"]

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
        result = process_oocds(
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
        query_count = result["total_query_count"]
        coll_found = result["coll_found"]
        colldict = result["colldict"]

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
        result = process_oocd_states_preemptive(
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
        oocds = result["oocds"]
        query_count = result["total_query_count"]
        coll_found = result["coll_found"]
        colldict = result["colldict"]

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
    mode="batch",
):
    """
    Simulate collision detection for a single edge in double buffer architecture.
    Supported modes:
        - 'simple': One-to-one prediction (1 prediction -> 1 detection task).
        - 'batch': Batch expansion (1 prediction (Link) -> N detection tasks (Spheres), all enqueued immediately).
        - 'hierarchical': Hierarchical dispatch (1 prediction (Link) -> N pending tasks (Spheres), dispatched over time).
    """

    active_index = edge_idx % num_predictions
    active_pred = predictions[active_index]

    qcoll_len_start = len(active_pred.qcoll)
    qnoncoll_len_start = len(active_pred.qnoncoll)

    edge_start_cycle = cycle
    edge_completed = False
    coll_found = 0

    while not edge_completed:
        # 1. OOCD Processing
        result = process_oocds_with_mode(
            mode=mode,
            oocds=oocds,
            qcoll=active_pred.qcoll,
            qnoncoll=active_pred.qnoncoll,
            linklist=active_pred.linklist,
            cycle=cycle,
            total_query_count=total_query_count,
            coll_found=coll_found,
            cycle_check=cycle_check,
            colldict=colldict,
            sample_rate=sample_rate,
            num_oocds=len(oocds),
            qnoncoll_len=qnoncoll_len,
            num_dedicated_oocds=num_dedicated_oocds,
            pending_spheres=getattr(active_pred, "pending_spheres", None),
        )

        # Unpack results based on mode (adapter logic)
        total_query_count = result["total_query_count"]
        coll_found = result["coll_found"]
        colldict = result["colldict"]
        if "oocds" in result:
            oocds = result["oocds"]

        # 2. Prediction Enqueuing (Parallel for all buffers)
        for pred in predictions:
            enqueue_predictions_with_mode(
                mode=mode,
                linklist=pred.linklist,
                linklist_coll=pred.linklist_coll,
                qcoll=pred.qcoll,
                qnoncoll=pred.qnoncoll,
                colldict=colldict,
                threshold=threshold,
                bins=bins,
                qcoll_len=qcoll_len,
                qnoncoll_len=qnoncoll_len,
                link_to_spheres=link_to_spheres,
                sphere_to_link=sphere_to_link,
                num_spheres_per_pose=num_spheres_per_pose,
                pose_cursor=pred.pose_cursor,
            )

        # 3. Check Completion
        is_buffers_empty = (
            len(active_pred.linklist) == 0
            and len(active_pred.qnoncoll) == 0
            and len(active_pred.qcoll) == 0
        )

        if mode == "hierarchical":
            is_buffers_empty = is_buffers_empty and not active_pred.pending_spheres

        everything_free = is_buffers_empty and not any(
            oocd.free_cycle > cycle for oocd in oocds
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


def init_double_buffer_resources(
    num_oocds,
    num_predictions,
    qcoll_len,
    qnoncoll_len,
    edges_data,
    edges_coll,
):
    """
    Initialize resources for double buffer simulation: OOCDs, Predictions, and pre-load data.
    """
    oocds = [OOCDState() for _ in range(num_oocds)]
    predictions = [Prediction(qcoll_len, qnoncoll_len) for _ in range(num_predictions)]

    # Initialize pose_cursor and pending_spheres for each prediction object
    for pred in predictions:
        pred.pose_cursor = [0]
        pred.pending_spheres = deque()

    next_load_edge_idx = 0
    # Pre-load initial edges into buffers
    for i in range(min(num_predictions, len(edges_data))):
        edge_flat, edge_coll_flat = csp_rearrange(
            edges_data[i], edges_coll[i], groupsize=8
        )
        predictions[i].linklist = edge_flat
        predictions[i].linklist_coll = edge_coll_flat
        next_load_edge_idx += 1

    return oocds, predictions, next_load_edge_idx


def reload_prediction_buffer(
    active_pred, oocds, edges_data, edges_coll, next_load_edge_idx
):
    """
    Reset the active prediction buffer and hardware states, then load the next edge if available.
    """
    # Reset OOCD states for the next run
    for oocd in oocds:
        oocd.reset()

    # Clear the active prediction buffer
    active_pred.qcoll.clear()
    active_pred.qnoncoll.clear()
    active_pred.pose_cursor[0] = 0
    if hasattr(active_pred, "pending_spheres"):
        active_pred.pending_spheres.clear()

    # Load next edge if available
    if next_load_edge_idx < len(edges_data):
        edge_flat, edge_coll_flat = csp_rearrange(
            edges_data[next_load_edge_idx],
            edges_coll[next_load_edge_idx],
            groupsize=8,
        )
        active_pred.linklist = edge_flat
        active_pred.linklist_coll = edge_coll_flat
        return next_load_edge_idx + 1

    return next_load_edge_idx


def compile_simulation_stats(
    cycle,
    num_oocds,
    total_query_count,
    cdu_idle_cycles,
    cycle_check,
    oocds,
    total_coll_edge_cycles,
    total_noncoll_edge_cycles,
    qcoll_lengths_at_start,
    qnoncoll_lengths_at_start,
    colldict,
):
    """
    Calculate and compile final simulation statistics.
    """
    # Add remaining fractional queries for OOCDs that are still busy or finished late
    for oocd in oocds:
        if oocd.free_cycle > cycle:
            total_query_count += (cycle_check - oocd.free_cycle + cycle) / cycle_check

    stats = {
        "cdu_idle_cycles": cdu_idle_cycles,
        "cdu_utilization": (
            1.0 - (cdu_idle_cycles / (cycle * num_oocds)) if cycle > 0 else 0.0
        ),
        "total_coll_edge_cycles": total_coll_edge_cycles,
        "total_noncoll_edge_cycles": total_noncoll_edge_cycles,
        "qcoll_lengths_at_start": qcoll_lengths_at_start,
        "qnoncoll_lengths_at_start": qnoncoll_lengths_at_start,
    }

    return total_query_count, colldict, cycle, stats


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
    mode="sphere",
):
    """
    双缓冲架构的并行碰撞检测仿真。

    Parameters:
    - mode: 'standard', 'sphere', or 'link'
    """
    # 1. Initialization and Pre-loading
    oocds, predictions, next_load_edge_idx = init_double_buffer_resources(
        num_oocds,
        num_predictions,
        qcoll_len,
        qnoncoll_len,
        edges_data,
        edges_coll,
    )

    cycle = 0
    total_query_count = 0.0
    cdu_idle_cycles = 0
    total_coll_edge_cycles = 0
    total_noncoll_edge_cycles = 0
    qcoll_lengths_at_start = []
    qnoncoll_lengths_at_start = []

    # 2. Main Edge Loop
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
            mode=mode,
        )

        qcoll_lengths_at_start.append(qcoll_len_start)
        qnoncoll_lengths_at_start.append(qnoncoll_len_start)

        if coll_found > 0:
            total_coll_edge_cycles += edge_cycles
        else:
            total_noncoll_edge_cycles += edge_cycles

        # 3. Buffer Swapping and Reloading
        active_index = edge_idx % num_predictions
        active_pred = predictions[active_index]

        next_load_edge_idx = reload_prediction_buffer(
            active_pred, oocds, edges_data, edges_coll, next_load_edge_idx
        )

    # 4. Final Statistics
    return compile_simulation_stats(
        cycle,
        num_oocds,
        total_query_count,
        cdu_idle_cycles,
        cycle_check,
        oocds,
        total_coll_edge_cycles,
        total_noncoll_edge_cycles,
        qcoll_lengths_at_start,
        qnoncoll_lengths_at_start,
        colldict,
    )
