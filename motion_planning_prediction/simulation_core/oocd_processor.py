"""
OOCD (Out-Of-Order Collision Detector) processing functions.
"""

from .data_structures import OOCDState, OOCDStatePreemptive
from .collision_prediction import update_collision_dict, submit_cht_write


def check_completion(oocd, cycle, query_count, coll_found, colldict, sample_rate):
    """
    检查OOCD是否完成任务，并更新统计信息和碰撞字典。
    """
    if oocd.busy == 1 and oocd.free_cycle <= cycle:
        query_count += 1
        if oocd.result == 0:
            coll_found = 1
        colldict = update_collision_dict(
            colldict, oocd.hash_key, oocd.result, sample_rate
        )
    return query_count, coll_found, colldict


def process_oocd_completion(
    oocds,
    cycle,
    query_count,
    coll_found,
    cht_scheduler,
    copu_id,
    sample_rate,
    num_oocds,
):
    """处理OOCD完成和CHT更新"""
    oocd_cycles_delta = 0
    for oocd_id in range(num_oocds):
        oocd = oocds[oocd_id]
        if oocd.busy == 1:
            oocd_cycles_delta += 1
            if oocd.free_cycle <= cycle:
                query_count += 1
                if oocd.result == 0:  # 碰撞
                    coll_found = True

                # 直接调用submit_cht_write更新CHT
                submit_cht_write(
                    cht_scheduler, copu_id, oocd.hash_key, oocd.result, sample_rate
                )

    return oocd_cycles_delta, query_count, coll_found


def attempt_standard_allocation(
    oocd_id,
    oocds,
    qcoll,
    qnoncoll,
    linklist,
    cycle,
    cycle_check,
    qnoncoll_len,
):
    """
    尝试为OOCD分配任务 (标准策略)。
    适用于 process_oocds 和 dispatch_new_tasks。
    """
    allocated = False
    if len(qcoll) > 0:
        task = qcoll[0]
        task_cycle = task[2] if len(task) > 2 else cycle_check

        oocds[oocd_id] = OOCDState(
            hash_key=task[0],
            result=task[1],
            busy=1,
            free_cycle=cycle + task_cycle,
        )
        qcoll.popleft()
        allocated = True
    elif len(qnoncoll) == qnoncoll_len or (len(linklist) == 0 and len(qnoncoll) > 0):
        task = qnoncoll[0]
        task_cycle = task[2] if len(task) > 2 else cycle_check

        oocds[oocd_id] = OOCDState(
            hash_key=task[0],
            result=task[1],
            busy=1,
            free_cycle=cycle + task_cycle,
        )
        qnoncoll.popleft()
        allocated = True

    return allocated


def attempt_dedicated_allocation(
    oocd_id,
    oocds,
    qcoll,
    qnoncoll,
    linklist,
    cycle,
    cycle_check,
    qnoncoll_len,
    is_dedicated,
):
    """
    尝试为OOCD分配任务（专用策略）。

    Dedicated OOCD: 优先处理qcoll（仅当qnoncoll满时），否则处理qnoncoll（满或linklist空时）
    Non-dedicated OOCD: 优先处理qcoll，其次处理qnoncoll
    """
    allocated = False

    # 尝试分配碰撞任务
    if len(qcoll) > 0:
        oocds[oocd_id] = OOCDState(
            hash_key=qcoll[0][0],
            result=qcoll[0][1],
            busy=1,
            free_cycle=cycle + cycle_check,
        )
        qcoll.popleft()
        allocated = True
    # 尝试分配非碰撞任务
    elif len(qnoncoll) > 0:
        if not is_dedicated or len(qnoncoll) >= qnoncoll_len or len(linklist) == 0:
            oocds[oocd_id] = OOCDState(
                hash_key=qnoncoll[0][0],
                result=qnoncoll[0][1],
                busy=1,
                free_cycle=cycle + cycle_check,
            )
            qnoncoll.popleft()
            allocated = True

    return allocated


def process_oocds(
    oocds,
    qcoll,
    qnoncoll,
    linklist,
    cycle,
    total_query_count,
    coll_found,
    cycle_check,
    colldict,
    sample_rate,
    num_oocds,
    qnoncoll_len,
):
    for oocd_id in range(num_oocds):
        oocd = oocds[oocd_id]

        # Check completion
        total_query_count, coll_found, colldict = check_completion(
            oocd, cycle, total_query_count, coll_found, colldict, sample_rate
        )

        # allocate new tasks
        if oocd.free_cycle <= cycle:
            allocated = attempt_standard_allocation(
                oocd_id,
                oocds,
                qcoll,
                qnoncoll,
                linklist,
                cycle,
                cycle_check,
                qnoncoll_len,
            )

            if not allocated:
                oocds[oocd_id] = OOCDState(hash_key="", result=1, busy=0, free_cycle=0)

    cdu_idle_this_cycle = sum(
        1 for oocd in oocds if oocd.free_cycle <= cycle and not oocd.busy
    )
    return {
        "total_query_count": total_query_count,
        "coll_found": coll_found,
        "cdu_idle_cycles": cdu_idle_this_cycle,
        "colldict": colldict,
        "oocds": oocds,
    }


def process_oocd_states_preemptive(
    oocds,
    qcoll,
    qnoncoll,
    current_time,
    cycle_check,
    query_count,
    coll_found,
    colldict,
    sample_rate,
):
    """
    处理OOCD状态和任务完成。
    """
    for oocd_id in range(len(oocds)):
        oocd = oocds[oocd_id]

        # Check completion
        query_count, coll_found, colldict = check_completion(
            oocd, current_time, query_count, coll_found, colldict, sample_rate
        )

        if oocd.free_cycle <= current_time:
            if len(qcoll) > 0:
                task = qcoll.popleft()
                oocds[oocd_id] = OOCDStatePreemptive(
                    hash_key=task[0],
                    result=task[1],
                    busy=1,
                    free_cycle=current_time + cycle_check,
                    task_type="COLL",
                )
            elif len(qnoncoll) > 0:
                task = qnoncoll.popleft()
                oocds[oocd_id] = OOCDStatePreemptive(
                    hash_key=task[0],
                    result=task[1],
                    busy=1,
                    free_cycle=current_time + cycle_check,
                    task_type="NONCOLL",
                )
            else:
                oocds[oocd_id] = OOCDStatePreemptive(
                    hash_key=0, result=0, busy=0, free_cycle=0, task_type=None
                )
    return {
        "oocds": oocds,
        "total_query_count": query_count,
        "coll_found": coll_found,
        "colldict": colldict,
    }


def handle_preemption(
    oocds, qcoll, qnoncoll, current_time, cycle_check, preemption_count
):
    """
    处理抢占：COLL抢占NONCOLL。
    """
    if len(qcoll) > 0:
        for oocd_id in range(len(oocds)):
            if oocds[oocd_id].busy == 1 and oocds[oocd_id].task_type == "NONCOLL":
                preempted_task = [oocds[oocd_id].hash_key, oocds[oocd_id].result]
                qnoncoll.append(preempted_task)

                task = qcoll.popleft()
                oocds[oocd_id] = OOCDStatePreemptive(
                    hash_key=task[0],
                    result=task[1],
                    busy=1,
                    free_cycle=current_time + cycle_check,
                    task_type="COLL",
                )
                preemption_count += 1
                break
    return oocds, preemption_count


def process_oocd_states_dedicated(
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
):
    """
    处理OOCD状态和任务完成（专用策略）。

    Dedicated OOCD: 等待qnoncoll满或linklist空，否则优先处理qcoll
    Non-dedicated OOCD: 优先处理qcoll，无qcoll时处理qnoncoll
    """
    for oocd_id in range(len(oocds)):
        oocd = oocds[oocd_id]

        # Check completion
        query_count, coll_found, colldict = check_completion(
            oocd, cycle, query_count, coll_found, colldict, sample_rate
        )

        if oocd.free_cycle <= cycle:
            is_dedicated = oocd_id < num_dedicated_oocds

            allocated = attempt_dedicated_allocation(
                oocd_id,
                oocds,
                qcoll,
                qnoncoll,
                linklist,
                cycle,
                cycle_check,
                qnoncoll_len,
                is_dedicated,
            )

            if not allocated:
                oocds[oocd_id] = OOCDState()

    return {
        "oocds": oocds,
        "total_query_count": query_count,
        "coll_found": coll_found,
        "colldict": colldict,
    }


def process_oocds_link(
    oocds,
    qcoll,
    qnoncoll,
    pending_spheres,
    linklist,
    cycle,
    total_query_count,
    coll_found,
    cycle_check,
    colldict,
    sample_rate,
    num_oocds,
    qnoncoll_len,
):
    """Process OOCDs when queues hold link-level tasks.

    qcoll/qnoncoll entries: [hash_key, [sphere_results], [optional cycles]].
    pending_spheres holds per-sphere tasks not yet dispatched.
    """
    for oocd_id in range(num_oocds):
        oocd = oocds[oocd_id]
        total_query_count, coll_found, colldict = check_completion(
            oocd, cycle, total_query_count, coll_found, colldict, sample_rate
        )

    free_ids = [idx for idx, oocd in enumerate(oocds) if oocd.free_cycle <= cycle]

    def assign_task(oocd_idx, task):
        task_cycle = task[2] if len(task) > 2 else cycle_check
        oocds[oocd_idx] = OOCDState(
            hash_key=task[0],
            result=task[1],
            busy=1,
            free_cycle=cycle + task_cycle,
        )

    while free_ids:
        if pending_spheres:
            oocd_idx = free_ids.pop(0)
            task = pending_spheres.popleft()
            assign_task(oocd_idx, task)
            continue

        # Prioritize qcoll; only take from qnoncoll when it's full or linklist is empty
        link_task = None
        if qcoll:
            link_task = qcoll.popleft()
        elif qnoncoll and (len(qnoncoll) == qnoncoll_len or len(linklist) == 0):
            link_task = qnoncoll.popleft()

        if not link_task:
            break
        link_hash = link_task[0]
        sphere_results = link_task[1]
        sphere_cycles = link_task[2] if len(link_task) > 2 else None

        per_sphere_tasks = []
        for idx, sphere_result in enumerate(sphere_results):
            cycle_val = sphere_cycles[idx] if sphere_cycles is not None else cycle_check
            per_sphere_tasks.append([link_hash, sphere_result, cycle_val])

        while free_ids and per_sphere_tasks:
            oocd_idx = free_ids.pop(0)
            task = per_sphere_tasks.pop(0)
            assign_task(oocd_idx, task)

        if per_sphere_tasks:
            pending_spheres.extend(per_sphere_tasks)

    cdu_idle_this_cycle = sum(
        1 for oocd in oocds if oocd.free_cycle <= cycle and not oocd.busy
    )

    return {
        "total_query_count": total_query_count,
        "coll_found": coll_found,
        "cdu_idle_cycles": cdu_idle_this_cycle,
        "colldict": colldict,
        "oocds": oocds,
    }


def dispatch_new_tasks(
    oocds,
    qcoll,
    qnoncoll,
    linklist,
    cycle,
    cycle_check,
    num_oocds,
    qnoncoll_size,
    num_dedicated_oocds=8,
):
    """分派新任务给空闲的OOCD（改进版：支持多OOCD并行分派）

    参数：
        num_dedicated_oocds: 专用OOCD数量。当>0时使用专用策略，否则使用标准策略
    """
    for oocd_id in range(num_oocds):
        oocd = oocds[oocd_id]
        if oocd.free_cycle <= cycle:
            # 根据num_dedicated_oocds选择分配策略
            is_dedicated = oocd_id < num_dedicated_oocds
            allocated = attempt_dedicated_allocation(
                oocd_id,
                oocds,
                qcoll,
                qnoncoll,
                linklist,
                cycle,
                cycle_check,
                qnoncoll_size,
                is_dedicated,
            )

            if not allocated:
                # 简化空闲状态管理：只在busy==1时重置为0
                if oocd.busy == 1:
                    oocds[oocd_id] = OOCDState()


def process_oocds_with_mode(
    mode,
    oocds,
    qcoll,
    qnoncoll,
    linklist,
    cycle,
    total_query_count,
    coll_found,
    cycle_check,
    colldict,
    sample_rate,
    num_oocds,
    qnoncoll_len,
    num_dedicated_oocds=1,
    pending_spheres=None,
):
    """
    Unified OOCD processing function supporting different modes.

    Args:
        mode (str): 'simple', 'batch', or 'hierarchical'.
    """
    if mode == "hierarchical":
        if pending_spheres is None:
            raise ValueError("pending_spheres required for 'hierarchical' mode")
        
        # Link mode typically does not use dedicated OOCDs logic in same way, 
        # or it is handled internally if needed. Here we map to process_oocds_link.
        return process_oocds_link(
            oocds,
            qcoll,
            qnoncoll,
            pending_spheres,
            linklist,
            cycle,
            total_query_count,
            coll_found,
            cycle_check,
            colldict,
            sample_rate,
            num_oocds,
            qnoncoll_len,
        )
    elif mode in ["simple", "batch"]:
        # Standard dedicated processing
        return process_oocd_states_dedicated(
            oocds,
            qcoll,
            qnoncoll,
            cycle,
            cycle_check,
            total_query_count,
            coll_found,
            colldict,
            sample_rate,
            num_dedicated_oocds,
            qnoncoll_len,
            linklist,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
