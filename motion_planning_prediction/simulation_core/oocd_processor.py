"""
OOCD (Out-Of-Order Collision Detector) processing functions.
"""

from .data_structures import OOCDState, OOCDStatePreemptive
from .collision_prediction import update_collision_dict


def process_oocds(
    oocds,
    qcoll,
    qnoncoll,
    linklist,
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
):
    for oocd_id in range(num_oocds):
        oocd = oocds[oocd_id]
        if oocd.busy == 1 and oocd.free_cycle <= cycle:
            total_query_count += 1
            if oocd.result == 0:
                coll_found = 1
            colldict = update_collision_dict(
                colldict, oocd.hash_key, oocd.result, sample_rate
            )
        if oocd.free_cycle <= cycle:
            if len(qcoll) > 0 and first_two_checked < cycle:
                first_two_running += 1
                if first_two_running == 1:
                    first_two_checked = cycle + cycle_check
                task_cycle = qcoll[0][2] if len(qcoll[0]) > 2 else cycle_check
                oocds[oocd_id] = OOCDState(
                    hash_key=qcoll[0][0],
                    result=qcoll[0][1],
                    busy=1,
                    free_cycle=cycle + task_cycle,
                )
                qcoll.popleft()
            elif (
                len(qnoncoll) == qnoncoll_len
                or (len(linklist) == 0 and len(qnoncoll) > 0)
            ) and first_two_checked < cycle:
                task_cycle = qnoncoll[0][2] if len(qnoncoll[0]) > 2 else cycle_check
                oocds[oocd_id] = OOCDState(
                    hash_key=qnoncoll[0][0],
                    result=qnoncoll[0][1],
                    busy=1,
                    free_cycle=cycle + task_cycle,
                )
                qnoncoll.popleft()
            else:
                oocds[oocd_id] = OOCDState(hash_key=0, result=0, busy=0, free_cycle=0)
    cdu_idle_this_cycle = sum(
        1 for oocd in oocds if oocd.free_cycle <= cycle and not oocd.busy
    )
    return (
        total_query_count,
        coll_found,
        first_two_running,
        first_two_checked,
        cdu_idle_this_cycle,
    )


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
        if oocd.busy == 1 and oocd.free_cycle <= current_time:
            query_count += 1
            if oocd.result == 0:
                coll_found = True
            colldict = update_collision_dict(
                colldict, oocd.hash_key, oocd.result, sample_rate
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
    return oocds, query_count, coll_found, colldict


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
    first_two_running,
    first_two_checked,
):
    """
    处理OOCD状态和任务完成（专用策略）。
    """
    for oocd_id in range(len(oocds)):
        oocd = oocds[oocd_id]
        if oocd.busy == 1 and oocd.free_cycle <= cycle:
            query_count += 1
            if oocd.result == 0:
                coll_found = 1
            colldict = update_collision_dict(
                colldict, oocd.hash_key, oocd.result, sample_rate
            )

        if oocd.free_cycle <= cycle:
            is_dedicated = oocd_id < num_dedicated_oocds
            task_assigned = False

            if is_dedicated:
                if len(qnoncoll) >= qnoncoll_len:
                    oocds[oocd_id] = OOCDState(
                        hash_key=qnoncoll[0][0],
                        result=qnoncoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                    )
                    qnoncoll.popleft()
                    task_assigned = True
                elif len(qcoll) > 0 and first_two_checked < cycle:
                    first_two_running += 1
                    if first_two_running == 1:
                        first_two_checked = cycle + cycle_check
                    oocds[oocd_id] = OOCDState(
                        hash_key=qcoll[0][0],
                        result=qcoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                    )
                    qcoll.popleft()
                    task_assigned = True
                elif len(linklist) == 0 and len(qnoncoll) > 0:
                    oocds[oocd_id] = OOCDState(
                        hash_key=qnoncoll[0][0],
                        result=qnoncoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                    )
                    qnoncoll.popleft()
                    task_assigned = True
            else:
                if len(qcoll) > 0 and first_two_checked < cycle:
                    first_two_running += 1
                    if first_two_running == 1:
                        first_two_checked = cycle + cycle_check
                    oocds[oocd_id] = OOCDState(
                        hash_key=qcoll[0][0],
                        result=qcoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                    )
                    qcoll.popleft()
                    task_assigned = True
                elif len(qnoncoll) > 0:
                    oocds[oocd_id] = OOCDState(
                        hash_key=qnoncoll[0][0],
                        result=qnoncoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                    )
                    qnoncoll.popleft()
                    task_assigned = True

            if not task_assigned:
                oocds[oocd_id] = OOCDState(hash_key=0, result=0, busy=0, free_cycle=0)
    return (
        oocds,
        query_count,
        coll_found,
        colldict,
        first_two_running,
        first_two_checked,
    )
