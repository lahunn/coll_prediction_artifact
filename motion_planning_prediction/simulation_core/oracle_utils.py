"""
Oracle utilities for calculating theoretical minimum cycles.
"""


def calculate_oracle_cycles(edge_coll_data, num_oocds, cycle_check):
    """
    根据num_oocds,计算单个edge的理论最小周期数消耗.

    如果edge 会发生碰撞,那么edge消耗的理论最小周期数消耗是单个cycle_check
    如果edge不会发生碰撞,那么edge消耗的理论最小周期数就是 ceil(edge中总的碰撞检查数/num_oocds) * cycle_check
    """
    has_collision = False
    total_checks = 0

    for pose_coll in edge_coll_data:
        if any(c == 0 for c in pose_coll):
            has_collision = True
            break
        total_checks += len(pose_coll)

    if has_collision:
        return cycle_check
    else:
        num_batches = (total_checks + num_oocds - 1) // num_oocds
        return num_batches * cycle_check


def calculate_oracle_cycles_for_edges(edges_coll_data, num_oocds, cycle_check):
    """
    统计edge数组的理论最小周期数消耗总和。

    Args:
        edges_coll_data: 边碰撞数据列表 [edge][pose][element]
        num_oocds: OOCD/CDU数量
        cycle_check: 单个检查周期时间

    Returns:
        int: 所有边的理论最小周期数消耗总和
    """
    total_theoretical_cycles = 0

    for edge_coll in edges_coll_data:
        edge_cycles = calculate_oracle_cycles(edge_coll, num_oocds, cycle_check)
        total_theoretical_cycles += edge_cycles

    return total_theoretical_cycles
