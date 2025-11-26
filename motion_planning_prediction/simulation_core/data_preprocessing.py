"""
Data preprocessing utilities for CSP rearrangement and COPU allocation.
"""


def recursive_binary_reorder(n):
    """
    将 [0,1,2,...,n-1] 按递归二分方式重排（使用位反转）。
    """
    if n <= 1:
        return list(range(n))

    num_bits = 0
    temp = n - 1
    while temp > 0:
        num_bits += 1
        temp >>= 1

    reorder = []
    for i in range(n):
        reversed_i = 0
        for bit in range(num_bits):
            reversed_i = (reversed_i << 1) | ((i >> bit) & 1)
        reorder.append(reversed_i)

    return reorder


def generate_recursive_reorder(num_poses, step_size=8):
    """
    生成递归式重排顺序（保持固定步长，只对组序列进行递归二分重排）。
    """
    group_count = min(step_size, (num_poses + step_size - 1) // step_size)
    group_order = recursive_binary_reorder(group_count)
    
    reorder = []
    for group_id in group_order:
        pose_idx = group_id
        while pose_idx < num_poses:
            reorder.append(pose_idx)
            pose_idx += step_size

    return reorder


def csp_rearrange(edge, edgeyarr, groupsize=8):
    """
    根据分层采样策略（CSP）重排路径上的姿态。
    """
    num_steps = len(edge)

    rearr = [edge[-1]]
    rearryarr = [edgeyarr[-1]]

    reorder_indices = recursive_binary_reorder(groupsize)
    for i in reorder_indices:
        for j in range(i, num_steps - 1, groupsize):
            rearr.append(edge[j])
            rearryarr.append(edgeyarr[j])

    group = []
    grouparr = []
    for pose, posecoll in zip(rearr, rearryarr):
        for link, linkcoll in zip(pose, posecoll):
            group.append(link)
            grouparr.append(linkcoll)

    return group, grouparr


def csp_rearrange_with_cycles(edge, edgeyarr, edge_cycles, groupsize=8):
    """
    根据分层采样策略（CSP）重排路径上的姿态，同时重排周期数据。

    Args:
        edge: 边数据 [pose][sphere]
        edgeyarr: 碰撞标记 [pose][sphere]
        edge_cycles: 周期数据 [pose][sphere]
        groupsize: 分组大小（默认8）

    Returns:
        group: 展平后的边数据
        grouparr: 展平后的碰撞标记
        group_cycles: 展平后的周期数据
    """
    num_steps = len(edge)

    rearr = [edge[-1]]
    rearryarr = [edgeyarr[-1]]
    rearr_cycles = [edge_cycles[-1]]

    reorder_indices = recursive_binary_reorder(groupsize)
    for i in reorder_indices:
        for j in range(i, num_steps - 1, groupsize):
            rearr.append(edge[j])
            rearryarr.append(edgeyarr[j])
            rearr_cycles.append(edge_cycles[j])

    group = []
    grouparr = []
    group_cycles = []

    for pose, posecoll, pose_cycles in zip(rearr, rearryarr, rearr_cycles):
        for sphere, sphere_coll, sphere_cycle in zip(pose, posecoll, pose_cycles):
            group.append(sphere)
            grouparr.append(sphere_coll)
            group_cycles.append(sphere_cycle)

    return group, grouparr, group_cycles


def allocate_edge_data_to_copus(
    edge_coords,
    edge_flags,
    edge_cycles,
    num_copus,
    use_recursive_reorder=True,
    step_size=8,
):
    """
    将单条edge的pose数据按轮转方式分配给所有COPU。
    """
    num_poses = len(edge_coords)

    if use_recursive_reorder:
        reorder = generate_recursive_reorder(num_poses, step_size)
    else:
        reorder = list(range(num_poses))

    copus_coords = [[] for _ in range(num_copus)]
    copus_flags = [[] for _ in range(num_copus)]
    copus_cycles = [[] for _ in range(num_copus)]

    for reordered_idx, original_pose_idx in enumerate(reorder):
        copu_id = reordered_idx % num_copus
        pose_coords = edge_coords[original_pose_idx]
        pose_flags = edge_flags[original_pose_idx]

        copus_coords[copu_id].extend(pose_coords)
        copus_flags[copu_id].extend(pose_flags)

        if edge_cycles is not None:
            pose_cycles = edge_cycles[original_pose_idx]
            copus_cycles[copu_id].extend(pose_cycles)
        else:
            copus_cycles[copu_id].extend([40 for _ in range(len(pose_coords))])

    return copus_coords, copus_flags, copus_cycles
