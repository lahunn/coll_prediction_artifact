import numpy as np
import random
import pickle
import sys


def reutrn_keyy(code):
    """Creates a hash key from a quantized code."""
    bitsize = len(code)
    keyy = ""
    for j in range(0, bitsize):
        if code[j] < 10:
            keyy = keyy + "0"
        keyy = keyy + str(code[j])
    return keyy


def csp_rearrange(edge, edgeyarr, groupsize=8):
    """
    根据分层采样策略（CSP）重排路径上的姿态。
    """
    num_steps = len(edge)

    # --- 1. 分层重排姿态 ---
    # 目标：将姿态从[0,1,2,3...]的顺序重排，优先检查关键姿态（如中点、四分位点等）。
    # 这是一种类似二分查找的策略，希望能更快地发现碰撞。

    # 首先放入路径的最后一个姿态
    rearr = [edge[-1]]
    rearryarr = [edgeyarr[-1]]

    # 分层采样顺序，例如对于8个姿态，顺序为 0, 4, 2, 6, 1, 5, 3, 7
    # 这个循环将该模式应用到整个路径上
    for i in [0, 4, 2, 6, 1, 5, 3, 7]:
        for j in range(i, num_steps - 1, 8):
            rearr.append(edge[j])
            rearryarr.append(edgeyarr[j])

    # --- 2. 展平数据结构 ---
    # 目标：将数据从“姿态列表（每个姿态又是一个连杆列表）”展平为单一的“连杆列表”。
    # [pose[link]] -> [link]
    group = []
    grouparr = []
    # 遍历重排后的每个姿态
    for pose, posecoll in zip(rearr, rearryarr):
        # 遍历该姿态下的每个连杆
        for link, linkcoll in zip(pose, posecoll):
            group.append(link)
            grouparr.append(linkcoll)

    return group, grouparr


def load_data(planner_type, benchid, dimension):
    """
    Loads motion trace data from a pickle file based on planner, benchmark ID, and dimension.
    """
    path_prefix = "../trace_files/motion_traces/"
    if dimension == "2D":
        if planner_type == "BIT":
            filename = f"{path_prefix}logfiles_BIT_2D/coord_motiom_{benchid}.pkl"
        elif planner_type == "GNN":
            filename = f"{path_prefix}logfiles_GNN_2D/coord_motiom_{benchid}.pkl"
        elif planner_type == "MPNET":
            filename = f"{path_prefix}logfiles_MPNET_2D/link_info_1_{benchid}.pkl"
        else:
            return None, None
    elif dimension == "nDOF":
        if planner_type == "BIT":
            filename = f"{path_prefix}logfiles_BIT_link/coord_motiom_{benchid}.pkl"
        elif planner_type == "GNN":
            filename = f"{path_prefix}logfiles_GNN_link/coord_gnn_motiom_{benchid}.pkl"
        elif planner_type == "MPNET":
            filename = f"{path_prefix}logfiles_MPNET_7D/coord_bench_3_{benchid}.pkl"
        else:
            return None, None
    else:
        return None, None

    try:
        with open(filename, "rb") as f:
            if planner_type == "MPNET":
                return pickle.load(f, encoding="latin1")
            else:
                return pickle.load(f)
    except FileNotFoundError:
        return None, None


def load_sphere_data(basename, benchid, data_folder):
    """
    Loads sphere collision data from a pickle file.
    Format: (sphere_link_data, sphere_link_coll_data)
    """
    filename = f"{data_folder}/{basename}_{benchid:04d}_sphere.pkl"
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            # 新格式: (sphere_link_data, sphere_link_coll_data)
            if isinstance(data, tuple) and len(data) == 2:
                return data
            # 兼容旧格式: (qarr, rarr, yarr)
            elif isinstance(data, tuple) and len(data) == 3:
                print(
                    f"Warning: Old format detected in {filename}, converting...",
                    file=sys.stderr,
                )
                qarr_sphere, rarr_sphere, yarr_sphere = data
                return qarr_sphere, yarr_sphere  # 返回坐标和碰撞标签
            else:
                return None, None
    except FileNotFoundError:
        print(f"Warning: Sphere data file not found at {filename}", file=sys.stderr)
        return None, None


def load_obb_data(basename, benchid, data_folder):
    """
    Loads OBB collision data from a pickle file.
    Format: (obb_link_data, obb_link_coll_data)
    """
    filename = f"{data_folder}/{basename}_{benchid:04d}_obb.pkl"
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            # 新格式: (obb_link_data, obb_link_coll_data)
            if isinstance(data, tuple) and len(data) == 2:
                return data
            else:
                return None, None
    except FileNotFoundError:
        print(f"Warning: OBB data file not found at {filename}", file=sys.stderr)
        return None, None


def update_collision_dict(colldict, hash_key, is_free, sample_rate):
    """
    Updates the collision history dictionary.
    """
    if hash_key in colldict:
        if (
            is_free == 1
            and random.random() <= sample_rate
            and colldict[hash_key][is_free] < 15
        ):
            colldict[hash_key][is_free] += 1
        elif colldict[hash_key][is_free] < 15 and is_free == 0:
            colldict[hash_key][is_free] += 1
    else:
        colldict[hash_key] = [0, 0]
        if (
            is_free == 1
            and random.random() <= sample_rate
            and colldict[hash_key][is_free] < 15
        ):
            colldict[hash_key][is_free] += 1
        elif colldict[hash_key][is_free] < 15 and is_free == 0:
            colldict[hash_key][is_free] += 1
    return colldict


def predict_collision(colldict, hash_key, threshold):
    """
    Predicts collision based on the history dictionary.
    """
    if hash_key in colldict:
        if colldict[hash_key][0] > colldict[hash_key][1] * threshold:
            return True  # Predict collision
        else:
            return False  # Predict free
    else:
        return False  # Predict free for unseen configurations


def simulate_parallel_collision_detection(
    linklist,
    linklist_coll,
    colldict,
    threshold,
    sample_rate,
    bins,
    qnoncoll_len=56,
    qcoll_len=8,
    cycle_check=40,
):
    """
    模拟并行的碰撞检测过程，该过程结合了硬件检测器 (OOCD) 和基于历史的碰撞预测。
    """
    # 初始化7个硬件碰撞检测器 (OOCD)，每个检测器的状态为 [key, 真实碰撞结果, 是否繁忙, 完成周期]
    oocds = [[0, 0, 0, 0] for _ in range(7)]
    # qcoll: 预测为会碰撞的配置队列
    # qnoncoll: 预测为不会碰撞的配置队列
    qcoll, qnoncoll = [], []
    cycle = 0  # 仿真周期计数器
    first_two_running = 0
    first_two_checked = 0
    coll_found = 0  # 是否发现真实碰撞的标志
    links_remaining = len(linklist)  # 剩余待处理的配置数量
    everything_free = 0  # 所有任务是否完成的标志
    query_count = 0.0  # 实际执行的硬件查询总数

    # 主循环：直到发现碰撞或所有任务完成
    while not coll_found and not everything_free:
        # --- 步骤1: 处理硬件检测器 (OOCD) 的状态 ---
        for oocd_id in range(len(oocds)):
            oocd = oocds[oocd_id]
            # 如果一个检测器任务已完成 (繁忙状态且到达完成周期)
            if oocd[2] == 1 and oocd[3] <= cycle:
                query_count += 1  # 增加硬件查询计数
                if oocd[1] == 0:  # 假设0代表真实发生碰撞
                    coll_found = 1
                # 根据真实的检测结果，更新碰撞历史表
                colldict = update_collision_dict(
                    colldict, oocd[0], oocd[1], sample_rate
                )

            # 如果一个检测器现在空闲 (到达完成周期)
            if oocd[3] <= cycle:
                # 优先从“预测碰撞”队列 (qcoll) 中取任务
                if len(qcoll) > 0 and first_two_checked < cycle:
                    first_two_running += 1
                    if first_two_running == 1:
                        first_two_checked = cycle + cycle_check
                    # 分配新任务给这个OOCD
                    oocds[oocd_id] = [qcoll[0][0], qcoll[0][1], 1, cycle + cycle_check]
                    del qcoll[0]
                # 如果qcoll为空，则从“预测不碰撞”队列 (qnoncoll) 中取任务
                elif (
                    len(qnoncoll) == qnoncoll_len
                    or (links_remaining == 0 and len(qnoncoll) > 0)
                    and first_two_checked < cycle
                ):
                    oocds[oocd_id] = [
                        qnoncoll[0][0],
                        qnoncoll[0][1],
                        1,
                        cycle + cycle_check,
                    ]
                    del qnoncoll[0]
                else:
                    # 如果两个队列都没有任务，则OOCD变为空闲状态
                    oocds[oocd_id] = [0, 0, 0, 0]

        # --- 步骤2: 预测下一个配置并放入相应队列 ---
        if len(linklist) > 0:
            link, linkcoll = linklist[0], linklist_coll[0]

            # 将配置数据“量化”以生成用于查询历史表的键 (key)
            code_quant = np.digitize(link, bins, right=True)
            keyy = reutrn_keyy(code_quant)

            # 使用历史表进行碰撞预测
            is_collision_predicted = predict_collision(colldict, keyy, threshold)

            # 根据预测结果，将配置放入不同的队列
            if is_collision_predicted:
                if len(qcoll) < qcoll_len:  # 如果队列未满
                    qcoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]
            else:
                if len(qnoncoll) < qnoncoll_len:  # 如果队列未满
                    qnoncoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]

        # --- 步骤3: 检查仿真是否结束 ---
        links_remaining = len(linklist)
        # 如果所有输入配置都已处理，所有检测器都空闲，且所有队列都为空
        if (
            links_remaining == 0
            and not any(oocd[3] > cycle for oocd in oocds)
            and not qnoncoll
            and not qcoll
        ):
            everything_free = 1  # 设置结束标志

        cycle += 1  # 时间周期前进

    # --- 步骤4: 计算仿真结束时仍在运行的任务 ---
    # 对于未完成的检查，按其已执行的比例计入查询总数
    for oocd in oocds:
        if oocd[3] > cycle:
            query_count += (cycle_check - oocd[3] + cycle) / cycle_check

    # 返回总查询数、更新后的碰撞历史表和是否找到碰撞的标志
    return query_count, colldict, coll_found
