import numpy as np
import random
import pickle
import sys
from collections import deque, namedtuple

# Constants
NUM_OOCDS = 7
MAX_COLLISION_COUNT = 15
DEFAULT_QNONCOLL_LEN = 56
DEFAULT_QCOLL_LEN = 8
DEFAULT_CYCLE_CHECK = 40

# Named tuple for OOCD state
OOCDState = namedtuple("OOCDState", ["hash_key", "result", "busy", "free_cycle"])


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

    # --- 1. 分层重排姿态（包括周期数据）---
    rearr = [edge[-1]]
    rearryarr = [edgeyarr[-1]]
    rearr_cycles = [edge_cycles[-1]]

    # 分层采样顺序
    for i in [0, 4, 2, 6, 1, 5, 3, 7]:
        for j in range(i, num_steps - 1, 8):
            rearr.append(edge[j])
            rearryarr.append(edgeyarr[j])
            rearr_cycles.append(edge_cycles[j])

    # --- 2. 展平数据结构 ---
    group = []
    grouparr = []
    group_cycles = []

    # 遍历重排后的每个姿态
    for pose, posecoll, pose_cycles in zip(rearr, rearryarr, rearr_cycles):
        # 遍历该姿态下的每个球体
        for sphere, sphere_coll, sphere_cycle in zip(pose, posecoll, pose_cycles):
            group.append(sphere)
            grouparr.append(sphere_coll)
            group_cycles.append(sphere_cycle)

    return group, grouparr, group_cycles


def load_motion_trace_data(planner_type, benchid, dimension):
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


def load_data(basename, benchid, data_folder, collision_model_type="link"):
    """
    Loads collision data from a pickle file.
    
    Args:
        basename: Base name of the dataset (e.g., "iiwa_7")
        benchid: Benchmark number
        data_folder: Path to the data folder
        collision_model_type: Type of collision model ("link" or "sphere", default="link")
    
    Returns:
        (collision_data, collision_flags) tuple or (None, None)
    
    File naming convention:
        {basename}_{benchid:04d}_{collision_model_type}.pkl
    """
    filename = f"{data_folder}/{basename}_{benchid:04d}_{collision_model_type}.pkl"
    
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) >= 2:
                return data[0], data[1]
    except FileNotFoundError:
        pass
    
    print(f"Warning: Collision data file not found at {filename}", file=sys.stderr)
    return None, None


def load_data_with_cycles(basename, benchid, data_folder, collision_model_type="link"):
    """
    Loads collision data with cycles from a pickle file.
    
    Args:
        basename: Base name of the dataset (e.g., "iiwa_7")
        benchid: Benchmark number
        data_folder: Path to the data folder
        collision_model_type: Type of collision model ("link" or "sphere", default="link")
    
    Returns:
        (collision_data, collision_flags, cycles) tuple or (None, None, None)
    
    File naming convention:
        - Sphere model: {basename}_{benchid:04d}_sphere_geometric_cycles.pkl
        - Link model: {basename}_{benchid:04d}_{collision_model_type}_cycles.pkl
    """
    if collision_model_type == "sphere":
        filename = f"{data_folder}/{basename}_{benchid:04d}_sphere_geometric_cycles.pkl"
    else:
        filename = f"{data_folder}/{basename}_{benchid:04d}_{collision_model_type}_cycles.pkl"
    
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 3:
                return data[0], data[1], data[2]
    except FileNotFoundError:
        pass
    
    print(
        f"Warning: Collision data with cycles file not found at {filename}",
        file=sys.stderr,
    )
    return None, None, None


def update_collision_dict(colldict, hash_key, is_free, sample_rate):
    """
    Updates the collision history dictionary.
    """
    if hash_key in colldict:
        if (
            is_free == 1
            and random.random() <= sample_rate
            and colldict[hash_key][is_free] < MAX_COLLISION_COUNT
        ):
            colldict[hash_key][is_free] += 1
        elif colldict[hash_key][is_free] < MAX_COLLISION_COUNT and is_free == 0:
            colldict[hash_key][is_free] += 1
    else:
        colldict[hash_key] = [0, 0]
        if (
            is_free == 1
            and random.random() <= sample_rate
            and colldict[hash_key][is_free] < MAX_COLLISION_COUNT
        ):
            colldict[hash_key][is_free] += 1
        elif colldict[hash_key][is_free] < MAX_COLLISION_COUNT and is_free == 0:
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


def calculate_accuracy(predictions, actuals):
    """
    计算预测准确率

    Args:
        predictions: 预测结果列表 (True/False)
        actuals: 实际结果列表 (0/1, 0表示碰撞)

    Returns:
        float: 准确率 (0-1)
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return 0.0

    correct = 0
    for pred, act in zip(predictions, actuals):
        # pred: True表示预测碰撞, False表示预测无碰撞
        # act: 0表示实际碰撞, 1表示实际无碰撞
        pred_collision = pred
        actual_collision = act == 0
        if pred_collision == actual_collision:
            correct += 1

    return correct / len(predictions)


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
    # 初始化硬件碰撞检测器 (OOCD)
    oocds = [
        OOCDState(hash_key=0, result=0, busy=0, free_cycle=0) for _ in range(num_oocds)
    ]
    # 使用deque替代list，提高队列操作效率
    qcoll = deque(maxlen=qcoll_len)  # 预测碰撞任务队列
    qnoncoll = deque(maxlen=qnoncoll_len)  # 预测无碰撞任务队列
    cycle = 0  # 仿真周期计数器
    first_two_running = 0  # 当前正在运行的前两个任务计数
    first_two_checked = 0  # 前两个任务开始处理的周期标记
    coll_found = 0  # 是否发现真实碰撞的标志
    links_remaining = len(linklist)  # 剩余待处理的配置数量
    everything_free = 0  # 所有任务是否完成的标志
    query_count = 0.0  # 实际执行的硬件查询总数

    # 主循环：直到发现碰撞或所有任务完成
    while not coll_found and not everything_free:
        # --- 步骤1: 处理硬件检测器 (OOCD) 的状态 ---
        dequeued_this_cycle = False  # 每个周期最多只出队一次
        for oocd_id in range(len(oocds)):
            oocd = oocds[oocd_id]
            # 如果一个检测器任务已完成 (繁忙状态且到达完成周期)
            if oocd.busy == 1 and oocd.free_cycle <= cycle:
                query_count += 1  # 增加硬件查询计数
                if oocd.result == 0:  # 假设0代表真实发生碰撞
                    coll_found = 1
                # 根据真实的检测结果，更新碰撞历史表
                colldict = update_collision_dict(
                    colldict, oocd.hash_key, oocd.result, sample_rate
                )

            # 如果一个检测器现在空闲 (到达完成周期) 并且本周期还未分配过任务
            if oocd.free_cycle <= cycle and not dequeued_this_cycle:
                # 优先从“预测碰撞”队列 (qcoll) 中取任务
                if len(qcoll) > 0 and first_two_checked < cycle:
                    first_two_running += 1
                    if first_two_running == 1:
                        first_two_checked = cycle + cycle_check
                    # 分配新任务给这个OOCD
                    oocds[oocd_id] = OOCDState(
                        hash_key=qcoll[0][0],
                        result=qcoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                    )
                    qcoll.popleft()
                    dequeued_this_cycle = True  # 标记本周期已出队
                # 如果qcoll为空，则从“预测不碰撞”队列 (qnoncoll) 中取任务
                elif (
                    len(qnoncoll) == qnoncoll_len
                    or (links_remaining == 0 and len(qnoncoll) > 0)
                    and first_two_checked < cycle
                ):
                    oocds[oocd_id] = OOCDState(
                        hash_key=qnoncoll[0][0],
                        result=qnoncoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                    )
                    qnoncoll.popleft()
                    dequeued_this_cycle = True  # 标记本周期已出队
                else:
                    # 如果两个队列都没有任务，则OOCD变为空闲状态
                    oocds[oocd_id] = OOCDState(
                        hash_key=0, result=0, busy=0, free_cycle=0
                    )

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

    # 返回总查询数、更新后的碰撞历史表和是否找到碰撞的标志
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
    num_oocds=NUM_OOCDS,
):
    """
    使用真实周期数的并行碰撞检测仿真。

    与 simulate_parallel_collision_detection 的区别：
    - 使用 linklist_cycles 中的真实周期数，而不是固定的 cycle_check
    - 每个 OOCD 根据实际任务的周期数来确定完成时间

    参数:
        linklist: 配置列表（已展平并重排）
        linklist_coll: 碰撞标志列表（已展平并重排）
        linklist_cycles: 周期数列表（已展平并重排，与 linklist 对应）
        其他参数与原函数相同
    """
    # 初始化硬件碰撞检测器 (OOCD)
    oocds = [
        OOCDState(hash_key=0, result=0, busy=0, free_cycle=0) for _ in range(num_oocds)
    ]
    # 使用deque替代list，提高队列操作效率
    qcoll = deque(maxlen=qcoll_len)  # 预测碰撞任务队列
    qnoncoll = deque(maxlen=qnoncoll_len)  # 预测无碰撞任务队列
    cycle = 0  # 仿真周期计数器
    first_two_running = 0  # 当前正在运行的前两个任务计数
    first_two_checked = 0  # 前两个任务开始处理的周期标记
    coll_found = 0  # 是否发现真实碰撞的标志
    links_remaining = len(linklist)  # 剩余待处理的配置数量
    everything_free = 0  # 所有任务是否完成的标志
    query_count = 0.0  # 实际执行的硬件查询总数

    # 主循环：直到发现碰撞或所有任务完成
    while not coll_found and not everything_free:
        # --- 步骤1: 处理硬件检测器 (OOCD) 的状态 ---
        dequeued_this_cycle = False  # 每个周期最多只出队一次
        for oocd_id in range(len(oocds)):
            oocd = oocds[oocd_id]
            # 如果一个检测器任务已完成 (繁忙状态且到达完成周期)
            if oocd.busy == 1 and oocd.free_cycle <= cycle:
                query_count += 1  # 增加硬件查询计数
                if oocd.result == 0:  # 假设0代表真实发生碰撞
                    coll_found = 1
                # 根据真实的检测结果，更新碰撞历史表
                colldict = update_collision_dict(
                    colldict, oocd.hash_key, oocd.result, sample_rate
                )

            # 如果一个检测器现在空闲 (到达完成周期) 并且本周期还未分配过任务
            if oocd.free_cycle <= cycle and not dequeued_this_cycle:
                # 优先从"预测碰撞"队列 (qcoll) 中取任务
                if len(qcoll) > 0 and first_two_checked < cycle:
                    first_two_running += 1
                    if first_two_running == 1:
                        first_two_checked = cycle + qcoll[0][2]
                    # 分配新任务给这个OOCD，使用真实周期数
                    oocds[oocd_id] = OOCDState(
                        hash_key=qcoll[0][0],
                        result=qcoll[0][1],
                        busy=1,
                        free_cycle=cycle + qcoll[0][2],  # 使用真实周期数
                    )
                    qcoll.popleft()
                    dequeued_this_cycle = True  # 标记本周期已出队
                # 如果qcoll为空，则从"预测不碰撞"队列 (qnoncoll) 中取任务
                elif (
                    len(qnoncoll) == qnoncoll_len
                    or (links_remaining == 0 and len(qnoncoll) > 0)
                    and first_two_checked < cycle
                ):
                    oocds[oocd_id] = OOCDState(
                        hash_key=qnoncoll[0][0],
                        result=qnoncoll[0][1],
                        busy=1,
                        free_cycle=cycle + qnoncoll[0][2],  # 使用真实周期数
                    )
                    qnoncoll.popleft()
                    dequeued_this_cycle = True  # 标记本周期已出队
                else:
                    # 如果两个队列都没有任务，则OOCD变为空闲状态
                    oocds[oocd_id] = OOCDState(
                        hash_key=0, result=0, busy=0, free_cycle=0
                    )

        # --- 步骤2: 预测下一个配置并放入相应队列 ---
        if len(linklist) > 0:
            link, linkcoll, link_cycle = (
                linklist[0],
                linklist_coll[0],
                linklist_cycles[0],
            )

            # 将配置数据"量化"以生成用于查询历史表的键 (key)
            code_quant = np.digitize(link, bins, right=True)
            keyy = reutrn_keyy(code_quant)

            # 使用历史表进行碰撞预测
            is_collision_predicted = predict_collision(colldict, keyy, threshold)

            # 根据预测结果，将配置放入不同的队列（包含周期数）
            if is_collision_predicted:
                if len(qcoll) < qcoll_len:  # 如果队列未满
                    qcoll.append([keyy, linkcoll, link_cycle])
                    del linklist[0]
                    del linklist_coll[0]
                    del linklist_cycles[0]
            else:
                if len(qnoncoll) < qnoncoll_len:  # 如果队列未满
                    qnoncoll.append([keyy, linkcoll, link_cycle])
                    del linklist[0]
                    del linklist_coll[0]
                    del linklist_cycles[0]

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
            # 注意：这里无法精确知道该OOCD任务的原始周期数
            # 简化处理：假设已完成的比例为 (已执行周期 / 总周期)
            # 但由于我们不知道原始周期数，这里保持简单，计为部分查询
            query_count += 0.5  # 简化处理

    # 返回总查询数、更新后的碰撞历史表和是否找到碰撞的标志
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

    返回值:
        query_count: 总查询数
        colldict: 更新后的碰撞历史表
        coll_found: 是否找到碰撞
        cycle: 总周期数
        predictions: 预测结果列表
        actuals: 实际结果列表
    """
    # 初始化硬件碰撞检测器 (OOCD)
    oocds = [
        OOCDState(hash_key=0, result=0, busy=0, free_cycle=0) for _ in range(num_oocds)
    ]
    # 使用deque替代list，提高队列操作效率
    qcoll = deque(maxlen=qcoll_len)  # 预测碰撞任务队列
    qnoncoll = deque(maxlen=qnoncoll_len)  # 预测无碰撞任务队列
    cycle = 0  # 仿真周期计数器
    first_two_running = 0  # 当前正在运行的前两个任务计数
    first_two_checked = 0  # 前两个任务开始处理的周期标记
    coll_found = 0  # 是否发现真实碰撞的标志
    links_remaining = len(linklist)  # 剩余待处理的配置数量
    everything_free = 0  # 所有任务是否完成的标志
    query_count = 0.0  # 实际执行的硬件查询总数

    # 准确率跟踪变量
    predictions = []  # 预测结果
    actuals = []  # 实际结果

    # 主循环：直到发现碰撞或所有任务完成
    while not coll_found and not everything_free:
        # --- 步骤1: 处理硬件检测器 (OOCD) 的状态 ---
        dequeued_this_cycle = False  # 每个周期最多只出队一次
        for oocd_id in range(len(oocds)):
            oocd = oocds[oocd_id]
            # 如果一个检测器任务已完成 (繁忙状态且到达完成周期)
            if oocd.busy == 1 and oocd.free_cycle <= cycle:
                query_count += 1  # 增加硬件查询计数
                if oocd.result == 0:  # 假设0代表真实发生碰撞
                    coll_found = 1
                # 根据真实的检测结果，更新碰撞历史表
                colldict = update_collision_dict(
                    colldict, oocd.hash_key, oocd.result, sample_rate
                )

            # 如果一个检测器现在空闲 (到达完成周期) 并且本周期还未分配过任务
            if oocd.free_cycle <= cycle and not dequeued_this_cycle:
                # 优先从"预测碰撞"队列 (qcoll) 中取任务
                if len(qcoll) > 0 and first_two_checked < cycle:
                    first_two_running += 1
                    if first_two_running == 1:
                        first_two_checked = cycle + cycle_check
                    # 分配新任务给这个OOCD
                    oocds[oocd_id] = OOCDState(
                        hash_key=qcoll[0][0],
                        result=qcoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                    )
                    qcoll.popleft()
                    dequeued_this_cycle = True  # 标记本周期已出队
                # 如果qcoll为空，则从"预测不碰撞"队列 (qnoncoll) 中取任务
                elif (
                    len(qnoncoll) == qnoncoll_len
                    or (links_remaining == 0 and len(qnoncoll) > 0)
                    and first_two_checked < cycle
                ):
                    oocds[oocd_id] = OOCDState(
                        hash_key=qnoncoll[0][0],
                        result=qnoncoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                    )
                    qnoncoll.popleft()
                    dequeued_this_cycle = True  # 标记本周期已出队
                else:
                    # 如果两个队列都没有任务，则OOCD变为空闲状态
                    oocds[oocd_id] = OOCDState(
                        hash_key=0, result=0, busy=0, free_cycle=0
                    )

        # --- 步骤2: 预测下一个配置并放入相应队列 ---
        if len(linklist) > 0:
            link, linkcoll = linklist[0], linklist_coll[0]

            # 将配置数据"量化"以生成用于查询历史表的键 (key)
            code_quant = np.digitize(link, bins, right=True)
            keyy = reutrn_keyy(code_quant)

            # 使用历史表进行碰撞预测
            is_collision_predicted = predict_collision(colldict, keyy, threshold)

            # 记录预测结果用于准确率计算
            predictions.append(is_collision_predicted)
            actuals.append(linkcoll)

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
            query_count += 0.5  # 简化处理

    # 返回总查询数、更新后的碰撞历史表和是否找到碰撞的标志，以及预测跟踪数据
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

    步骤：
    1. 处理OOCD状态和任务完成
    2. 处理新任务到达：预测并放入队列
    3. 处理抢占：COLL抢占NONCOLL
    4. 检查仿真是否结束
    5. 时间周期前进
    6. 循环直到结束
    7. 计算未完成任务

    返回值:
        query_count: 总查询数
        colldict: 更新后的碰撞历史表
        coll_found: 是否找到碰撞
        current_time: 总仿真时间
        preemption_count: 抢占事件发生次数
    """
    # 扩展OOCD状态以包含任务类型
    OOCDStatePreemptive = namedtuple(
        "OOCDStatePreemptive", ["hash_key", "result", "busy", "free_cycle", "task_type"]
    )

    # 初始化硬件碰撞检测器 (OOCD)
    oocds = [
        OOCDStatePreemptive(hash_key=0, result=0, busy=0, free_cycle=0, task_type=None)
        for _ in range(num_oocds)
    ]
    # 使用deque替代list，提高队列操作效率
    qcoll = deque(maxlen=qcoll_len)  # 预测碰撞任务队列
    qnoncoll = deque(maxlen=qnoncoll_len)  # 预测无碰撞任务队列
    current_time = 0  # 当前仿真时间
    coll_found = False  # 是否发现真实碰撞的标志
    everything_free = False  # 所有任务是否完成的标志
    query_count = 0.0  # 实际执行的硬件查询总数
    preemption_count = 0  # 抢占事件计数器

    # 主循环：直到发现碰撞或所有任务完成
    while not coll_found and not everything_free:
        # --- 步骤1: 处理OOCD状态和任务完成 ---
        dequeued_this_cycle = False  # 每个周期最多只出队一次
        for oocd_id in range(num_oocds):
            oocd = oocds[oocd_id]
            # 如果一个检测器任务已完成 (繁忙状态且到达完成周期)
            if oocd.busy == 1 and oocd.free_cycle <= current_time:
                query_count += 1  # 增加硬件查询计数
                if oocd.result == 0:  # 假设0代表真实发生碰撞
                    coll_found = True
                # 根据真实的检测结果，更新碰撞历史表
                colldict = update_collision_dict(
                    colldict, oocd.hash_key, oocd.result, sample_rate
                )

            # 如果一个检测器现在空闲 (到达完成周期) 并且本周期还未分配过任务
            if oocd.free_cycle <= current_time and not dequeued_this_cycle:
                # 优先从"预测碰撞"队列 (qcoll) 中取任务
                if len(qcoll) > 0:
                    # 分配COLL任务
                    task = qcoll.popleft()
                    oocds[oocd_id] = OOCDStatePreemptive(
                        hash_key=task[0],
                        result=task[1],
                        busy=1,
                        free_cycle=current_time + cycle_check,
                        task_type="COLL",
                    )
                    dequeued_this_cycle = True  # 标记本周期已出队
                # 如果qcoll为空，则从"预测不碰撞"队列 (qnoncoll) 中取任务
                elif len(qnoncoll) > 0:
                    # 分配NONCOLL任务
                    task = qnoncoll.popleft()
                    oocds[oocd_id] = OOCDStatePreemptive(
                        hash_key=task[0],
                        result=task[1],
                        busy=1,
                        free_cycle=current_time + cycle_check,
                        task_type="NONCOLL",
                    )
                    dequeued_this_cycle = True  # 标记本周期已出队
                else:
                    # 如果两个队列都没有任务，则OOCD变为空闲状态
                    oocds[oocd_id] = OOCDStatePreemptive(
                        hash_key=0, result=0, busy=0, free_cycle=0, task_type=None
                    )

        # --- 步骤2: 处理新任务到达 ---
        if len(linklist) > 0:
            link, linkcoll = linklist[0], linklist_coll[0]

            # 将配置数据"量化"以生成用于查询历史表的键 (key)
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

        # --- 步骤3: 处理抢占 ---
        if len(qcoll) > 0:
            # 查找一个正在运行NONCOLL任务的OOCD进行抢占
            for oocd_id in range(num_oocds):
                if oocds[oocd_id].busy == 1 and oocds[oocd_id].task_type == "NONCOLL":
                    # 抢占：将NONCOLL任务放回队列
                    preempted_task = [oocds[oocd_id].hash_key, oocds[oocd_id].result]
                    qnoncoll.append(preempted_task)

                    # 分配COLL任务给此OOCD
                    task = qcoll.popleft()
                    oocds[oocd_id] = OOCDStatePreemptive(
                        hash_key=task[0],
                        result=task[1],
                        busy=1,
                        free_cycle=current_time + cycle_check,
                        task_type="COLL",
                    )
                    preemption_count += 1  # 增加抢占计数
                    break  # 只抢占一个OOCD

        # --- 步骤4: 检查仿真是否结束 ---
        links_remaining = len(linklist)
        # 如果所有输入配置都已处理，所有检测器都空闲，且所有队列都为空
        if (
            links_remaining == 0
            and not any(oocd.free_cycle > current_time for oocd in oocds)
            and not qnoncoll
            and not qcoll
        ):
            everything_free = True  # 设置结束标志

        current_time += 1  # 时间周期前进

    # --- 步骤7: 计算仿真结束时仍在运行的任务 ---
    # 对于未完成的检查，按其已执行的比例计入查询总数
    for oocd in oocds:
        if oocd.free_cycle > current_time and oocd.busy == 1:
            executed_cycles = current_time - (oocd.free_cycle - cycle_check)
            query_count += executed_cycles / cycle_check

    # 返回总查询数、更新后的碰撞历史表和是否找到碰撞的标志
    return query_count, colldict, coll_found, current_time, preemption_count
