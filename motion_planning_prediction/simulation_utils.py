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


def calculate_bins(quant_min, quant_max, quant_bits):
    """
    计算量化分箱的边界

    Args:
        quant_min: 坐标的最小值
        quant_max: 坐标的最大值
        quant_bits: 量化位数（每个维度的比特数）

    Returns:
        bins: 分箱边界数组
    """
    num_bins = 2**quant_bits
    bins = np.linspace(quant_min, quant_max, num_bins)
    return bins


def return_keyy(code, quant_bits):
    """
    将量化编码转换为二进制字符串

    Args:
        code: 量化编码数组，例如 [3, 5, 2]（每个元素是量化值）
        quant_bits: 每个量化值的比特宽度（例如4表示每个值用4位表示）

    Returns:
        keyy: 二进制编码字符串，例如 "001101010010"（每个元素转为二进制后拼接）

    说明：
        假定 quant_bits=4，则每个量化值用4个比特表示
        例如：code=[3, 5, 2], quant_bits=4 -> "0011" + "0101" + "0010" = "001101010010"
        最终返回的二进制字符串长度为 len(code) * quant_bits
    """
    bitsize = len(code)
    keyy = ""

    for j in range(bitsize):
        # 将每个量化值转为二进制，用零补齐到quant_bits位
        binary_str = format(int(code[j]), f"0{quant_bits}b")
        keyy = keyy + binary_str

    return keyy


def compute_hash_keyy(link_coords, bins):
    """
    Args:
        link_coords: 单个link的坐标列表（7D: [x, y, z, qx, qy, qz, qw] 或 [x, y, z, radius]）
        bins: 分箱边界数组

    Returns:
        hash_key: 量化编码后的hash key字符串
    """
    # 只对坐标部分[0:3]进行量化
    code_quant = np.digitize(link_coords[0:3], bins, right=True)
    # 从bins计算quant_bits
    quant_bits = (len(bins) - 1).bit_length()
    # 转换为hash key字符串
    keyy = return_keyy(code_quant, quant_bits)
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
        filename = (
            f"{data_folder}/{basename}_{benchid:04d}_{collision_model_type}_cycles.pkl"
        )

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
            quant_bits = (len(bins) - 1).bit_length()
            keyy = return_keyy(code_quant, quant_bits)

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
            quant_bits = (len(bins) - 1).bit_length()
            keyy = return_keyy(code_quant, quant_bits)

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
            quant_bits = (len(bins) - 1).bit_length()
            keyy = return_keyy(code_quant, quant_bits)

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
            quant_bits = (len(bins) - 1).bit_length()
            keyy = return_keyy(code_quant, quant_bits)

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


def generate_recursive_reorder(num_poses, step_size=8):
    """
    生成递归式重排顺序（保持固定步长，只对组序列进行递归二分重排）。
    """
    # 步骤1：对组号进行递归二分重排
    group_count = min(step_size, (num_poses + step_size - 1) // step_size)
    group_order = recursive_binary_reorder(group_count)
    # group_order = list(range(step_size))
    # 步骤2：按重排后的组号顺序生成pose列表
    reorder = []
    for group_id in group_order:
        pose_idx = group_id
        while pose_idx < num_poses:
            reorder.append(pose_idx)
            pose_idx += step_size

    return reorder


def recursive_binary_reorder(n):
    """
    将 [0,1,2,...,n-1] 按递归二分方式重排（使用位反转）。
    """
    if n <= 1:
        return list(range(n))

    # 计算需要的位数
    num_bits = 0
    temp = n - 1
    while temp > 0:
        num_bits += 1
        temp >>= 1

    reorder = []
    for i in range(n):
        # 对i进行位反转
        reversed_i = 0
        for bit in range(num_bits):
            reversed_i = (reversed_i << 1) | ((i >> bit) & 1)
        reorder.append(reversed_i)

    return reorder


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

    # 步骤1：确定pose顺序（可选递归重排）
    if use_recursive_reorder:
        reorder = generate_recursive_reorder(num_poses, step_size)
    else:
        reorder = list(range(num_poses))

    # 步骤2：初始化所有COPU的数据列表
    copus_coords = [[] for _ in range(num_copus)]
    copus_flags = [[] for _ in range(num_copus)]
    copus_cycles = [[] for _ in range(num_copus)]

    # 步骤3：按轮转方式将poses分配给各COPU
    for reordered_idx, original_pose_idx in enumerate(reorder):
        copu_id = reordered_idx % num_copus
        pose_coords = edge_coords[original_pose_idx]  # List[link_coord]
        pose_flags = edge_flags[original_pose_idx]  # List[link_flag]

        # 展平link数据
        copus_coords[copu_id].extend(pose_coords)
        copus_flags[copu_id].extend(pose_flags)

        # 处理周期数据
        if edge_cycles is not None:
            pose_cycles = edge_cycles[original_pose_idx]
            copus_cycles[copu_id].extend(pose_cycles)
        else:
            copus_cycles[copu_id].extend([40 for _ in range(len(pose_coords))])

    return copus_coords, copus_flags, copus_cycles


def analyze_multi_copu_performance(results):
    """
    分析多COPU系统的性能指标
    """
    cht_stats = results["cht_stats"]
    copu_stats = results["copus"]

    # KPI 1: 系统吞吐量
    total_queries = sum(c["total_queries"] for c in copu_stats)
    system_throughput = total_queries / max(1, results["total_cycles"])

    # KPI 2: COPU平均利用率
    avg_utilization = sum(c["oocd_utilization"] for c in copu_stats) / len(copu_stats)

    # KPI 3: CHT冲突率
    cht_conflict_rate = cht_stats["conflict_rate"]

    # KPI 4: 负载平衡
    query_counts = [c["total_queries"] for c in copu_stats]
    if len(query_counts) > 1:
        load_balance = np.std(query_counts) / (np.mean(query_counts) + 1e-6)
    else:
        load_balance = 0.0

    return {
        "system_throughput": system_throughput,
        "avg_copu_utilization": avg_utilization,
        "cht_conflict_rate": cht_conflict_rate,
        "load_balance_variance": load_balance,
        "total_cycles": results["total_cycles"],
        "total_queries": total_queries,
        "num_copus": len(copu_stats),
        "per_copu_queries": query_counts,
    }


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

    参数:
        num_dedicated_oocds: 专门用于QCOLL任务的CDU数量。
                             这些CDU优先处理QCOLL，除非QNONCOLL已满。
                             剩余的CDU (num_oocds - num_dedicated_oocds) 为共享CDU，
                             无优先级地处理QCOLL和QNONCOLL任务。
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
                is_dedicated = oocd_id < num_dedicated_oocds
                task_assigned = False

                if is_dedicated:
                    # 专用CDU策略:
                    # 1. 如果QNONCOLL已满，优先处理QNONCOLL以解除阻塞
                    if len(qnoncoll) >= qnoncoll_len:
                        oocds[oocd_id] = OOCDState(
                            hash_key=qnoncoll[0][0],
                            result=qnoncoll[0][1],
                            busy=1,
                            free_cycle=cycle + cycle_check,
                        )
                        qnoncoll.popleft()
                        task_assigned = True
                    # 2. 否则，专门处理QCOLL
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
                    # 3. 如果没有更多待处理的链接（清空阶段）且QNONCOLL有任务，处理QNONCOLL
                    elif len(linklist) == 0 and len(qnoncoll) > 0:
                        oocds[oocd_id] = OOCDState(
                            hash_key=qnoncoll[0][0],
                            result=qnoncoll[0][1],
                            busy=1,
                            free_cycle=cycle + cycle_check,
                        )
                        qnoncoll.popleft()
                        task_assigned = True
                    # 4. 否则空闲
                else:
                    # 共享CDU策略 (无优先级/混合使用):
                    # 只要有任务就处理，不区分优先级

                    # 尝试QCOLL
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
                    # 尝试QNONCOLL (无"满"限制)
                    elif len(qnoncoll) > 0:
                        oocds[oocd_id] = OOCDState(
                            hash_key=qnoncoll[0][0],
                            result=qnoncoll[0][1],
                            busy=1,
                            free_cycle=cycle + cycle_check,
                        )
                        qnoncoll.popleft()
                        task_assigned = True

                if task_assigned:
                    dequeued_this_cycle = True
                else:
                    # 如果没有分配任务，保持空闲
                    oocds[oocd_id] = OOCDState(
                        hash_key=0, result=0, busy=0, free_cycle=0
                    )

        # --- 步骤2: 预测下一个配置并放入相应队列 ---
        if len(linklist) > 0:
            link, linkcoll = linklist[0], linklist_coll[0]

            # 将配置数据“量化”以生成用于查询历史表的键 (key)
            code_quant = np.digitize(link, bins, right=True)
            quant_bits = (len(bins) - 1).bit_length()
            keyy = return_keyy(code_quant, quant_bits)

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


def calculate_oracle_cycles(edge_coll_data, num_oocds, cycle_check):
    """
    根据num_oocds,计算单个edge的理论最小周期数消耗.

    如果edge 会发生碰撞,那么edge消耗的理论最小周期数消耗是单个cycle_check
    如果edge不会发生碰撞,那么edge消耗的理论最小周期数就是 ceil(edge中总的碰撞检查数/num_oocds) * cycle_check
    """
    has_collision = False
    total_checks = 0

    for pose_coll in edge_coll_data:
        # 检查当前pose是否有碰撞 (0表示碰撞)
        if any(c == 0 for c in pose_coll):
            has_collision = True
            break
        total_checks += len(pose_coll)

    if has_collision:
        return cycle_check
    else:
        # 使用向上取整计算批次: (total + num - 1) // num
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
):
    """
    双缓冲架构的并行碰撞检测仿真。

    核心机制：
    - 两组队列 (Bank A 和 Bank B)，分别处理不同的 edge
    - 当一个 edge 处理完成（碰撞或全部完成）时，CDU 切换到另一组队列
    - 两个预测器同时工作，为当前 edge 和下一 edge 生成任务

    Args:
        edges_data: 边数据列表 [edge][pose][element]
        edges_coll: 边碰撞标志 [edge][pose][element]
        其他参数同基础仿真函数

    Returns:
        query_count: 总查询数
        colldict: 更新后的碰撞历史表
        total_cycles: 总周期数
        stats: 统计信息字典
    """
    # 初始化 CDU 阵列
    oocds = [
        OOCDState(hash_key=0, result=0, busy=0, free_cycle=0) for _ in range(num_oocds)
    ]

    # Bank A 队列
    qcoll_a = deque(maxlen=qcoll_len)
    qnoncoll_a = deque(maxlen=qnoncoll_len)
    linklist_a = []
    linklist_coll_a = []

    # Bank B 队列
    qcoll_b = deque(maxlen=qcoll_len)
    qnoncoll_b = deque(maxlen=qnoncoll_len)
    linklist_b = []
    linklist_coll_b = []

    # 状态变量
    active_bank_is_a = True  # True = Bank A 供给 CDU, False = Bank B
    current_edge_idx = 0
    cycle = 0
    total_query_count = 0.0
    bank_swap_count = 0
    cdu_idle_cycles = 0

    first_two_running = 0
    first_two_checked = 0

    # 主循环
    while True:
        # 选择当前活跃的 Bank
        if active_bank_is_a:
            qcoll = qcoll_a
            qnoncoll = qnoncoll_a
            linklist = linklist_a
            linklist_coll = linklist_coll_a
            # Staging Bank
            qcoll_staging = qcoll_b
            qnoncoll_staging = qnoncoll_b
            linklist_staging = linklist_b
            linklist_coll_staging = linklist_coll_b
        else:
            qcoll = qcoll_b
            qnoncoll = qnoncoll_b
            linklist = linklist_b
            linklist_coll = linklist_coll_b
            # Staging Bank
            qcoll_staging = qcoll_a
            qnoncoll_staging = qnoncoll_a
            linklist_staging = linklist_a
            linklist_coll_staging = linklist_coll_a

        # --- 步骤1: 处理 CDU 状态 ---
        coll_found = 0
        dequeued_this_cycle = False
        for oocd_id in range(len(oocds)):
            oocd = oocds[oocd_id]

            if oocd.busy == 1 and oocd.free_cycle <= cycle:
                total_query_count += 1
                if oocd.result == 0:
                    coll_found = 1
                colldict = update_collision_dict(
                    colldict, oocd.hash_key, oocd.result, sample_rate
                )

            if oocd.free_cycle <= cycle and not dequeued_this_cycle:
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
                    dequeued_this_cycle = True
                elif (
                    len(qnoncoll) == qnoncoll_len
                    or (len(linklist) == 0 and len(qnoncoll) > 0)
                ) and first_two_checked < cycle:
                    oocds[oocd_id] = OOCDState(
                        hash_key=qnoncoll[0][0],
                        result=qnoncoll[0][1],
                        busy=1,
                        free_cycle=cycle + cycle_check,
                    )
                    qnoncoll.popleft()
                    dequeued_this_cycle = True
                else:
                    oocds[oocd_id] = OOCDState(
                        hash_key=0, result=0, busy=0, free_cycle=0
                    )

        # 计算 CDU 空闲数
        cdu_idle_this_cycle = sum(
            1 for oocd in oocds if oocd.free_cycle <= cycle and not oocd.busy
        )
        cdu_idle_cycles += cdu_idle_this_cycle

        # --- 步骤2: 为 Active Bank 预测并入队 ---
        if len(linklist) > 0:
            link, linkcoll = linklist[0], linklist_coll[0]
            code_quant = np.digitize(link, bins, right=True)
            quant_bits = (len(bins) - 1).bit_length()
            keyy = return_keyy(code_quant, quant_bits)
            is_collision_predicted = predict_collision(colldict, keyy, threshold)

            if is_collision_predicted:
                if len(qcoll) < qcoll_len:
                    qcoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]
            else:
                if len(qnoncoll) < qnoncoll_len:
                    qnoncoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]

        # --- 步骤2b: 为 Staging Bank 预测并入队 ---
        if len(linklist_staging) > 0:
            link_s, linkcoll_s = linklist_staging[0], linklist_coll_staging[0]
            code_quant_s = np.digitize(link_s, bins, right=True)
            quant_bits_s = (len(bins) - 1).bit_length()
            keyy_s = return_keyy(code_quant_s, quant_bits_s)
            is_collision_predicted_s = predict_collision(colldict, keyy_s, threshold)

            if is_collision_predicted_s:
                if len(qcoll_staging) < qcoll_len:
                    qcoll_staging.append([keyy_s, linkcoll_s])
                    del linklist_staging[0]
                    del linklist_coll_staging[0]
            else:
                if len(qnoncoll_staging) < qnoncoll_len:
                    qnoncoll_staging.append([keyy_s, linkcoll_s])
                    del linklist_staging[0]
                    del linklist_coll_staging[0]

        # --- 步骤3: 检查当前 edge 是否完成 ---
        everything_free = (
            len(linklist) == 0
            and not any(oocd.free_cycle > cycle for oocd in oocds)
            and len(qnoncoll) == 0
            and len(qcoll) == 0
        )

        # 当前 edge 完成（碰撞或全部完成）
        if coll_found or everything_free:
            # 加载下一个 edge 到当前 Bank
            if current_edge_idx < len(edges_data):
                edge_flat, edge_coll_flat = csp_rearrange(
                    edges_data[current_edge_idx],
                    edges_coll[current_edge_idx],
                    groupsize=8,
                )
                if active_bank_is_a:
                    linklist_a.extend(edge_flat)
                    linklist_coll_a.extend(edge_coll_flat)
                else:
                    linklist_b.extend(edge_flat)
                    linklist_coll_b.extend(edge_coll_flat)
                current_edge_idx += 1
            else:
                break
            # 切换 Bank
            if everything_free and (
                len(qcoll_staging) > 0
                or len(qnoncoll_staging) > 0
                or len(linklist_staging) > 0
            ):
                active_bank_is_a = not active_bank_is_a
                bank_swap_count += 1
                first_two_running = 0
                first_two_checked = 0

        cycle += 1

    # 计算未完成任务
    for oocd in oocds:
        if oocd.free_cycle > cycle:
            total_query_count += (cycle_check - oocd.free_cycle + cycle) / cycle_check

    stats = {
        "total_edges": len(edges_data),
        "bank_swap_count": bank_swap_count,
        "cdu_idle_cycles": cdu_idle_cycles,
        "cdu_utilization": 1.0 - (cdu_idle_cycles / (cycle * num_oocds))
        if cycle > 0
        else 0.0,
    }

    return total_query_count, colldict, cycle, stats
