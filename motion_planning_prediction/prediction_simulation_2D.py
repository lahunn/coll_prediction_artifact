import sys

import numpy as np
import matplotlib.pyplot as plt

import random
import pickle


def plot(code, ytest, name):
    principalComponents = code.data.cpu().numpy()
    # print(principalComponents)
    coll = []
    collfree = []
    for i in range(0, len(ytest)):
        if ytest[i] > 0.5:
            collfree.append(principalComponents[i])
        else:
            coll.append(principalComponents[i])
    coll1 = np.array(coll)
    collfree1 = np.array(collfree)
    plt.scatter(
        collfree1[:, 0],
        collfree1[:, 1],
        label="Collision free",
        color="blue",
        alpha=0.3,
    )
    plt.scatter(coll1[:, 0], coll1[:, 1], color="red", label="Colliding", alpha=0.3)
    plt.savefig(name)
    plt.clf()
    plt.close()


def reutrn_keyy(code):
    bitsize = len(code)
    keyy = ""
    for j in range(0, bitsize):
        if code[j] < 10:
            keyy = keyy + "0"
        keyy = keyy + str(code[j])
    return keyy


# training for prediction

consider_dir = False


def csp_rearrange(edge, edgeyarr, groupsize=8):
    """
    CSP重排序函数：按照分层采样策略重新排列路径上的姿态

    重排序策略：
    1. 先检查最后一个姿态(终点)
    2. 然后按照0,4,2,6,1,5,3,7的顺序进行分层采样
    3. 这种策略可以快速覆盖整个路径空间

    Args:
        edge: 路径上的姿态列表
        edgeyarr: 对应的碰撞标签列表
        groupsize: 分组大小(默认8)

    Returns:
        group: 重排序后的链接列表
        grouparr: 重排序后的碰撞标签列表
    """
    num_steps = len(edge)
    rearr = [edge[-1]]  # 首先添加最后一个姿态
    rearryarr = [edgeyarr[-1]]

    # 按照分层采样顺序重排
    for i in [0, 4, 2, 6, 1, 5, 3, 7]:
        for j in range(i, num_steps - 1, 8):
            rearr.append(edge[j])
            rearryarr.append(edgeyarr[j])

    # 将姿态展开为链接级别的数据
    group = []
    grouparr = []
    for pose, posecoll in zip(rearr, rearryarr):
        for link, linkcoll in zip(pose, posecoll):
            group.append(link)
            grouparr.append(linkcoll)
    return group, grouparr


def update_collision_dict(colldict, hash_key, is_free, sample_rate):
    """
    更新碰撞历史字典（持续学习）

    Args:
        colldict: 碰撞字典 {hash_key: [碰撞次数, 自由次数]}
        hash_key: 配置的哈希键
        is_free: 碰撞检测结果 (0=碰撞, 1=自由)
        sample_rate: 自由样本采样率

    Returns:
        updated_colldict: 更新后的碰撞字典
    """
    if hash_key in colldict:
        # 自由样本按概率采样更新
        if (
            is_free == 1
            and random.random() <= sample_rate
            and colldict[hash_key][is_free] < 15
        ):
            colldict[hash_key][is_free] += 1
        # 碰撞样本总是更新
        elif colldict[hash_key][is_free] < 15 and is_free == 0:
            colldict[hash_key][is_free] += 1
    else:
        # 新配置：初始化统计
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
    基于碰撞历史字典预测配置是否会发生碰撞

    Args:
        colldict: 碰撞字典 {hash_key: [碰撞次数, 自由次数]}
        hash_key: 配置的哈希键
        threshold: 碰撞预测阈值

    Returns:
        is_collision_predicted: 预测结果 (True=预测碰撞, False=预测自由)
    """
    if hash_key in colldict:
        # 碰撞预测逻辑：如果 碰撞次数 > 阈值 × 自由次数，预测为碰撞
        if colldict[hash_key][0] > colldict[hash_key][1] * threshold:
            return True  # 预测为碰撞
        else:
            return False  # 预测为自由
    else:
        # 未见过的配置：默认预测为自由
        return False


def simulate_parallel_collision_detection(
    linklist, linklist_coll, colldict, sample_phase, threshold, sample_rate
):
    """
    模拟并行碰撞检测过程，使用OOCD（乱序碰撞检测器）和预测优化

    Args:
        linklist: 重排序后的链接配置列表
        linklist_coll: 对应的碰撞标签列表
        colldict: 碰撞字典 {hash_key: [碰撞次数, 自由次数]}
        sample_phase: 采样阶段标志 (0=评估模式, 1=采样模式)
        threshold: 碰撞预测阈值
        sample_rate: 自由样本采样率

    Returns:
        query_count: 碰撞检测查询次数
        updated_colldict: 更新后的碰撞字典
        collision_found: 是否发现碰撞
    """
    # === 检测参数初始化 ===
    qnoncoll_len = 56  # 非碰撞队列最大长度
    qcoll_len = 8  # 碰撞队列最大长度
    cycle_check = 40  # 每次碰撞检测所需的周期数

    # === OOCD初始化 ===
    # 7个并行的乱序碰撞检测器
    oocds = []
    for i in range(0, 7):
        # [hash_key, is_free, is_valid, completion_cycle]
        oocds.append([0, 0, 0, 0])

    # === 队列和状态初始化 ===
    qcoll = []  # 碰撞队列（高优先级）
    qnoncoll = []  # 非碰撞队列（低优先级）
    cycle = 0  # 当前周期
    first_two_running = 0  # 已启动的初始检测数
    first_two_checked = 0  # 第一批检测完成时间

    # === 检测状态 ===
    coll_found = 0  # 是否发现碰撞
    links_remaining = len(linklist)  # 剩余待检测的链接数
    everything_free = 0  # 是否所有检测都完成且无碰撞
    query_count = 0  # 查询次数计数器

    # === 主循环：模拟并行碰撞检测过程 ===
    while coll_found == 0 and everything_free == 0:
        # === 步骤1：处理已完成的检测并调度新任务 ===
        for oocd_id in range(0, len(oocds)):
            oocd = oocds[oocd_id]

            # 如果检测器有效且检测已完成
            if oocd[2] == 1 and oocd[3] <= cycle:
                # 计数查询次数
                if sample_phase == 0:
                    query_count += 1

                # 检查检测结果
                if oocd[1] == 0:  # 发现碰撞
                    coll_found = 1

                # === 更新碰撞字典（持续学习） ===
                colldict = update_collision_dict(
                    colldict, oocd[0], oocd[1], sample_rate
                )

            # === 调度新的检测任务 ===
            if oocd[3] <= cycle:
                # 优先级1：碰撞队列
                if len(qcoll) > 0 and first_two_checked < cycle:
                    first_two_running += 1
                    if first_two_running == 1:
                        first_two_checked = cycle + cycle_check
                    oocds[oocd_id] = [
                        qcoll[0][0],
                        qcoll[0][1],
                        1,
                        cycle + cycle_check,
                    ]
                    del qcoll[0]
                # 优先级2：非碰撞队列
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
                    # 没有任务，保持空闲
                    oocds[oocd_id] = [0, 0, 0, 0]

        # === 步骤2：从重排序列表中取出下一个链接进行预测 ===
        if len(linklist) > 0:
            link = linklist[0]
            linkcoll = linklist_coll[0]

            # 量化为哈希键
            code_quant = np.digitize(link, bins, right=True)
            keyy = reutrn_keyy(code_quant)

            # 使用碰撞字典进行预测
            is_collision_predicted = predict_collision(colldict, keyy, threshold)

            if is_collision_predicted:
                # 预测为碰撞：加入碰撞队列（高优先级）
                if len(qcoll) < qcoll_len:
                    qcoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]
            else:
                # 预测为自由：加入非碰撞队列（低优先级）
                if len(qnoncoll) < qnoncoll_len:
                    qnoncoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]

        # 更新剩余链接数
        links_remaining = len(linklist_coll)

        # === 步骤3：检查是否所有检测都完成 ===
        if links_remaining == 0:
            everything_free = 1

            # 检查是否还有正在进行的检测
            for oocd in oocds:
                if oocd[3] > cycle:
                    everything_free = 0

            # 检查队列是否为空
            if len(qnoncoll) > 0 or len(qcoll) > 0:
                everything_free = 0

        # 周期递增
        cycle += 1

    # === 处理未完成的检测 ===
    for oocd in oocds:
        if oocd[3] > cycle:
            # 计算完成百分比
            query_count += (cycle_check - oocd[3] + cycle) / cycle_check

    return query_count, colldict, coll_found


# distributing the dataset into two components X and Y


binnumber = 32

intervalsize = 2 / binnumber
bins = np.zeros(binnumber)
start = -1
if sys.argv[4] == "MPNET":
    intervalsize = 40 / binnumber
    bins = np.zeros(binnumber)
    start = -20


for i in range(0, binnumber):
    bins[i] = start
    start += intervalsize

# === 全局统计变量初始化 ===
all_csp = 0  # CSP方法的总查询次数(未使用)
all_prediction = 0  # 预测方法的总查询次数
all_oracle = 0  # Oracle理想方法的总查询次数
globalcolldict = {}  # 全局碰撞字典(未使用)
colldict = {}  # 当前场景的碰撞字典: {hash_key: [碰撞次数, 自由次数]}

# === 累计统计变量 ===
fall_serial = 0  # 串行检测的累计查询次数(未使用)
fall_parallel = 0  # 并行检测的累计查询次数(未使用)
fall_prediction = 0  # 预测方法的累计查询次数
fall_oracle = 0  # Oracle方法的累计查询次数

# === 队列和检测参数 ===
qnoncoll_len = 56  # 非碰撞队列最大长度(初始值)
qnoncoll_len = int(1 * int(sys.argv[3]))  # 从命令行参数设置非碰撞队列长度
qcoll_len = 8  # 碰撞队列最大长度(固定为8)
cycle_check = 40  # 每次碰撞检测所需的周期数(模拟检测延迟)
# for benchid in [0,10,11,12,13,14,15,16,17,18,19,1,20,21,22,23,24,25,26,27,28,29,2,30,31,32,33,34,35,36,37,38,39,3,40,41,42,43,44,45,46,47,48,49,4,50,51,52,53,54,55,56,57,58,59,5,60,61,62,63,64,65,66,67,68,69,6,70,71,72,73,74,75,76,77,78,79,7,80,81,82,83,84,85,86,87,88,89,8,90,91,92,93,94,95,96,97,98,99,9]:
benchrange = range(0, 201)
if sys.argv[4] == "GNN":
    benchrange = range(1, 201)

for benchid in benchrange:
    """
    遍历所有基准测试场景
    对每个场景计算三种方法的碰撞检测查询次数：
    1. all_prediction: CSP+预测方法的查询次数
    2. all_oracle: Oracle理想方法的查询次数
    """

    # === 重置当前场景的统计变量 ===
    all_parallel = 0  # 并行方法的查询次数(未使用)
    all_prediction = 0  # 预测方法的查询次数
    all_oracle = 0  # Oracle方法的查询次数
    colldict = {}  # 重置碰撞字典
    benchidstr = str(benchid)

    # === 根据规划器类型选择数据文件 ===
    if sys.argv[4] == "BIT":
        filename = (
            "../trace_files/motion_traces/logfiles_BIT_2D/coord_motiom_"
            + str(benchid)
            + ".pkl"
        )
    elif sys.argv[4] == "GNN":
        filename = (
            "../trace_files/motion_traces/logfiles_GNN_2D/coord_motiom_"
            + str(benchid)
            + ".pkl"
        )
    elif sys.argv[4] == "MPNET":
        filename = (
            "../trace_files/motion_traces/logfiles_MPNET_2D/link_info_1_"
            + str(benchid)
            + ".pkl"
        )

    try:
        f = open(filename, "rb")
    except:
        continue
    if sys.argv[4] == "MPNET":
        (edge_link_data, edge_link_coll_data) = pickle.load(f, encoding="latin1")
    else:
        (edge_link_data, edge_link_coll_data) = pickle.load(f)

    f.close()

    for k, v in colldict.items():
        sc = 2
        colldict[k][0] = int(colldict[k][0] / sc)
        colldict[k][1] = int(colldict[k][1] / sc)

    # each entry is a tuple (hash code, and output)
    for edge, edge_coll in zip(edge_link_data, edge_link_coll_data):
        sample_phase = 0
        if len(edge_coll) == 1:
            sample_phase = 0
            # all_sample+=1
        # print(edge_coll)
        cycle = 0
        first_two_running = 0
        first_two_checked = 0
        oocds = []
        for i in range(0, 7):
            # hash code, is feasible, is valid, when free
            # oocds[i] = [hash_key, is_free, is_valid, completion_cycle]
            oocds.append([0, 0, 0, 0])
        qcoll = []
        qnoncoll = []

        if len(edge_coll) == 0:
            continue

        # === Oracle方法：计算理想情况下的最少查询次数 ===
        coll_found = 0
        # 遍历所有姿态，检查是否存在碰撞
        for pose, pose_coll in zip(edge, edge_coll):
            for link, link_coll in zip(pose, pose_coll):
                if link_coll == 0:  # 发现碰撞
                    coll_found = 1
                    break

        # Oracle计数逻辑：
        # - 如果有碰撞：只需1次查询就能发现（完美预知第一个碰撞位置）
        # - 如果无碰撞：需要检查所有姿态（必须验证整条路径安全）
        if sample_phase == 0:
            if coll_found == 1:
                all_oracle += 1  # 有碰撞：查询次数+1
            else:
                all_oracle += len(edge_coll)  # 无碰撞：查询次数+姿态数量

        # === CSP+预测方法：按重排序后的顺序进行碰撞检测 ===
        linklist, linklist_coll = csp_rearrange(edge, edge_coll, groupsize=4)

        # 使用封装的函数进行并行碰撞检测模拟
        edge_query_count, colldict, coll_found = simulate_parallel_collision_detection(
            linklist.copy(),  # 传入副本避免修改原列表
            linklist_coll.copy(),
            colldict,
            sample_phase,
            float(sys.argv[1]),  # threshold
            float(sys.argv[2]),  # sample_rate
        )

        # 累加当前路径的查询次数
        if sample_phase == 0:
            all_prediction += edge_query_count
        # print(all_prediction)
        # print(colldict)
        # break
        # break
        # print(coll_found,all_serial,all_prediction)
    # for k,v in colldict.items():
    #    print(k,v,"\n")
    # print(benchid,"Overall parallel %d Overall serial %d Overall prediction %d Overall oracle %d"%(all_parallel,all_serial,all_prediction,all_oracle))
    # print(benchid,"%d %d %d %d"%(all_parallel,all_serial,all_prediction,all_oracle))
    # === 累计到全局统计 ===
    fall_oracle += all_oracle
    # fall_parallel+=all_parallel
    fall_prediction += all_prediction

    # === 输出当前场景的结果 ===
    # 格式：prediction_queries oracle_queries
    # 这些数据会被保存到CSV文件，用于后续的性能分析
    print(all_prediction, all_oracle)
    # fall_serial+=all_serial
    # for k,v in colldict.items():
    #    print(v)

# print(fall_prediction,fall_oracle)
