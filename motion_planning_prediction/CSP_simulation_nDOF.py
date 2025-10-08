"""
CSP (Collision Space Partitioning) 碰撞检测调度模拟 - nDOF场景
===============================================================

功能说明:
    模拟使用CSP静态启发式顺序的并行碰撞检测过程,不使用预测策略。
    作为性能评估的基线(baseline),用于与预测优化方法进行对比。

CSP方法特点:
    - 使用静态的csp_rearrange()函数重排序碰撞检测任务
    - 按照固定的启发式顺序(0,4,2,6,1,5,3,7)进行分层采样
    - 不使用碰撞历史表(colldict)进行预测和优先级调度
    - 所有任务按相同优先级处理,填充到单一队列中

输入参数:
    sys.argv[1]: 运动规划算法类型 ("MPNET"/"BIT"/"GNN")

输出结果:
    每个benchmark输出一行: <预测成本> <Oracle理想成本>
    用于后续与预测方法(prediction_simulation_nDOF.py)进行性能对比
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import pickle


def plot(code, ytest, name):
    """
    可视化辅助函数(当前未使用)
    用于绘制编码空间中碰撞和非碰撞样本的分布
    """
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
    """
    将量化后的配置向量转换为哈希键字符串

    Args:
        code: 量化后的配置向量 (每个元素是0-15的整数)

    Returns:
        keyy: 哈希键字符串 (如: "080811" 表示 [8,8,11])

    说明:
        - 每个维度占2位数字,不足10的补0
        - 用于构建碰撞历史表(CHT)的键
    """
    bitsize = len(code)
    keyy = ""
    for j in range(0, bitsize):
        if code[j] < 10:
            keyy = keyy + "0"  # 补零,确保每个维度占2位
        keyy = keyy + str(code[j])
    return keyy


# 是否考虑运动方向信息(当前未使用)
consider_dir = False


def csp_rearrange(edge, edgeyarr, groupsize=8):
    """
    CSP重排序函数: 按照静态启发式顺序重新排列路径姿态

    核心思想:
        使用分层采样策略,快速覆盖整个路径空间,尽早发现碰撞

    重排序策略:
        1. 首先检查最后一个姿态(目标点)
        2. 然后按照 [0,4,2,6,1,5,3,7] 的顺序进行分层采样
        3. 每8个姿态中按此顺序选取,实现粗到细的覆盖

    Args:
        edge: 路径上的姿态列表 [[link1,link2,...], [link1,link2,...], ...]
        edgeyarr: 对应的碰撞标签列表 [[0/1,0/1,...], [0/1,0/1,...], ...]
        groupsize: 分组大小(未使用,保留参数)

    Returns:
        group: 重排序后展平的链接配置列表
        grouparr: 对应的碰撞标签列表

    示例:
        假设路径有16个姿态 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        重排后顺序: [15, 0,8, 4,12, 2,10, 6,14, 1,9, 5,13, 3,11, 7,15]
        这样可以快速覆盖整个路径的不同区域
    """
    num_steps = len(edge)

    # 首先添加最后一个姿态(目标点) - 优先检查是否能到达目标
    rearr = [edge[-1]]
    rearryarr = [edgeyarr[-1]]

    # 按照分层采样顺序重排: [0,4,2,6,1,5,3,7]
    # 这是一个二分策略: 先检查0和4(两端), 再检查2和6(四分点), 以此类推
    for i in [0, 4, 2, 6, 1, 5, 3, 7]:
        # 每隔8个姿态采样一次,实现粗粒度覆盖
        for j in range(i, num_steps - 1, 8):
            rearr.append(edge[j])
            rearryarr.append(edgeyarr[j])

    # 将姿态级别的数据展平为链接级别
    # 每个姿态包含多个链接(机器人的不同部位)
    group = []
    grouparr = []
    for pose, posecoll in zip(rearr, rearryarr):
        for link, linkcoll in zip(pose, posecoll):
            group.append(link)
            grouparr.append(linkcoll)
    return group, grouparr


# ========================================
# 量化参数设置: 将连续配置空间离散化
# ========================================
# 目的: 将机器人连续的关节角度量化到离散的区间中,用于构建哈希键

binnumber = 16  # 每个维度分成16个区间(2^4)
intervalsize = 2 / binnumber  # 每个区间的大小: 2/16 = 0.125
bins = np.zeros(binnumber)  # 区间边界数组
start = -1  # 起始值(假设关节角度范围[-1, 1])

# 构建分箱边界: [-1.0, -0.875, -0.75, ..., 0.875]
for i in range(0, binnumber):
    bins[i] = start
    start += intervalsize

# ========================================
# 全局统计变量初始化
# ========================================
all_csp = 0  # CSP方法的总查询数(未使用)
all_prediction = 0  # 当前benchmark的碰撞检测查询数
all_oracle = 0  # 当前benchmark的Oracle理想成本
globalcolldict = {}  # 全局碰撞字典(未使用)
colldict = {}  # 当前benchmark的碰撞历史表(CHT) - CSP方法中仅用于统计,不用于预测

# 累积统计变量(跨所有benchmark)
fall_serial = 0  # 串行方法累积成本(未使用)
fall_parallel = 0  # 并行方法累积成本(未使用)
fall_prediction = 0  # CSP方法累积成本
fall_oracle = 0  # Oracle方法累积成本

# ========================================
# 并行碰撞检测器(OOCD)配置参数
# ========================================
qnoncoll_len = 56  # 非碰撞队列最大长度(CSP方法中所有任务都进此队列)
qcoll_len = 8  # 碰撞队列最大长度(CSP方法中不使用,保留参数)
cycle_check = 40  # 每次碰撞检测所需周期数(模拟硬件延迟)

# 性能记录数组(用于详细分析)
cycle_count = []  # 记录每个benchmark中每条路径的周期数
comp_count = []  # 记录每个benchmark中每条路径的计算成本

# ========================================
# 确定要评估的benchmark范围
# ========================================
benchrange = range(2000, 2200)  # 默认: BIT/GNN使用2000-2199共200个benchmark
if sys.argv[1] == "MPNET":
    # MPNET只有特定的58个有效benchmark
    benchrange = [
        0,
        13,
        14,
        15,
        16,
        17,
        19,
        1,
        20,
        21,
        23,
        24,
        25,
        27,
        28,
        29,
        2,
        30,
        32,
        34,
        35,
        36,
        37,
        39,
        3,
        44,
        45,
        46,
        49,
        4,
        53,
        55,
        56,
        57,
        58,
        59,
        5,
        63,
        64,
        65,
        70,
        71,
        75,
        7,
        82,
        83,
        85,
        87,
        88,
        8,
        90,
        91,
        92,
        93,
        95,
        96,
        97,
        98,
    ]

# ========================================
# 主循环: 遍历所有benchmark场景
# ========================================
for benchid in benchrange:
    # 为当前benchmark初始化记录数组
    cycle_count.append([])  # 当前benchmark的周期记录
    comp_count.append([])  # 当前benchmark的计算成本记录

    # 重置当前benchmark的统计变量
    all_parallel = 0  # 并行方法成本(未使用)
    all_prediction = 0  # CSP方法碰撞检测查询数
    all_oracle = 0  # Oracle理想成本
    colldict = {}  # 碰撞历史表(CHT) - CSP中仅记录,不预测

    # ========================================
    # 根据算法类型加载对应的轨迹数据文件
    # ========================================
    benchidstr = str(benchid)
    if sys.argv[1] == "BIT":
        # BIT*算法生成的轨迹数据
        filename = (
            "../trace_files/motion_traces/logfiles_BIT_link/coord_motiom_"
            + str(benchid)
            + ".pkl"
        )
    elif sys.argv[1] == "GNN":
        # GNN算法生成的轨迹数据
        filename = (
            "../trace_files/motion_traces/logfiles_GNN_link/coord_gnn_motiom_"
            + str(benchid)
            + ".pkl"
        )
    elif sys.argv[1] == "MPNET":
        # MPNet算法生成的轨迹数据
        filename = (
            "../trace_files/motion_traces/logfiles_MPNET_7D/coord_bench_3_"
            + str(benchid)
            + ".pkl"
        )

    # 尝试加载数据文件,如果不存在则跳过
    try:
        f = open(filename, "rb")
    except:
        continue  # 文件不存在,跳到下一个benchmark

    # 加载轨迹数据
    # edge_link_data: 路径列表,每条路径包含多个姿态,每个姿态包含多个链接的配置
    # edge_link_coll_data: 对应的碰撞标签(0=碰撞, 1=自由)
    (edge_link_data, edge_link_coll_data) = pickle.load(f)
    f.close()

    # 缩放碰撞历史表的计数(用于防止过拟合,当前CSP方法中未实际使用)
    for k, v in colldict.items():
        sc = 2  # 缩放因子
        colldict[k][0] = int(colldict[k][0] / sc)  # 碰撞计数缩放
        colldict[k][1] = int(colldict[k][1] / sc)  # 自由计数缩放

    # ========================================
    # 遍历当前benchmark中的所有路径(edge)
    # ========================================
    all_prediction_this_edge = 0  # 当前路径的碰撞检测成本
    for edge, edge_coll in zip(edge_link_data, edge_link_coll_data):
        # ========================================
        # 初始化并行碰撞检测模拟的变量
        # ========================================
        cycle = 0  # 当前周期计数器
        first_two_running = 0  # 前两个检测器运行计数(未使用)
        first_two_checked = 0  # 前两个检测器检查时间(未使用)

        # 初始化7个并行的OOCD(Out-of-Order Collision Detector)执行槽位
        # 每个槽位结构: [hash_key, collision_result, is_active, finish_cycle]
        oocds = []
        for i in range(0, 7):
            # [哈希键, 碰撞结果(0/1), 是否激活(0/1), 完成周期]
            oocds.append([0, 0, 0, 0])

        # 初始化任务队列
        qcoll = []  # 高风险队列(CSP中不使用)
        qnoncoll = []  # 普通队列(CSP中所有任务都进此队列)

        # 跳过空路径
        if len(edge_coll) == 0:
            continue

        # ========================================
        # 计算Oracle理想成本(理论下界)
        # ========================================
        # Oracle知道真实碰撞信息,可以最优地安排检测顺序
        coll_found = 0
        for pose, pose_coll in zip(edge, edge_coll):
            for link, link_coll in zip(pose, pose_coll):
                if link_coll == 0:  # 发现碰撞
                    coll_found = 1
                    break
        if coll_found == 1:
            # 如果路径有碰撞,Oracle只需检测1次就能发现
            all_oracle += 1
        else:
            # 如果路径无碰撞,需要检测所有链接(7个OOCD × 姿态数)
            all_oracle += 7 * len(edge_coll)

        # ========================================
        # CSP重排序: 将路径按启发式顺序重新排列
        # ========================================
        linklist, linklist_coll = csp_rearrange(edge, edge_coll, groupsize=4)
        # linklist: 重排序后的链接配置列表
        # linklist_coll: 对应的碰撞标签列表

        # ========================================
        # 并行碰撞检测主循环
        # ========================================
        coll_found = 0  # 是否发现碰撞(终止条件)
        links_remaining = len(linklist)  # 剩余待检测的链接数
        everything_free = 0  # 是否所有检测都完成(终止条件)

        while coll_found == 0 and everything_free == 0:
            # ========================================
            # 阶段1: 回收完成的执行槽位
            # ========================================
            for oocd_id in range(0, len(oocds)):
                oocd = oocds[oocd_id]

                # 检查槽位是否激活且已完成(finish_cycle <= 当前cycle)
                if oocd[2] == 1 and oocd[3] <= cycle:
                    # 统计碰撞检测成本
                    all_prediction += 1  # 全局计数
                    all_prediction_this_edge += 1  # 当前路径计数

                    # 检查是否发现碰撞
                    if oocd[1] == 0:  # 碰撞结果为0表示碰撞
                        coll_found = 1  # 设置终止标志

                    # 更新碰撞历史表(CSP中仅记录,不用于预测)
                    if oocd[0] in colldict:
                        colldict[oocd[0]][oocd[1]] += 1
                    else:
                        colldict[oocd[0]] = [0, 0]  # [碰撞计数, 自由计数]
                        colldict[oocd[0]][oocd[1]] += 1

                # ========================================
                # 阶段2: 为空闲槽位分配新任务
                # ========================================
                if oocd[3] <= cycle:  # 槽位空闲或刚完成
                    if len(qnoncoll) > 0:
                        # 从队列中取出任务并分配给槽位
                        oocds[oocd_id] = [
                            qnoncoll[0][0],  # 哈希键
                            qnoncoll[0][1],  # 碰撞标签
                            1,  # 激活标志
                            cycle + cycle_check,  # 完成周期 = 当前周期 + 延迟
                        ]
                        del qnoncoll[0]  # 从队列中移除
                    else:
                        # 队列为空,槽位保持空闲
                        oocds[oocd_id] = [0, 0, 0, 0]
            # ========================================
            # 阶段3: 从待检测链接列表填充队列
            # ========================================
            if len(linklist) > 0:
                link = linklist[0]  # 取队列头部链接配置
                linkcoll = linklist_coll[0]  # 对应的碰撞标签

                # 量化链接配置并生成哈希键
                code_quant = np.digitize(link, bins, right=True)
                keyy = reutrn_keyy(code_quant)

                # 将任务添加到非碰撞队列
                if True:  # 占位条件(可用于动态调度策略)
                    if len(qnoncoll) < qnoncoll_len:
                        qnoncoll.append([keyy, linkcoll])
                        # 调试输出: 特定键的调度情况
                        # if keyy=="080811":
                        #    print(qnoncoll)
                        #    print(linklist[0],linklist_coll[0])
                        del linklist[0]  # 从待检测列表移除
                        del linklist_coll[0]
            # ========================================
            # 阶段4: 检查终止条件
            # ========================================
            links_remaining = len(linklist_coll)
            if links_remaining == 0:
                everything_free = 1  # 假设所有任务完成

                # 检查是否有执行中的任务
                for oocd in oocds:
                    if oocd[3] > cycle:  # 槽位仍在执行
                        everything_free = 0

                # 检查是否有队列中的待处理任务
                if len(qnoncoll) > 0:
                    everything_free = 0
                if len(qcoll) > 0:
                    everything_free = 0

            cycle += 1  # 周期计数器递增
        # ========================================
        # 路径检测完成后处理
        # ========================================
        # 处理循环提前终止时仍在执行的任务(部分完成)
        for oocd_id in range(0, len(oocds)):
            oocd = oocds[oocd_id]
            if oocd[3] > cycle:  # 任务未完成
                # 按比例计算已消耗的周期成本
                # (cycle_check - 剩余周期) / cycle_check = 已完成比例
                partial_cost = (cycle_check - oocd[3] + cycle) / cycle_check
                all_prediction += partial_cost
                all_prediction_this_edge += partial_cost

        # 记录当前路径的统计数据
        cycle_count[-1].append(cycle)  # 总周期数
        comp_count[-1].append(all_prediction_this_edge)  # 总检测成本
    # ========================================
    # 基准测试完成后统计
    # ========================================
    # 累加到全局统计变量
    fall_oracle += all_oracle  # Oracle总成本
    fall_prediction += all_prediction  # 实际检测总成本

    # 输出当前基准测试结果
    print(all_prediction, all_oracle)
