"""perf_prediction_simulation_mpnet.py
微架构级并行碰撞检测调度/性能仿真脚本（MPNET 数据集专用）。

核心思想：
1. 读取预先离线生成的路径文件 (edge_link_data, edge_link_coll_data)，每条路径包含多帧 pose，每帧含多个 link（对应需要做的碰撞查询单元任务）。
2. 将 (pose, link) 重新线性化 (csp_rearrange) 形成一个待调度的查询序列 linklist。
3. 用固定数量的“并行碰撞检测单元” (oocds) 模拟硬件功能单元/执行槽位，每个任务执行固定 latency (cycle_check)。
4. 通过两个队列：
    - qcoll    : 预测为高碰撞概率（优先）
    - qnoncoll : 预测为低碰撞概率（批量填充）
    实现一种经验驱动的调度策略。
5. 历史预测表 (colldict) ：key 为量化后的 link 几何特征哈希，值为 [碰撞次数, 自由次数]。简单规则：如果历史碰撞计数高于 (自由计数 * 阈值系数)，则优先入高风险队列。
6. 调度循环逐 cycle 前进：
    - 检查已完成的执行槽位：统计成本、更新预测历史、检测是否发现碰撞(终止条件)。
    - 分配可用槽位：优先取 qcoll 中任务；若 qnoncoll 满或无更多待分派且有剩余则取 qnoncoll。
    - 从剩余 linklist 继续按上述规则填充队列（相当于模拟前端取指 / 预测分类）。
    - 终止条件：发现碰撞 (coll_found) 或所有待检查任务与队列均为空且单元全部空闲 (everything_free)。
7. 成本模型：
    - 碰撞任务计 1 单位（all_prediction_this_edge +=1）。
    - Free 任务计 0.7（抽象表示较轻路径 / 可早停收益）。
    - 未完成任务在结束时按剩余执行比例折算（部分完成的周期占比）。
8. 汇总输出：打印 (并行核数, 平均每条路径消耗的周期, 平均预测加权查询成本)。

与其它脚本区别：该脚本关注“预测策略”性能；另有 perf_csp_* (无预测) 与 perf_oracle_* (理想顺序下界)。
"""

import sys, os, argparse

import numpy as np
import matplotlib.pyplot as plt

import random
from tqdm import tqdm
import pickle
import pandas as pd


# 0.5 0.25
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
    """将 (pose, pose_labels) 重排成线性 link 序列（CSP 风格顺序）。

    策略：
    1. 先取最后一个 pose (往往更靠近目标/更关键)。
    2. 其余 pose 以内置次序索引模式 [0,4,2,6,1,5,3,7] 分组抽取（类似分层 / 交错以提高早期发现冲突概率）。
    3. 展平成 (linklist, linklist_coll)。
    返回：扁平化后的 link 向量数组 & 对应标签 (0=碰撞,1=自由)。
    """
    # receives a list of poses in the edge and answers for that
    num_steps = len(edge)
    rearr = [edge[-1]]
    rearryarr = [edgeyarr[-1]]
    for i in [0, 4, 2, 6, 1, 5, 3, 7]:
        for j in range(i, num_steps - 1, 8):
            rearr.append(edge[j])
            rearryarr.append(edgeyarr[j])
    group = []
    grouparr = []
    for pose, posecoll in zip(rearr, rearryarr):
        for link, linkcoll in zip(pose, posecoll):
            group.append(link)
            grouparr.append(linkcoll)
    return group, grouparr


# distributing the dataset into two components X and Y

# ====== 离散化相关参数：将连续 link 特征向量逐元素量化到固定 bins 以形成哈希键 ======
binnumber = 16  # 每维分箱数量（定宽分箱），越大 → 哈希更稀疏
intervalsize = 2 / binnumber
bins = np.zeros(binnumber)
start = -1
for i in range(0, binnumber):
    bins[i] = start
    start += intervalsize

# ====== 全局累积统计变量 ======
all_csp = 0  # （未在最终输出使用，保留）
all_prediction = 0  # 当前 bench 的预测策略查询累计（含权重）
all_oracle = 0  # 当前 bench 的“oracle”理想最小参考成本
globalcolldict = {}  # 跨 bench 可共享（此脚本中未复用）
colldict = {}  # 预测历史表：key -> [collision_count, free_count]
fall_serial = 0  # 预留：串行基线
fall_parallel = 0  # 预留：并行基线
fall_prediction = 0  # 跨 bench 累积预测成本
fall_oracle = 0  # 跨 bench 累积 oracle 成本

# ====== 微架构/资源参数 ======
qnoncoll_len = 56  # 低风险队列容量（批量缓冲）
qcoll_len = 8  # 高风险队列容量（小而快）
cycle_check = 40  # 统一固定 latency（所有任务同一执行时长）
cycle_check_free = 30  # （未真正使用于执行时间区分，可能为未来扩展）

# ====== 记录每条路径的周期/成本明细 ======
cycle_count = []  # 每条路径消耗的周期集合（用于统计均值）
comp_count = []  # 每条路径预测成本集合（加权）
count_edges = 0  # 已处理路径数量
obb_conv_count = 0  # OBB 转换/特征量化计数（近似前端准备开销）
total_cycles = 0  # 全部路径周期总和
for benchid in tqdm([
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
]):
    # for benchid in tqdm([0,13,14]):
    # for benchid in (range(2000,2200)):

    cycle_count.append([])  # 为该 bench 累积其下所有路径的周期
    comp_count.append([])  # 为该 bench 累积其下所有路径的预测成本
    all_parallel = 0
    all_prediction = 0
    all_oracle = 0
    colldict = {}
    benchidstr = str(benchid)
    filename = ("../trace_files/motion_traces/logfiles_MPNET_7D/coord_bench_3_" + str(benchid) + ".pkl")
    try:
        f = open(filename, "rb")
    except:
        continue

    (edge_link_data, edge_link_coll_data) = pickle.load(f)
    f.close()

    for k, v in colldict.items():
        sc = 2
        colldict[k][0] = int(colldict[k][0] / sc)
        colldict[k][1] = int(colldict[k][1] / sc)

    # each entry is a tuple (hash code, and output)

    # ====== 遍历该 bench 下所有路径（edge） ======
    for edge, edge_coll in zip(edge_link_data, edge_link_coll_data):
        count_edges += 1
        all_prediction_this_edge = 0  # 加权查询成本（当前路径级）
        cycle = 0
        first_two_running = 0
        first_two_checked = 0
        # 初始化并行执行槽位
        oocds = [[0, 0, 0, 0] for _ in range(int(sys.argv[3]))]
        qcoll = []  # 高风险队列
        qnoncoll = []  # 低风险队列

        if len(edge_coll) == 0:
            continue

        # 先判断该路径是否存在碰撞以更新 oracle 下界
        coll_found = 0
        for pose, pose_coll in zip(edge, edge_coll):
            for link, link_coll in zip(pose, pose_coll):
                if link_coll == 0:
                    coll_found = 1
                    break
            if coll_found:
                break
        if coll_found == 1:
            all_oracle += 1
        else:
            all_oracle += 7 * len(edge_coll)

        # 生成线性任务序列
        linklist, linklist_coll = csp_rearrange(edge, edge_coll, groupsize=4)
        coll_found = 0  # 重置（用于模拟阶段）
        links_remaining = len(linklist)
        everything_free = 0

        # ====== 主循环：cycle 级执行调度仿真 ======
        while coll_found == 0 and everything_free == 0:
            # 1. 回收已完成槽位并更新统计 / 预测表
            for oocd_id in range(len(oocds)):
                oocd = oocds[oocd_id]
                if oocd[2] == 1 and oocd[3] <= cycle:  # 该执行槽已完成
                    all_prediction += 1
                    if oocd[1] == 0:
                        all_prediction_this_edge += 1
                        coll_found = 1
                    else:
                        all_prediction_this_edge += 0.7

                    # 更新历史表（带随机采样，限制最大计数15防止饱和）
                    if oocd[0] not in colldict:
                        colldict[oocd[0]] = [0, 0]
                    if oocd[1] == 1:
                        if (random.random() <= float(sys.argv[2]) and colldict[oocd[0]][1] < 15):
                            colldict[oocd[0]][1] += 1
                    else:  # 碰撞
                        if colldict[oocd[0]][0] < 15:
                            colldict[oocd[0]][0] += 1

                # 2. 若槽位空闲则尝试派发新任务
                if oocd[3] <= cycle and coll_found == 0:
                    if len(qcoll) > 0 and first_two_checked < cycle:
                        first_two_running += 1
                        if first_two_running == 1:
                            first_two_checked = cycle + cycle_check
                        task = qcoll.pop(0)
                        oocds[oocd_id] = [task[0], task[1], 1, cycle + cycle_check]
                    elif ((len(qnoncoll) == qnoncoll_len or (links_remaining == 0 and len(qnoncoll) > 0))
                          and first_two_checked < cycle):
                        task = qnoncoll.pop(0)
                        oocds[oocd_id] = [task[0], task[1], 1, cycle + cycle_check]
                    else:
                        oocds[oocd_id] = [0, 0, 0, 0]

            # 3. 取前端新任务放入队列（若未发现碰撞）
            if len(linklist) > 0 and coll_found == 0:
                link = linklist[0]
                linkcoll = linklist_coll[0]
                obb_conv_count += 1
                code_quant = np.digitize(link, bins, right=True)
                keyy = reutrn_keyy(code_quant)

                if keyy in colldict and colldict[keyy][0] > colldict[keyy][1] * float(sys.argv[1]):
                    # 预测高风险
                    if len(qcoll) < qcoll_len:
                        qcoll.append([keyy, linkcoll])
                        del linklist[0]
                        del linklist_coll[0]
                else:
                    if len(qnoncoll) < qnoncoll_len:
                        qnoncoll.append([keyy, linkcoll])
                        del linklist[0]
                        del linklist_coll[0]

            links_remaining = len(linklist_coll)

            # 4. 判断是否所有任务完成
            if coll_found == 0 and links_remaining == 0:
                everything_free = 1
                if any(o[3] > cycle for o in oocds):
                    everything_free = 0
                if qnoncoll or qcoll:
                    everything_free = 0

            cycle += 1

        # 5. 对仍在执行的槽位做部分折算
        for oocd in oocds:
            if oocd[2] == 1 and oocd[3] > cycle:
                frac = (cycle_check - oocd[3] + cycle) / cycle_check
                all_prediction += frac
                # 按 free 权重处理（此处保持原始脚本逻辑：未区分 label）
                all_prediction_this_edge += 0.7 * frac

        cycle_count[-1].append(cycle)
        total_cycles += cycle
        comp_count[-1].append(all_prediction_this_edge)
        # print(all_prediction)
        # print(colldict)
        # break
        # break
        # print(coll_found,all_serial,all_prediction)
    # for k,v in colldict.items():
    #    print(k,v,"\n")
    # print(benchid,"Overall parallel %d Overall serial %d Overall prediction %d Overall oracle %d"%(all_parallel,all_serial,all_prediction,all_oracle))
    # print(benchid,"%d %d %d %d"%(all_parallel,all_serial,all_prediction,all_oracle))
    fall_oracle += all_oracle
    # fall_parallel+=all_parallel
    fall_prediction += all_prediction
    # print(all_prediction,all_oracle)
    # fall_serial+=all_serial
    # for k,v in colldict.items():
    #    print(v)
# print(fall_parallel,fall_serial,fall_prediction,fall_oracle)
# print((fall_parallel-fall_prediction)/fall_parallel,(fall_parallel-fall_oracle)/fall_parallel)
# print((fall_parallel-fall_prediction)/fall_parallel,(fall_parallel-fall_oracle)/fall_parallel)
# print("cycles")
# for i,j in zip(cycle_count,comp_count):
#    print(np.mean(i),np.mean(j))
# print("overall")
# print(fall_prediction,fall_oracle,fall_prediction/count_edges,total_cycles/count_edges,obb_conv_count/count_edges)

# print(fall_prediction,fall_oracle,fall_prediction/count_edges,total_cycles/count_edges,obb_conv_count/count_edges)

print(
    sys.argv[3],  # 并行单元数量
    total_cycles / count_edges,  # 平均每条路径（edge）消耗的模拟周期
    fall_prediction / count_edges,  # 平均加权查询成本
)
