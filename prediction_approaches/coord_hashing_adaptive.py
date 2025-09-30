import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from collections import deque

# 该脚本实现了一种自适应哈希策略。
# 在这种策略中,敏感度'S'会根据最近的碰撞频率动态调整。


def plot(code, ytest, name):
    # 绘图函数,用于将二维哈希码可视化,区分碰撞和非碰撞点。
    principalComponents = code.data.cpu().numpy()
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


# --- 参数定义 ---
# sys.argv[1]: 用于离散化的箱子(bin)数量的比特位数,例如,4意味着2^4=16个箱子。
# sys.argv[2]: S_min (高碰撞频率下的敏感度)
# sys.argv[3]: S_max (低碰撞频率下的敏感度)
# sys.argv[4]: 对于非碰撞状态,以一定概率更新哈希表,用于控制哈希表的学习速度。

binnumber = 2 ** int(sys.argv[1])  # 计算箱子总数
S_min = float(sys.argv[2])  # 最小敏感度
S_max = float(sys.argv[3])  # 最大敏感度
update_prob = float(sys.argv[4])  # 更新概率

# --- 哈希空间初始化 ---
intervalsize = 2.24 / binnumber  # 计算每个箱子的间隔大小
bins = np.zeros(binnumber)
start = -1.12
# 初始化每个箱子的起始边界
for i in range(0, binnumber):
    bins[i] = start
    start += intervalsize

# --- 统计变量初始化 ---
all_onezero = 0  # 统计所有被错误预测为碰撞的非碰撞状态(False Positives)
all_zerozero = 0  # 统计所有被正确预测为碰撞的碰撞状态(True Positives)
all_total_colliding = 0  # 统计所有实际发生的碰撞总数
colldict = {}  # 哈希表,用于存储每个哈希键对应的[碰撞次数, 非碰撞次数]

# 创建一个固定长度的队列,用于记录最近1000次碰撞检测的结果。
# 1代表非碰撞,0代表碰撞。
collision_history = deque(maxlen=1000)

# --- 主循环: 遍历所有基准测试场景 ---
# 这个循环通过从低密度障碍物场景切换到高密度场景来模拟一个变化的环境。
for benchid in range(0, 100):
    benchidstr = str(benchid)

    # 前50个基准测试加载低密度场景,后50个加载高密度场景。
    if benchid < 50:
        trace_path = f"../trace_files/scene_benchmarks/moving_1030_10_low/obstacles_{benchidstr}_coord.pkl"
    else:
        high_benchid = benchid
        trace_path = f"../trace_files/scene_benchmarks/moving_1030_10_high/obstacles_{high_benchid}_coord.pkl"

    if not os.path.exists(trace_path):
        continue

    # 从pickle文件中加载数据
    with open(trace_path, "rb") as f:
        xtest_pred, dirr_pred, label_pred = pickle.load(f)

    # 将连续的坐标数据离散化,映射到对应的箱子索引
    code_pred_quant = np.digitize(xtest_pred, bins, right=True)

    bitsize = len(code_pred_quant[0])  # 获取哈希码的位数
    total_colliding_in_bench = 0

    # 以步长为7遍历所有状态
    for bini in range(0, len(code_pred_quant), 7):
        predicted = 1  # 默认预测为非碰撞
        true_ans = 1  # 默认真实状态为非碰撞

        # --- 自适应敏感度计算 ---
        if len(collision_history) > 0:
            # 碰撞频率 = 历史记录中0的个数 / 历史记录长度
            collision_freq = (collision_history.maxlen - sum(collision_history)) / len(
                collision_history
            )
            # 使用线性插值计算当前的敏感度S
            # 碰撞频率越高,S越接近S_min; 频率越低,S越接近S_max。
            current_S = S_max - (S_max - S_min) * collision_freq
        else:
            current_S = S_max  # 在开始阶段,使用较高的敏感度

        # 检查一个运动轨迹中的7个连续状态点
        for i in range(bini, bini + 7):
            keyy = ""
            # 生成哈希键
            for j in range(0, bitsize):
                keyy += str(code_pred_quant[i, j]).zfill(2)

            if keyy in colldict:
                # 如果哈希键存在,则根据动态S值进行预测
                # 碰撞计数 > S * 非碰撞计数,则预测为碰撞
                if colldict[keyy][0] > (current_S * colldict[keyy][1]):
                    predicted = 0
                    if label_pred[i] > 0.5:  # 如果真实为非碰撞,则为误报
                        all_onezero += 1

                # 更新哈希表
                if (
                    label_pred[i] > 0.5 and random.random() <= update_prob
                ) or label_pred[i] < 0.5:
                    colldict[keyy][int(label_pred[i].item())] += 1
            else:
                # 如果哈希键不存在,则创建新条目并更新
                if (
                    label_pred[i] > 0.5 and random.random() <= update_prob
                ) or label_pred[i] < 0.5:
                    colldict[keyy] = [0, 0]
                    colldict[keyy][int(label_pred[i].item())] += 1

            # 更新碰撞历史记录,并检查是否可以提前退出
            collision_history.append(label_pred[i])
            if label_pred[i] < 0.5:  # 如果当前点真实为碰撞
                true_ans = 0
                if predicted == 0:  # 如果预测也为碰撞,则是正确预测
                    all_zerozero += 1
                    break  # 提前退出内层循环

        if true_ans == 0:
            total_colliding_in_bench += 1
            all_total_colliding += 1

    if total_colliding_in_bench == 0:
        continue

# --- 最终结果计算 ---
# 精确度 = 正确预测的碰撞 / (正确预测的碰撞 + 错误预测的碰撞)
precision = (
    all_zerozero * 100 / (all_zerozero + all_onezero)
    if (all_zerozero + all_onezero) > 0
    else 0
)
# 召回率 = 正确预测的碰撞 / 所有真实发生的碰撞
recall = all_zerozero * 100 / all_total_colliding if all_total_colliding > 0 else 0

print(f"{precision:.2f},{recall:.2f}")
