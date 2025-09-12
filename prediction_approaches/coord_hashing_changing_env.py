import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle
import pandas as pd

# 这是coord_hashing.py的一个修改版,用于在变化的环境中(从低密度到高密度)
# 评估使用固定敏感度S的策略,作为自适应策略的基准。


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
    coll1 = (np.array(coll))
    collfree1 = (np.array(collfree))
    plt.scatter(collfree1[:, 0], collfree1[:, 1], label="Collision free", color="blue", alpha=0.3)
    plt.scatter(coll1[:, 0], coll1[:, 1], color="red", label="Colliding", alpha=0.3)
    plt.savefig(name)
    plt.clf()
    plt.close()


# --- 参数定义 ---
# sys.argv[1]: 用于离散化的箱子(bin)数量的比特位数。
# sys.argv[2]: 固定的敏感度S。如果S < 0,则使用随机预测作为基线。
# sys.argv[3]: 对于非碰撞状态,以一定概率更新哈希表。

binnumber = 2**int(sys.argv[1])  # 计算箱子总数
fixed_S = float(sys.argv[2])  # 固定的敏感度S
update_prob = float(sys.argv[3])  # 更新概率

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
colldict = {}  # 哈希表

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

    # 将连续的坐标数据离散化
    code_pred_quant = np.digitize(xtest_pred, bins, right=True)

    bitsize = len(code_pred_quant[0])  # 获取哈希码的位数
    total_colliding_in_bench = 0

    # 以步长为7遍历所有状态
    for bini in range(0, len(code_pred_quant), 7):
        predicted = 1  # 默认预测为非碰撞
        true_ans = 1  # 默认真实状态为非碰撞

        # 检查一个运动轨迹中的7个连续状态点
        for i in range(bini, bini + 7):
            keyy = ""
            # 生成哈希键
            for j in range(0, bitsize):
                keyy += str(code_pred_quant[i, j]).zfill(2)

            if keyy in colldict:
                # --- 预测逻辑 ---
                # 基于固定的S值进行预测
                if fixed_S >= 0 and colldict[keyy][0] > (fixed_S * colldict[keyy][1]):
                    predicted = 0
                    if label_pred[i] > 0.5:  # 误报
                        all_onezero += 1
                # 如果S为负数,则执行随机预测策略作为基线
                elif fixed_S < 0:
                    if random.random() < 0.5:
                        predicted = 0
                        if label_pred[i] > 0.5:  # 误报
                            all_onezero += 1

                # 更新哈希表
                if (label_pred[i] > 0.5 and random.random() <= update_prob) or label_pred[i] < 0.5:
                    colldict[keyy][int(label_pred[i].item())] += 1
            else:
                # 如果哈希键不存在,则创建新条目并更新
                if (label_pred[i] > 0.5 and random.random() <= update_prob) or label_pred[i] < 0.5:
                    colldict[keyy] = [0, 0]
                    colldict[keyy][int(label_pred[i].item())] += 1

            if label_pred[i] < 0.5:  # 如果当前点真实为碰撞
                true_ans = 0
                if predicted == 0:  # 如果预测也为碰撞,则是正确预测
                    all_zerozero += 1
                    break  # 提前退出

        if true_ans == 0:
            total_colliding_in_bench += 1
            all_total_colliding += 1

    if total_colliding_in_bench == 0:
        continue

# --- 最终结果计算 ---
# 精确度 = 正确预测的碰撞 / (正确预测的碰撞 + 错误预测的碰撞)
precision = all_zerozero * 100 / (all_zerozero + all_onezero) if (all_zerozero + all_onezero) > 0 else 0
# 召回率 = 正确预测的碰撞 / 所有真实发生的碰撞
recall = all_zerozero * 100 / all_total_colliding if all_total_colliding > 0 else 0

print(f"{precision:.2f},{recall:.2f}")
