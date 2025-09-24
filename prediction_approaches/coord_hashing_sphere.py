# 坐标哈希算法评估脚本 - 球体版本
# 通过离散化球体位置和半径空间并构建哈希表来预测机器人运动轨迹的碰撞风险
# 使用球体的位置坐标(x,y,z)和半径作为哈希键值
# 使用命令行参数: <密度等级> <量化位数> <碰撞阈值> <自由样本采样率>

import sys

import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


def plot(code, ytest, name):
    """绘制二维散点图显示碰撞和非碰撞样本的分布"""
    # 从编码数据中提取主成分
    principalComponents = code.data.cpu().numpy()
    # print(principalComponents)
    coll = []  # 碰撞样本
    collfree = []  # 无碰撞样本
    # 根据标签分离碰撞和无碰撞样本
    for i in range(0, len(ytest)):
        if ytest[i] > 0.5:
            collfree.append(principalComponents[i])  # 标签>0.5为无碰撞
        else:
            coll.append(principalComponents[i])  # 标签≤0.5为碰撞
    coll1 = np.array(coll)
    collfree1 = np.array(collfree)
    # 绘制散点图：蓝色为无碰撞，红色为碰撞
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


# 球体哈希算法：使用球体位置(x,y,z)和半径构建哈希键


# 设置量化参数：将连续坐标空间离散化为哈希桶
# distributing the dataset into two components X and Y

# 根据命令行参数计算分桶数量：binnumber = 2^argv[2]
binnumber = 2 ** int(sys.argv[2])
# 计算每个桶的区间大小（总范围2.24，区间[-1.12, 1.12)）
intervalsize = 2.24 / binnumber
bins = np.zeros(binnumber)
start = -1.12
# 构建分桶边界数组
for i in range(0, binnumber):
    bins[i] = start
    start += intervalsize

# 初始化全局累计统计变量
all_onezero = 0  # 全局false positive计数(真实无碰撞但预测碰撞)
all_zerozero = 0  # 全局true positive计数(真实碰撞且预测碰撞)
all_total = 0  # 全局总样本数
all_total_colliding = 0  # 全局真实碰撞总数 len(label_pred)-np.sum(label_pred)
globalcolldict = {}  # 全局碰撞字典(未使用)
colldict = {}  # 当前场景的碰撞统计字典
# print("Total colliding,zerozero,onezero,random_baseline,Prediction_accuracy,Fraction_predicted,link_colliding,link_zerozero,link_onezero")

# 主循环：遍历100个基准场景进行评估
for benchid in range(0, 100):
    benchidstr = str(benchid)
    # 根据密度参数选择不同的数据集 - 修改为读取球体数据
    if sys.argv[1] == "low":
        f = open(
            "../trace_files/scene_benchmarks/moving_1030_10_low/obstacles_"
            + benchidstr
            + "_sphere.pkl",
            "rb",
        )
    elif sys.argv[1] == "mid":
        f = open(
            "../trace_files/scene_benchmarks/moving_1030_10_mid/obstacles_"
            + benchidstr
            + "_sphere.pkl",
            "rb",
        )
    else:
        f = open(
            "../trace_files/scene_benchmarks/moving_1030_10_high/obstacles_"
            + benchidstr
            + "_sphere.pkl",
            "rb",
        )
    # 加载球体数据：球体位置、球体半径、球体碰撞标签
    qarr_sphere, rarr_sphere, yarr_sphere = pickle.load(f)
    # print(qarr_sphere.shape, rarr_sphere.shape, yarr_sphere.shape)
    f.close()

    # 构建球体测试数据
    xtest_pred = qarr_sphere  # 球体位置 [N, 3]
    radius_pred = rarr_sphere  # 球体半径 [N, 1]
    label_pred = yarr_sphere.flatten()  # 球体碰撞标签 [N,]
    # 对球体位置进行量化离散化
    code_pred_quant = np.digitize(xtest_pred, bins, right=True)
    # 对球体半径进行量化离散化 (使用相同的区间)
    radius_pred_quant = np.digitize(radius_pred.flatten(), bins, right=True)
    # 对球体半径进行量化离散化 (使用相同的区间)
    radius_pred_quant = np.digitize(radius_pred.flatten(), bins, right=True)
    # print(len(code_pred_quant))
    # 重置当前场景的碰撞统计字典
    colldict = {}

    # 获取坐标维度数（每个样本的坐标分量数）
    bitsize = len(code_pred_quant[0])
    # 初始化当前场景的统计变量
    prediction_true = 0
    onezero = 0  # false positive (真实自由但预测碰撞)
    zerozero = 0  # true positive (真实碰撞且预测碰撞)
    zeroone = 0  # false negative (真实碰撞但预测自由)
    total_colliding = 0  # 当前场景真实碰撞总数

    # link级别的统计变量
    link_colliding = 0
    link_zerozero = 0
    link_onezero = 0
    all_total += len(code_pred_quant)

    # 按单个球体遍历数据（每个球体独立处理）
    for i in range(len(code_pred_quant)):
        # 初始化预测结果为1（无碰撞）
        predicted = 1
        # 获取真实答案
        true_ans = int(label_pred[i])

        # 构建当前球体的哈希键：位置(x,y,z) + 半径
        keyy = ""
        # 添加球体位置信息到键中
        for j in range(bitsize):  # 位置的x,y,z坐标
            if code_pred_quant[i, j] < 10:
                keyy = keyy + "0"
            keyy = keyy + str(code_pred_quant[i, j])

        # 添加球体半径信息到键中
        if radius_pred_quant[i] < 10:
            keyy = keyy + "0"
        keyy = keyy + str(radius_pred_quant[i])

        # 检查键是否已存在于碰撞字典中
        if keyy in colldict:
            # 判断碰撞阈值：碰撞次数 > 阈值 × 自由次数
            if colldict[keyy][0] > (float(sys.argv[3]) * colldict[keyy][1]):
                predicted = 0  # 预测为碰撞
                if true_ans == 1:  # 真实无碰撞但预测碰撞
                    link_onezero += 1

            # 更新统计（持续学习模式）
            if (
                true_ans == 1 and random.random() <= float(sys.argv[4])
            ) or true_ans == 0:
                colldict[keyy][true_ans] += 1
        else:
            # 新键：初始化统计并按规则更新
            if (
                true_ans == 1 and random.random() <= float(sys.argv[4])
            ) or true_ans == 0:
                colldict[keyy] = [0, 0]  # [碰撞计数, 自由计数]
                colldict[keyy][true_ans] += 1

        # 根据真实值和预测值更新混淆矩阵统计
        if true_ans == 0 and predicted == 0:
            zerozero += 1  # 真正例：真实碰撞且预测碰撞
            all_zerozero += 1
            link_zerozero += 1
        elif true_ans == 1 and predicted == 0:
            onezero += 1  # 假正例：真实无碰撞但预测碰撞
            all_onezero += 1
        elif true_ans == 0 and predicted == 1:
            zeroone += 1  # 假负例：真实碰撞但预测无碰撞

        # 统计真实碰撞总数和连杆碰撞
        if true_ans == 0:
            total_colliding += 1
            all_total_colliding += 1
            link_colliding += 1

    # 过滤条件：跳过没有碰撞或没有正确预测碰撞的场景
    if total_colliding == 0 or zerozero == 0:
        continue

# 输出最终评估指标
# 精确率 = TP / (TP + FP) * 100%
# 召回率 = TP / (TP + FN) * 100% = TP / 总碰撞数 * 100%
print(
    "%.2f,%.2f"
    % (
        all_zerozero * 100 / (all_zerozero + all_onezero),
        all_zerozero * 100 / all_total_colliding,
    )
)
