"""
编码坐标哈希碰撞预测脚本 (Encoded Coordinate Hashing)

与coord_hashing.py的区别:
- coord_hashing.py: 直接使用原始关节坐标
- encoord_hashing.py: 使用神经网络编码后的低维特征

命令行参数:
    sys.argv[1]: 密度等级 ("low", "mid", "high")
    sys.argv[2]: 量化位数 (如: 8)
    sys.argv[3]: 碰撞阈值 (如: 0.1)
    sys.argv[4]: 自由样本采样率 (如: 0.3)
    sys.argv[5]: 编码器类型/ID (用于加载对应的编码文件)

示例: python encoord_hashing.py mid 8 0.1 0.3 encoder1
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


def plot(code, ytest, name):
    """
    绘制编码空间中的碰撞和非碰撞样本分布

    这是一个可视化辅助函数，用于理解编码器学到的特征空间

    Args:
        code: 编码后的特征 (通常是神经网络的输出)
        ytest: 真实标签 (>0.5为自由, <=0.5为碰撞)
        name: 保存图像的文件名
    """
    principalComponents = code.data.cpu().numpy()
    # print(principalComponents)
    coll = []  # 碰撞样本
    collfree = []  # 自由样本

    # 根据标签分离碰撞和非碰撞样本
    for i in range(0, len(ytest)):
        if ytest[i] > 0.5:
            collfree.append(principalComponents[i])
        else:
            coll.append(principalComponents[i])

    coll1 = np.array(coll)
    collfree1 = np.array(collfree)

    # 绘制散点图：蓝色=自由, 红色=碰撞
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


# ========== 配置参数 ==========
consider_dir = False  # 是否考虑运动方向信息 (当前未使用)

# ========== 设置量化参数 ==========
# 将编码后的连续特征空间离散化为哈希桶
# 量化的目的: 将相似的配置映射到同一个哈希键

binnumber = 2 ** int(sys.argv[2])  # 桶数量 = 2^量化位数 (如: 2^8=256个桶)
intervalsize = 2.24 / binnumber  # 每个桶的区间大小
bins = np.zeros(binnumber)  # 分桶边界数组

# 构建分桶边界: 假设编码特征范围是[-1.12, 1.12]
start = -1.12
for i in range(0, binnumber):
    bins[i] = start
    start += intervalsize

# ========== 全局统计变量 (累积所有100个benchmark的结果) ==========
all_onezero = 0  # 误报总数 (False Positives): 预测碰撞但实际自由
all_zerozero = 0  # 真阳性总数 (True Positives): 正确预测碰撞
all_total = 0  # 总样本数
all_total_colliding = 0  # 真实碰撞总数 (用于计算召回率)

globalcolldict = {}  # 全局碰撞字典 (未使用)
colldict = {}  # 当前benchmark的碰撞历史表 (CHT)
# 格式: {hash_key: [碰撞次数, 自由次数]}

# ========== 主循环: 遍历100个基准场景 ==========
for benchid in range(0, 100):
    benchidstr = str(benchid)

    # ========== 根据密度等级选择数据文件 ==========
    # 加载两个文件:
    # 1. f: 原始坐标和标签数据 (用于获取label)
    # 2. f1: 神经网络编码后的特征 (用于哈希)
    if sys.argv[1] == "low":
        f = open(
            "../trace_files/scene_benchmarks/moving_1030_10_low/obstacles_"
            + benchidstr
            + "_coord.pkl",
            "rb",
        )
        f1 = open(
            "../trace_files/scene_benchmarks/low_obstacle_encord/encodecoord_"
            + sys.argv[5]  # 编码器类型/ID
            + "_"
            + benchidstr
            + ".pkl",
            "rb",
        )
    elif sys.argv[1] == "mid":
        f = open(
            "../trace_files/scene_benchmarks/moving_1030_10_mid/obstacles_"
            + benchidstr
            + "_coord.pkl",
            "rb",
        )
        f1 = open(
            "../trace_files/scene_benchmarks/low_obstacle_encord/encodecoord_"
            + sys.argv[5]
            + "_"
            + benchidstr
            + ".pkl",
            "rb",
        )
    else:  # high
        f = open(
            "../trace_files/scene_benchmarks/moving_1030_10_high/obstacles_"
            + benchidstr
            + "_coord.pkl",
            "rb",
        )
        f1 = open(
            "../trace_files/scene_benchmarks/high_obstacle_encord/encodecoord_"
            + sys.argv[5]
            + "_"
            + benchidstr
            + ".pkl",
            "rb",
        )

    # ========== 加载数据 ==========
    xtest_pred, dirr_pred, label_pred = pickle.load(f)  # 原始坐标、方向、标签
    code = pickle.load(f1)  # 神经网络编码后的特征 (低维表示)
    f.close()
    f1.close()

    # ========== 量化编码特征 ==========
    # 将连续的编码特征离散化到预定义的桶中
    code_pred_quant = np.digitize(code, bins, right=True)

    # ========== 重置当前benchmark的CHT ==========
    # 注意: 与coord_hashing.py不同,这里每个benchmark独立评估
    colldict = {}  # 碰撞历史表 {hash_key: [碰撞次数, 自由次数]}

    # ========== 初始化本benchmark的统计变量 ==========
    bitsize = len(code_pred_quant[0])  # 编码特征的维度
    prediction_true = 0  # 预测正确数 (未使用)
    onezero = 0  # 误报数 (FP)
    zerozero = 0  # 真阳性数 (TP)
    zeroone = 0  # 漏报数 (FN)
    total_colliding = 0  # 真实碰撞数

    link_colliding = 0  # 链接级别的碰撞数 (未使用)
    link_zerozero = 0  # 链接级别的TP (未使用)
    link_onezero = 0  # 链接级别的FP (未使用)

    all_total += len(code_pred_quant)  # 累加总样本数

    # ========== 轨迹级别评估: 以7个状态为一组 ==========
    # 注意: 这里group_size=7,与coord_hashing.py的11不同
    # 可能因为使用编码特征后,需要检查的状态点更少
    for bini in range(0, len(code_pred_quant), 7):
        # if bini>=2800:
        #   break
        predicted = 1  # 初始假设: 预测为自由(非碰撞)
        true_ans = 1  # 初始假设: 真实为自由

        # ========== 检查一组7个连续状态点 ==========
        for i in range(bini, bini + 7):
            # ========== 生成哈希键 ==========
            # 将量化后的编码特征转换为字符串键
            # 例如: [1, 5, 10] -> "010510"
            keyy = ""
            for j in range(0, bitsize):
                if code_pred_quant[i, j] < 10:
                    keyy = keyy + "0"  # 补零,保证每个维度占2位
                keyy = keyy + str(code_pred_quant[i, j])

            # 可选: 添加运动方向信息到键中
            if consider_dir:
                keyy = keyy + dirr_pred[i]

            # ========== 碰撞预测和CHT更新 ==========
            if keyy in colldict:
                # 键已存在: 使用CHT进行预测
                # 预测规则: 碰撞次数 > 阈值 × 自由次数
                # if colldict[keyy][0]>0:#colldict[keyy][1]:
                if colldict[keyy][0] > (
                    float(sys.argv[3]) * colldict[keyy][1]
                ):  # 碰撞占优
                    # print(colldict[keyy],keyy)
                    predicted = 0  # 预测为碰撞
                    if label_pred[i] > 0.5:
                        link_onezero += 1  # 误报

                # ========== 更新CHT (增量学习) ==========
                # 采样策略:
                # - 碰撞样本: 总是更新
                # - 自由样本: 以sys.argv[4]的概率采样更新
                if (
                    label_pred[i] > 0.5 and random.random() <= float(sys.argv[4])
                ) or label_pred[i] < 0.5:
                    colldict[keyy][int(label_pred[i])] += 1
            else:
                # 键不存在: 初始化CHT条目
                if (
                    label_pred[i] > 0.5 and random.random() <= float(sys.argv[4])
                ) or label_pred[i] < 0.5:
                    colldict[keyy] = [0, 0]
                    colldict[keyy][int(label_pred[i])] += 1

            # ========== 检查真实标签 ==========
            if label_pred[i] < 0.5:
                # 真实为碰撞
                true_ans = 0
                link_colliding += 1
                if predicted == 0:
                    # 预测也为碰撞 (TP)
                    link_zerozero += 1
                    break  # 找到碰撞,提前退出该组
        # ========== 统计该组的预测结果 ==========
        # print(keyy,predicted)
        if true_ans == 0 and predicted == 0:
            # True Positive: 正确预测碰撞
            zerozero += 1
            all_zerozero += 1
        elif true_ans == 1 and predicted == 0:
            # False Positive: 误报为碰撞
            onezero += 1
            all_onezero += 1
            # print(colldict[keyy])
        elif true_ans == 0 and predicted == 1:
            # False Negative: 漏报碰撞
            zeroone += 1

        if true_ans == 0:
            # 累计真实碰撞数
            total_colliding += 1
            all_total_colliding += 1

    # ========== 跳过无效benchmark ==========
    # 如果该benchmark没有碰撞或没有预测到碰撞,跳过
    if total_colliding == 0 or zerozero == 0:
        continue

# ========== 输出最终结果 ==========
# 格式: 精确率, 召回率
print(
    "%.2f,%.2f"
    % (
        all_zerozero * 100 / (all_zerozero + all_onezero),  # 精确率 = TP / (TP + FP)
        all_zerozero * 100 / all_total_colliding,  # 召回率 = TP / (TP + FN)
    )
)


# ========== 代码总结 ==========
"""
核心流程:
1. 加载神经网络编码后的特征 (code)
2. 量化编码特征到离散桶中
3. 生成哈希键并建立CHT
4. 使用CHT进行碰撞预测
5. 增量更新CHT
6. 统计精确率和召回率

与coord_hashing.py的主要区别:
1. 输入数据: 编码特征 vs 原始坐标
2. Group size: 7 vs 11
3. 特征空间: 低维编码 vs 高维原始空间
4. 预期: 编码特征可能提供更好的泛化能力

优点:
- 编码特征可能捕获更抽象的碰撞模式
- 低维空间可能需要更小的CHT
- 神经网络学到的特征可能更适合预测

缺点:
- 依赖预训练的编码器
- 编码器质量影响最终性能
- 增加了系统复杂度
"""
