# 坐标哈希算法评估脚本 - 球体版本
# 通过离散化球体位置和半径空间并构建哈希表来预测机器人运动轨迹的碰撞风险
# 使用球体的位置坐标(x,y,z)和半径作为哈希键值
# 使用命令行参数: <密度等级> <坐标量化位数> <半径量化位数> <碰撞阈值> <自由样本采样率>

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


# 解析命令行参数
if len(sys.argv) != 6:
    print(
        "用法: python coord_hashing_sphere.py <密度等级> <坐标量化位数> <半径量化位数> <碰撞阈值> <自由样本采样率>"
    )
    print("示例: python coord_hashing_sphere.py mid 8 6 0.1 0.3")
    sys.exit(1)

# 解析命令行参数到变量
density_level = sys.argv[1]  # 密度等级: "low", "mid", "high"
coord_quantize_bits = int(sys.argv[2])  # 坐标量化位数 (如: 8)
radius_quantize_bits = int(sys.argv[3])  # 半径量化位数 (如: 6)
collision_threshold = float(sys.argv[4])  # 碰撞阈值 (如: 0.1)
free_sample_rate = float(sys.argv[5])  # 自由样本采样率 (如: 0.3)

# print(
#     f"参数设置: 密度={density_level}, 坐标量化={coord_quantize_bits}位, 半径量化={radius_quantize_bits}位, "
#     + f"碰撞阈值={collision_threshold}, 采样率={free_sample_rate}"
# )


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

# 控制参数：是否考虑球体半径
consider_radius = False  # True: 哈希键包含半径信息，False: 仅使用位置信息


# 设置量化参数：将连续坐标空间离散化为哈希桶
# 首先收集所有场景的数据来确定坐标和半径的范围
# print("正在计算坐标和半径的数据范围...")

all_positions = []
all_radii = []

# 遍历所有场景收集数据范围
for benchid in range(0, 100):
    benchidstr = str(benchid)
    try:
        if density_level == "low":
            f = open(
                "../trace_generation/scene_benchmarks/dens3_rs/obstacles_"
                + benchidstr
                + "_sphere.pkl",
                "rb",
            )
        elif density_level == "mid":
            f = open(
                "../trace_generation/scene_benchmarks/dens3_rs/obstacles_"
                + benchidstr
                + "_sphere.pkl",
                "rb",
            )
        else:
            f = open(
                "../trace_generation/scene_benchmarks/dens3_rs/obstacles_"
                + benchidstr
                + "_sphere.pkl",
                "rb",
            )

        qarr_sphere, rarr_sphere, yarr_sphere = pickle.load(f)
        f.close()

        all_positions.append(qarr_sphere)
        all_radii.append(rarr_sphere.flatten())
    except FileNotFoundError:
        continue

# 合并所有数据
all_positions = np.vstack(all_positions)  # [N_total, 3]
all_radii = np.concatenate(all_radii)  # [N_total,]

# 计算每个坐标轴的范围
x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])
r_min, r_max = np.min(all_radii), np.max(all_radii)

# print(f"X轴范围: [{x_min:.4f}, {x_max:.4f}]")
# print(f"Y轴范围: [{y_min:.4f}, {y_max:.4f}]")
# print(f"Z轴范围: [{z_min:.4f}, {z_max:.4f}]")
# print(f"半径范围: [{r_min:.4f}, {r_max:.4f}]")

# 根据命令行参数计算分桶数量
# 坐标分桶数量: binnumber_coord = 2^coord_quantize_bits
# 半径分桶数量: binnumber_radius = 2^radius_quantize_bits
binnumber_coord = 2**coord_quantize_bits
binnumber_radius = 2**radius_quantize_bits


# 为每个坐标轴创建独立的分桶边界
def create_bins(min_val, max_val, num_bins):
    """创建等间距的分桶边界"""
    # 添加小量边距避免边界值问题
    margin = (max_val - min_val) * 0.01
    return np.linspace(min_val - margin, max_val + margin, num_bins + 1)[:-1]


# 创建各轴独立的分桶边界
x_bins = create_bins(x_min, x_max, binnumber_coord)
y_bins = create_bins(y_min, y_max, binnumber_coord)
z_bins = create_bins(z_min, z_max, binnumber_coord)
r_bins = create_bins(r_min, r_max, binnumber_radius)

# print(f"坐标轴使用 {binnumber_coord} 个桶, 半径使用 {binnumber_radius} 个桶进行离散化")

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
    if density_level == "low":
        f = open(
            "../trace_generation/scene_benchmarks/dens3_rs/obstacles_"
            + benchidstr
            + "_sphere.pkl",
            "rb",
        )
    elif density_level == "mid":
        f = open(
            "../trace_generation/scene_benchmarks/dens3_rs/obstacles_"
            + benchidstr
            + "_sphere.pkl",
            "rb",
        )
    else:
        f = open(
            "../trace_generation/scene_benchmarks/dens3_rs/obstacles_"
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

    # 对球体位置进行分轴量化离散化
    code_pred_quant = np.zeros_like(xtest_pred, dtype=int)
    code_pred_quant[:, 0] = np.digitize(xtest_pred[:, 0], x_bins, right=True)  # X轴
    code_pred_quant[:, 1] = np.digitize(xtest_pred[:, 1], y_bins, right=True)  # Y轴
    code_pred_quant[:, 2] = np.digitize(xtest_pred[:, 2], z_bins, right=True)  # Z轴

    # 对球体半径进行独立量化离散化
    radius_pred_quant = np.digitize(radius_pred.flatten(), r_bins, right=True)
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

        # 构建当前球体的哈希键：位置(x,y,z) + 半径(可选)
        keyy = ""
        # 添加球体位置信息到键中
        for j in range(bitsize):  # 位置的x,y,z坐标
            if code_pred_quant[i, j] < 10:
                keyy = keyy + "0"
            keyy = keyy + str(code_pred_quant[i, j])

        # 根据全局变量决定是否添加球体半径信息到键中
        if consider_radius:
            if radius_pred_quant[i] < 10:
                keyy = keyy + "0"
            keyy = keyy + str(radius_pred_quant[i])

        # 检查键是否已存在于碰撞字典中
        if keyy in colldict:
            # 判断碰撞阈值：碰撞次数 > 阈值 × 自由次数
            if colldict[keyy][0] > (collision_threshold * colldict[keyy][1]):
                predicted = 0  # 预测为碰撞
                if true_ans == 1:  # 真实无碰撞但预测碰撞
                    link_onezero += 1

            # 更新统计（持续学习模式）
            if (true_ans == 1 and random.random() <= free_sample_rate) or true_ans == 0:
                colldict[keyy][true_ans] += 1
        else:
            # 新键：初始化统计并按规则更新
            if (true_ans == 1 and random.random() <= free_sample_rate) or true_ans == 0:
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

# 计算精确率和召回率
precision = (
    all_zerozero * 100 / (all_zerozero + all_onezero)
    if (all_zerozero + all_onezero) > 0
    else 0
)
recall = all_zerozero * 100 / all_total_colliding if all_total_colliding > 0 else 0


# 输出详细结果：参数设置和性能指标
# print("密度, 坐标量化位数, 半径量化位数, 碰撞阈值, 采样率, 精确率, 召回率")

print(
    f"{density_level}, {coord_quantize_bits}, {radius_quantize_bits}, {collision_threshold}, {free_sample_rate}, {precision:.2f}%, {recall:.2f}%"
)
