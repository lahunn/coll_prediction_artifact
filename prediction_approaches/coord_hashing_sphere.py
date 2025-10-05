# 坐标哈希算法评估脚本 - 球体版本
# 通过离散化球体位置和半径空间并构建哈希表来预测机器人运动轨迹的碰撞风险
# 使用球体的位置坐标(x,y,z)和半径作为哈希键值
# 使用命令行参数: <密度等级> <坐标量化位数> <半径量化位数> <碰撞阈值> <自由样本采样率>

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collision_prediction_strategies import (
    FixedThresholdStrategy,
    evaluate_strategy_on_spheres,
)


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
    if density_level == "low":
        f = open(
            "../trace_generation/scene_benchmarks/dens6_rs/obstacles_"
            + benchidstr
            + "_sphere.pkl",
            "rb",
        )
    elif density_level == "mid":
        f = open(
            "../trace_generation/scene_benchmarks/dens9_rs/obstacles_"
            + benchidstr
            + "_sphere.pkl",
            "rb",
        )
    else:
        f = open(
            "../trace_generation/scene_benchmarks/dens12_rs/obstacles_"
            + benchidstr
            + "_sphere.pkl",
            "rb",
        )

    qarr_sphere, rarr_sphere, yarr_sphere = pickle.load(f)
    f.close()

    all_positions.append(qarr_sphere)
    all_radii.append(rarr_sphere.flatten())

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

# 创建固定阈值策略
strategy = FixedThresholdStrategy(
    threshold=collision_threshold,
    update_prob=free_sample_rate,
    max_count=255,  # 8-bit SRAM存储
)

# 主循环：遍历100个基准场景进行评估
for benchid in range(0, 100):
    # 🔑 修复方案2: 重置strategy的历史和统计 (每个benchmark独立评估)
    strategy.reset_collision_history()  # 清空colldict
    # strategy.reset_statistics()  # 重置统计变量

    benchidstr = str(benchid)
    # 根据密度参数选择不同的数据集 - 修改为读取球体数据
    if density_level == "low":
        f = open(
            "../trace_generation/scene_benchmarks/dens6_rs/obstacles_"
            + benchidstr
            + "_sphere.pkl",
            "rb",
        )
    elif density_level == "mid":
        f = open(
            "../trace_generation/scene_benchmarks/dens9_rs/obstacles_"
            + benchidstr
            + "_sphere.pkl",
            "rb",
        )
    else:
        f = open(
            "../trace_generation/scene_benchmarks/dens12_rs/obstacles_"
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

    # 使用策略评估球体
    evaluate_strategy_on_spheres(
        strategy,
        code_pred_quant,
        radius_pred_quant,
        label_pred,
        consider_radius=consider_radius,
    )

# 输出最终评估指标
# 计算精确率和召回率
precision, recall = strategy.get_metrics()

# 输出详细结果：参数设置和性能指标
print(
    f"{density_level}, {coord_quantize_bits}, {radius_quantize_bits}, {collision_threshold}, {free_sample_rate}, {precision:.2f}%, {recall:.2f}%"
)
