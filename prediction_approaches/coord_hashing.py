# 坐标哈希算法评估脚本
# 通过离散化坐标空间并构建哈希表来预测机器人运动轨迹的碰撞风险
# 使用命令行参数: <密度等级> <量化位数> <碰撞阈值> <自由样本采样率>

# 使用示例：
# python coord_hashing.py mid 8 0.1 0.3    # 中等密度场景，8位量化，0.1碰撞阈值，30%自由样本采样率
# python coord_hashing.py high 10 0.05 0.5  # 高密度场景，10位量化，0.05碰撞阈值，50%自由样本采样率
# python coord_hashing.py low 6 0.2 0.2    # 低密度场景，6位量化，0.2碰撞阈值，20%自由样本采样率
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collision_prediction_strategies import (
    FixedThresholdStrategy,
    evaluate_strategy_on_trajectory,
)

# 解析命令行参数
if len(sys.argv) != 5:
    print(
        "用法: python coord_hashing.py <密度等级> <量化位数> <碰撞阈值> <自由样本采样率>"
    )
    print("示例: python coord_hashing.py mid 8 0.1 0.3")
    sys.exit(1)

# 解析命令行参数到变量
density_level = sys.argv[1]  # 密度等级: "low", "mid", "high"
quantize_bits = int(sys.argv[2])  # 量化位数 (如: 8)
collision_threshold = float(sys.argv[3])  # 碰撞阈值 (如: 0.1)
free_sample_rate = float(sys.argv[4])  # 自由样本采样率 (如: 0.3)
num_links = 11

# print(
#     f"参数设置: 密度={density_level}, 量化={quantize_bits}位, 碰撞阈值={collision_threshold}, "
#     + f"采样率={free_sample_rate}, 链接数={num_links}"
# )


# 是否考虑运动方向（当前设为False，仅考虑位置）
consider_dir = False

# 从解析的变量获取链接数
# num_links = 11  # 默认11个链接（已在参数解析中定义）


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


# 设置量化参数：将连续坐标空间离散化为哈希桶

# 根据解析的参数计算分桶数量：binnumber = 2^quantize_bits
binnumber = 2**quantize_bits
# 计算每个桶的区间大小（总范围2.24，区间[-1.12, 1.12)）
intervalsize = 2.24 / binnumber
bins = np.zeros(binnumber)
start = -1.12
# 构建分桶边界数组
for i in range(0, binnumber):
    bins[i] = start
    start += intervalsize

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

    benchidstr = str(benchid)
    # 根据密度参数选择不同的数据集
    if density_level == "low":
        f = open(
            "../trace_generation/scene_benchmarks/dens6_rs/obstacles_"
            + benchidstr
            + "_coord.pkl",
            "rb",
        )
        # f=open("../trace_generation/scene_benchmarks/dens6/obstacles_"+benchidstr+"_coord.pkl","rb")
    elif density_level == "mid":
        f = open(
            "../trace_generation/scene_benchmarks/dens9_rs/obstacles_"
            + benchidstr
            + "_coord.pkl",
            "rb",
        )
    else:
        f = open(
            "../trace_generation/scene_benchmarks/dens12_rs/obstacles_"
            + benchidstr
            + "_coord.pkl",
            "rb",
        )
    ##f=open("../trace_files/scene_benchmarks/moving_3050_10_mid/obstacles_"+benchidstr+"_coord.pkl","rb")
    # 加载测试数据：坐标、方向、碰撞标签
    xtest_pred, dirr_pred, label_pred = pickle.load(f)
    # print(xtest_pred,label_pred)
    f.close()
    # 对坐标进行量化离散化
    code_pred_quant = np.digitize(xtest_pred, bins, right=True)

    # 使用策略评估轨迹
    evaluate_strategy_on_trajectory(
        strategy, code_pred_quant, label_pred, group_size=num_links
    )

# 输出最终评估指标
# 计算精确率和召回率
precision, recall = strategy.get_metrics()

# 输出详细结果：参数设置和性能指标
print(
    f"{density_level}, {quantize_bits}, {collision_threshold}, {free_sample_rate},  {precision:.2f}%, {recall:.2f}%"
)
