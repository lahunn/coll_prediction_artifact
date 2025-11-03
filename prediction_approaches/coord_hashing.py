"""
坐标哈希算法评估脚本
通过离散化坐标空间并构建哈希表来预测机器人运动轨迹的碰撞风险

使用示例：
python coord_hashing.py dens9 8 0.1 0.3 100    # 中等密度场景，8位量化，0.1碰撞阈值，30%自由样本采样率，100个问题
python coord_hashing.py dens12 10 0.05 0.5 50  # 高密度场景，10位量化，0.05碰撞阈值，50%自由样本采样率，50个问题
python coord_hashing.py dens6 6 0.2 0.2 100    # 低密度场景，6位量化，0.2碰撞阈值，20%自由样本采样率，100个问题
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collision_prediction_strategies import (
    FixedThresholdStrategy,
    evaluate_strategy_on_trajectory,
)
from utils.utils import calculate_expected_checks, calculate_baseline_expectation

# 添加 trace_generation 目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from trace_generation.robot_as.ana_parameters import obb_num, obb_cost


def plot(code, ytest, name):
    """绘制二维散点图显示碰撞和非碰撞样本的分布"""
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


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) != 6:
        print(
            "用法: python coord_hashing.py <密度等级> <量化位数> <碰撞阈值> <自由样本采样率> <问题数量>"
        )
        print("示例: python coord_hashing.py dens6 8 0.1 0.3 100")
        sys.exit(1)

    density_level = sys.argv[1]
    quantize_bits = int(sys.argv[2])
    collision_threshold = float(sys.argv[3])
    free_sample_rate = float(sys.argv[4])
    num_problems = int(sys.argv[5])
    num_links = 11

    # 设置量化参数
    binnumber = 2**quantize_bits
    intervalsize = 2.24 / binnumber
    bins = np.zeros(binnumber)
    start = -1.12
    for i in range(0, binnumber):
        bins[i] = start
        start += intervalsize

    # 创建固定阈值策略
    strategy = FixedThresholdStrategy(
        threshold=collision_threshold,
        update_prob=free_sample_rate,
        max_count=255,
    )

    # 主循环：遍历num_problems个基准场景进行评估
    all_labels = []  # 收集所有问题的标签

    for benchid in range(0, num_problems):
        strategy.reset_collision_history()

        benchidstr = str(benchid)
        f = open(
            f"../trace_files/scene_benchmarks/{density_level}_rs/obstacles_{benchidstr}_coord.pkl",
            "rb",
        )
        xtest_pred, dirr_pred, label_pred = pickle.load(f)
        f.close()

        all_labels.append(label_pred)

        code_pred_quant = np.digitize(xtest_pred, bins, right=True)
        evaluate_strategy_on_trajectory(
            strategy, code_pred_quant, label_pred, group_size=num_links
        )

    # 合并所有标签
    all_labels = np.concatenate(all_labels)

    # 输出最终评估指标
    precision, recall, ele_precision, ele_recall = strategy.get_metrics()
    all_collision_ratio, ele_collision_ratio = strategy.get_collision_ratio(all_labels)

    # 计算预期成本（姿态级）
    if ele_precision > 0 and ele_recall > 0 and ele_collision_ratio > 0:
        expected_checks = calculate_expected_checks(
            R=ele_collision_ratio,
            C=ele_recall / 100.0,
            A=ele_precision / 100.0,
            N=obb_num,
        )
        pred_cost = expected_checks * obb_cost
        baseline_checks = calculate_baseline_expectation(
            N=obb_num, R=ele_collision_ratio
        )
        baseline_cost = baseline_checks * obb_cost

        speedup = baseline_cost / pred_cost if pred_cost > 0 else 0
    else:
        pred_cost = float("inf")
        baseline_cost = float("inf")
        speedup = 0

    # 输出姿态级和元素级指标
    print(
        f"{density_level}, {quantize_bits}, {collision_threshold}, {free_sample_rate}, "
        f"Pose: {precision:.2f}%, {recall:.2f}%, {all_collision_ratio:.4f}, "
        f"Elem: {ele_precision:.2f}%, {ele_recall:.2f}%, {ele_collision_ratio:.4f}, "
        f"Cost: {pred_cost:.2f}, {baseline_cost:.2f}, {speedup:.2f}"
    )


if __name__ == "__main__":
    main()
