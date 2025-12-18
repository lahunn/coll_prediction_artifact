"""
坐标哈希算法评估脚本 - 球体版本
通过离散化球体位置和半径空间并构建哈希表来预测机器人运动轨迹的碰撞风险
使用球体的位置坐标(x,y,z)和半径作为哈希键值

使用示例：
python coord_hashing_sphere.py <密度等级> <坐标量化位数> <半径量化位数> <碰撞阈值> <自由样本采样率> <问题数量>"
python coord_hashing_sphere.py dens9 8 6 0.1 0.3 100
python coord_hashing_sphere.py dens6 4 2 0.05 0.5 50
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collision_prediction_strategies import (
    FixedThresholdStrategy,
    evaluate_strategy_on_spheres,
)
from utils.utils import calculate_expected_checks, calculate_baseline_expectation

# 添加 trace_generation 目录到 Python 路径
from trace_generation.config.ana_parameters import get_robot_params


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


def create_bins(min_val, max_val, num_bins):
    """创建等间距的分桶边界"""
    margin = (max_val - min_val) * 0.01
    return np.linspace(min_val - margin, max_val + margin, num_bins + 1)[:-1]


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) < 7 or len(sys.argv) > 8:
        print(
            "用法: python coord_hashing_sphere.py <密度等级> <坐标量化位数> <半径量化位数> <碰撞阈值> <自由样本采样率> <问题数量> [机器人名称]"
        )
        print("示例: python coord_hashing_sphere.py dens6 8 6 0.1 0.3 100 franka")
        sys.exit(1)

    density_level = sys.argv[1]
    coord_quantize_bits = int(sys.argv[2])
    radius_quantize_bits = int(sys.argv[3])
    collision_threshold = float(sys.argv[4])
    free_sample_rate = float(sys.argv[5])
    num_problems = int(sys.argv[6])
    robot_name = sys.argv[7] if len(sys.argv) == 8 else "franka"

    # 获取机器人参数
    robot_params = get_robot_params(robot_name)
    sphere_num = robot_params["sphere_num"]
    sphere_cost = robot_params["sphere_cost"]

    # consider_radius = False
    consider_radius = True

    # 收集所有场景的数据来确定坐标和半径的范围
    all_positions = []
    all_radii = []

    for benchid in range(0, num_problems):
        benchidstr = str(benchid)
        f = open(
            f"../trace_files/scene_benchmarks/{density_level}_rs/obstacles_{benchidstr}_sphere.pkl",
            "rb",
        )

        qarr_sphere, rarr_sphere, yarr_sphere = pickle.load(f)
        f.close()

        all_positions.append(qarr_sphere)
        all_radii.append(rarr_sphere.flatten())

    # 合并所有数据
    all_positions = np.vstack(all_positions)
    all_radii = np.concatenate(all_radii)

    # 计算坐标范围（统一量化）
    coord_min, coord_max = np.min(all_positions), np.max(all_positions)
    r_min, r_max = np.min(all_radii), np.max(all_radii)

    # 计算分桶数量
    binnumber_coord = 2**coord_quantize_bits
    binnumber_radius = 2**radius_quantize_bits

    # 创建统一的坐标分桶边界
    coord_bins = create_bins(coord_min, coord_max, binnumber_coord)
    r_bins = create_bins(r_min, r_max, binnumber_radius)

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
            f"../trace_files/scene_benchmarks/{density_level}_rs/obstacles_{benchidstr}_sphere.pkl",
            "rb",
        )
        qarr_sphere, rarr_sphere, yarr_sphere = pickle.load(f)
        f.close()

        # 构建球体测试数据
        xtest_pred = qarr_sphere
        radius_pred = rarr_sphere
        label_pred = yarr_sphere.flatten()
        all_labels.append(label_pred)

        # 对球体位置进行统一量化离散化
        code_pred_quant = np.digitize(xtest_pred, coord_bins, right=True)

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
            N=sphere_num,
        )
        pred_cost = expected_checks * sphere_cost

        baseline_checks = sphere_num
        baseline_cost = baseline_checks * sphere_cost

        # Compute cost ratio relative to baseline; report as percentage
        speedup_pct = (
            (pred_cost / baseline_cost * 100.0) if baseline_cost > 0 else float("inf")
        )
    else:
        pred_cost = float("inf")
        baseline_cost = float("inf")
        speedup_pct = 0

    # 输出姿态级和元素级指标
    print(
        f"{density_level}, {coord_quantize_bits}, {radius_quantize_bits}, {collision_threshold}, {free_sample_rate}, "
        f"Pose: {precision:.2f}%, {recall:.2f}%, {all_collision_ratio:.4f}, "
        f"Elem: {ele_precision:.2f}%, {ele_recall:.2f}%, {ele_collision_ratio:.4f}, "
        f"Cost: {pred_cost:.2f}, {baseline_cost:.2f}, {speedup_pct:.2f}%"
    )


if __name__ == "__main__":
    main()
