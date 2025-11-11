#!/usr/bin/env python3
"""
碰撞预测策略对比测试脚本
比较固定阈值策略和自适应阈值策略的性能
"""

import sys
import os
import numpy as np
import pickle
from collision_prediction_strategies import (
    FixedThresholdStrategy,
    AdaptiveThresholdStrategy,
    evaluate_strategy_on_trajectory,
    find_sim_cost,
)


def load_benchmark_data(benchid, density="low"):
    """
    加载基准测试数据

    Args:
        benchid: 基准测试ID
        density: 密度级别 ("low", "mid", "high")

    Returns:
        tuple: (xtest_pred, dirr_pred, label_pred) 或 None
    """
    benchidstr = str(benchid)

    if density == "low":
        trace_path = f"../trace_generation/scene_benchmarks/dens3_rs/obstacles_{benchidstr}_coord.pkl"
    elif density == "mid":
        trace_path = f"../trace_generation/scene_benchmarks/dens6_rs/obstacles_{benchidstr}_coord.pkl"
    else:
        trace_path = f"../trace_generation/scene_benchmarks/dens9_rs/obstacles_{benchidstr}_coord.pkl"

    if not os.path.exists(trace_path):
        return None

    with open(trace_path, "rb") as f:
        xtest_pred, dirr_pred, label_pred = pickle.load(f)

    return xtest_pred, dirr_pred, label_pred


def quantize_coordinates(xtest_pred, num_bins=32):
    """
    量化坐标数据

    Args:
        xtest_pred: 原始坐标数据
        num_bins: 分桶数量

    Returns:
        np.ndarray: 量化后的坐标
    """
    intervalsize = 2.24 / num_bins
    bins = np.zeros(num_bins)
    start = -1.12

    for i in range(num_bins):
        bins[i] = start
        start += intervalsize

    code_pred_quant = np.digitize(xtest_pred, bins, right=True)
    return code_pred_quant


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) < 5:
        print(
            "用法: python test_strategies.py <bin_bits> <固定阈值> <S_min> <S_max> [update_prob]"
        )
        print("示例: python test_strategies.py 5 0.1 0.01 1.0 0.5")
        sys.exit(1)

    bin_bits = int(sys.argv[1])
    fixed_threshold = float(sys.argv[2])
    s_min = float(sys.argv[3])
    s_max = float(sys.argv[4])
    update_prob = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

    num_bins = 2**bin_bits

    print("=" * 70)
    print("碰撞预测策略对比测试")
    print("=" * 70)
    print("参数配置:")
    print(f"  - 分桶数量: {num_bins} (2^{bin_bits})")
    print(f"  - 固定阈值: {fixed_threshold}")
    print(f"  - 自适应阈值范围: [{s_min}, {s_max}]")
    print(f"  - 更新概率: {update_prob}")
    print("=" * 70)

    # 创建两种策略
    fixed_strategy = FixedThresholdStrategy(
        threshold=fixed_threshold, update_prob=update_prob
    )

    adaptive_strategy = AdaptiveThresholdStrategy(
        s_min=s_min, s_max=s_max, update_prob=update_prob
    )

    # 测试场景：前50个低密度，后50个高密度
    print("\n开始评估...")
    print(
        f"{'ID':<5} {'密度':<8} {'固定-精确率':<12} {'固定-召回率':<12} {'固定-成本':<10} "
        f"{'自适应-精确率':<14} {'自适应-召回率':<14} {'自适应-成本':<12} {'自适应阈值':<12} {'碰撞占优比':<12}"
    )
    print("-" * 130)

    # 记录每个场景的collision_dominant_ratio和计算成本
    ratio_history = []
    cost_history = []  # 存储 (benchid, density, fixed_cost, adaptive_cost)

    for benchid in range(0, 100):
        # 确定密度
        density = "low" if benchid < 50 else "high"

        # 加载数据
        data = load_benchmark_data(benchid, density)
        if data is None:
            continue

        xtest_pred, dirr_pred, label_pred = data

        # 量化坐标
        code_pred_quant = quantize_coordinates(xtest_pred, num_bins)

        # 评估固定阈值策略
        evaluate_strategy_on_trajectory(
            fixed_strategy, code_pred_quant, label_pred, group_size=11
        )

        # 评估自适应阈值策略
        evaluate_strategy_on_trajectory(
            adaptive_strategy, code_pred_quant, label_pred, group_size=11
        )

        # 获取评估指标
        fixed_prec, fixed_rec = fixed_strategy.get_metrics()
        adaptive_prec, adaptive_rec = adaptive_strategy.get_metrics()

        # 计算真实碰撞率 R
        fixed_collision_ratio = fixed_strategy.get_collision_ratio()
        adaptive_collision_ratio = adaptive_strategy.get_collision_ratio()

        # 计算计算成本 (使用 find_sim_cost)
        # find_sim_cost(R, C, A, N) 其中 R=碰撞率, C=召回率/100, A=精确率/100, N=11
        fixed_cost = 0.0
        adaptive_cost = 0.0

        if fixed_prec > 0 and fixed_rec > 0 and fixed_collision_ratio > 0:
            fixed_cost = find_sim_cost(
                R=fixed_collision_ratio, C=fixed_rec / 100.0, A=fixed_prec / 100.0, N=11
            )

        if adaptive_prec > 0 and adaptive_rec > 0 and adaptive_collision_ratio > 0:
            adaptive_cost = find_sim_cost(
                R=adaptive_collision_ratio,
                C=adaptive_rec / 100.0,
                A=adaptive_prec / 100.0,
                N=11,
            )

        # 记录collision_dominant_ratio和成本
        ratio = adaptive_strategy.get_collision_dominant_ratio()
        ratio_history.append((benchid, density, ratio))
        cost_history.append((benchid, density, fixed_cost, adaptive_cost))

        # 每10个场景输出一次结果
        if (benchid + 1) % 10 == 0:
            adaptive_threshold = adaptive_strategy.get_current_threshold()

            print(
                f"{benchid:<5} {density:<8} {fixed_prec:>10.2f}% {fixed_rec:>11.2f}% {fixed_cost:>9.2f} "
                f"{adaptive_prec:>12.2f}% {adaptive_rec:>13.2f}% {adaptive_cost:>11.2f} "
                f"{adaptive_threshold:>11.4f} {ratio:>11.4f}"
            )
        fixed_strategy.reset_collision_history()
        adaptive_strategy.reset_collision_history()
        fixed_strategy.reset_statistics()
        adaptive_strategy.reset_statistics()

    # 最终结果
    print("=" * 130)

    # 计算平均成本
    all_fixed_costs = [c for _, _, c, _ in cost_history if c > 0]
    all_adaptive_costs = [c for _, _, _, c in cost_history if c > 0]

    avg_fixed_cost = np.mean(all_fixed_costs) if all_fixed_costs else 0.0
    avg_adaptive_cost = np.mean(all_adaptive_costs) if all_adaptive_costs else 0.0

    print("\n最终结果 (所有场景平均):")
    print("  固定阈值策略:")
    print(f"    - 平均计算成本: {avg_fixed_cost:.4f}")
    print("  自适应阈值策略:")
    print(f"    - 平均计算成本: {avg_adaptive_cost:.4f}")

    if avg_fixed_cost > 0 and avg_adaptive_cost > 0:
        improvement = ((avg_fixed_cost - avg_adaptive_cost) / avg_fixed_cost) * 100
        print(f"\n  成本改善: {improvement:.2f}%")

    # 统计计算成本的变化
    print("\n" + "=" * 70)
    print("计算成本统计:")
    print("=" * 70)

    # 按密度分组统计成本
    low_fixed_costs = [fc for bid, d, fc, _ in cost_history if d == "low" and fc > 0]
    low_adaptive_costs = [ac for bid, d, _, ac in cost_history if d == "low" and ac > 0]
    high_fixed_costs = [fc for bid, d, fc, _ in cost_history if d == "high" and fc > 0]
    high_adaptive_costs = [
        ac for bid, d, _, ac in cost_history if d == "high" and ac > 0
    ]

    if low_fixed_costs and low_adaptive_costs:
        print("\n低密度场景 (场景0-49):")
        print(f"  固定阈值策略平均成本: {np.mean(low_fixed_costs):.4f}")
        print(f"  自适应阈值策略平均成本: {np.mean(low_adaptive_costs):.4f}")
        improvement = (
            (np.mean(low_fixed_costs) - np.mean(low_adaptive_costs))
            / np.mean(low_fixed_costs)
        ) * 100
        print(f"  成本改善: {improvement:.2f}%")

    if high_fixed_costs and high_adaptive_costs:
        print("\n高密度场景 (场景50-99):")
        print(f"  固定阈值策略平均成本: {np.mean(high_fixed_costs):.4f}")
        print(f"  自适应阈值策略平均成本: {np.mean(high_adaptive_costs):.4f}")
        improvement = (
            (np.mean(high_fixed_costs) - np.mean(high_adaptive_costs))
            / np.mean(high_fixed_costs)
        ) * 100
        print(f"  成本改善: {improvement:.2f}%")

    # 统计collision_dominant_ratio的变化
    print("\n" + "=" * 70)
    print("碰撞占优比例 (collision_dominant_ratio) 变化统计:")
    print("=" * 70)

    # 按密度分组统计
    low_density_ratios = [r for bid, d, r in ratio_history if d == "low"]
    high_density_ratios = [r for bid, d, r in ratio_history if d == "high"]

    if low_density_ratios:
        print("\n低密度场景 (场景0-49):")
        print(f"  平均值: {np.mean(low_density_ratios):.4f}")
        print(f"  最小值: {np.min(low_density_ratios):.4f}")
        print(f"  最大值: {np.max(low_density_ratios):.4f}")
        print(f"  标准差: {np.std(low_density_ratios):.4f}")

    if high_density_ratios:
        print("\n高密度场景 (场景50-99):")
        print(f"  平均值: {np.mean(high_density_ratios):.4f}")
        print(f"  最小值: {np.min(high_density_ratios):.4f}")
        print(f"  最大值: {np.max(high_density_ratios):.4f}")
        print(f"  标准差: {np.std(high_density_ratios):.4f}")

    # 显示变化趋势
    print("\n变化趋势 (每10个场景):")
    print(f"{'场景范围':<15} {'密度':<10} {'平均碰撞占优比':<15}")
    print("-" * 40)
    for i in range(0, 100, 10):
        segment_ratios = [r for bid, d, r in ratio_history if i <= bid < i + 10]
        if segment_ratios:
            density_label = "低密度" if i < 50 else "高密度"
            print(
                f"{i:02d}-{i + 9:02d}{'':>7} {density_label:<10} {np.mean(segment_ratios):>14.4f}"
            )

    print("\n" + "=" * 70)

    # 输出CSV格式的数据（便于后续处理）
    print("\n计算成本变化 (场景ID,密度,固定成本,自适应成本):")
    for bid, density, fixed_cost, adaptive_cost in cost_history[::10]:  # 每10个输出一次
        if fixed_cost > 0 and adaptive_cost > 0:
            print(f"{bid},{density},{fixed_cost:.4f},{adaptive_cost:.4f}")

    print("\n碰撞占优比例变化 (场景ID,密度,比例):")
    for bid, density, ratio in ratio_history[::10]:  # 每10个输出一次
        print(f"{bid},{density},{ratio:.4f}")


if __name__ == "__main__":
    main()
