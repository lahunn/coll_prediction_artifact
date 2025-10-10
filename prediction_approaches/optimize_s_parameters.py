#!/usr/bin/env python3
"""
S参数优化脚本
在不同障碍物密度下寻找最佳的S参数（阈值），以计算成本作为优化目标
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


obb_num = 11
obb_cost = 42

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
        trace_path = f"../trace_generation/scene_benchmarks/dens9_rs/obstacles_{benchidstr}_coord.pkl"
    else:
        trace_path = f"../trace_generation/scene_benchmarks/dens12_rs/obstacles_{benchidstr}_coord.pkl"

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


def evaluate_fixed_threshold(threshold, density, bench_ids, num_bins, update_prob):
    """
    评估固定阈值策略在指定场景下的性能

    Args:
        threshold: 固定阈值
        density: 密度级别
        bench_ids: 场景ID列表
        num_bins: 分桶数量
        update_prob: 更新概率

    Returns:
        tuple: (平均成本, 平均精确率, 平均召回率)
    """
    strategy = FixedThresholdStrategy(threshold=threshold, update_prob=update_prob)
    all_costs = []
    prec, rec = 0, 0
    for benchid in bench_ids:
        data = load_benchmark_data(benchid, density)
        if data is None:
            continue

        xtest_pred, dirr_pred, label_pred = data
        code_pred_quant = quantize_coordinates(xtest_pred, num_bins)

        # 评估策略
        evaluate_strategy_on_trajectory(
            strategy, code_pred_quant, label_pred, group_size=11
        )

        # 计算成本
        prec, rec = strategy.get_metrics()
        collision_ratio = strategy.get_collision_ratio()

        if prec > 0 and rec > 0 and collision_ratio > 0:
            cost = find_sim_cost(
                R=collision_ratio, C=rec / 100.0, A=prec / 100.0, N=obb_num
            ) * obb_cost
            all_costs.append(cost)

        # 重置以准备下一个场景
        strategy.reset_collision_history()
        strategy.reset_statistics()

    avg_cost = np.mean(all_costs) if all_costs else float("inf")

    return avg_cost, prec, rec


def evaluate_adaptive_threshold(
    s_min, s_max, density, bench_ids, num_bins, update_prob
):
    """
    评估自适应阈值策略在指定场景下的性能

    Args:
        s_min: 最小敏感度
        s_max: 最大敏感度
        density: 密度级别
        bench_ids: 场景ID列表
        num_bins: 分桶数量
        update_prob: 更新概率

    Returns:
        tuple: (平均成本, 平均精确率, 平均召回率)
    """
    strategy = AdaptiveThresholdStrategy(
        s_min=s_min, s_max=s_max, update_prob=update_prob
    )
    all_costs = []

    for benchid in bench_ids:
        data = load_benchmark_data(benchid, density)
        if data is None:
            continue

        xtest_pred, dirr_pred, label_pred = data
        code_pred_quant = quantize_coordinates(xtest_pred, num_bins)

        # 评估策略
        evaluate_strategy_on_trajectory(
            strategy, code_pred_quant, label_pred, group_size=11
        )

        # 计算成本
        prec, rec = strategy.get_metrics()
        collision_ratio = strategy.get_collision_ratio()

        if prec > 0 and rec > 0 and collision_ratio > 0:
            cost = find_sim_cost(R=collision_ratio, C=rec / 100.0, A=prec / 100.0, N=obb_num) * obb_cost
            all_costs.append(cost)

        # 重置以准备下一个场景
        strategy.reset_collision_history()
        strategy.reset_statistics()

    avg_cost = np.mean(all_costs) if all_costs else float("inf")
    final_prec, final_rec = strategy.get_metrics()

    return avg_cost, final_prec, final_rec


def optimize_fixed_threshold(density, bench_ids, num_bins, update_prob):
    """
    优化固定阈值策略的阈值参数

    Args:
        density: 密度级别
        bench_ids: 场景ID列表
        num_bins: 分桶数量
        update_prob: 更新概率

    Returns:
        tuple: (最佳阈值, 最低成本, 精确率, 召回率)
    """
    print(f"\n优化固定阈值策略 - {density}密度")
    print("-" * 70)

    # 搜索空间：0 和 2的幂次方，从2^-5 (1/32) 到 2^5 (32)
    # 生成: 0, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32
    threshold_candidates = [0] + [2**i for i in range(-5, 6)]

    best_threshold = None
    best_cost = float("inf")
    best_prec = 0
    best_rec = 0

    results = []

    for threshold in threshold_candidates:
        avg_cost, prec, rec = evaluate_fixed_threshold(
            threshold, density, bench_ids, num_bins, update_prob
        )

        results.append((threshold, avg_cost, prec, rec))

        print(
            f"  阈值={threshold:8.4f}, 平均成本={avg_cost:7.4f}, "
            f"精确率={prec:6.2f}%, 召回率={rec:6.2f}%"
        )

        if avg_cost < best_cost:
            best_cost = avg_cost
            best_threshold = threshold
            best_prec = prec
            best_rec = rec

    print(f"\n✅ 最佳固定阈值: {best_threshold:.4f}")
    print(f"   最低成本: {best_cost:.4f}")
    print(f"   精确率: {best_prec:.2f}%")
    print(f"   召回率: {best_rec:.2f}%")

    return best_threshold, best_cost, best_prec, best_rec, results


def optimize_adaptive_threshold(density, bench_ids, num_bins, update_prob):
    """
    优化自适应阈值策略的S_min和S_max参数

    Args:
        density: 密度级别
        bench_ids: 场景ID列表
        num_bins: 分桶数量
        update_prob: 更新概率

    Returns:
        tuple: (最佳s_min, 最佳s_max, 最低成本, 精确率, 召回率)
    """
    print(f"\n优化自适应阈值策略 - {density}密度")
    print("-" * 70)

    # 搜索空间
    s_min_candidates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    s_max_candidates = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

    best_s_min = None
    best_s_max = None
    best_cost = float("inf")
    best_prec = 0
    best_rec = 0

    results = []

    for s_min in s_min_candidates:
        for s_max in s_max_candidates:
            # s_max必须大于s_min
            if s_max <= s_min:
                continue

            avg_cost, prec, rec = evaluate_adaptive_threshold(
                s_min, s_max, density, bench_ids, num_bins, update_prob
            )

            results.append((s_min, s_max, avg_cost, prec, rec))

            print(
                f"  S_min={s_min:6.3f}, S_max={s_max:5.2f}, 平均成本={avg_cost:7.4f}, "
                f"精确率={prec:6.2f}%, 召回率={rec:6.2f}%"
            )

            if avg_cost < best_cost:
                best_cost = avg_cost
                best_s_min = s_min
                best_s_max = s_max
                best_prec = prec
                best_rec = rec

    print(f"\n✅ 最佳自适应参数: S_min={best_s_min:.3f}, S_max={best_s_max:.2f}")
    print(f"   最低成本: {best_cost:.4f}")
    print(f"   精确率: {best_prec:.2f}%")
    print(f"   召回率: {best_rec:.2f}%")

    return best_s_min, best_s_max, best_cost, best_prec, best_rec, results


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python optimize_s_parameters.py <bin_bits> [update_prob]")
        print("示例: python optimize_s_parameters.py 5 0.5")
        sys.exit(1)

    bin_bits = int(sys.argv[1])
    update_prob = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    num_bins = 2**bin_bits

    print("=" * 70)
    print("S参数优化 - 基于计算成本")
    print("=" * 70)
    print("配置:")
    print(f"  - 分桶数量: {num_bins} (2^{bin_bits})")
    print(f"  - 更新概率: {update_prob}")
    print("  - 优化目标: 最小化计算成本")
    print("=" * 70)

    # 定义不同密度的场景ID范围
    densities = {
        "low": list(range(0, 100)),  # 场景0-99 (低密度)
        "mid": list(range(0, 100)),  # 场景0-99 (中等密度)
        "high": list(range(0, 100)),  # 场景0-99 (高密度)
    }

    all_results = {}

    # 对每种密度进行优化
    for density_name, bench_ids in densities.items():
        print(f"\n{'=' * 70}")
        print(f"优化密度级别: {density_name.upper()}")
        print(f"场景范围: {bench_ids[0]}-{bench_ids[-1]}")
        print(f"{'=' * 70}")

        # 优化固定阈值策略
        (
            best_fixed_threshold,
            best_fixed_cost,
            best_fixed_prec,
            best_fixed_rec,
            fixed_results,
        ) = optimize_fixed_threshold(density_name, bench_ids, num_bins, update_prob)

        # # 优化自适应阈值策略
        # (
        #     best_s_min,
        #     best_s_max,
        #     best_adaptive_cost,
        #     best_adaptive_prec,
        #     best_adaptive_rec,
        #     adaptive_results,
        # ) = optimize_adaptive_threshold(density_name, bench_ids, num_bins, update_prob)

        # 保存结果
        all_results[density_name] = {
            "fixed": {
                "threshold": best_fixed_threshold,
                "cost": best_fixed_cost,
                "precision": best_fixed_prec,
                "recall": best_fixed_rec,
                "all_results": fixed_results,
            }
        }

    # 输出最终总结
    print("\n" + "=" * 70)
    print("优化结果总结")
    print("=" * 70)

    for density_name in ["low", "mid", "high"]:
        print(f"\n【{density_name.upper()}密度场景】")
        print("-" * 70)

        fixed_data = all_results[density_name]["fixed"]

        print("固定阈值策略:")
        print(f"  最佳阈值: {fixed_data['threshold']:.4f}")
        print(f"  平均成本: {fixed_data['cost']:.4f}")
        print(f"  精确率: {fixed_data['precision']:.2f}%")
        print(f"  召回率: {fixed_data['recall']:.2f}%")

        # 如果有自适应策略结果，也输出
        if "adaptive" in all_results[density_name]:
            adaptive_data = all_results[density_name]["adaptive"]
            print("\n自适应阈值策略:")
            print(
                f"  最佳参数: S_min={adaptive_data['s_min']:.3f}, S_max={adaptive_data['s_max']:.2f}"
            )
            print(f"  平均成本: {adaptive_data['cost']:.4f}")
            print(f"  精确率: {adaptive_data['precision']:.2f}%")
            print(f"  召回率: {adaptive_data['recall']:.2f}%")

            # 计算改善
            if fixed_data["cost"] > 0:
                improvement = (
                    (fixed_data["cost"] - adaptive_data["cost"]) / fixed_data["cost"]
                ) * 100
                print(f"\n  自适应策略成本改善: {improvement:.2f}%")

    # 输出CSV格式的最优参数
    print("\n" + "=" * 70)
    print("最优参数 (CSV格式)")
    print("=" * 70)
    print("密度,策略,参数,成本,精确率,召回率")
    for density_name in ["low", "mid", "high"]:
        fixed_data = all_results[density_name]["fixed"]

        print(
            f"{density_name},固定阈值,{fixed_data['threshold']:.4f},"
            f"{fixed_data['cost']:.4f},{fixed_data['precision']:.2f},{fixed_data['recall']:.2f}"
        )

        # 如果有自适应策略结果，也输出
        if "adaptive" in all_results[density_name]:
            adaptive_data = all_results[density_name]["adaptive"]
            print(
                f'{density_name},自适应阈值,"{adaptive_data["s_min"]:.3f},{adaptive_data["s_max"]:.2f}",'
                f"{adaptive_data['cost']:.4f},{adaptive_data['precision']:.2f},{adaptive_data['recall']:.2f}"
            )

    print("\n" + "=" * 70)
    print("✅ 优化完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
