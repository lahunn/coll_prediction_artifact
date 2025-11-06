#!/usr/bin/env python3
"""
S参数优化脚本
在不同障碍物密度下寻找最佳的S参数（阈值），以计算成本作为优化目标
"""
# python optimize_s_parameters.py 4 0.5

import sys
import os
import numpy as np
import pickle
import csv
from collision_prediction_strategies import (
    FixedThresholdStrategy,
    evaluate_strategy_on_trajectory,
)
from utils.utils import calculate_expected_checks, calculate_baseline_expectation

# 添加 trace_generation 目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from trace_generation.config.ana_parameters import get_robot_params


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
        trace_path = f"../trace_generation/scene_benchmarks/dens6_rs/obstacles_{benchidstr}_coord.pkl"
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


def evaluate_fixed_threshold(threshold, density, bench_ids, num_bins, update_prob, robot_params):
    """
    评估固定阈值策略在指定场景下的性能

    Args:
        threshold: 固定阈值
        density: 密度级别
        bench_ids: 场景ID列表
        num_bins: 分桶数量
        update_prob: 更新概率
        robot_params: 机器人参数字典

    Returns:
        tuple: (平均成本, 平均baseline成本, 平均精确率, 平均召回率, 平均碰撞概率)
    """
    obb_num = robot_params["obb_num"]
    obb_cost = robot_params["obb_cost"]
    
    strategy = FixedThresholdStrategy(threshold=threshold, update_prob=update_prob)
    all_costs = []
    all_baseline_costs = []
    all_collision_ratios = []
    ele_prec, ele_rec = 0.0, 0.0
    for benchid in bench_ids:
        data = load_benchmark_data(benchid, density)
        if data is None:
            continue

        xtest_pred, dirr_pred, label_pred = data
        code_pred_quant = quantize_coordinates(xtest_pred, num_bins)

        # 评估策略
        evaluate_strategy_on_trajectory(
            strategy, code_pred_quant, label_pred, group_size=obb_num
        )

        # 计算成本
        prec, rec, ele_prec, ele_rec = strategy.get_metrics()
        all_collision_ratio, ele_collision_ratio = strategy.get_collision_ratio(label_pred)
        collision_ratio = ele_collision_ratio  # 使用元素级碰撞率

        if ele_prec > 0 and ele_rec > 0 and collision_ratio > 0:
            # 使用 calculate_expected_checks 计算预测器成本（使用元素级指标）
            expected_checks = calculate_expected_checks(
                R=collision_ratio, C=ele_rec / 100.0, A=ele_prec / 100.0, N=obb_num
            )
            cost = expected_checks * obb_cost
            all_costs.append(cost)

            # 使用 calculate_baseline_expectation 计算baseline成本
            baseline_checks = calculate_baseline_expectation(
                N=obb_num, R=collision_ratio
            )
            baseline_cost = baseline_checks * obb_cost
            all_baseline_costs.append(baseline_cost)

            # 收集碰撞概率
            all_collision_ratios.append(collision_ratio)

        # 重置以准备下一个场景
        strategy.reset_collision_history()
        strategy.reset_statistics()

    avg_cost = np.mean(all_costs) if all_costs else float("inf")
    avg_baseline_cost = (
        np.mean(all_baseline_costs) if all_baseline_costs else float("inf")
    )
    avg_collision_ratio = np.mean(all_collision_ratios) if all_collision_ratios else 0.0

    return avg_cost, avg_baseline_cost, ele_prec, ele_rec, avg_collision_ratio


def optimize_fixed_threshold(density, bench_ids, num_bins, update_prob, robot_params):
    """
    优化固定阈值策略的阈值参数

    Args:
        density: 密度级别
        bench_ids: 场景ID列表
        num_bins: 分桶数量
        update_prob: 更新概率
        robot_params: 机器人参数字典

    Returns:
        tuple: (最佳阈值, 最低成本, baseline成本, 精确率, 召回率, 碰撞概率, 所有结果)
    """
    print(f"\n优化固定阈值策略 - {density}密度")
    print("-" * 70)

    # 搜索空间：0 和 2的幂次方，从2^-5 (1/32) 到 2^5 (32)
    # 生成: 0, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32
    threshold_candidates = [0] + [2**i for i in range(-5, 6)]

    best_threshold = None
    best_cost = float("inf")
    best_baseline_cost = float("inf")
    best_prec = 0
    best_rec = 0
    best_collision_ratio = 0.0

    results = []

    for threshold in threshold_candidates:
        avg_cost, avg_baseline_cost, prec, rec, collision_ratio = (
            evaluate_fixed_threshold(
                threshold, density, bench_ids, num_bins, update_prob, robot_params
            )
        )

        results.append(
            (threshold, avg_cost, avg_baseline_cost, prec, rec, collision_ratio)
        )

        print(
            f"  阈值={threshold:8.4f}, 平均成本={avg_cost:7.4f}, "
            f"baseline成本={avg_baseline_cost:7.4f}, "
            f"精确率={prec:6.2f}%, 召回率={rec:6.2f}%, 碰撞率={collision_ratio:.4f}"
        )

        if avg_cost < best_cost:
            best_cost = avg_cost
            best_baseline_cost = avg_baseline_cost
            best_threshold = threshold
            best_prec = prec
            best_rec = rec
            best_collision_ratio = collision_ratio

    print(f"\n✅ 最佳固定阈值: {best_threshold:.4f}")
    print(f"   最低成本: {best_cost:.4f}")
    print(f"   Baseline成本: {best_baseline_cost:.4f}")
    print(f"   精确率: {best_prec:.2f}%")
    print(f"   召回率: {best_rec:.2f}%")
    print(f"   碰撞率: {best_collision_ratio:.4f}")

    return (
        best_threshold,
        best_cost,
        best_baseline_cost,
        best_prec,
        best_rec,
        best_collision_ratio,
        results,
    )


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python optimize_s_parameters.py <bin_bits> [update_prob] [robot_name]")
        print("示例: python optimize_s_parameters.py 4 0.5 franka")
        sys.exit(1)

    bin_bits = int(sys.argv[1])
    update_prob = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    robot_name = sys.argv[3] if len(sys.argv) > 3 else "franka"
    
    # 获取机器人参数
    robot_params = get_robot_params(robot_name)
    
    num_bins = 2**bin_bits
    num_problems = 100
    print("=" * 70)
    print("S参数优化 - 基于计算成本")
    print("=" * 70)
    print("配置:")
    print(f"  - 机器人: {robot_name}")
    print(f"  - OBB数量: {robot_params['obb_num']}")
    print(f"  - 分桶数量: {num_bins} (2^{bin_bits})")
    print(f"  - 更新概率: {update_prob}")
    print("  - 优化目标: 最小化计算成本")
    print("=" * 70)

    # 定义不同密度的场景ID范围
    densities = {
        "low": list(range(0, num_problems)),  # 场景0-99 (低密度)
        "mid": list(range(0, num_problems)),  # 场景0-99 (中等密度)
        "high": list(range(0, num_problems)),  # 场景0-99 (高密度)
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
            best_fixed_baseline_cost,
            best_fixed_prec,
            best_fixed_rec,
            best_fixed_collision_ratio,
            fixed_results,
        ) = optimize_fixed_threshold(density_name, bench_ids, num_bins, update_prob, robot_params)

        # 保存结果
        all_results[density_name] = {
            "fixed": {
                "threshold": best_fixed_threshold,
                "cost": best_fixed_cost,
                "baseline_cost": best_fixed_baseline_cost,
                "precision": best_fixed_prec,
                "recall": best_fixed_rec,
                "collision_ratio": best_fixed_collision_ratio,
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
        print(f"  Baseline成本: {fixed_data['baseline_cost']:.4f}")
        if fixed_data["baseline_cost"] > 0:
            speedup = fixed_data["baseline_cost"] / fixed_data["cost"]
            print(f"  加速比: {speedup:.2f}x")
        print(f"  精确率: {fixed_data['precision']:.2f}%")
        print(f"  召回率: {fixed_data['recall']:.2f}%")
        print(f"  碰撞率: {fixed_data['collision_ratio']:.4f}")

    # 输出CSV格式的最优参数
    print("\n" + "=" * 70)
    print("最优参数 (CSV格式)")
    print("=" * 70)
    print("密度,策略,参数,成本,Baseline成本,加速比,精确率,召回率,碰撞率")
    for density_name in ["low", "mid", "high"]:
        fixed_data = all_results[density_name]["fixed"]
        speedup = (
            fixed_data["baseline_cost"] / fixed_data["cost"]
            if fixed_data["cost"] > 0
            else 0
        )

        print(
            f"{density_name},固定阈值,{fixed_data['threshold']:.4f},"
            f"{fixed_data['cost']:.4f},{fixed_data['baseline_cost']:.4f},{speedup:.2f},"
            f"{fixed_data['precision']:.2f},{fixed_data['recall']:.2f},"
            f"{fixed_data['collision_ratio']:.4f}"
        )

    print("\n" + "=" * 70)
    print("✅ 优化完成!")
    print("=" * 70)

    # 保存结果到CSV文件
    os.makedirs("result_files", exist_ok=True)
    output_csv = f"result_files/s_params_optimization_{num_bins}bins.csv"

    print(f"\n正在保存结果到: {output_csv}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(
            [
                "密度",
                "策略类型",
                "阈值/参数",
                "平均成本",
                "Baseline成本",
                "加速比",
                "精确率(%)",
                "召回率(%)",
                "碰撞率",
                "配置",
            ]
        )

        # 写入每种密度的结果
        for density_name in ["low", "mid", "high"]:
            fixed_data = all_results[density_name]["fixed"]
            speedup = (
                fixed_data["baseline_cost"] / fixed_data["cost"]
                if fixed_data["cost"] > 0
                else 0
            )

            # 固定阈值策略结果
            writer.writerow(
                [
                    density_name,
                    "固定阈值",
                    f"{fixed_data['threshold']:.4f}",
                    f"{fixed_data['cost']:.4f}",
                    f"{fixed_data['baseline_cost']:.4f}",
                    f"{speedup:.2f}",
                    f"{fixed_data['precision']:.2f}",
                    f"{fixed_data['recall']:.2f}",
                    f"{fixed_data['collision_ratio']:.4f}",
                    f"bins={num_bins}, update_prob={update_prob}",
                ]
            )

    print(f"✅ 结果已保存到: {output_csv}")

    # 同时保存详细的所有测试结果
    detailed_csv = f"result_files/s_params_detailed_{num_bins}bins.csv"
    print(f"正在保存详细结果到: {detailed_csv}")

    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(
            [
                "密度",
                "策略类型",
                "参数值",
                "平均成本",
                "Baseline成本",
                "加速比",
                "精确率(%)",
                "召回率(%)",
                "碰撞率",
            ]
        )

        # 写入固定阈值策略的所有测试结果
        for density_name in ["low", "mid", "high"]:
            fixed_data = all_results[density_name]["fixed"]
            for (
                threshold,
                cost,
                baseline_cost,
                prec,
                rec,
                collision_ratio,
            ) in fixed_data["all_results"]:
                speedup = baseline_cost / cost if cost > 0 else 0
                writer.writerow(
                    [
                        density_name,
                        "固定阈值",
                        f"{threshold:.4f}",
                        f"{cost:.4f}",
                        f"{baseline_cost:.4f}",
                        f"{speedup:.2f}",
                        f"{prec:.2f}",
                        f"{rec:.2f}",
                        f"{collision_ratio:.4f}",
                    ]
                )

    print(f"✅ 详细结果已保存到: {detailed_csv}")


if __name__ == "__main__":
    main()
