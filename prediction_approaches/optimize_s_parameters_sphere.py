#!/usr/bin/env python3
"""
球体碰撞预测的S参数优化脚本
在不同障碍物密度下寻找最佳的S参数（阈值），以计算成本作为优化目标
"""

import sys
import os
import numpy as np
import pickle
import csv
from collision_prediction_strategies import (
    FixedThresholdStrategy,
    AdaptiveThresholdStrategy,
    evaluate_strategy_on_spheres,
)
from utils.utils import calculate_expected_checks, calculate_baseline_expectation


sphere_num = 63
sphere_cost = 18


def load_sphere_benchmark_data(benchid, density="low"):
    """
    加载球体基准测试数据

    Args:
        benchid: 基准测试ID
        density: 密度级别 ("low", "mid", "high")

    Returns:
        tuple: (qarr_sphere, rarr_sphere, yarr_sphere) 或 None
    """
    benchidstr = str(benchid)

    if density == "low":
        trace_path = f"../trace_generation/scene_benchmarks/dens6_rs/obstacles_{benchidstr}_sphere.pkl"
    elif density == "mid":
        trace_path = f"../trace_generation/scene_benchmarks/dens9_rs/obstacles_{benchidstr}_sphere.pkl"
    else:
        trace_path = f"../trace_generation/scene_benchmarks/dens12_rs/obstacles_{benchidstr}_sphere.pkl"

    if not os.path.exists(trace_path):
        return None

    with open(trace_path, "rb") as f:
        qarr_sphere, rarr_sphere, yarr_sphere = pickle.load(f)

    return qarr_sphere, rarr_sphere, yarr_sphere


def compute_sphere_bins(density, num_bins_coord, num_bins_radius):
    """
    计算球体位置和半径的分桶边界

    Args:
        density: 密度级别 ("low", "mid", "high")
        num_bins_coord: 坐标分桶数量
        num_bins_radius: 半径分桶数量

    Returns:
        tuple: (x_bins, y_bins, z_bins, r_bins)
    """
    # 收集所有场景的数据来确定范围
    all_positions = []
    all_radii = []

    for benchid in range(0, 100):
        data = load_sphere_benchmark_data(benchid, density)
        if data is None:
            continue

        qarr_sphere, rarr_sphere, yarr_sphere = data
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

    # 为每个坐标轴创建独立的分桶边界
    def create_bins(min_val, max_val, num_bins):
        """创建等间距的分桶边界"""
        margin = (max_val - min_val) * 0.01
        return np.linspace(min_val - margin, max_val + margin, num_bins + 1)[:-1]

    x_bins = create_bins(x_min, x_max, num_bins_coord)
    y_bins = create_bins(y_min, y_max, num_bins_coord)
    z_bins = create_bins(z_min, z_max, num_bins_coord)
    r_bins = create_bins(r_min, r_max, num_bins_radius)

    return x_bins, y_bins, z_bins, r_bins


def quantize_sphere_data(qarr_sphere, rarr_sphere, x_bins, y_bins, z_bins, r_bins):
    """
    量化球体位置和半径数据

    Args:
        qarr_sphere: 球体位置数据 [N, 3]
        rarr_sphere: 球体半径数据 [N, 1]
        x_bins, y_bins, z_bins: 位置分桶边界
        r_bins: 半径分桶边界

    Returns:
        tuple: (position_quant, radius_quant)
    """
    # 对球体位置进行分轴量化离散化
    position_quant = np.zeros_like(qarr_sphere, dtype=int)
    position_quant[:, 0] = np.digitize(qarr_sphere[:, 0], x_bins, right=True)  # X轴
    position_quant[:, 1] = np.digitize(qarr_sphere[:, 1], y_bins, right=True)  # Y轴
    position_quant[:, 2] = np.digitize(qarr_sphere[:, 2], z_bins, right=True)  # Z轴

    # 对球体半径进行独立量化离散化
    radius_quant = np.digitize(rarr_sphere.flatten(), r_bins, right=True)

    return position_quant, radius_quant


def evaluate_fixed_threshold_sphere(
    threshold,
    density,
    bench_ids,
    num_bins_coord,
    num_bins_radius,
    update_prob,
    consider_radius=False,
):
    """
    评估固定阈值策略在球体数据上的性能

    Args:
        threshold: 固定阈值
        density: 密度级别
        bench_ids: 场景ID列表
        num_bins_coord: 坐标分桶数量
        num_bins_radius: 半径分桶数量
        update_prob: 更新概率
        consider_radius: 是否考虑半径

    Returns:
        tuple: (平均成本, 平均baseline成本, 平均精确率, 平均召回率)
    """
    # 计算分桶边界
    x_bins, y_bins, z_bins, r_bins = compute_sphere_bins(
        density, num_bins_coord, num_bins_radius
    )

    strategy = FixedThresholdStrategy(threshold=threshold, update_prob=update_prob)
    all_costs = []
    all_baseline_costs = []
    prec, rec = 0, 0

    for benchid in bench_ids:
        data = load_sphere_benchmark_data(benchid, density)
        if data is None:
            continue

        qarr_sphere, rarr_sphere, yarr_sphere = data
        position_quant, radius_quant = quantize_sphere_data(
            qarr_sphere, rarr_sphere, x_bins, y_bins, z_bins, r_bins
        )

        # 评估策略
        evaluate_strategy_on_spheres(
            strategy,
            position_quant,
            radius_quant,
            yarr_sphere.flatten(),
            consider_radius=consider_radius,
        )

        # 计算成本
        prec, rec = strategy.get_metrics()
        collision_ratio = strategy.get_collision_ratio()

        if prec > 0 and rec > 0 and collision_ratio > 0:
            # 使用 calculate_expected_checks 计算预测器成本
            expected_checks = calculate_expected_checks(
                R=collision_ratio, C=rec / 100.0, A=prec / 100.0, N=sphere_num
            )
            cost = expected_checks * sphere_cost
            all_costs.append(cost)
            
            # 使用 calculate_baseline_expectation 计算baseline成本
            baseline_checks = calculate_baseline_expectation(N=sphere_num, R=collision_ratio)
            baseline_cost = baseline_checks * sphere_cost
            all_baseline_costs.append(baseline_cost)

        # 重置以准备下一个场景
        strategy.reset_collision_history()
        strategy.reset_statistics()

    avg_cost = np.mean(all_costs) if all_costs else float("inf")
    avg_baseline_cost = np.mean(all_baseline_costs) if all_baseline_costs else float("inf")

    return avg_cost, avg_baseline_cost, prec, rec


def evaluate_adaptive_threshold_sphere(
    s_min,
    s_max,
    density,
    bench_ids,
    num_bins_coord,
    num_bins_radius,
    update_prob,
    consider_radius=False,
):
    """
    评估自适应阈值策略在球体数据上的性能

    Args:
        s_min: 最小敏感度
        s_max: 最大敏感度
        density: 密度级别
        bench_ids: 场景ID列表
        num_bins_coord: 坐标分桶数量
        num_bins_radius: 半径分桶数量
        update_prob: 更新概率
        consider_radius: 是否考虑半径

    Returns:
        tuple: (平均成本, 平均baseline成本, 平均精确率, 平均召回率)
    """
    # 计算分桶边界
    x_bins, y_bins, z_bins, r_bins = compute_sphere_bins(
        density, num_bins_coord, num_bins_radius
    )

    strategy = AdaptiveThresholdStrategy(
        s_min=s_min, s_max=s_max, update_prob=update_prob
    )
    all_costs = []
    all_baseline_costs = []

    for benchid in bench_ids:
        data = load_sphere_benchmark_data(benchid, density)
        if data is None:
            continue

        qarr_sphere, rarr_sphere, yarr_sphere = data
        position_quant, radius_quant = quantize_sphere_data(
            qarr_sphere, rarr_sphere, x_bins, y_bins, z_bins, r_bins
        )

        # 评估策略
        evaluate_strategy_on_spheres(
            strategy,
            position_quant,
            radius_quant,
            yarr_sphere.flatten(),
            consider_radius=consider_radius,
        )

        # 计算成本
        prec, rec = strategy.get_metrics()
        collision_ratio = strategy.get_collision_ratio()

        if prec > 0 and rec > 0 and collision_ratio > 0:
            # 使用 calculate_expected_checks 计算预测器成本
            expected_checks = calculate_expected_checks(
                R=collision_ratio, C=rec / 100.0, A=prec / 100.0, N=sphere_num
            )
            cost = expected_checks * sphere_cost
            all_costs.append(cost)
            
            # 使用 calculate_baseline_expectation 计算baseline成本
            baseline_checks = calculate_baseline_expectation(N=sphere_num, R=collision_ratio)
            baseline_cost = baseline_checks * sphere_cost
            all_baseline_costs.append(baseline_cost)

        # 重置以准备下一个场景
        strategy.reset_collision_history()
        strategy.reset_statistics()

    avg_cost = np.mean(all_costs) if all_costs else float("inf")
    avg_baseline_cost = np.mean(all_baseline_costs) if all_baseline_costs else float("inf")
    final_prec, final_rec = strategy.get_metrics()

    return avg_cost, avg_baseline_cost, final_prec, final_rec


def optimize_fixed_threshold_sphere(
    density,
    bench_ids,
    num_bins_coord,
    num_bins_radius,
    update_prob,
    consider_radius=False,
):
    """
    优化球体数据上固定阈值策略的阈值参数

    Args:
        density: 密度级别
        bench_ids: 场景ID列表
        num_bins_coord: 坐标分桶数量
        num_bins_radius: 半径分桶数量
        update_prob: 更新概率
        consider_radius: 是否考虑半径

    Returns:
        tuple: (最佳阈值, 最低成本, baseline成本, 精确率, 召回率, 所有结果)
    """
    print(f"\n优化固定阈值策略 (球体) - {density}密度")
    print("-" * 70)

    # 搜索空间：0 和 2的幂次方，从2^-5 (1/32) 到 2^5 (32)
    threshold_candidates = [0] + [2**i for i in range(-5, 6)]

    best_threshold = None
    best_cost = float("inf")
    best_baseline_cost = float("inf")
    best_prec = 0
    best_rec = 0

    results = []

    for threshold in threshold_candidates:
        avg_cost, avg_baseline_cost, prec, rec = evaluate_fixed_threshold_sphere(
            threshold,
            density,
            bench_ids,
            num_bins_coord,
            num_bins_radius,
            update_prob,
            consider_radius,
        )

        results.append((threshold, avg_cost, avg_baseline_cost, prec, rec))

        print(
            f"  阈值={threshold:8.4f}, 平均成本={avg_cost:7.4f}, "
            f"baseline成本={avg_baseline_cost:7.4f}, "
            f"精确率={prec:6.2f}%, 召回率={rec:6.2f}%"
        )

        if avg_cost < best_cost:
            best_cost = avg_cost
            best_baseline_cost = avg_baseline_cost
            best_threshold = threshold
            best_prec = prec
            best_rec = rec

    print(f"\n✅ 最佳固定阈值: {best_threshold:.4f}")
    print(f"   最低成本: {best_cost:.4f}")
    print(f"   Baseline成本: {best_baseline_cost:.4f}")
    print(f"   精确率: {best_prec:.2f}%")
    print(f"   召回率: {best_rec:.2f}%")

    return best_threshold, best_cost, best_baseline_cost, best_prec, best_rec, results


def optimize_adaptive_threshold_sphere(
    density,
    bench_ids,
    num_bins_coord,
    num_bins_radius,
    update_prob,
    consider_radius=False,
):
    """
    优化球体数据上自适应阈值策略的S_min和S_max参数

    Args:
        density: 密度级别
        bench_ids: 场景ID列表
        num_bins_coord: 坐标分桶数量
        num_bins_radius: 半径分桶数量
        update_prob: 更新概率
        consider_radius: 是否考虑半径

    Returns:
        tuple: (最佳s_min, 最佳s_max, 最低成本, baseline成本, 精确率, 召回率, 所有结果)
    """
    print(f"\n优化自适应阈值策略 (球体) - {density}密度")
    print("-" * 70)

    # 搜索空间
    s_min_candidates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    s_max_candidates = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

    best_s_min = None
    best_s_max = None
    best_cost = float("inf")
    best_baseline_cost = float("inf")
    best_prec = 0
    best_rec = 0

    results = []

    for s_min in s_min_candidates:
        for s_max in s_max_candidates:
            # s_max必须大于s_min
            if s_max <= s_min:
                continue

            avg_cost, avg_baseline_cost, prec, rec = evaluate_adaptive_threshold_sphere(
                s_min,
                s_max,
                density,
                bench_ids,
                num_bins_coord,
                num_bins_radius,
                update_prob,
                consider_radius,
            )

            results.append((s_min, s_max, avg_cost, avg_baseline_cost, prec, rec))

            print(
                f"  S_min={s_min:6.3f}, S_max={s_max:5.2f}, 平均成本={avg_cost:7.4f}, "
                f"baseline成本={avg_baseline_cost:7.4f}, "
                f"精确率={prec:6.2f}%, 召回率={rec:6.2f}%"
            )

            if avg_cost < best_cost:
                best_cost = avg_cost
                best_baseline_cost = avg_baseline_cost
                best_s_min = s_min
                best_s_max = s_max
                best_prec = prec
                best_rec = rec

    print(f"\n✅ 最佳自适应参数: S_min={best_s_min:.3f}, S_max={best_s_max:.2f}")
    print(f"   最低成本: {best_cost:.4f}")
    print(f"   Baseline成本: {best_baseline_cost:.4f}")
    print(f"   精确率: {best_prec:.2f}%")
    print(f"   召回率: {best_rec:.2f}%")

    return best_s_min, best_s_max, best_cost, best_baseline_cost, best_prec, best_rec, results


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) < 3:
        print(
            "用法: python optimize_s_parameters_sphere.py <coord_bin_bits> <radius_bin_bits> [update_prob] [consider_radius]"
        )
        print("示例: python optimize_s_parameters_sphere.py 4 0 0.5 0")
        sys.exit(1)

    coord_bin_bits = int(sys.argv[1])
    radius_bin_bits = int(sys.argv[2])
    update_prob = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    consider_radius = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False

    num_bins_coord = 2**coord_bin_bits
    num_bins_radius = 2**radius_bin_bits

    print("=" * 70)
    print("球体碰撞预测的S参数优化 - 基于计算成本")
    print("=" * 70)
    print("配置:")
    print(f"  - 坐标分桶数量: {num_bins_coord} (2^{coord_bin_bits})")
    print(f"  - 半径分桶数量: {num_bins_radius} (2^{radius_bin_bits})")
    print(f"  - 更新概率: {update_prob}")
    print(f"  - 考虑半径: {'是' if consider_radius else '否'}")
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
            best_fixed_baseline_cost,
            best_fixed_prec,
            best_fixed_rec,
            fixed_results,
        ) = optimize_fixed_threshold_sphere(
            density_name,
            bench_ids,
            num_bins_coord,
            num_bins_radius,
            update_prob,
            consider_radius,
        )

        # 优化自适应阈值策略（如果需要）
        # (
        #     best_s_min,
        #     best_s_max,
        #     best_adaptive_cost,
        #     best_adaptive_baseline_cost,
        #     best_adaptive_prec,
        #     best_adaptive_rec,
        #     adaptive_results,
        # ) = optimize_adaptive_threshold_sphere(
        #     density_name, bench_ids, num_bins_coord, num_bins_radius, update_prob, consider_radius
        # )

        # 保存结果
        all_results[density_name] = {
            "fixed": {
                "threshold": best_fixed_threshold,
                "cost": best_fixed_cost,
                "baseline_cost": best_fixed_baseline_cost,
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
        print(f"  Baseline成本: {fixed_data['baseline_cost']:.4f}")
        if fixed_data['baseline_cost'] > 0:
            speedup = fixed_data['baseline_cost'] / fixed_data['cost']
            print(f"  加速比: {speedup:.2f}x")
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
            print(f"  Baseline成本: {adaptive_data['baseline_cost']:.4f}")
            if adaptive_data['baseline_cost'] > 0:
                speedup = adaptive_data['baseline_cost'] / adaptive_data['cost']
                print(f"  加速比: {speedup:.2f}x")
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
    print("密度,策略,参数,成本,Baseline成本,加速比,精确率,召回率")
    for density_name in ["low", "mid", "high"]:
        fixed_data = all_results[density_name]["fixed"]
        speedup = fixed_data['baseline_cost'] / fixed_data['cost'] if fixed_data['cost'] > 0 else 0

        print(
            f"{density_name},固定阈值,{fixed_data['threshold']:.4f},"
            f"{fixed_data['cost']:.4f},{fixed_data['baseline_cost']:.4f},{speedup:.2f},"
            f"{fixed_data['precision']:.2f},{fixed_data['recall']:.2f}"
        )

        # 如果有自适应策略结果，也输出
        if "adaptive" in all_results[density_name]:
            adaptive_data = all_results[density_name]["adaptive"]
            speedup = adaptive_data['baseline_cost'] / adaptive_data['cost'] if adaptive_data['cost'] > 0 else 0
            print(
                f'{density_name},自适应阈值,"{adaptive_data["s_min"]:.3f},{adaptive_data["s_max"]:.2f}",'
                f"{adaptive_data['cost']:.4f},{adaptive_data['baseline_cost']:.4f},{speedup:.2f},"
                f"{adaptive_data['precision']:.2f},{adaptive_data['recall']:.2f}"
            )

    print("\n" + "=" * 70)
    print("✅ 优化完成!")
    print("=" * 70)

    # 保存结果到CSV文件
    os.makedirs("result_files", exist_ok=True)
    radius_info = "with_radius" if consider_radius else "no_radius"
    output_csv = f"result_files/s_params_sphere_{num_bins_coord}coord_{num_bins_radius}radius_{radius_info}.csv"

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
                "配置",
            ]
        )

        # 写入每种密度的结果
        for density_name in ["low", "mid", "high"]:
            fixed_data = all_results[density_name]["fixed"]
            speedup = fixed_data['baseline_cost'] / fixed_data['cost'] if fixed_data['cost'] > 0 else 0

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
                    f"coord_bins={num_bins_coord}, radius_bins={num_bins_radius}, "
                    f"update_prob={update_prob}, consider_radius={consider_radius}",
                ]
            )

            # 如果有自适应策略结果，也写入
            if "adaptive" in all_results[density_name]:
                adaptive_data = all_results[density_name]["adaptive"]
                speedup = adaptive_data['baseline_cost'] / adaptive_data['cost'] if adaptive_data['cost'] > 0 else 0
                writer.writerow(
                    [
                        density_name,
                        "自适应阈值",
                        f"S_min={adaptive_data['s_min']:.3f}, S_max={adaptive_data['s_max']:.2f}",
                        f"{adaptive_data['cost']:.4f}",
                        f"{adaptive_data['baseline_cost']:.4f}",
                        f"{speedup:.2f}",
                        f"{adaptive_data['precision']:.2f}",
                        f"{adaptive_data['recall']:.2f}",
                        f"coord_bins={num_bins_coord}, radius_bins={num_bins_radius}, "
                        f"update_prob={update_prob}, consider_radius={consider_radius}",
                    ]
                )

    print(f"✅ 结果已保存到: {output_csv}")

    # 同时保存详细的所有测试结果
    detailed_csv = f"result_files/s_params_sphere_detailed_{num_bins_coord}coord_{num_bins_radius}radius_{radius_info}.csv"
    print(f"正在保存详细结果到: {detailed_csv}")

    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(
            ["密度", "策略类型", "参数值", "平均成本", "Baseline成本", "加速比", "精确率(%)", "召回率(%)"]
        )

        # 写入固定阈值策略的所有测试结果
        for density_name in ["low", "mid", "high"]:
            fixed_data = all_results[density_name]["fixed"]
            for threshold, cost, baseline_cost, prec, rec in fixed_data["all_results"]:
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
                    ]
                )

            # 如果有自适应策略的详细结果，也写入
            if "adaptive" in all_results[density_name]:
                adaptive_data = all_results[density_name]["adaptive"]
                for s_min, s_max, cost, baseline_cost, prec, rec in adaptive_data["all_results"]:
                    speedup = baseline_cost / cost if cost > 0 else 0
                    writer.writerow(
                        [
                            density_name,
                            "自适应阈值",
                            f"S_min={s_min:.3f}, S_max={s_max:.2f}",
                            f"{cost:.4f}",
                            f"{baseline_cost:.4f}",
                            f"{speedup:.2f}",
                            f"{prec:.2f}",
                            f"{rec:.2f}",
                        ]
                    )

    print(f"✅ 详细结果已保存到: {detailed_csv}")


if __name__ == "__main__":
    main()
