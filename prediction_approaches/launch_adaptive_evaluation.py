#!/usr/bin/env python3
"""
该脚本用于在动态变化的环境中(从低密度到高密度)评估和比较
固定S策略和自适应S策略的性能。
生成的CSV格式供 plot_adaptive_evaluation.py 绘图使用。
"""

import os
import sys
import numpy as np
import pickle
from collision_prediction_strategies import (
    FixedThresholdStrategy,
    AdaptiveThresholdStrategy,
    evaluate_strategy_on_trajectory,
)

# 添加项目根目录到 Python 路径以导入 trace_generation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trace_generation.config.ana_parameters import get_robot_params

def load_benchmark_data(benchid, density):
    """加载基准测试数据"""
    benchidstr = str(benchid)
    # 映射密度名称到文件夹名称
    density_map = {
        "low": "dens6",
        "mid": "dens9",
        "high": "dens12"
    }
    folder = density_map.get(density, "dens6")
    
    trace_path = f"trace_files/scene_benchmarks/{folder}_rs/obstacles_{benchidstr}_coord.pkl"
    
    # 如果在 prediction_approaches 目录下运行，需要向上跳一级
    if not os.path.exists(trace_path):
        trace_path = f"../{trace_path}"
        
    if not os.path.exists(trace_path):
        return None

    with open(trace_path, "rb") as f:
        xtest_pred, dirr_pred, label_pred = pickle.load(f)
    return xtest_pred, dirr_pred, label_pred

def quantize_coordinates(xtest_pred, bin_bits):
    """量化坐标"""
    num_bins = 2**bin_bits
    bins = np.linspace(-1.12, 1.12, num_bins + 1)[:-1]
    return np.digitize(xtest_pred, bins, right=True)

def run_evaluation(strategy, bin_bits, robot_params):
    """在一系列变化的场景中评估策略"""
    # 模拟环境变化：30个低密度 -> 30个中密度 -> 30个高密度
    # 注意：历史(colldict)在这些场景间保持不重置，以体现自适应过程
    # 但统计量(all_zerozero等)是在整个90个场景上累计的
    strategy.reset_collision_history()
    strategy.reset_statistics()
    
    obb_num = robot_params["obb_num"]
    
    densities = [("low", range(0, 30)), ("mid", range(0, 30)), ("high", range(0, 30))]
    
    for density_name, bench_range in densities:
        for benchid in bench_range:
            data = load_benchmark_data(benchid, density_name)
            if data is None:
                continue
            
            xtest, dirr, label = data
            code_quant = quantize_coordinates(xtest, bin_bits)
            
            # evaluate_strategy_on_trajectory 会调用 strategy.update_history
            # 从而在处理过程中学习。统计变量也会持续增加。
            evaluate_strategy_on_trajectory(strategy, code_quant, label, group_size=obb_num)
            
    # 获取整个过程的姿态级指标
    precision, recall, _, _ = strategy.get_metrics()
    return precision, recall

def main():
    RESULT_FILE = "result_files/adaptive_evaluation.csv"
    os.makedirs("result_files", exist_ok=True)
    
    bin_bits = 4
    update_prob = 1.0
    robot_name = "franka"
    robot_params = get_robot_params(robot_name)
    
    # 定义所有要评估的策略，顺序必须与 plot_adaptive_evaluation.py 一致
    strategies = [
        ("Fixed S: Random", FixedThresholdStrategy(threshold=4.0, update_prob=update_prob)),
        ("Fixed S: 2.0", FixedThresholdStrategy(threshold=2.0, update_prob=update_prob)),
        ("Fixed S: 1.0", FixedThresholdStrategy(threshold=1.0, update_prob=update_prob)),
        ("Fixed S: 0.5", FixedThresholdStrategy(threshold=0.5, update_prob=update_prob)),
        ("Fixed S: 0.125", FixedThresholdStrategy(threshold=0.125, update_prob=update_prob)),
        ("Fixed S: 0.031", FixedThresholdStrategy(threshold=0.03125, update_prob=update_prob)),
        ("Adaptive S (0.1-1.0)", AdaptiveThresholdStrategy(s_min=0.1, s_max=1.0, update_prob=update_prob)),
        ("Adaptive S (0.1-1.5)", AdaptiveThresholdStrategy(s_min=0.1, s_max=1.5, update_prob=update_prob)),
        ("Adaptive S (0.05-2.0)", AdaptiveThresholdStrategy(s_min=0.05, s_max=2.0, update_prob=update_prob)),
    ]
    
    results = []
    print(f"Starting evaluation on changing environments (Robot: {robot_name})...")
    print(f"{'Strategy':<25} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 55)
    
    for name, strategy in strategies:
        prec, rec = run_evaluation(strategy, bin_bits, robot_params)
        results.append((prec, rec))
        print(f"{name:<25} | {prec:>9.2f}% | {rec:>9.2f}%")
        
    # 写入CSV (不要表头)
    with open(RESULT_FILE, "w") as f:
        for prec, rec in results:
            f.write(f"{prec:.6f},{rec:.6f}\n")
            
    print(f"\nResults saved to {RESULT_FILE}")
    print("You can now run 'python plot/plot_adaptive_evaluation.py' to generate the figure.")

if __name__ == "__main__":
    main()
