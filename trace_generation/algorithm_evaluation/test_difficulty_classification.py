#!/usr/bin/env python3
"""
测试问题集难度划分准确性脚本

通过调用 BIT* 算法运行不同难度级别（G1-G5）的问题，
并统计碰撞检测次数、时间等指标，验证难度划分是否符合预期。
"""

import os
import pickle
import numpy as np
import argparse
import time
from collections import defaultdict

from trace_generation.core.robot.modular_env import ModularEnv
from trace_generation.algorithm_evaluation.algorithm.bit_star import BITStar

def run_benchmark(args):
    difficulty_levels = ["G1", "G2", "G3", "G4", "G5"]
    results = defaultdict(list)
    
    print(f"开始测试机器人: {args.robot_name}")
    print(f"使用碰撞模型: {args.collision_model}")
    print(f"每个级别测试问题数: {args.num_test}")
    print("-" * 50)

    # 初始化环境
    env = ModularEnv(
        args.robot_name, 
        GUI=args.visualize, 
        collision_model_type=args.collision_model,
        enable_self_collision=args.enable_self_collision
    )

    for level in difficulty_levels:
        problems_path = os.path.join(args.problems_root, level, "problems.pkl")
        if not os.path.exists(problems_path):
            print(f"警告: 跳过 {level}, 未找到文件: {problems_path}")
            continue

        with open(problems_path, "rb") as f:
            problems = pickle.load(f)

        # 随机抽取或按顺序选取测试题目
        test_indices = np.random.choice(len(problems), min(args.num_test, len(problems)), replace=False)
        print(f"\n正在测试难度级别: {level} ({len(test_indices)} 个问题)")

        level_metrics = []
        for i, idx in enumerate(test_indices):
            obstacles, start, goal, _ = problems[idx]
            
            # 加载障碍物
            env.load_obstacles(obstacles)
            env.init_state = start
            env.goal_state = goal
            
            # 重置碰撞检测计数器
            env.collision_env.data_manager.reset()
            
            planner = BITStar(env)
            start_time = time.time()
            
            # 执行规划
            result = planner.plan(
                pathLengthLimit=args.path_limit, 
                time_budget=args.max_time, 
                dump_log=False
            )
            
            duration = time.time() - start_time
            coll_checks = env.collision_env.data_manager.edge_fp_call_count
            cost = result.get("cost", float("inf"))
            
            success = cost < float("inf")
            
            metric = {
                "success": success,
                "coll_checks": coll_checks,
                "time": duration,
                "cost": cost if success else None
            }
            level_metrics.append(metric)
            
            if (i + 1) % 5 == 0:
                print(f"  进度: {i+1}/{len(test_indices)}...")

        results[level] = level_metrics

    env.close()

    # 打印汇总报告
    print("\n" + "="*60)
    print(f"{'难度':<6} | {'成功率':<8} | {'平均碰撞次数':<15} | {'平均耗时(s)':<12}")
    print("-" * 60)
    
    for level in difficulty_levels:
        if level not in results: continue
        
        metrics = results[level]
        successes = [m["success"] for m in metrics]
        checks = [m["coll_checks"] for m in metrics]
        times = [m["time"] for m in metrics]
        
        avg_success = np.mean(successes) * 100
        avg_checks = np.mean(checks)
        avg_time = np.mean(times)
        
        print(f"{level:<6} | {avg_success:>6.1f}% | {avg_checks:>15.1f} | {avg_time:>12.3f}")
    
    print("="*60)
    print("测试完成。如果平均碰撞次数随 G1->G5 递增，说明划分准确。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIT* 难度划分准确性验证")
    parser.add_argument("--robot-name", type=str, default="iiwa", help="机器人名称")
    parser.add_argument("--collision-model", type=str, default="sphere", choices=["link", "sphere"], help="使用的碰撞模型")
    parser.add_argument("--num-test", type=int, default=20, help="每个级别测试的问题数量")
    parser.add_argument("--problems-root", type=str, default="../../trace_files/problems", help="问题集根目录")
    parser.add_argument("--max-time", type=float, default=60.0, help="BIT* 单个问题最大允许时间")
    parser.add_argument("--path-limit", type=float, default=1.2, help="BIT* 路径长度限制")
    parser.add_argument("--visualize", action="store_true", help="是否开启可视化")
    parser.add_argument("--enable-self-collision", action="store_true", help="启用自碰撞检测")

    args = parser.parse_args()
    run_benchmark(args)