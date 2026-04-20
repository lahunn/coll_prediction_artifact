#!/usr/bin/env python3
"""
简化版机器人路径规划问题数据集生成脚本（仅使用 Sphere 模型）

该脚本全程使用 Sphere 碰撞检测模型来生成障碍物、寻找起点终点并进行路径规划。
支持障碍物数量在指定范围内随机，且文件名中不再包含障碍物数量信息。
"""

import os
import pickle
import numpy as np
import argparse
import shutil
import gc
from time import time

from trace_generation.core.robot.modular_env import ModularEnv
from trace_generation.algorithm_evaluation.algorithm.bit_star import BITStar
from trace_generation.utils.planning_utils import uniform_sample, distance

EDGE_COUNT_LIMIT = 10 # 过滤掉过于简单的场景

def reconstruct_path(edges, start, goal):
    """从边字典重构路径"""
    from collections import deque
    def to_tuple(state): return tuple(state.flatten())

    path = deque([goal])
    current = to_tuple(goal)
    start_tuple = to_tuple(start)

    for _ in range(10000):
        if current not in edges: return None
        parent = edges[current]
        
        if parent == start_tuple:
            path.appendleft(np.array(parent))
            break
            
        path.appendleft(np.array(parent))
        current = parent
    else: return None
    return list(path)

def redistribute_problems_by_difficulty(robot_name, config_dim, filename_to_info, problems, output_file):
    """根据 sphere_edge_count 将问题划分为不同难度级别 (G1-G5)"""
    print("\n开始根据 Sphere 模型的复杂度重新划分难度级别...")
    
    source_dir = "../../trace_files/bit_traces"
    collision_dir = "../../trace_files/scene_benchmarks/bit_collision_data"
    difficulty_levels = ["G1", "G2", "G3", "G4", "G5"]
    
    problems_output_root = os.path.dirname(output_file) if output_file else "../../trace_files/problems"
    
    # 建立目录
    for level in difficulty_levels:
        os.makedirs(os.path.join(source_dir, level), exist_ok=True)
        os.makedirs(os.path.join(collision_dir, level), exist_ok=True)
        os.makedirs(os.path.join(problems_output_root, level), exist_ok=True)

    # 提取数据进行分位数计算
    items = []
    base_filename = f"{robot_name}_{config_dim}"
    for coll_file, info in filename_to_info.items():
        try:
            idx = info['success_idx']
            pair_file = f"{base_filename}_{idx:04d}.pkl"
            items.append({
                'idx': idx, 
                'pair_file': pair_file, 
                'coll_file': coll_file, 
                'count': info['edge_count']
            })
        except:
            continue

    if not items: return

    counts = np.array([x['count'] for x in items])
    quantiles = np.percentile(counts, [20, 40, 60, 80, 100])
    print(f"Sphere Edge Count 分位数: {quantiles}")

    level_counters = {l: 0 for l in difficulty_levels}
    problems_by_level = {l: [] for l in difficulty_levels}

    for item in items:
        # 确定难度
        level = difficulty_levels[0]
        for i, q in enumerate(quantiles):
            if item['count'] <= q:
                level = difficulty_levels[i]
                break
        
        level_counters[level] += 1
        if 1 <= item['idx'] <= len(problems):
            problems_by_level[level].append(problems[item['idx'] - 1])

        # 重命名并移动文件 (文件名保持简洁)
        new_pair = f"{base_filename}_{level_counters[level]:04d}.pkl"
        new_coll = f"{base_filename}_{level_counters[level]:04d}_sphere.pkl"

        old_pair_path = os.path.join(source_dir, item['pair_file'])
        old_coll_path = os.path.join(collision_dir, item['coll_file'])
        
        if os.path.exists(old_pair_path):
            shutil.move(old_pair_path, os.path.join(source_dir, level, new_pair))
        if os.path.exists(old_coll_path):
            shutil.move(old_coll_path, os.path.join(collision_dir, level, new_coll))

    # 保存各级别的 problems.pkl
    for level in difficulty_levels:
        level_path = os.path.join(problems_output_root, level)
        with open(os.path.join(level_path, "problems.pkl"), "wb") as f:
            pickle.dump(problems_by_level[level], f)
    print("难度划分完成。")

def generate_problem_dataset_sphere(args):
    """主生成函数"""
    print(f"启动单模型(Sphere)数据集生成: {args.robot_name}, 目标数量: {args.num_problems}")
    print(f"障碍物范围: {args.min_obstacles} - {args.max_obstacles}")
    
    env = ModularEnv(args.robot_name, GUI=args.visualize, collision_model_type="sphere", 
                     enable_self_collision=args.enable_self_collision)
    config_dim = env.config_dim
    
    output_file = args.output_file or f"../../trace_files/problems/{args.robot_name}_{config_dim}_{args.num_problems}_sphere.pkl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    obstacle_config_dir = "../../trace_files/bit_traces"
    collision_data_dir = "../../trace_files/scene_benchmarks/bit_collision_data"
    os.makedirs(obstacle_config_dir, exist_ok=True)
    os.makedirs(collision_data_dir, exist_ok=True)

    problems = []
    success_count = 0
    filename_to_info = {}
    base_name = f"{args.robot_name}_{config_dim}"

    while success_count < args.num_problems:
        # 随机选择障碍物数量
        num_obs = np.random.randint(args.min_obstacles, args.max_obstacles + 1)
        
        print(f"生成问题 {success_count + 1}/{args.num_problems} (障碍物: {num_obs})...")
        
        # 1. 生成环境
        env.generate_random_obstacles(
            num_obstacles=num_obs,
            workspace_range=(args.workspace_min, args.workspace_max),
            voxel_size_range=(args.voxel_size_min, args.voxel_size_max),
            safe_zone_center=(0.0, 0.0, 0.0),
            safe_zone_radius=args.safe_zone_radius
        )
        obstacles = env.obstacle_manager.obstacles

        # 2. 采样起点终点
        start = uniform_sample(env.robot_env.lower_bounds, env.robot_env.upper_bounds, config_dim)
        goal = uniform_sample(env.robot_env.lower_bounds, env.robot_env.upper_bounds, config_dim)

        if not env._state_fp(start) or not env._state_fp(goal) or distance(start, goal) < 1.0:
            continue

        # 3. 规划
        env.collision_env.config_list = []
        env.collision_env.data_manager.reset()
        env.init_state, env.goal_state = start, goal
        
        planner = BITStar(env)
        result = planner.plan(pathLengthLimit=args.path_length_limit, time_budget=args.max_time, dump_log=False)
        
        cost = result.get("cost", float("inf"))
        edge_count = env.collision_env.data_manager.edge_fp_call_count

        if cost < float("inf") and edge_count > EDGE_COUNT_LIMIT:
            path = reconstruct_path(result.get("edges", {}), start, goal)
            if path:
                success_count += 1
                problems.append((obstacles, start, goal, path))

                # 保存障碍物和配置轨迹
                pair_path = os.path.join(obstacle_config_dir, f"{base_name}_{success_count:04d}.pkl")
                with open(pair_path, "wb") as f:
                    pickle.dump({"obstacles": obstacles, "configs": env.collision_env.config_list.copy()}, f)

                # 保存碰撞检测原始数据 (文件名中不再出现 num_obstacles)
                coll_name = f"{base_name}_{success_count:04d}_sphere.pkl"
                coll_path = os.path.join(collision_data_dir, coll_name)
                env.collision_env.data_manager.save_collision_data(coll_path)
                
                filename_to_info[coll_name] = {
                    'edge_count': edge_count,
                    'success_idx': success_count
                }
                print(f"  ✓ 成功! Edge counts: {edge_count}")
        
        del planner
        gc.collect()

    env.close()
    
    # 最终汇总与划分
    with open(output_file, "wb") as f:
        pickle.dump(problems, f)
        
    redistribute_problems_by_difficulty(args.robot_name, config_dim, 
                                        filename_to_info, problems, output_file)
    print(f"数据集生成完成，保存在: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单模型路径规划数据集生成器")
    parser.add_argument("--robot-name", type=str, default="iiwa")
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument("--min-obstacles", type=int, default=5)
    parser.add_argument("--max-obstacles", type=int, default=9)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--path-length-limit", type=float, default=1.2)
    parser.add_argument("--workspace-min", type=float, default=-0.8)
    parser.add_argument("--workspace-max", type=float, default=0.8)
    parser.add_argument("--voxel-size-min", type=float, default=0.12)
    parser.add_argument("--voxel-size-max", type=float, default=0.20)
    parser.add_argument("--safe-zone-radius", type=float, default=0.15)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--enable-self-collision", action="store_true")

    generate_problem_dataset_sphere(parser.parse_args())
