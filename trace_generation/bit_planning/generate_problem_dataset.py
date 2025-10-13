#!/usr/bin/env python3
"""
生成机器人路径规划问题数据集

该脚本用于生成包含障碍物、起点、终点和路径的问题集，
保存为.pkl文件供训练和评估使用。

每个问题包含:
    - obstacles: List[Tuple[halfExtents, basePosition]] - 体素障碍物列表
    - start: np.ndarray - 起始配置
    - goal: np.ndarray - 目标配置
    - path: List[np.ndarray] - 从起点到终点的路径（配置序列）
"""

import sys
import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm
import pybullet as p

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.robot_env import RobotEnv
from algorithm.bit_star import BITStar


def generate_random_obstacles(
    num_obstacles=10,
    workspace_range=(-1.0, 1.0),
    voxel_size_range=(0.05, 0.15),
    safe_zone_center=(0.0, 0.0, 0.0),
    safe_zone_radius=0.3,
):
    """生成随机障碍物，避开机器人基座附近的安全区域"""
    obstacles = []
    w_min, w_max = workspace_range
    safe_center = np.array(safe_zone_center)

    for _ in range(num_obstacles):
        max_attempts = 100
        for attempt in range(max_attempts):
            half_size = np.random.uniform(voxel_size_range[0], voxel_size_range[1], size=3)
            position = np.random.uniform(w_min, w_max, size=3)
            distance_to_base = np.linalg.norm(position - safe_center)
            min_safe_distance = safe_zone_radius + np.max(half_size)

            if distance_to_base > min_safe_distance:
                obstacles.append((half_size, position))
                break

    return obstacles


def visualize_problem(env, obstacles, start=None, goal=None, path=None):
    """可视化问题场景（需要GUI模式）"""
    import time

    print(f"\n障碍物数量: {len(obstacles)}")

    if start is not None:
        env.set_config(start)
        time.sleep(1)
        collision = not env._state_fp(start)
        print(f"起点: {'碰撞' if collision else '无碰撞'}")
        time.sleep(1)

    if goal is not None:
        env.set_config(goal)
        time.sleep(1)
        collision = not env._state_fp(goal)
        print(f"终点: {'碰撞' if collision else '无碰撞'}")
        time.sleep(1)

    if path is not None and len(path) > 0:
        print(f"路径长度: {len(path)}")
        for i, config in enumerate(path):
            env.set_config(config)
            time.sleep(0.05)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def generate_single_problem(
    env, obstacles, max_planning_time=60.0, max_sample_attempts=100, visualize=False, planner=None
):
    """生成单个路径规划问题"""
    if planner is None:
        sys.exit(f"错误：必须提供规划器实例")

    for attempt in range(max_sample_attempts):
        start = env.uniform_sample()
        goal = env.uniform_sample()

        if not env._state_fp(start) or not env._state_fp(goal):
            continue

        distance = env.distance(start, goal)
        if distance < 0.5:
            continue

        env.init_state = start
        env.goal_state = goal
        planner.reset(start, goal)

        samples, edges, collision_count, cost, num_samples, planning_time = (
            planner.plan(pathLengthLimit=float("inf"), time_budget=max_planning_time)
        )

        if cost < float("inf"):
            path = reconstruct_path(edges, start, goal)
            if path is not None and len(path) > 1:
                if visualize:
                    visualize_problem(env, obstacles, start, goal, path)
                return (obstacles, start, goal, path)

    return None


def reconstruct_path(edges, start, goal):
    """从边字典重构路径"""
    from collections import deque

    def to_tuple(state):
        return tuple(state.flatten())

    path = deque([goal])
    current = to_tuple(goal)
    start_tuple = to_tuple(start)

    for _ in range(10000):
        if current not in edges:
            return None

        parent = edges[current]

        if parent == start_tuple:
            path.appendleft(np.array(parent))
            break

        path.appendleft(np.array(parent))
        current = parent
    else:
        return None

    return list(path)


def generate_problem_dataset(
    robot_file,
    num_problems=3000,
    num_obstacles=10,
    output_file=None,
    max_planning_time=60.0,
    workspace_range=(-1.5, 1.5),
    voxel_size_range=(0.05, 0.12),
    safe_zone_radius=0.4,
    visualize=False,
):
    """生成完整的问题数据集"""
    print(f"机器人: {robot_file}, 问题数: {num_problems}, 障碍物: {num_obstacles}")

    env = RobotEnv(GUI=visualize, robot_file=robot_file)
    config_dim = env.config_dim

    if output_file is None:
        robot_name = robot_file.split("/")[-1].split(".")[0]
        output_file = f"maze_files/{robot_name}_{config_dim}_{num_problems}.pkl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    problems = []
    success_count = 0

    initial_obstacles = generate_random_obstacles(
        num_obstacles=num_obstacles,
        workspace_range=workspace_range,
        voxel_size_range=voxel_size_range,
        safe_zone_center=(0.0, 0.0, 0.0),
        safe_zone_radius=safe_zone_radius,
    )
    env.init_obstacle_bodies(num_obstacles, initial_obstacles)

    planner = BITStar(env)
    pbar = tqdm(total=num_problems, desc="生成问题", unit="问题")

    while success_count < num_problems:
        env.randomize_obstacle_poses(
            workspace_range=workspace_range,
            safe_zone_center=(0.0, 0.0, 0.0),
            safe_zone_radius=safe_zone_radius,
            max_attempts_per_obstacle=100
        )

        problem = generate_single_problem(
            env, env.obstacles, max_planning_time=max_planning_time, visualize=visualize, planner=planner
        )

        if problem is not None:
            problems.append(problem)
            success_count += 1
            pbar.update(1)
            pbar.set_postfix({"成功": success_count, "路径长度": len(problem[3])})

    pbar.close()
    env.cleanup_obstacles()

    with open(output_file, "wb") as f:
        pickle.dump(problems, f)

    path_lengths = [len(p[3]) for p in problems]
    print(f"\n完成! 保存到: {output_file}")
    print(f"路径长度 - 平均: {np.mean(path_lengths):.2f}, 最小: {np.min(path_lengths)}, 最大: {np.max(path_lengths)}")

    return problems


def main():
    parser = argparse.ArgumentParser(description="生成机器人路径规划问题数据集")

    parser.add_argument("--robot-file", type=str, default="kuka_iiwa/model_0.urdf")
    parser.add_argument("--num-problems", type=int, default=3000)
    parser.add_argument("--num-obstacles", type=int, default=10)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--workspace-min", type=float, default=-0.8)
    parser.add_argument("--workspace-max", type=float, default=0.8)
    parser.add_argument("--voxel-size-min", type=float, default=0.05)
    parser.add_argument("--voxel-size-max", type=float, default=0.12)
    parser.add_argument("--safe-zone-radius", type=float, default=0.3)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    generate_problem_dataset(
        robot_file=args.robot_file,
        num_problems=args.num_problems,
        num_obstacles=args.num_obstacles,
        output_file=args.output_file,
        max_planning_time=args.max_time,
        workspace_range=(args.workspace_min, args.workspace_max),
        voxel_size_range=(args.voxel_size_min, args.voxel_size_max),
        safe_zone_radius=args.safe_zone_radius,
        visualize=args.visualize,
    )

    return 0


if __name__ == "__main__":
    exit(main())
