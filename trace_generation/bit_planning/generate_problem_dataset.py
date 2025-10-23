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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.collision_env import CollisionEnv
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
            half_size = np.random.uniform(
                voxel_size_range[0], voxel_size_range[1], size=3
            )
            position = np.random.uniform(w_min, w_max, size=3)

            # 确保 z 坐标不小于 0（障碍物顶部不低于地面）
            if position[2] + half_size[2] < 0:
                continue

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
    env,
    obstacles,
    max_planning_time=60.0,
    max_sample_attempts=100,
    visualize=False,
    planner=None,
):
    """生成单个路径规划问题"""
    if planner is None:
        sys.exit("错误：必须提供规划器实例")

    for attempt in range(max_sample_attempts):
        start = env.uniform_sample()
        goal = env.uniform_sample()

        if not env._state_fp(start) or not env._state_fp(goal):
            continue

        distance = env.distance(start, goal)
        if distance < 1.5:
            continue

        env.init_state = start
        env.goal_state = goal
        planner.reset(start, goal)

        samples, edges, cost, num_samples, planning_time = planner.plan(
            pathLengthLimit=float("inf"), time_budget=max_planning_time
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
    robot_name,
    num_problems=3000,
    num_obstacles=10,
    output_file=None,
    max_planning_time=60.0,
    workspace_range=(-1.0, 1.0),
    voxel_size_range=(0.12, 0.20),
    safe_zone_radius=0.5,
    visualize=False,
):
    """生成完整的问题数据集"""
    robot_name = robot_name
    # 不再传递config_output_file,使用内存记录
    print(f"机器人: {robot_file}, 问题数: {num_problems}, 障碍物: {num_obstacles}")
    env = CollisionEnv(GUI=visualize, robot_file=robot_file, config_output_file="dummy")
    config_dim = env.config_dim

    if output_file is None:
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

    # 创建保存目录
    obstacle_config_dir = "../../trace_files/bit_traces"
    collision_data_dir = "../../trace_files/scene_benchmarks/bit_collision_data"
    os.makedirs(obstacle_config_dir, exist_ok=True)
    os.makedirs(collision_data_dir, exist_ok=True)

    base_filename = f"{robot_name}_{config_dim}"

    while success_count < num_problems:
        print(f"\n正在生成问题 {success_count + 1}/{num_problems} ...")

        env.randomize_obstacle_poses(
            workspace_range=workspace_range,
            safe_zone_center=(0.0, 0.0, 0.0),
            safe_zone_radius=safe_zone_radius,
            max_attempts_per_obstacle=100,
        )

        # 在生成问题前重置数据收集列表
        env.config_list = []
        env.obb_link_data = []
        env.obb_link_coll_data = []
        env.sphere_link_data = []
        env.sphere_link_coll_data = []

        problem = generate_single_problem(
            env,
            env.obstacles,
            max_planning_time=max_planning_time,
            visualize=visualize,
            planner=planner,
        )

        if problem is not None:
            problems.append(problem)
            success_count += 1

            # 生成文件名
            pair_filename = f"{base_filename}_{success_count:04d}.pkl"
            pair_filepath = os.path.join(obstacle_config_dir, pair_filename)

            obb_filename = f"{base_filename}_{success_count:04d}_obb.pkl"
            sphere_filename = f"{base_filename}_{success_count:04d}_sphere.pkl"
            obb_filepath = os.path.join(collision_data_dir, obb_filename)
            sphere_filepath = os.path.join(collision_data_dir, sphere_filename)

            # 保存障碍物-配置对
            obstacle_config_pair = {
                "obstacles": problem[0],
                "configs": env.config_list.copy(),
            }

            with open(pair_filepath, "wb") as f:
                pickle.dump(obstacle_config_pair, f)

            # 保存碰撞检测数据
            env.save_collision_data(obb_filepath, sphere_filepath)

    env.cleanup_obstacles()
    env.close()

    with open(output_file, "wb") as f:
        pickle.dump(problems, f)

    # 统计信息
    path_lengths = [len(prob[3]) for prob in problems]

    print(f"\n完成! 保存到: {output_file}")
    print(f"障碍物-配置配对文件保存到: {obstacle_config_dir}/")
    print(f"  文件数量: {success_count}")
    print(f"  文件命名格式: {base_filename}_XXXX.pkl (例: {base_filename}_0001.pkl)")
    print(f"碰撞检测数据保存到: {collision_data_dir}/")
    print(f"  OBB文件格式: {base_filename}_XXXX_obb.pkl")
    print(f"  Sphere文件格式: {base_filename}_XXXX_sphere.pkl")
    print(
        f"路径长度 - 平均: {np.mean(path_lengths):.2f}, 最小: {np.min(path_lengths)}, 最大: {np.max(path_lengths)}"
    )

    return problems


def main():
    parser = argparse.ArgumentParser(description="生成机器人路径规划问题数据集")

    parser.add_argument("--robot-file", type=str, default="kuka_iiwa/model_0.urdf")
    parser.add_argument("--robot-name", type=str, default="kuka_iiwa")
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
        robot_name=args.robot_name,
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
