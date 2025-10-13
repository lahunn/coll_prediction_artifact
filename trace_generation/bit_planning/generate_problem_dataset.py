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
    """
    生成随机障碍物，避开机器人基座附近的安全区域

    Args:
        num_obstacles: 障碍物数量
        workspace_range: 工作空间范围 (min, max)
        voxel_size_range: 体素尺寸范围 (min, max)
        safe_zone_center: 安全区域中心 (通常是机器人基座位置)
        safe_zone_radius: 安全区域半径

    Returns:
        obstacles: List[Tuple[halfExtents, basePosition]]
    """
    obstacles = []
    w_min, w_max = workspace_range
    safe_center = np.array(safe_zone_center)

    for _ in range(num_obstacles):
        max_attempts = 100
        for attempt in range(max_attempts):
            # 随机生成体素半边长
            half_size = np.random.uniform(
                voxel_size_range[0], voxel_size_range[1], size=3
            )

            # 随机生成体素中心位置
            position = np.random.uniform(w_min, w_max, size=3)

            # 检查是否在安全区域外
            distance_to_base = np.linalg.norm(position - safe_center)

            # 障碍物中心需要距离安全区域足够远（考虑障碍物本身的尺寸）
            min_safe_distance = safe_zone_radius + np.max(half_size)

            if distance_to_base > min_safe_distance:
                obstacles.append((half_size, position))
                break

    return obstacles


def visualize_problem(env, obstacles, start=None, goal=None, path=None):
    """
    可视化问题场景（需要GUI模式）

    Args:
        env: RobotEnv环境实例
        obstacles: 障碍物列表
        start: 起始配置（可选）
        goal: 目标配置（可选）
        path: 路径（可选）
    """
    import time

    print("\n" + "=" * 70)
    print("可视化模式")
    print("=" * 70)
    print(f"障碍物数量: {len(obstacles)}")

    if start is not None:
        print(f"起点配置: {start}")
        print("设置机器人到起点位置...")
        env.set_config(start)
        time.sleep(1)

        # 检查起点碰撞
        collision = not env._state_fp(start)
        print(f"起点碰撞检测: {'❌ 碰撞' if collision else '✓ 无碰撞'}")
        time.sleep(2)

    if goal is not None:
        print(f"\n目标配置: {goal}")
        print("设置机器人到目标位置...")
        env.set_config(goal)
        time.sleep(1)

        # 检查终点碰撞
        collision = not env._state_fp(goal)
        print(f"终点碰撞检测: {'❌ 碰撞' if collision else '✓ 无碰撞'}")
        time.sleep(2)

    if path is not None and len(path) > 0:
        print(f"\n路径长度: {len(path)}")
        print("播放路径动画...")
        for i, config in enumerate(path):
            env.set_config(config)
            if i % 5 == 0:  # 每5个节点打印一次
                print(f"  步骤 {i + 1}/{len(path)}")
            time.sleep(0.05)
        print("✓ 路径播放完成")

    print("\n按 Ctrl+C 停止可视化...")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n可视化结束")


def generate_single_problem(
    env, obstacles, max_planning_time=60.0, max_sample_attempts=10000, visualize=False
):
    """
    生成单个路径规划问题

    Args:
        env: RobotEnv环境实例
        obstacles: 障碍物列表
        max_planning_time: 最大规划时间(秒)
        max_sample_attempts: 采样起点终点的最大尝试次数
        visualize: 是否可视化（需要GUI模式）

    Returns:
        tuple: (obstacles, start, goal, path) 如果成功，否则返回None
    """
    # 采样无碰撞的起点和终点
    start = None
    goal = None

    for attempt in range(max_sample_attempts):
        try:
            # 采样起点和终点
            start = env.uniform_sample()
            goal = env.uniform_sample()

            # 检查起点和终点是否无碰撞
            if not env._state_fp(start) or not env._state_fp(goal):
                if visualize and attempt == 0:
                    print("\n起点或终点发生碰撞，显示场景...")
                    visualize_problem(env, obstacles, start, goal, None)
                continue

            # 检查起点和终点是否足够远
            distance = env.distance(start, goal)
            if distance < 0.5:  # 太近的话跳过
                continue

            # 设置环境的起点和终点
            env.init_state = start
            env.goal_state = goal

            # 使用BIT*算法规划路径
            planner = BITStar(env)

            # 执行规划
            # pathLengthLimit设为INF表示不限制路径长度，time_budget设置最大规划时间
            samples, edges, collision_count, cost, num_samples, planning_time = (
                planner.plan(
                    pathLengthLimit=float("inf"), time_budget=max_planning_time
                )
            )

            # 检查是否找到路径
            if cost < float("inf"):
                # 从edges字典重构路径
                path = reconstruct_path(edges, start, goal)

                if path is not None and len(path) > 1:
                    # 如果启用可视化，显示结果
                    if visualize:
                        visualize_problem(env, obstacles, start, goal, path)
                    return (obstacles, start, goal, path)
            else:
                # 即使规划失败，如果可视化模式也显示起点和终点
                if visualize and attempt == 0:
                    print("\n规划失败，显示起点和终点...")
                    visualize_problem(env, obstacles, start, goal, None)

        except Exception as e:
            print(f"  规划失败 (尝试 {attempt + 1}/{max_sample_attempts}): {e}")
            if visualize and attempt == 0 and start is not None:
                print("显示失败的场景...")
                visualize_problem(env, obstacles, start, goal, None)
            continue

    return None


def reconstruct_path(edges, start, goal):
    """
    从边字典重构路径

    Args:
        edges: dict - {child: parent} 映射
        start: 起始状态
        goal: 目标状态

    Returns:
        path: List[np.ndarray] - 从起点到终点的路径
    """

    # 将numpy数组转换为可哈希的元组用于字典查找
    def to_tuple(state):
        return tuple(state.flatten())

    def from_tuple(t):
        return np.array(t)

    # 从goal回溯到start
    path = [goal]
    current = to_tuple(goal)

    # edges是 {child: parent} 格式
    max_iterations = 10000
    iterations = 0

    while iterations < max_iterations:
        if current not in edges:
            break

        parent = edges[current]
        parent_state = from_tuple(parent)
        path.append(parent_state)

        # 检查是否到达起点
        if np.allclose(parent_state, start, atol=1e-6):
            break

        current = parent
        iterations += 1

    if iterations >= max_iterations or not np.allclose(path[-1], start, atol=1e-6):
        return None

    # 反转路径使其从start到goal
    path.reverse()
    return path


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
    """
    生成完整的问题数据集

    Args:
        robot_file: 机器人URDF文件路径
        num_problems: 要生成的问题数量
        num_obstacles: 每个问题的障碍物数量
        output_file: 输出文件路径
        max_planning_time: 每个问题的最大规划时间
        workspace_range: 工作空间范围
        voxel_size_range: 体素尺寸范围
        safe_zone_radius: 机器人基座周围的安全区域半径
        visualize: 是否启用可视化模式（会打开GUI）

    Returns:
        problems: List[Tuple[obstacles, start, goal, path]]
    """
    print("=" * 70)
    print("机器人路径规划问题数据集生成器")
    print("=" * 70)
    print(f"机器人文件: {robot_file}")
    print(f"目标问题数: {num_problems}")
    print(f"障碍物数量: {num_obstacles}")
    print(f"工作空间范围: {workspace_range}")
    print(f"体素尺寸范围: {voxel_size_range}")
    print(f"最大规划时间: {max_planning_time}秒")
    print(f"安全区域半径: {safe_zone_radius}米")
    print("=" * 70)

    if visualize:
        print("⚠️  可视化模式已启用 - 将打开GUI窗口")
        print("=" * 70)

    # 创建环境实例（只创建一次）
    env = RobotEnv(GUI=visualize, robot_file=robot_file)
    config_dim = env.config_dim
    print(f"机器人自由度: {config_dim}")

    # 确定输出文件名
    if output_file is None:
        robot_name = robot_file.split("/")[-1].split(".")[0]
        output_file = f"maze_files/{robot_name}_{config_dim}_{num_problems}.pkl"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"输出文件: {output_file}")
    print("=" * 70)

    problems = []
    success_count = 0

    # 用于跟踪已创建的障碍物ID，以便清除
    obstacle_ids = []

    # 使用tqdm显示进度
    pbar = tqdm(total=num_problems, desc="生成问题", unit="问题")

    while success_count < num_problems:
        # 清除之前的障碍物
        for obs_id in obstacle_ids:
            try:
                p.removeBody(obs_id)
            except Exception:
                pass
        obstacle_ids.clear()

        # 生成随机障碍物，避开机器人基座附近的安全区域
        obstacles = generate_random_obstacles(
            num_obstacles=num_obstacles,
            workspace_range=workspace_range,
            voxel_size_range=voxel_size_range,
            safe_zone_center=(0.0, 0.0, 0.0),
            safe_zone_radius=safe_zone_radius,
        )

        # 更新环境的障碍物列表
        env.obstacles = obstacles

        # 在仿真环境中创建障碍物，并记录ID
        for halfExtents, basePosition in obstacles:
            obs_id = env.create_voxel(halfExtents, basePosition)
            if obs_id is not None:
                obstacle_ids.append(obs_id)

        # 生成问题
        problem = generate_single_problem(
            env, obstacles, max_planning_time=max_planning_time, visualize=visualize
        )

        if problem is not None:
            problems.append(problem)
            success_count += 1
            pbar.update(1)
            pbar.set_postfix({"成功": success_count, "路径长度": len(problem[3])})

    pbar.close()

    # 保存到pkl文件
    print(f"\n正在保存 {len(problems)} 个问题到 {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(problems, f)

    print("=" * 70)
    print("✓ 数据集生成完成!")
    print(f"✓ 成功生成 {len(problems)} 个问题")
    print(f"✓ 已保存到: {output_file}")
    print("=" * 70)

    # 打印统计信息
    path_lengths = [len(p[3]) for p in problems]
    print("\n路径长度统计:")
    print(f"  平均: {np.mean(path_lengths):.2f}")
    print(f"  最小: {np.min(path_lengths)}")
    print(f"  最大: {np.max(path_lengths)}")
    print(f"  中位数: {np.median(path_lengths):.2f}")

    return problems


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成机器人路径规划问题数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--robot-file",
        type=str,
        default="kuka_iiwa/model_0.urdf",
        help="机器人URDF文件路径",
    )

    parser.add_argument(
        "--num-problems", type=int, default=3000, help="要生成的问题数量"
    )

    parser.add_argument(
        "--num-obstacles", type=int, default=10, help="每个问题的障碍物数量"
    )

    parser.add_argument(
        "--output-file", type=str, default=None, help="输出文件路径（默认自动生成）"
    )

    parser.add_argument(
        "--max-time", type=float, default=60.0, help="每个问题的最大规划时间（秒）"
    )

    parser.add_argument(
        "--workspace-min", type=float, default=-0.8, help="工作空间最小坐标"
    )

    parser.add_argument(
        "--workspace-max", type=float, default=0.8, help="工作空间最大坐标"
    )

    parser.add_argument(
        "--voxel-size-min", type=float, default=0.05, help="体素最小尺寸"
    )

    parser.add_argument(
        "--voxel-size-max", type=float, default=0.12, help="体素最大尺寸"
    )

    parser.add_argument(
        "--safe-zone-radius",
        type=float,
        default=0.3,
        help="机器人基座周围的安全区域半径(米)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="启用可视化模式（打开GUI窗口显示障碍物和机器人状态）",
    )

    args = parser.parse_args()

    # 生成数据集
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
