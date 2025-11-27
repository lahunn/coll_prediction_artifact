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

import os
import pickle
import numpy as np
import argparse

# 添加项目路径

from trace_generation.core.robot.modular_env import ModularEnv
from trace_generation.bit_planning.algorithm.bit_star import BITStar
from trace_generation.utils.planning_utils import uniform_sample, distance

EDGE_COUNT_LIMIT = 100


def visualize_problem(modular_env, obstacles, start=None, goal=None, path=None):
    """可视化问题场景（需要GUI模式）"""
    import time

    print(f"\n障碍物数量: {len(obstacles)}")

    if start is not None:
        modular_env.robot_env.set_config(start)
        time.sleep(1)
        collision = not modular_env._state_fp(start)
        print(f"起点: {'碰撞' if collision else '无碰撞'}")
        time.sleep(1)

    if goal is not None:
        modular_env.robot_env.set_config(goal)
        time.sleep(1)
        collision = not modular_env._state_fp(goal)
        print(f"终点: {'碰撞' if collision else '无碰撞'}")
        time.sleep(1)

    if path is not None and len(path) > 0:
        print(f"路径长度: {len(path)}")
        for i, config in enumerate(path):
            modular_env.robot_env.set_config(config)
            time.sleep(0.05)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def generate_single_problem(
    modular_env,
    num_obstacles,
    workspace_range,
    voxel_size_range,
    safe_zone_center,
    safe_zone_radius,
    max_planning_time=60.0,
    max_sample_attempts=100,
    visualize=False,
):
    """生成单个路径规划问题"""

    for attempt in range(max_sample_attempts):
        # 生成随机障碍物并加载到环境中
        modular_env.generate_random_obstacles(
            num_obstacles=num_obstacles,
            workspace_range=workspace_range,
            voxel_size_range=voxel_size_range,
            safe_zone_center=safe_zone_center,
            safe_zone_radius=safe_zone_radius,
        )

        # 获取当前障碍物
        obstacles = modular_env.obstacle_manager.obstacles

        start = uniform_sample(
            modular_env.robot_env.lower_bounds,
            modular_env.robot_env.upper_bounds,
            modular_env.robot_env.config_dim,
        )
        goal = uniform_sample(
            modular_env.robot_env.lower_bounds,
            modular_env.robot_env.upper_bounds,
            modular_env.robot_env.config_dim,
        )

        if not modular_env._state_fp(start) or not modular_env._state_fp(goal):
            continue
        dist = distance(start, goal)
        if dist < 1.0:
            continue

        print(f"find a valid start-goal pair with distance: {dist:.2f}")
        modular_env.init_state = start
        modular_env.goal_state = goal
        planner = BITStar(modular_env)

        samples, edges, collision_check_count, cost, num_samples, planning_time = (
            planner.plan(pathLengthLimit=float("inf"), time_budget=max_planning_time)
        )

        if cost < float("inf"):
            path = reconstruct_path(edges, start, goal)
            if path is not None and len(path) > 1:
                if visualize:
                    visualize_problem(modular_env, obstacles, start, goal, path)
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
    enable_self_collision=False,
):
    """生成完整的问题数据集"""
    # 不再传递config_output_file,使用内存记录
    print(f"机器人: {robot_name}, 问题数: {num_problems}, 障碍物: {num_obstacles}")
    print("碰撞检测模型: link + sphere (双模型生成)")
    print(f"自碰撞检测: {'启用' if enable_self_collision else '禁用'}")

    # 创建两个模块化环境：link 和 sphere
    modular_env_link = ModularEnv(
        robot_name,
        map_file=None,
        GUI=visualize,
        collision_model_type="link",
        enable_self_collision=enable_self_collision,
    )
    modular_env_sphere = ModularEnv(
        robot_name,
        map_file=None,
        GUI=False,  # 第二个环境不需要GUI
        collision_model_type="sphere",
        enable_self_collision=enable_self_collision,
    )
    config_dim = modular_env_link.config_dim

    if output_file is None:
        output_file = f"maze_files/{robot_name}_{config_dim}_{num_problems}.pkl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    problems = []
    success_count = 0
    # 创建保存目录
    obstacle_config_dir = "../../trace_files/bit_traces"
    collision_data_dir = "../../trace_files/scene_benchmarks/bit_collision_data"
    os.makedirs(obstacle_config_dir, exist_ok=True)
    os.makedirs(collision_data_dir, exist_ok=True)

    base_filename = f"{robot_name}_{config_dim}"

    while success_count < num_problems:
        print(f"\n正在生成问题 {success_count + 1}/{num_problems} ...")

        # 使用link环境生成问题
        modular_env_link.collision_env.config_list = []
        modular_env_link.collision_env.data_manager.reset()

        problem = generate_single_problem(
            modular_env_link,
            num_obstacles,
            workspace_range,
            voxel_size_range,
            safe_zone_center=(0.0, 0.0, 0.0),
            safe_zone_radius=safe_zone_radius,
            max_planning_time=max_planning_time,
            visualize=visualize,
        )

        if problem is None:
            continue

        obstacles, start, goal, path_link = problem
        link_edge_count = modular_env_link.collision_env.data_manager.edge_fp_call_count
        if link_edge_count <= EDGE_COUNT_LIMIT:
            continue
        # 使用sphere环境对同一问题重新规划
        print("  使用sphere模型重新规划...")
        modular_env_sphere.collision_env.config_list = []
        modular_env_sphere.collision_env.data_manager.reset()

        # 加载相同的障碍物
        modular_env_sphere.load_obstacles(obstacles)

        # 执行规划
        modular_env_sphere.init_state = start
        modular_env_sphere.goal_state = goal
        planner_sphere = BITStar(modular_env_sphere)
        _, edges_sphere, _, cost_sphere, _, _ = planner_sphere.plan(
            pathLengthLimit=float("inf"), time_budget=max_planning_time
        )

        path_sphere = None
        if cost_sphere < float("inf"):
            path_sphere = reconstruct_path(edges_sphere, start, goal)
        if path_sphere is None or len(path_sphere) <= 1:
            continue
        sphere_edge_count = (
            modular_env_sphere.collision_env.data_manager.edge_fp_call_count
        )

        # 只有两种模型都成功且edge调用数>EDGE_COUNT_LIMIT 才保存
        problems.append(problem)
        success_count += 1

        # 生成文件名
        pair_filename = f"{base_filename}_{success_count:04d}.pkl"
        pair_filepath = os.path.join(obstacle_config_dir, pair_filename)

        # 保存障碍物-配置对（只保存一次）
        obstacle_config_pair = {
            "obstacles": obstacles,
            "configs": modular_env_link.collision_env.config_list.copy(),
        }

        with open(pair_filepath, "wb") as f:
            pickle.dump(obstacle_config_pair, f)

        # 保存link碰撞检测数据
        coll_filename_link = f"{base_filename}_{success_count:04d}_link.pkl"
        coll_filepath_link = os.path.join(collision_data_dir, coll_filename_link)
        modular_env_link.collision_env.data_manager.save_collision_data(
            coll_filepath_link
        )

        # 保存sphere碰撞检测数据
        coll_filename_sphere = f"{base_filename}_{success_count:04d}_sphere.pkl"
        coll_filepath_sphere = os.path.join(collision_data_dir, coll_filename_sphere)
        modular_env_sphere.collision_env.data_manager.save_collision_data(
            coll_filepath_sphere
        )

        print(f"  ✓ Link edges: {link_edge_count}, Sphere edges: {sphere_edge_count}")

    modular_env_link.close()
    modular_env_sphere.close()

    with open(output_file, "wb") as f:
        pickle.dump(problems, f)

    # 统计信息
    path_lengths = [len(prob[3]) for prob in problems]

    print(f"\n完成! 保存到: {output_file}")
    print(f"障碍物-配置配对文件保存到: {obstacle_config_dir}/")
    print(f"  文件数量: {success_count}")
    print(f"  文件命名格式: {base_filename}_XXXX.pkl (例: {base_filename}_0001.pkl)")
    print(f"碰撞检测数据保存到: {collision_data_dir}/")
    print(f"  Link文件格式: {base_filename}_XXXX_link.pkl")
    print(f"  Sphere文件格式: {base_filename}_XXXX_sphere.pkl")
    print(
        f"路径长度 - 平均: {np.mean(path_lengths):.2f}, 最小: {np.min(path_lengths)}, 最大: {np.max(path_lengths)}"
    )

    return problems


def main():
    parser = argparse.ArgumentParser(description="生成机器人路径规划问题数据集")

    parser.add_argument(
        "--robot-file",
        type=str,
        default="kuka_iiwa/model_0.urdf",
        help="(Deprecated) Robot URDF file path",
    )
    parser.add_argument(
        "--robot-name", type=str, default="kuka_iiwa", help="Robot name identifier"
    )
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
    parser.add_argument(
        "--enable-self-collision",
        action="store_true",
        help="Enable self-collision detection",
    )

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
        enable_self_collision=args.enable_self_collision,
    )

    return 0


if __name__ == "__main__":
    exit(main())
