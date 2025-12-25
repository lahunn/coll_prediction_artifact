#!/usr/bin/env python3
"""
生成机器人路径规划问题数据集（连续变化障碍物版本）

该脚本用于生成包含连续变化障碍物、固定起点、不同终点的问题集，
保存为.pkl文件供训练和评估使用。

主要特点:
- 所有问题使用相同的起点
- 每个问题使用不同的终点
- 障碍物位置连续变化（每次移动0.02），当障碍物离开工作空间时补充新的

每个问题包含:
    - obstacles: List[Tuple[halfExtents, basePosition]] - 体素障碍物列表
    - start: np.ndarray - 起始配置（固定）
    - goal: np.ndarray - 目标配置（变化）
    - path: List[np.ndarray] - 从起点到终点的路径（配置序列）
"""

import os
import pickle
import numpy as np
import argparse

from trace_generation.core.robot.modular_env import ModularEnv
from trace_generation.bit_planning.algorithm.bit_star import BITStar
from trace_generation.utils.planning_utils import uniform_sample, distance

EDGE_COUNT_LIMIT = 10


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


def is_in_safe_zone(position, safe_zone_center, safe_zone_radius):
    """检查位置是否在安全区内"""
    return (
        np.linalg.norm(np.array(position) - np.array(safe_zone_center))
        < safe_zone_radius
    )


def generate_candidate_obstacles(
    num_candidates,
    workspace_range,
    voxel_size_range,
    safe_zone_center,
    safe_zone_radius,
):
    """预先生成候选障碍物（靠近边界）"""
    candidates = []
    for _ in range(num_candidates):
        while True:
            # 简化：随机在工作空间内生成
            position = np.random.uniform(
                [workspace_range[0] + 0.05, -0.5, workspace_range[0] + 0.05],
                [workspace_range[1] - 0.05, 0.5, workspace_range[1] - 0.05],
            )
            if not is_in_safe_zone(position, safe_zone_center, safe_zone_radius):
                break
        size = np.random.uniform(voxel_size_range[0], voxel_size_range[1])
        half_extents = (size / 2, size / 2, size / 2)
        candidates.append((half_extents, tuple(position)))
    return candidates


def move_obstacles(obstacles, move_vector):
    """移动障碍物位置"""
    new_obstacles = []
    for half_extents, position in obstacles:
        new_position = tuple(np.array(position) + move_vector)
        new_obstacles.append((half_extents, new_position))
    return new_obstacles


def generate_problem_dataset(
    robot_name,
    num_problems=3000,
    num_obstacles=10,
    output_file=None,
    max_planning_time=60.0,
    workspace_range=(-1.0, 1.0),
    voxel_size_range=(0.12, 0.20),
    safe_zone_radius=0.5,
    enable_self_collision=False,
    move_step=0.02,
    move_direction=(1, 0, 0),  # 沿x轴正方向移动
):
    """生成完整的问题数据集"""
    print(f"机器人: {robot_name}, 问题数: {num_problems}, 障碍物: {num_obstacles}")
    print("碰撞检测模型: link + sphere (双模型生成)")
    print(f"自碰撞检测: {'启用' if enable_self_collision else '禁用'}")
    print(f"障碍物移动步长: {move_step}, 方向: {move_direction}")

    # 创建两个模块化环境：link 和 sphere
    modular_env_link = ModularEnv(
        robot_name,
        map_file=None,
        GUI=False,
        collision_model_type="link",
        enable_self_collision=enable_self_collision,
    )
    modular_env_sphere = ModularEnv(
        robot_name,
        map_file=None,
        GUI=False,
        collision_model_type="sphere",
        enable_self_collision=enable_self_collision,
    )
    config_dim = modular_env_link.config_dim

    if output_file is None:
        output_file = (
            f"maze_files/{robot_name}_{config_dim}_{num_problems}_continuous.pkl"
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 初始化障碍物
    current_obstacles = generate_candidate_obstacles(
        num_obstacles,
        workspace_range,
        voxel_size_range,
        (0.0, 0.0, 0.0),
        safe_zone_radius,
    )
    modular_env_link.load_obstacles(current_obstacles)
    modular_env_sphere.load_obstacles(current_obstacles)

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

        # 生成起点和终点
        start = uniform_sample(
            modular_env_link.robot_env.lower_bounds,
            modular_env_link.robot_env.upper_bounds,
            modular_env_link.robot_env.config_dim,
        )
        goal = uniform_sample(
            modular_env_link.robot_env.lower_bounds,
            modular_env_link.robot_env.upper_bounds,
            modular_env_link.robot_env.config_dim,
        )
        dist = distance(start, goal)
        if dist < 1.0:
            continue  # 距离太近，重新生成

        # 检查起点和终点无碰撞
        if not modular_env_link._state_fp(start) or not modular_env_link._state_fp(
            goal
        ):
            continue

        print(f"  起点-终点距离: {dist:.2f}")

        # 使用link环境规划
        modular_env_link.collision_env.config_list = []
        modular_env_link.collision_env.data_manager.reset()
        modular_env_link.init_state = start
        modular_env_link.goal_state = goal
        planner_link = BITStar(modular_env_link)

        samples, edges, collision_check_count, cost, num_samples, planning_time = (
            planner_link.plan(
                pathLengthLimit=float("inf"), time_budget=max_planning_time
            )
        )

        if cost >= float("inf"):
            continue

        path_link = reconstruct_path(edges, start, goal)
        if path_link is None or len(path_link) <= 1:
            continue

        link_edge_count = modular_env_link.collision_env.data_manager.edge_fp_call_count
        if link_edge_count <= EDGE_COUNT_LIMIT or link_edge_count >= 200:
            continue

        # 使用sphere环境重新规划
        print("  使用sphere模型重新规划...")
        modular_env_sphere.collision_env.config_list = []
        modular_env_sphere.collision_env.data_manager.reset()
        modular_env_sphere.init_state = start
        modular_env_sphere.goal_state = goal
        planner_sphere = BITStar(modular_env_sphere)
        _, edges_sphere, _, cost_sphere, _, _ = planner_sphere.plan(
            pathLengthLimit=float("inf"), time_budget=max_planning_time
        )

        if cost_sphere >= float("inf"):
            continue

        path_sphere = reconstruct_path(edges_sphere, start, goal)
        if path_sphere is None or len(path_sphere) <= 1:
            continue

        sphere_edge_count = (
            modular_env_sphere.collision_env.data_manager.edge_fp_call_count
        )
        # 当边调用数差异过大时，丢弃该问题
        diff = abs(sphere_edge_count - link_edge_count)
        if diff > min(link_edge_count, sphere_edge_count):
            continue

        # 保存问题
        problem = (current_obstacles, start, goal, path_link)
        problems.append(problem)
        success_count += 1

        # 保存文件
        pair_filename = f"{base_filename}_{success_count:04d}.pkl"
        pair_filepath = os.path.join(obstacle_config_dir, pair_filename)
        obstacle_config_pair = {
            "obstacles": current_obstacles,
            "configs": modular_env_link.collision_env.config_list.copy(),
        }
        with open(pair_filepath, "wb") as f:
            pickle.dump(obstacle_config_pair, f)

        # 保存碰撞数据
        coll_filename_link = f"{base_filename}_{success_count:04d}_ctn_link.pkl"
        coll_filepath_link = os.path.join(collision_data_dir, coll_filename_link)
        modular_env_link.collision_env.data_manager.save_collision_data(
            coll_filepath_link
        )

        coll_filename_sphere = f"{base_filename}_{success_count:04d}_ctn_sphere.pkl"
        coll_filepath_sphere = os.path.join(collision_data_dir, coll_filename_sphere)
        modular_env_sphere.collision_env.data_manager.save_collision_data(
            coll_filepath_sphere
        )

        print(f"  ✓ Link edges: {link_edge_count}, Sphere edges: {sphere_edge_count}")

        # 移动障碍物
        move_vector = np.array(move_direction) * move_step
        current_obstacles = move_obstacles(current_obstacles, move_vector)

        # 检查并替换进入safe_zone的障碍物
        new_obstacles = []
        for obs in current_obstacles:
            half_extents, position = obs
            if is_in_safe_zone(position, (0.0, 0.0, 0.0), safe_zone_radius):
                # 生成新的障碍物
                new_obs = generate_candidate_obstacles(
                    1,
                    workspace_range,
                    voxel_size_range,
                    (0.0, 0.0, 0.0),
                    safe_zone_radius,
                )[0]
                new_obstacles.append(new_obs)
            else:
                new_obstacles.append(obs)
        current_obstacles = new_obstacles

        # 重新加载障碍物到环境
        modular_env_link.load_obstacles(current_obstacles)
        modular_env_sphere.load_obstacles(current_obstacles)

    modular_env_link.close()
    modular_env_sphere.close()

    with open(output_file, "wb") as f:
        pickle.dump(problems, f)

    # 统计信息
    path_lengths = [len(prob[3]) for prob in problems]

    print(f"\n完成! 保存到: {output_file}")
    print(f"障碍物-配置配对文件保存到: {obstacle_config_dir}/")
    print(f"  文件数量: {success_count}")
    print(f"  文件命名格式: {base_filename}_XXXX.pkl")
    print(f"碰撞检测数据保存到: {collision_data_dir}/")
    print(f"  Link文件格式: {base_filename}_XXXX_link.pkl")
    print(f"  Sphere文件格式: {base_filename}_XXXX_sphere.pkl")
    print(
        f"路径长度 - 平均: {np.mean(path_lengths):.2f}, 最小: {np.min(path_lengths)}, 最大: {np.max(path_lengths)}"
    )

    return problems


def main():
    parser = argparse.ArgumentParser(
        description="生成机器人路径规划问题数据集（连续变化障碍物）"
    )

    parser.add_argument(
        "--robot-name", type=str, default="iiwa", help="Robot name identifier"
    )
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--num-obstacles", type=int, default=10)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--max-time", type=float, default=5.0)
    parser.add_argument("--workspace-min", type=float, default=-0.8)
    parser.add_argument("--workspace-max", type=float, default=0.8)
    parser.add_argument("--voxel-size-min", type=float, default=0.12)
    parser.add_argument("--voxel-size-max", type=float, default=0.20)
    parser.add_argument("--safe-zone-radius", type=float, default=0.15)
    parser.add_argument(
        "--move-step", type=float, default=0.02, help="Obstacle move step"
    )
    parser.add_argument(
        "--move-direction", type=str, default="1,0,0", help="Move direction as x,y,z"
    )
    parser.add_argument(
        "--enable-self-collision",
        action="store_true",
        help="Enable self-collision detection",
    )

    args = parser.parse_args()

    move_direction = tuple(map(float, args.move_direction.split(",")))

    generate_problem_dataset(
        robot_name=args.robot_name,
        num_problems=args.num_problems,
        num_obstacles=args.num_obstacles,
        output_file=args.output_file,
        max_planning_time=args.max_time,
        workspace_range=(args.workspace_min, args.workspace_max),
        voxel_size_range=(args.voxel_size_min, args.voxel_size_max),
        safe_zone_radius=args.safe_zone_radius,
        enable_self_collision=args.enable_self_collision,
        move_step=args.move_step,
        move_direction=move_direction,
    )

    return 0


if __name__ == "__main__":
    exit(main())
