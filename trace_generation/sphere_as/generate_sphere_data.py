#!/usr/bin/env python3
"""
对比ModularEnv和球体碰撞检测结果

该程序读取obstacle_config_file，
使用ModularEnv和球体模型分别计算碰撞数据，并进行对比。
"""

import sys
import os
import pickle
import argparse

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from sphere_method import SphereEnv
from robot_as.robot_method import RobotEnv


def compare_collision(
    obstacle_config_file,
    collision_data_file,
    robot_name="franka",
    output_file=None,
    benchmark_id=None,
    enable_self_collision=False,
):
    """
    对比OBB和球体碰撞检测结果

    Args:
        obstacle_config_file: 障碍物-配置文件路径
        collision_data_file: OBB碰撞数据文件路径
        robot_name: 机器人名称
        output_file: 球体碰撞数据输出文件路径（可选）
        benchmark_id: 基准测试ID
        enable_self_collision: 是否启用自碰撞检测
    """
    print(f"加载obstacle_config_file: {obstacle_config_file}")
    with open(obstacle_config_file, "rb") as f:
        obstacle_data = pickle.load(f)
    obstacles = obstacle_data["obstacles"]
    configs = obstacle_data["configs"]

    print(f"加载collision_data_file: {collision_data_file}")
    with open(collision_data_file, "rb") as f:
        obb_data, obb_link_coll_data = pickle.load(f)

    print(f"障碍物数量: {len(obstacles)}")
    print(f"边数量: {len(configs)}")
    print(f"自碰撞检测: {'启用' if enable_self_collision else '禁用'}")

    # 创建机器人环境和球体环境
    robot_env = RobotEnv(
        robot_name, OBB_GUI=False, enable_self_collision=enable_self_collision
    )
    sphere_env = SphereEnv(robot_env=robot_env, robot_name=robot_name, SPH_GUI=False)

    # 加载障碍物
    sphere_env.load_obstacles(obstacles)

    inconsistent_count = 0
    sphere_collision_obb_free_count = 0  # sphere碰撞，obb无碰撞
    sphere_free_obb_collision_count = 0  # sphere无碰撞，obb碰撞

    if len(configs) > len(obb_link_coll_data):
        print("警告: OBB数据中缺少部分edge")

    # 处理每个edge
    for i, edge_configs in enumerate(configs):
        obb_edge = obb_link_coll_data[i]
        if not obb_edge:
            continue

        edge_sphere_coords = []
        edge_sphere_colls = []

        if len(edge_configs) > len(obb_edge):
            print(f"警告: OBB数据中edge {i}缺少pose")
            continue

        # 初始化edge层面的碰撞标志
        obb_edge_collision = any(
            any(coll == 0 for coll in pose_colls) for pose_colls in obb_edge
        )
        sphere_edge_collision = False

        # 处理edge中的每个pose
        for j, config in enumerate(edge_configs):
            # 获取球体碰撞数据
            collision, coords, colls = sphere_env.get_sphere_collision_data(config)
            edge_sphere_coords.append(coords)
            edge_sphere_colls.append(colls)

            # 更新sphere edge碰撞标志
            if any(coll == 0 for coll in colls):
                sphere_edge_collision = True

        # 在edge层面检查一致性
        if obb_edge_collision != sphere_edge_collision:
            print(
                f"Edge {i}: OBB={obb_edge_collision}, Sphere={sphere_edge_collision} - 不一致!"
            )
            inconsistent_count += 1
            if not obb_edge_collision and sphere_edge_collision:
                sphere_collision_obb_free_count += 1
            elif obb_edge_collision and not sphere_edge_collision:
                sphere_free_obb_collision_count += 1

        # 存储球体数据（无论是否一致）
        if edge_sphere_coords:
            sphere_env.store_sphere_data(
                edge_sphere_coords, edge_sphere_colls, is_edge=True
            )

    # 清理资源
    sphere_env.cleanup_obstacles()
    sphere_env.close()
    robot_env.close()

    print(f"对比完成，发现 {inconsistent_count} 个不一致的配置")
    print(f"  Sphere碰撞但OBB无碰撞: {sphere_collision_obb_free_count}")
    print(f"  Sphere无碰撞但OBB碰撞: {sphere_free_obb_collision_count}")

    # 保存球体碰撞数据（如果指定了输出文件）
    if output_file:
        sphere_env.save_collision_data(output_file)


def main():
    parser = argparse.ArgumentParser(description="对比OBB和球体碰撞检测结果")

    parser.add_argument(
        "--obstacle-config-file", type=str, required=True, help="障碍物-配置文件路径"
    )
    parser.add_argument(
        "--collision-data-file", type=str, required=True, help="OBB碰撞数据文件路径"
    )
    parser.add_argument("--robot-name", type=str, default="franka", help="机器人名称")
    parser.add_argument("--benchmark-id", type=int, help="基准测试ID，用于命名输出文件")
    parser.add_argument("--output-file", type=str, help="球体碰撞数据输出文件路径")
    parser.add_argument(
        "--enable-self-collision",
        action="store_true",
        help="启用自碰撞检测",
    )

    args = parser.parse_args()

    compare_collision(
        args.obstacle_config_file,
        args.collision_data_file,
        args.robot_name,
        args.output_file,
        args.benchmark_id,
        args.enable_self_collision,
    )

    return 0


if __name__ == "__main__":
    exit(main())
