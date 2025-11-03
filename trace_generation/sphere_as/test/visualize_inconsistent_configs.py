#!/usr/bin/env python3
"""
可视化不一致配置程序

该程序加载不一致配置数据，并在可视化界面中显示。
提供两个选项：
1. 加载URDF并可视化机器人
2. 加载球体模型并可视化

使用方法：
python visualize_inconsistent_configs.py --input-file path/to/inconsistent.pkl --option urdf
python visualize_inconsistent_configs.py --input-file path/to/inconsistent.pkl --option spheres
"""

import sys
import os
import pickle
import argparse
import select

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from robot_as.robot_method import RobotEnv
from sphere_method import SphereEnv


def visualize_urdf(inconsistent_poses, obstacles, robot_file):
    """
    使用URDF可视化不一致配置

    Args:
        inconsistent_poses: 不一致配置列表
        obstacles: 障碍物列表
        robot_file: 机器人URDF文件路径
    """
    print("初始化URDF可视化环境...")

    # 初始化机器人环境（GUI模式）
    robot_env = RobotEnv(robot_file, OBB_GUI=True)

    # 加载障碍物
    obstacle_ids = []
    if obstacles:
        print(f"加载 {len(obstacles)} 个障碍物")
        import pybullet as p

        for half_extents, base_position in obstacles:
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=robot_env.physics_client,
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=robot_env.physics_client,
            )
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=base_position,
                physicsClientId=robot_env.physics_client,
            )
            obstacle_ids.append(obstacle_id)

    print(f"共 {len(inconsistent_poses)} 个不一致配置")

    for i, pose_data in enumerate(inconsistent_poses):
        print(f"\n显示配置 {i + 1}/{len(inconsistent_poses)}")
        print(f"Edge索引: {pose_data['edge_idx']}")
        print(f"OBB碰撞: {pose_data['obb_collision']}")
        print(f"Sphere碰撞: {pose_data['sphere_collision']}")

        # 按顺序逐个加载所有config
        configs = pose_data.get("configs", [])
        if len(configs) == 0:
            print("  无配置数据，跳过")
            continue

        while True:
            for j, config in enumerate(configs):
                print(f"  配置 {j + 1}/{len(configs)}: {config}")
                if config is not None:
                    robot_env.set_config(config)

            # 检查是否有键盘输入
            if select.select([sys.stdin], [], [], 0)[0]:
                # 有输入，读取并丢弃，然后退出循环
                sys.stdin.readline()
                break

    # 清理障碍物
    import pybullet as p

    for obstacle_id in obstacle_ids:
        p.removeBody(obstacle_id, physicsClientId=robot_env.physics_client)

    robot_env.close()
    print("URDF可视化完成")


def visualize_spheres(inconsistent_poses, obstacles, robot_name):
    """
    使用球体模型可视化不一致配置

    Args:
        inconsistent_poses: 不一致配置列表
        obstacles: 障碍物列表
        robot_name: 机器人名称
    """
    print("初始化球体可视化环境...")

    # 初始化球体环境（GUI模式）
    sphere_env = SphereEnv(robot_name=robot_name, SPH_GUI=True)

    # 加载障碍物
    if obstacles:
        print(f"加载 {len(obstacles)} 个障碍物")
        sphere_env.load_obstacles(obstacles)

    print(f"共 {len(inconsistent_poses)} 个不一致配置")

    for i, pose_data in enumerate(inconsistent_poses):
        print(f"\n显示配置 {i + 1}/{len(inconsistent_poses)}")
        print(f"Edge索引: {pose_data['edge_idx']}")
        print(f"OBB碰撞: {pose_data['obb_collision']}")
        print(f"Sphere碰撞: {pose_data['sphere_collision']}")

        # 按顺序逐个加载所有config
        configs = pose_data.get("configs", [])
        if len(configs) == 0:
            print("  无配置数据，跳过")
            continue

        while True:
            for j, config in enumerate(configs):
                print(f"  配置 {j + 1}/{len(configs)}: {config}")
                if config is not None:
                    sphere_env._update_sphere_positions(config)

            # 检查是否有键盘输入
            if select.select([sys.stdin], [], [], 0)[0]:
                # 有输入，读取并丢弃，然后退出循环
                sys.stdin.readline()
                break

    sphere_env.close()
    print("球体可视化完成")


def main():
    parser = argparse.ArgumentParser(description="可视化不一致配置程序")

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="不一致配置数据文件路径（_inconsistent.pkl）",
    )
    parser.add_argument(
        "--option",
        choices=["urdf", "spheres"],
        required=True,
        help="可视化选项：urdf 或 spheres",
    )
    parser.add_argument(
        "--robot-file",
        type=str,
        default="/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf",
        help="机器人URDF文件路径（urdf选项时需要）",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="franka",
        help="机器人名称（spheres选项时需要）",
    )

    args = parser.parse_args()

    # 加载不一致数据
    print(f"加载不一致配置数据: {args.input_file}")
    with open(args.input_file, "rb") as f:
        data = pickle.load(f)

    inconsistent_poses = data["inconsistent_poses"]
    obstacles = data["obstacles"]

    print(f"加载完成，不一致配置数量: {len(inconsistent_poses)}")

    # 根据选项调用相应函数
    if args.option == "urdf":
        visualize_urdf(inconsistent_poses, obstacles, args.robot_file)
    elif args.option == "spheres":
        visualize_spheres(inconsistent_poses, obstacles, args.robot_name)

    return 0


if __name__ == "__main__":
    exit(main())
