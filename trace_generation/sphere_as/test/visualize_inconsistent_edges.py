#!/usr/bin/env python3
"""
可视化不一致的edge数据

包括加载不一致edge数据、robot URDF可视化和球体建模可视化。
"""

import pickle
import os
import sys
import select
import argparse

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from robot_as.robot_method import RobotEnv
from sphere_method import SphereEnv
from robot_as.modular_env import ModularEnv


def load_inconsistent_edges(file_path):
    """
    加载不一致的edge数据

    Args:
        file_path: 不一致edge数据文件路径

    Returns:
        dict: 包含obstacles, edge_configs, obb_edge_collision, sphere_edge_collision的字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data


def verify_collision(edge_data, method, robot_file, robot_name):
    """
    复核碰撞检查结果

    Args:
        edge_data: edge数据
        method: 'urdf' 或 'sphere'
        robot_file: URDF文件路径
        robot_name: 机器人名称
    """
    print(f"开始复核碰撞检查 ({method})...")

    edge_configs = edge_data.get("edge_configs", [])
    obstacles = edge_data.get("obstacles", [])

    if method == "urdf":
        # 使用ModularEnv
        modular_env = ModularEnv(robot_file)
        # 加载障碍物
        if obstacles:
            modular_env.obstacle_manager.load_and_init_obstacles_from_data(obstacles)

        for i, edge in enumerate(edge_configs):
            # edge_collision = []
            # for config in edge:
            #     if config is not None:
            #         _,_,collision = modular_env._state_fp_probe(config)
            #         edge_collision.append(collision)

            _, _, edge_collision = (
                modular_env.collision_env._collect_edge_collision_data(edge)
            )
            # 比较与原OBB结果
            original_collision = edge_data["obb_edge_collision"][i]
            if edge_collision != original_collision:
                print(
                    f"Edge {i}: 复核碰撞 = {edge_collision}, 原OBB = {original_collision} - 不一致，退出"
                )
                # modular_env.close()
                # sys.exit(1)
            else:
                print(f"Edge {i}: 复核碰撞 = {edge_collision}")

        modular_env.close()

    elif method == "sphere":
        # 使用SphereEnv
        sphere_env = SphereEnv(robot_name=robot_name)
        if obstacles:
            sphere_env.init_obstacle_bodies(len(obstacles), obstacles)

        for i, edge in enumerate(edge_configs):
            edge_collision = False
            for config in edge:
                if config is not None:
                    collision, _, colls = sphere_env.get_sphere_collision_data(config)
                    if collision:
                        edge_collision = True
            # 比较与原Sphere结果
            original_collision = edge_data["sphere_edge_collision"][i]
            if edge_collision != original_collision:
                print(
                    f"Edge {i}: 复核碰撞 = {edge_collision}, 原Sphere = {original_collision} - 不一致，退出"
                )
                sphere_env.close()
                sys.exit(1)
            print(f"Edge {i}: 复核碰撞 = {edge_collision}")

        sphere_env.close()

    print("复核完成。")


def visualize_robot_urdf(
    edge_data,
    robot_file="/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf",
):
    """
    使用URDF可视化不一致edge

    Args:
        edge_data: 从load_inconsistent_edges加载的数据
        robot_file: 机器人URDF文件路径
    """
    # 复核碰撞检查
    verify_collision(edge_data, "urdf", robot_file, None)

    print("初始化URDF可视化环境...")

    # 初始化机器人环境（GUI模式）
    robot_env = RobotEnv(robot_file, OBB_GUI=True)

    # 加载障碍物
    obstacles = edge_data.get("obstacles", [])
    obstacle_ids = []
    if obstacles:
        print(f"加载 {len(obstacles)} 个障碍物")
        import pybullet as p

        for obs in obstacles:
            # 假设obs是[half_extents, base_position]格式
            if len(obs) == 2:
                half_extents, base_position = obs
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

    edge_configs = edge_data.get("edge_configs", [])
    print(f"共 {len(edge_configs)} 个不一致edge")

    for i, edge in enumerate(edge_configs):
        print(f"\n显示edge {i + 1}/{len(edge_configs)}")

        while True:
            for j, config in enumerate(edge):
                print(f"  配置 {j + 1}/{len(edge)}: {config}")
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


def visualize_sphere_model(edge_data, robot_name="franka"):
    """
    使用球体模型可视化不一致edge

    Args:
        edge_data: 从load_inconsistent_edges加载的数据
        robot_name: 机器人名称
    """
    # 复核碰撞检查
    verify_collision(edge_data, "sphere", None, robot_name)

    print("初始化球体可视化环境...")

    # 初始化球体环境（GUI模式）
    sphere_env = SphereEnv(robot_name=robot_name, SPH_GUI=True)

    # 加载障碍物
    obstacles = edge_data.get("obstacles", [])
    if obstacles:
        print(f"加载 {len(obstacles)} 个障碍物")
        sphere_env.init_obstacle_bodies(len(obstacles), obstacles)

    edge_configs = edge_data.get("edge_configs", [])
    print(f"共 {len(edge_configs)} 个不一致edge")

    for i, edge in enumerate(edge_configs):
        print(f"\n显示edge {i + 1}/{len(edge_configs)}")

        while True:
            for j, config in enumerate(edge):
                print(f"  配置 {j + 1}/{len(edge)}: {config}")
                if config is not None:
                    sphere_env._update_sphere_positions(config)

            # 检查是否有键盘输入
            if select.select([sys.stdin], [], [], 0)[0]:
                # 有输入，读取并丢弃，然后退出循环
                sys.stdin.readline()
                break

    sphere_env.close()
    print("球体可视化完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化不一致的edge数据")
    parser.add_argument(
        "--file-path",
        type=str,
        default="../inconsistent_edge/inconsistent_edges_2.pkl",
        help="不一致edge数据文件路径",
    )
    parser.add_argument("--urdf", action="store_true", help="使用URDF可视化")
    parser.add_argument("--sphere", action="store_true", help="使用球体模型可视化")
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
        help="机器人名称（sphere选项时需要）",
    )

    args = parser.parse_args()

    try:
        data = load_inconsistent_edges(args.file_path)
        print("加载成功，数据键:", list(data.keys()))

        if args.urdf:
            visualize_robot_urdf(data, args.robot_file)
        elif args.sphere:
            visualize_sphere_model(data, args.robot_name)
        else:
            print("请指定 --urdf 或 --sphere")
    except FileNotFoundError as e:
        print(e)
