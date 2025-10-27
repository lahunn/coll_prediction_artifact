#!/usr/bin/env python3
"""
使用示例: 展示如何使用collision_env保存碰撞数据
"""

import sys

sys.path.append("..")
from environment.collision_env import CollisionEnv
from environment.robot_env import RobotEnv
from environment.obstacle_manager import ObstacleManager
import numpy as np


def main():
    print("=== 碰撞数据收集示例 ===\n")

    # 初始化环境
    robot_urdf = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf"

    # 创建机器人环境
    robot_env = RobotEnv(robot_urdf, OBB_GUI=False)

    # 创建碰撞检测环境
    env = CollisionEnv(robot_env=robot_env, config_output_file="test_configs.pkl")

    # 创建障碍物管理器
    obstacle_manager = ObstacleManager(env.robot_env.physics_client)

    # 创建障碍物
    obstacles = [
        (np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.0, 0.3])),
        (np.array([0.15, 0.15, 0.1]), np.array([-0.3, 0.4, 0.2])),
    ]
    obstacle_ids = obstacle_manager.init_obstacle_bodies(len(obstacles), obstacles)
    env.load_obstacle_body_ids(obstacle_ids)
    print(f"✓ 创建了 {len(obstacles)} 个障碍物\n")

    # 测试场景1: 单点检测 (_state_fp)
    print("场景1: 单点检测")
    state1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result1 = env._state_fp(state1)
    print(f"  配置1: {'无碰撞' if result1 else '碰撞'}")
    print(f"  收集边数: {len(env.data_manager.obb_link_data)}\n")

    # 测试场景2: 边检测 (_edge_fp)
    print("场景2: 边检测")
    state_start = np.array([0.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0])
    state_end = np.array([0.5, -0.5, 0.3, -1.0, 0.0, 1.5, 0.0])
    result2 = env._edge_fp(state_start, state_end, RRT_EPS=0.25)
    print(f"  边检测: {'无碰撞' if result2 else '碰撞'}")
    print(f"  总收集边数: {len(env.data_manager.obb_link_data)}\n")

    # 保存结果
    print("保存数据...")
    with open("test_configs.pkl", "wb") as f:
        import pickle

        pickle.dump(env.config_list, f)

    env.data_manager.save_collision_data("test_obb_data.pkl")

    print("\n✓ 配置数据: test_configs.pkl")
    print(f"✓ OBB数据: test_obb_data.pkl ({len(env.data_manager.obb_link_data)} 条边)")

    # 清理
    obstacle_manager.cleanup_obstacles()
    env.close()
    robot_env.close()
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()
