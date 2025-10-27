#!/usr/bin/env python3
"""
测试程序：读取无碰撞问题集并可视化

读取 kuka_iiwa_13_200_no_coll.pkl 中的问题集，
使用 ModularEnv 加载障碍物和机器人，按顺序可视化每个配置。
"""

import sys
import os
import pickle
import time

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from environment.modular_env import ModularEnv


def load_obstacles_to_env(modular_env, obstacles):
    """
    将特定的障碍物列表加载到环境中

    Args:
        modular_env: ModularEnv 实例
        obstacles: 障碍物列表 [(halfExtents, basePosition), ...]
    """
    # 加载障碍物到环境中
    modular_env.obstacle_manager.load_and_init_obstacles_from_data(obstacles)

    # 更新碰撞环境中的障碍物
    modular_env.collision_env.load_obstacle_body_ids(
        modular_env.obstacle_manager.obstacle_body_ids
    )


def visualize_problem_sequence(modular_env, problem, delay=1.0):
    """
    可视化单个问题的配置序列

    Args:
        modular_env: ModularEnv 实例
        problem: (obstacles, start, goal, path)
        delay: 每个配置的显示延迟（秒）
    """
    obstacles, start, goal, path = problem

    print(f"可视化问题: 障碍物数量 {len(obstacles)}, 路径长度 {len(path)}")

    # 加载障碍物
    load_obstacles_to_env(modular_env, obstacles)

    # 显示起点
    print("显示起点...")
    modular_env.robot_env.set_config(start)
    time.sleep(delay)

    # 显示终点
    print("显示终点...")
    modular_env.robot_env.set_config(goal)
    time.sleep(delay)

    # 显示路径中的每个配置
    print("显示路径...")
    for i, config in enumerate(path):
        print(f"配置 {i + 1}/{len(path)}")
        modular_env.robot_env.set_config(config)
        time.sleep(delay * 0.5)  # 路径点显示稍快

    print("问题可视化完成")


def main():
    # 文件路径
    pkl_file = "maze_files/kuka_iiwa_13_200_no_coll.pkl"

    if not os.path.exists(pkl_file):
        print(f"错误: 文件 {pkl_file} 不存在")
        return 1

    # 读取问题集
    print(f"读取问题集: {pkl_file}")
    with open(pkl_file, "rb") as f:
        problems = pickle.load(f)

    print(f"加载了 {len(problems)} 个问题")

    # 创建模块化环境（启用GUI）
    robot_file = "kuka_iiwa/model_0.urdf"
    print(f"创建环境: {robot_file} (GUI模式)")
    modular_env = ModularEnv(robot_file, GUI=True)

    try:
        # 逐个可视化问题
        for i, problem in enumerate(problems):
            print(f"\n=== 可视化问题 {i + 1}/{len(problems)} ===")
            visualize_problem_sequence(modular_env, problem, delay=1.0)

            # 等待用户输入继续或退出
            try:
                user_input = input("按 Enter 继续下一个问题，输入 'q' 退出: ")
                if user_input.lower() == "q":
                    break
            except KeyboardInterrupt:
                print("\n用户中断")
                break

    except Exception as e:
        print(f"可视化过程中出错: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 清理环境
        modular_env.obstacle_manager.cleanup_obstacles()
        modular_env.close()
        print("环境已清理")

    return 0


if __name__ == "__main__":
    exit(main())
