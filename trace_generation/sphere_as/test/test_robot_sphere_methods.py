#!/usr/bin/env python3
"""
测试 robot_env 中的 _are_links_adjacent 方法和 sphere_env 中的自碰撞方法
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
# 导入模块
from robot_as.modular_env import ModularEnv
from sphere_method import SphereEnv
from utils.planning_utils import uniform_sample


def test_are_links_adjacent():
    """测试 _are_links_adjacent 方法"""
    print("=== 测试 _are_links_adjacent 方法 ===")

    # 初始化ModularEnv
    robot_file = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf"
    modular_env = ModularEnv(robot_file)
    robot_env = modular_env.robot_env

    print(f"有效碰撞link数量: {len(robot_env.valid_collision_links)}")
    print(f"有效碰撞links: {robot_env.valid_collision_links}")

    # 测试相邻links
    print("\n测试相邻links:")
    for i in range(len(robot_env.valid_collision_links) - 1):
        link1 = robot_env.valid_collision_links[i]
        link2 = robot_env.valid_collision_links[i + 1]
        result = robot_env._are_links_adjacent(link1, link2)
        print(f"  Link {link1} 和 Link {link2} 相邻: {result}")

    # 测试不相邻links
    print("\n测试不相邻links:")
    if len(robot_env.valid_collision_links) > 2:
        link1 = robot_env.valid_collision_links[0]
        link3 = (
            robot_env.valid_collision_links[2]
            if len(robot_env.valid_collision_links) > 2
            else robot_env.valid_collision_links[-1]
        )
        result = robot_env._are_links_adjacent(link1, link3)
        print(f"  Link {link1} 和 Link {link3} 相邻: {result}")

    # 测试无效links
    print("\n测试无效links:")
    result = robot_env._are_links_adjacent(-999, robot_env.valid_collision_links[0])
    print(
        f"  无效Link -999 和有效Link {robot_env.valid_collision_links[0]} 相邻: {result}"
    )

    modular_env.close()


def test_self_collision_geometric():
    """测试 sphere_env 中的自碰撞方法"""
    print("\n=== 测试自碰撞方法 _check_self_collision_geometric ===")

    # 初始化ModularEnv（无GUI）
    robot_file = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf"
    modular_env = ModularEnv(robot_file)

    # 初始化球体环境（无GUI）
    sphere_env = SphereEnv(robot_file)

    # 获取关节限位（假设franka的限位）

    # 生成100个随机配置
    test_configs = modular_env.robot_env.sample_n_points(1000)

    consistent_count = 0
    inconsistent_count = 0
    modular_collision_count = 0
    sphere_collision_count = 0

    print(f"测试 {len(test_configs)} 个随机配置...")

    for i, joint_state in enumerate(test_configs):
        # 检查modular_env的自碰撞（使用_state_fp，True表示无碰撞）
        modular_free = modular_env._state_fp(joint_state)
        modular_collision = not modular_free  # True if collision

        # 检查sphere_env的自碰撞
        sphere_env._update_sphere_positions(joint_state)
        sphere_collision, _ = sphere_env._check_self_collision_geometric(joint_state)
        
        modular_collision_count += int(modular_collision)
        sphere_collision_count += int(sphere_collision)

        if modular_collision == sphere_collision:
            consistent_count += 1
        else:
            inconsistent_count += 1

        if (i + 1) % 10 == 0:
            print(f"已测试 {i + 1} 个配置")

    print("\n测试结果:")
    print(f"  总配置数: {len(test_configs)}")
    print(f"  一致数量: {consistent_count}")
    print(f"  不一致数量: {inconsistent_count}")
    print(f"  Modular检测到碰撞: {modular_collision_count}")
    print(f"  Sphere检测到碰撞: {sphere_collision_count}")
    if inconsistent_count > 0:
        print(f"  不一致率: {inconsistent_count / len(test_configs) * 100:.2f}%")

    modular_env.close()
    sphere_env.close()


def main():
    """主测试函数"""
    print("开始测试 robot_env 和 sphere_env 的方法...")

    try:
        # test_are_links_adjacent()
        test_self_collision_geometric()
        print("\n所有测试完成!")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
