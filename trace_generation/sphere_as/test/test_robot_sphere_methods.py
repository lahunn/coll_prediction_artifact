#!/usr/bin/env python3
"""
测试 robot_env 中的 _are_links_adjacent 方法和 sphere_env 中的自碰撞方法
"""

import sys
import os
import numpy as np
import pybullet as p

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
# 导入模块
from robot_as.modular_env import ModularEnv
from sphere_method import SphereEnv

robot_name = "iiwa"


def print_link_names():
    """输出link名与link序号的对应关系"""
    print("=== Link名与序号对应关系 ===")

    # 初始化ModularEnv
    modular_env = ModularEnv(robot_name)
    robot_env = modular_env.robot_env

    print("所有关节信息:")
    num_joints = p.getNumJoints(
        robot_env.robotId, physicsClientId=robot_env.physics_client
    )
    for i in range(num_joints):
        joint_info = p.getJointInfo(
            robot_env.robotId, i, physicsClientId=robot_env.physics_client
        )
        link_name = joint_info[12].decode("utf-8") if joint_info[12] else "N/A"
        print(f"  关节 {i}: link名='{link_name}', 序号={i}")

    print(f"\n有效碰撞links ({len(robot_env.valid_collision_links)} 个):")
    for idx in robot_env.valid_collision_links:
        if idx == -1:
            print(f"  Link序号: {idx}, 名称: 'base'")
        else:
            joint_info = p.getJointInfo(
                robot_env.robotId, idx, physicsClientId=robot_env.physics_client
            )
            link_name = joint_info[12].decode("utf-8") if joint_info[12] else "N/A"
            print(f"  Link序号: {idx}, 名称: '{link_name}'")

    modular_env.close()


def test_are_links_adjacent():
    """测试 _are_links_adjacent 方法"""
    print("=== 测试 _are_links_adjacent 方法 ===")

    # 初始化ModularEnv
    modular_env = ModularEnv(robot_name)
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

    # 初始化ModularEnv（无GUI，启用自碰撞检测）
    modular_env = ModularEnv(robot_name)

    # 初始化球体环境（无GUI）
    sphere_env = SphereEnv(robot_env=modular_env.robot_env, robot_name=robot_name)

    # 获取关节限位（假设franka的限位）

    # 生成100个随机配置
    test_configs = modular_env.robot_env.sample_n_points(100)

    consistent_count = 0
    inconsistent_count = 0
    modular_collision_count = 0
    sphere_collision_count = 0

    print(f"测试 {len(test_configs)} 个随机配置...")

    for i, joint_state in enumerate(test_configs):
        # 检查modular_env的自碰撞（使用_state_fp，True表示无碰撞）
        modular_free, _, modular_link_colls = (
            modular_env.collision_env._point_in_free_space(joint_state)
        )
        modular_collision = not modular_free  # True if collision

        # 检查sphere_env的自碰撞
        sphere_env._update_sphere_positions(joint_state)
        sphere_collision, sphere_colls = sphere_env._check_sphere_collision(joint_state)

        modular_collision_count += int(modular_collision)
        sphere_collision_count += int(sphere_collision)

        if modular_collision == sphere_collision:
            consistent_count += 1
        else:
            # 获取碰撞的link和sphere序号
            colliding_links = [
                j for j, coll in enumerate(modular_link_colls) if coll == 0
            ]
            colliding_spheres = [j for j, coll in enumerate(sphere_colls) if coll == 0]
            print(
                f"配置 {i} 不一致: Modular碰撞={modular_collision}, Sphere碰撞={sphere_collision}"
            )
            print(f"  碰撞links: {colliding_links}")
            print(f"  碰撞spheres: {colliding_spheres}")
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


def test_random_obstacles_collision():
    """测试随机障碍物和随机pose的碰撞检测"""
    print("\n=== 测试随机障碍物和随机pose的碰撞检测 ===")

    # 初始化ModularEnv（无GUI）
    modular_env = ModularEnv(robot_name)

    # 初始化球体环境（无GUI）
    sphere_env = SphereEnv(robot_env=modular_env.robot_env, robot_name=robot_name)

    # 生成随机障碍物
    num_obstacles = 15
    workspace_range = (-1.0, 1.0)
    voxel_size_range = (0.05, 0.15)
    safe_zone_center = (0.0, 0.0, 0.0)
    safe_zone_radius = 0.3

    obstacles = modular_env.generate_random_obstacles(
        num_obstacles=num_obstacles,
        workspace_range=workspace_range,
        voxel_size_range=voxel_size_range,
        safe_zone_center=safe_zone_center,
        safe_zone_radius=safe_zone_radius,
    )

    # 初始化球体环境的障碍物
    sphere_env.load_obstacles(obstacles)

    print(f"生成了 {num_obstacles} 个随机障碍物")

    # 生成随机pose
    num_poses = 10000
    test_configs = modular_env.robot_env.sample_n_points(num_poses)

    print(f"测试 {num_poses} 个随机pose...")

    modular_collision_count = 0
    sphere_collision_count = 0
    inconsistent_count = 0

    for i, joint_state in enumerate(test_configs):
        # ModularEnv 碰撞检测
        modular_free, _, modular_link_colls = (
            modular_env.collision_env._point_in_free_space(joint_state)
        )
        modular_collision = not modular_free
        modular_collision_count += int(modular_collision)

        # SphereEnv 碰撞检测
        sphere_collision, sphere_colls = sphere_env._check_sphere_collision(joint_state)
        sphere_collision_count += int(sphere_collision)

        if modular_collision != sphere_collision:
            inconsistent_count += 1
            # 获取碰撞的link和sphere序号
            colliding_links = [
                j for j, coll in enumerate(modular_link_colls) if coll == 0
            ]
            colliding_spheres = [j for j, coll in enumerate(sphere_colls) if coll == 0]
            print(
                f"配置 {i}: sphere_collision={sphere_collision}, modular_collision={modular_collision}"
            )
            print(f"  碰撞links: {colliding_links}")
            print(f"  碰撞spheres: {colliding_spheres}")

        if (i + 1) % 10 == 0:
            print(f"已测试 {i + 1} 个pose")

    print("\n测试结果:")
    print(f"  总pose数: {num_poses}")
    print(f"  Modular检测到碰撞: {modular_collision_count}")
    print(f"  Sphere检测到碰撞: {sphere_collision_count}")
    print(f"  不一致数量: {inconsistent_count}")
    if num_poses > 0:
        print(f"  不一致率: {inconsistent_count / num_poses * 100:.2f}%")

    modular_env.close()
    sphere_env.close()


def main():
    """主测试函数"""
    print("开始测试 robot_env 和 sphere_env 的方法...")

    try:
        print_link_names()
        # test_are_links_adjacent()
        # test_self_collision_geometric()
        test_random_obstacles_collision()
        print("\n所有测试完成!")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
