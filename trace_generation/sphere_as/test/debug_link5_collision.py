#!/usr/bin/env python3
"""
调试 link5 的异常碰撞问题
"""

import sys
import os
import numpy as np
import pybullet as p

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from robot_as.modular_env import ModularEnv

robot_name = "iiwa"


def debug_link5_collision():
    """调试 link5 的碰撞问题"""
    print("=== 调试 link5 碰撞问题 ===\n")

    # 初始化环境（带GUI方便观察）
    modular_env = ModularEnv(robot_name, GUI=False)
    robot_env = modular_env.robot_env

    # 生成一些随机配置
    test_configs = modular_env.robot_env.sample_n_points(100)

    print(f"Valid collision links: {robot_env.valid_collision_links}\n")

    # 找到一个导致 link5 碰撞的配置
    problem_config = None
    for i, joint_state in enumerate(test_configs):
        modular_free, _, modular_link_colls = (
            modular_env.collision_env._point_in_free_space(joint_state)
        )

        if not modular_free:
            colliding_links = [
                j for j, coll in enumerate(modular_link_colls) if coll == 0
            ]
            if colliding_links == [5]:
                problem_config = joint_state
                print(f"找到问题配置 {i}:")
                print(f"  关节角度: {joint_state}")
                print(f"  碰撞 links: {colliding_links}\n")
                break

    if problem_config is None:
        print("未找到导致 link5 单独碰撞的配置")
        modular_env.close()
        return

    # 设置机器人到问题配置
    robot_env.set_config(problem_config)

    # 执行碰撞检测
    p.performCollisionDetection(physicsClientId=robot_env.physics_client)

    # 详细检查 link5 的碰撞情况
    link5_idx = robot_env.valid_collision_links[5]  # 获取 link5 的实际索引

    print(f"Link5 的实际索引: {link5_idx}")

    # 获取 link5 的碰撞信息
    joint_info = p.getJointInfo(
        robot_env.robotId, link5_idx, physicsClientId=robot_env.physics_client
    )
    print(f"Link5 名称: {joint_info[12].decode('utf-8')}")

    # 检查 link5 与所有物体的接触
    print("\n检查 link5 与机器人自身的接触:")
    contacts = p.getContactPoints(
        bodyA=robot_env.robotId,
        bodyB=robot_env.robotId,
        linkIndexA=link5_idx,
        physicsClientId=robot_env.physics_client,
    )

    print(f"找到 {len(contacts)} 个接触点:")
    for i, contact in enumerate(contacts):
        linkIndexB = contact[4]  # 对方的 link 索引
        contactDistance = contact[8]
        contactNormal = contact[7]
        positionOnA = contact[5]
        positionOnB = contact[6]

        # 获取对方 link 的名称
        if linkIndexB == -1:
            linkB_name = "base"
        else:
            joint_info_b = p.getJointInfo(
                robot_env.robotId, linkIndexB, physicsClientId=robot_env.physics_client
            )
            linkB_name = joint_info_b[12].decode("utf-8")

        print(f"\n  接触点 {i}:")
        print(f"    对方 link 索引: {linkIndexB} ({linkB_name})")
        print(f"    接触距离: {contactDistance:.6f}")
        print(f"    接触法向: {contactNormal}")
        print(f"    link5 接触位置: {positionOnA}")
        print(f"    对方接触位置: {positionOnB}")

    # 检查相邻关系
    print("\n检查 link5 的相邻关系:")
    for other_link_idx in robot_env.valid_collision_links:
        if other_link_idx != link5_idx:
            is_adjacent = robot_env._are_links_adjacent(link5_idx, other_link_idx)
            if is_adjacent:
                if other_link_idx == -1:
                    other_name = "base"
                else:
                    joint_info_other = p.getJointInfo(
                        robot_env.robotId,
                        other_link_idx,
                        physicsClientId=robot_env.physics_client,
                    )
                    other_name = joint_info_other[12].decode("utf-8")
                print(f"  Link5 与 link{other_link_idx} ({other_name}) 相邻")

    # 检查碰撞过滤设置
    print("\n检查 PyBullet 的碰撞过滤设置:")
    print("(注意: PyBullet 的 getCollisionFilterPair 可能不可用)")

    modular_env.close()


if __name__ == "__main__":
    debug_link5_collision()
