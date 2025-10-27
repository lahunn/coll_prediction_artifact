#!/usr/bin/env python3
"""
测试程序：调用 _setup_joint_info 方法，获取并输出所有有效关节信息
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import pybullet as p
from environment.collision_env import CollisionEnv
from environment.robot_env import RobotEnv


def test_joint_info():
    """测试关节信息获取功能"""
    print("=== 测试关节信息获取 ===")

    # 创建 RobotEnv 实例
    robot_env = RobotEnv("../../../data/robots/franka_description/franka_panda.urdf")

    # 创建 CollisionEnv 实例
    env = CollisionEnv(robot_env=robot_env)

    try:
        # 输出关节基本信息
        print(f"机器人ID: {env.robot_env.robotId}")
        print(f"配置维度 (有效关节数量): {env.robot_env.config_dim}")
        print(f"有效关节索引列表: {env.robot_env.valid_joints}")
        print(f"关节限位列表: {env.robot_env.pose_range}")

        # 获取总关节数量
        num_joints = p.getNumJoints(env.robot_env.robotId, physicsClientId=env.robot_env.physics_client)
        print(f"总关节数量: {num_joints}")

        # 输出所有关节信息（用于对比）
        print("\n=== 所有关节信息 (用于对比) ===")
        for i in range(num_joints):
            joint_info = p.getJointInfo(env.robot_env.robotId, i, physicsClientId=env.robot_env.physics_client)
            joint_name = joint_info[1].decode('utf-8') if joint_info[1] else "Unknown"
            joint_type = joint_info[2]
            joint_type_name = {
                p.JOINT_REVOLUTE: "REVOLUTE",
                p.JOINT_PRISMATIC: "PRISMATIC",
                p.JOINT_FIXED: "FIXED",
                p.JOINT_POINT2POINT: "POINT2POINT",
                p.JOINT_SPHERICAL: "SPHERICAL"
            }.get(joint_type, "UNKNOWN")

            lower_limit = joint_info[8]
            upper_limit = joint_info[9]

            is_valid = "✓" if i in env.robot_env.valid_joints else "✗"

            print(f"关节 {i} {is_valid}:")
            print(f"  名称: {joint_name}")
            print(f"  类型: {joint_type_name} ({joint_type})")
            print(f"  限位: [{lower_limit}, {upper_limit}]")
            print()

        # 输出关节范围信息
        print("=== 有效关节范围信息 ===")
        for i, (joint_id, limits) in enumerate(zip(env.robot_env.valid_joints, env.robot_env.pose_range)):
            print(f"关节 {i} (ID: {joint_id}): 范围 {limits}")

        # 输出预计算的边界
        print(f"\n下限边界: {env.robot_env.lower_bounds}")
        print(f"上限边界: {env.robot_env.upper_bounds}")
        print(f"边界数组: {env.robot_env.bound}")

        # 获取并输出有效关节详细信息
        print("\n=== 有效关节详细信息 ===")
        for i, joint_id in enumerate(env.robot_env.valid_joints):
            joint_info = p.getJointInfo(env.robot_env.robotId, joint_id, physicsClientId=env.robot_env.physics_client)
            joint_name = joint_info[1].decode('utf-8') if joint_info[1] else "Unknown"
            joint_type = joint_info[2]
            joint_type_name = {
                p.JOINT_REVOLUTE: "REVOLUTE",
                p.JOINT_PRISMATIC: "PRISMATIC",
                p.JOINT_FIXED: "FIXED",
                p.JOINT_POINT2POINT: "POINT2POINT",
                p.JOINT_SPHERICAL: "SPHERICAL"
            }.get(joint_type, "UNKNOWN")

            lower_limit = joint_info[8]
            upper_limit = joint_info[9]

            print(f"关节 {i}:")
            print(f"  ID: {joint_id}")
            print(f"  名称: {joint_name}")
            print(f"  类型: {joint_type_name} ({joint_type})")
            print(f"  原始限位: [{lower_limit}, {upper_limit}]")
            print(f"  处理后限位: {env.robot_env.pose_range[i]}")
            print()

        # 测试关节配置获取
        print("=== 当前关节配置 ===")
        current_config = env.robot_env.get_robot_config()
        print(f"当前配置: {current_config}")
        print(f"配置长度: {len(current_config)}")

        # 测试关节配置设置
        print("\n=== 测试关节配置设置 ===")
        test_config = [0.1] * env.robot_env.config_dim
        print(f"设置测试配置: {test_config}")
        env.robot_env.set_config(test_config)

        # 验证设置后的配置
        new_config = env.robot_env.get_robot_config()
        print(f"设置后的配置: {new_config}")

        # 测试修改关节初始状态
        print("\n=== 测试修改关节初始状态 ===")

        # 首先显示所有关节的当前状态（包括固定关节）
        print("所有关节的当前状态:")
        for i in range(num_joints):
            joint_state = p.getJointState(env.robot_env.robotId, i, physicsClientId=env.robot_env.physics_client)
            joint_info = p.getJointInfo(env.robot_env.robotId, i, physicsClientId=env.robot_env.physics_client)
            joint_name = joint_info[1].decode('utf-8') if joint_info[1] else "Unknown"
            joint_type = joint_info[2]
            joint_type_name = {
                p.JOINT_REVOLUTE: "REVOLUTE",
                p.JOINT_PRISMATIC: "PRISMATIC",
                p.JOINT_FIXED: "FIXED"
            }.get(joint_type, "UNKNOWN")

            print(f"关节 {i} ({joint_name}): 类型={joint_type_name}, 位置={joint_state[0]:.4f}")

        # 修改特定关节的状态（例如夹爪手指关节）
        print("\n修改夹爪手指关节状态:")

        # 找到手指关节的ID
        finger_joint_ids = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(env.robot_env.robotId, i, physicsClientId=env.robot_env.physics_client)
            joint_name = joint_info[1].decode('utf-8') if joint_info[1] else ""
            if "finger" in joint_name:
                finger_joint_ids.append(i)

        print(f"找到的手指关节ID: {finger_joint_ids}")

        # 设置手指关节到一个打开的状态 (0.02 是中间位置)
        for joint_id in finger_joint_ids:
            joint_info = p.getJointInfo(env.robot_env.robotId, joint_id, physicsClientId=env.robot_env.physics_client)
            joint_name = joint_info[1].decode('utf-8') if joint_info[1] else ""
            print(f"设置关节 {joint_id} ({joint_name}) 到位置 0.02")
            p.resetJointState(
                env.robot_env.robotId,
                joint_id,
                0.02,  # 位置
                0.0,   # 速度
                physicsClientId=env.robot_env.physics_client
            )

        # 验证修改后的状态
        print("\n修改后的手指关节状态:")
        for joint_id in finger_joint_ids:
            joint_state = p.getJointState(env.robot_env.robotId, joint_id, physicsClientId=env.robot_env.physics_client)
            joint_info = p.getJointInfo(env.robot_env.robotId, joint_id, physicsClientId=env.robot_env.physics_client)
            joint_name = joint_info[1].decode('utf-8') if joint_info[1] else ""
            print(f"关节 {joint_id} ({joint_name}): 位置={joint_state[0]:.4f}")

        print("\n=== 测试完成 ===")

    finally:
        # 清理资源
        env.close()
        robot_env.close()


if __name__ == "__main__":
    test_joint_info()