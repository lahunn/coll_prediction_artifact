#!/usr/bin/env python3
"""
诊断脚本：分析pred_trace_generation.py中的碰撞检测问题

该脚本将：
1. 检查机器人和障碍物的位置
2. 验证碰撞检测逻辑
3. 测试不同的机器人配置
4. 生成可视化输出帮助调试
"""

import pybullet as p
import math
import random
import sys
import os

# 添加当前目录到Python路径
sys.path.append(".")


def connect_pybullet_with_gui():
    """连接PyBullet并启用GUI"""
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath("/usr/local/lib/python3.8/dist-packages/pybullet_data")
    return physics_client


def load_scene_and_robot(scene_file, robot_urdf):
    """加载场景和机器人"""
    print(f"加载场景: {scene_file}")
    print(f"加载机器人: {robot_urdf}")

    # 加载场景
    scene_objects = []
    obstacle_ids = []

    try:
        scene_objects = p.loadMJCF(scene_file)
        if scene_objects:
            obstacle_ids = scene_objects[2:] if len(scene_objects) > 2 else []
            print(f"场景加载成功，包含 {len(obstacle_ids)} 个障碍物")
        else:
            print("警告：场景加载失败或为空")
    except Exception as e:
        print(f"场景加载错误: {e}")

    # 加载机器人
    robot_id = None
    try:
        robot_id = p.loadURDF(robot_urdf, [0, 0, 0])
        print(f"机器人加载成功，ID: {robot_id}")
    except Exception as e:
        print(f"机器人加载错误: {e}")
        return None, []

    return robot_id, obstacle_ids


def analyze_positions(robot_id, obstacle_ids):
    """分析机器人和障碍物的位置"""
    print("\n=== 位置分析 ===")

    # 分析机器人位置
    if robot_id is not None:
        robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
        print(f"机器人基座位置: {robot_pos}")
        print(f"机器人基座方向: {robot_orn}")

        # 分析机器人连杆
        num_joints = p.getNumJoints(robot_id)
        print(f"机器人关节数量: {num_joints}")

        for i in range(min(num_joints, 5)):  # 只显示前5个连杆
            link_state = p.getLinkState(robot_id, i)
            link_pos = link_state[0]
            print(f"  连杆 {i} 位置: {link_pos}")

    # 分析障碍物位置
    print(f"\n障碍物数量: {len(obstacle_ids)}")
    for i, obs_id in enumerate(obstacle_ids):
        try:
            obs_pos, obs_orn = p.getBasePositionAndOrientation(obs_id)
            print(f"  障碍物 {i} (ID: {obs_id}) 位置: {obs_pos}")
        except Exception as e:
            print(f"  障碍物 {i} 位置获取失败: {e}")


def get_joint_limits(robot_id):
    """获取关节限制"""
    joint_limits = []
    valid_joints = []

    if robot_id is None:
        return joint_limits, valid_joints

    num_joints = p.getNumJoints(robot_id)

    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] != p.JOINT_FIXED:  # 非固定关节
            valid_joints.append(i)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]

            # 处理无限制关节
            if lower_limit == 0 and upper_limit == -1:
                lower_limit, upper_limit = -math.pi, math.pi

            joint_limits.append((lower_limit, upper_limit))
            print(f"关节 {i}: [{lower_limit:.3f}, {upper_limit:.3f}]")

    return joint_limits, valid_joints


def test_collision_detection(
    robot_id, obstacle_ids, joint_limits, valid_joints, num_tests=50
):
    """测试碰撞检测功能"""
    print(f"\n=== 碰撞检测测试 ({num_tests} 次) ===")

    collision_count = 0

    for test_i in range(num_tests):
        # 生成随机关节配置
        joint_config = []
        for lower, upper in joint_limits:
            angle = random.uniform(lower, upper)
            joint_config.append(angle)

        # 设置机器人配置
        for i, angle in enumerate(joint_config):
            if i < len(valid_joints):
                p.resetJointState(robot_id, valid_joints[i], angle)

        # 检查碰撞
        has_collision = False

        # 方法1：检查与所有障碍物的碰撞
        for obstacle_id in obstacle_ids:
            contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
            if len(contacts) > 0:
                has_collision = True
                print(f"  测试 {test_i}: 发现碰撞！(与障碍物 {obstacle_id})")
                break

        if has_collision:
            collision_count += 1

            # 详细分析碰撞
            print(f"    关节配置: {[f'{angle:.3f}' for angle in joint_config[:7]]}")

            # 显示接触点信息
            for obstacle_id in obstacle_ids:
                contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
                if len(contacts) > 0:
                    contact = contacts[0]
                    print(f"    接触点位置: {contact[5]}")
                    print(f"    接触法向量: {contact[7]}")

        # 每10次测试输出进度
        if (test_i + 1) % 10 == 0:
            print(
                f"  已完成 {test_i + 1}/{num_tests} 次测试，碰撞次数: {collision_count}"
            )

    print(f"\n总结：{num_tests} 次测试中发现 {collision_count} 次碰撞")
    print(f"碰撞率: {collision_count / num_tests * 100:.1f}%")

    return collision_count


def test_extreme_configurations(robot_id, obstacle_ids, joint_limits, valid_joints):
    """测试极端配置以强制产生碰撞"""
    print("\n=== 极端配置测试 ===")

    # 测试1：所有关节最大角度
    print("测试1：所有关节最大角度")
    joint_config = [upper for lower, upper in joint_limits]
    for i, angle in enumerate(joint_config):
        if i < len(valid_joints):
            p.resetJointState(robot_id, valid_joints[i], angle)

    collision_found = check_collisions(robot_id, obstacle_ids, "最大角度")

    # 测试2：所有关节最小角度
    print("测试2：所有关节最小角度")
    joint_config = [lower for lower, upper in joint_limits]
    for i, angle in enumerate(joint_config):
        if i < len(valid_joints):
            p.resetJointState(robot_id, valid_joints[i], angle)

    collision_found = check_collisions(robot_id, obstacle_ids, "最小角度")

    # 测试3：特定的高风险配置
    print("测试3：特定的高风险配置")
    # 尝试让机器人伸展到最远
    joint_config = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]  # 中等角度
    for i, angle in enumerate(joint_config):
        if i < len(valid_joints) and i < len(joint_config):
            p.resetJointState(robot_id, valid_joints[i], angle)

    collision_found = check_collisions(robot_id, obstacle_ids, "高风险配置")


def check_collisions(robot_id, obstacle_ids, config_name):
    """检查当前配置下的碰撞"""
    collision_found = False

    for i, obstacle_id in enumerate(obstacle_ids):
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
        if len(contacts) > 0:
            print(f"  {config_name}: 与障碍物 {i} (ID: {obstacle_id}) 发生碰撞!")
            print(f"    接触点数量: {len(contacts)}")
            collision_found = True

    if not collision_found:
        print(f"  {config_name}: 无碰撞")

    return collision_found


def analyze_workspace(robot_id, joint_limits, valid_joints):
    """分析机器人工作空间"""
    print("\n=== 工作空间分析 ===")

    # 测试几个典型配置的末端执行器位置
    test_configs = [
        ([0] * len(joint_limits), "零位配置"),
        ([limit[1] * 0.5 for limit in joint_limits], "中位配置"),
        ([limit[1] for limit in joint_limits], "最大配置"),
        ([limit[0] for limit in joint_limits], "最小配置"),
    ]

    for config, name in test_configs:
        # 设置关节配置
        for i, angle in enumerate(config):
            if i < len(valid_joints):
                p.resetJointState(robot_id, valid_joints[i], angle)

        # 获取末端执行器位置（最后一个连杆）
        num_joints = p.getNumJoints(robot_id)
        if num_joints > 0:
            end_effector_state = p.getLinkState(robot_id, num_joints - 1)
            end_pos = end_effector_state[0]
            print(f"  {name}: 末端位置 {end_pos}")


def main():
    """主程序"""
    print("=== 碰撞检测诊断脚本 ===")

    # 解析命令行参数
    if len(sys.argv) != 4:
        print(
            "用法: python collision_diagnosis.py <numqueries> <foldername> <filenumber>"
        )
        print("示例: python collision_diagnosis.py 1000 ./scene_benchmarks/dens3 10")
        return

    numqueries = int(sys.argv[1])
    foldername = sys.argv[2]
    filenumber = sys.argv[3]

    # 设置文件路径
    scene_file = foldername + "/obstacles_" + filenumber + ".xml"
    robot_urdf = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/jaco_7/jaco_7s.urdf"

    print(f"场景文件: {scene_file}")
    print(f"机器人文件: {robot_urdf}")

    # 验证文件存在
    if not os.path.exists(scene_file):
        print(f"错误：场景文件不存在: {scene_file}")
        return

    if not os.path.exists(robot_urdf):
        print(f"错误：机器人文件不存在: {robot_urdf}")
        return

    # 连接PyBullet
    physics_client = connect_pybullet_with_gui()

    try:
        # 加载场景和机器人
        robot_id, obstacle_ids = load_scene_and_robot(scene_file, robot_urdf)

        if robot_id is None:
            print("错误：机器人加载失败")
            return

        # 分析位置
        analyze_positions(robot_id, obstacle_ids)

        # 获取关节限制
        joint_limits, valid_joints = get_joint_limits(robot_id)
        print(f"\n有效关节数量: {len(valid_joints)}")

        # 分析工作空间
        analyze_workspace(robot_id, joint_limits, valid_joints)

        # 测试极端配置
        test_extreme_configurations(robot_id, obstacle_ids, joint_limits, valid_joints)

        # 随机测试碰撞检测
        num_tests = min(50, numqueries // 10)  # 适量的测试次数
        collision_count = test_collision_detection(
            robot_id, obstacle_ids, joint_limits, valid_joints, num_tests
        )

        print("\n=== 诊断结论 ===")
        if collision_count == 0:
            print("问题确认：随机配置下无法检测到碰撞")
            print("可能原因：")
            print("1. 机器人与障碍物位置不匹配（机器人在地面上，障碍物在地面下）")
            print("2. 关节限制过于保守，无法达到障碍物位置")
            print("3. 场景文件中的机器人与实际加载的机器人不一致")
            print("4. 障碍物尺寸或位置设置问题")
        else:
            print(f"碰撞检测正常，{num_tests}次测试中检测到{collision_count}次碰撞")

        input("按Enter键关闭...")

    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
