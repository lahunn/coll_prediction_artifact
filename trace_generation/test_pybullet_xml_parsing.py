#!/usr/bin/env python3
"""
测试PyBullet能否解析现有的obstacles_0.xml文件
"""

import pybullet as p
import time
import sys
import os


def test_pybullet_xml_parsing():
    """测试PyBullet解析XML文件的能力"""

    # 文件路径
    xml_file = "scene_benchmarks/dens3/obstacles_0.xml"
    robot_urdf = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/panda/panda.urdf"

    print(f"Testing PyBullet XML parsing with: {xml_file}")
    print("=" * 50)

    # 检查文件是否存在
    if not os.path.exists(xml_file):
        print(f"❌ Error: File {xml_file} does not exist!")
        return False

    # 连接PyBullet
    try:
        p.connect(p.GUI)
        print("✅ PyBullet GUI connected successfully")
    except Exception as e:
        print(f"❌ Failed to connect PyBullet: {e}")
        return False

    # 设置为零重力
    p.setGravity(0, 0, 0)

    success = True

    # 尝试直接加载XML文件
    try:
        print("\n🔄 Attempting to load XML file directly...")
        objects = p.loadMJCF(xml_file)
        print(f"✅ Successfully loaded XML! Objects: {objects}")
        print(f"   Ground + {len(objects) - 1} obstacles loaded")

        # 确保所有障碍物完全固定，无法拖拽
        for obj_id in objects:
            # 将质量设为0，使物体变为静态
            p.changeDynamics(obj_id, -1, mass=0)
            # 禁用线性和角速度阻尼以确保完全静止
            p.changeDynamics(obj_id, -1, linearDamping=0, angularDamping=0)
        print("   🔒 All obstacles set to static (mass=0, cannot be moved)")
    except Exception as e:
        print(f"❌ Failed to load XML directly: {e}")
        success = False

    # 单独加载机器人URDF
    robot_id = None
    try:
        print("\n🔄 Loading robot URDF separately...")
        if os.path.exists(robot_urdf):
            robot_id = p.loadURDF(robot_urdf, basePosition=[0, 0, 0], useFixedBase=True)
            print(f"✅ Robot loaded with ID: {robot_id}")
            print("   🔒 Robot base fixed at origin")

            # 获取机器人信息
            num_joints = p.getNumJoints(robot_id)
            print(f"   Robot has {num_joints} joints")

            # 设置一个简单的机器人姿态用于测试
            for i in range(min(7, num_joints)):  # 设置前7个关节
                joint_info = p.getJointInfo(robot_id, i)
                if joint_info[2] != p.JOINT_FIXED:  # 非固定关节
                    p.resetJointState(robot_id, i, 0.1 * i)  # 简单的测试姿态
            print("   Set test robot configuration")
        else:
            print(f"⚠️  Robot file not found: {robot_urdf}")
            print("   Continuing with obstacles only...")
    except Exception as e:
        print(f"❌ Failed to load robot: {e}")

    # 保持仿真运行几秒钟以观察结果
    if success:
        print("\n✅ Running simulation for 5 seconds...")
        print("   💡 You should see obstacles and robot in the GUI")
        print("   🔒 All objects should remain fixed in position")
        for i in range(120000):  # 500秒，240Hz
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    else:
        print("\n❌ XML file cannot be parsed by PyBullet directly")
        print("💡 Recommendation: Convert to URDF or MuJoCo format")

    # 显示最终状态
    print("\n📊 Final status:")
    print(f"   Environment loaded: {'✅' if success else '❌'}")
    print(f"   Robot loaded: {'✅' if robot_id is not None else '❌'}")

    # 断开连接
    p.disconnect()
    print("\n🔚 Test completed")

    return success


if __name__ == "__main__":
    success = test_pybullet_xml_parsing()
    sys.exit(0 if success else 1)
