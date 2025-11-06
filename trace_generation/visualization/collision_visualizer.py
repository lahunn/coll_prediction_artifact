#!/usr/bin/env python3
"""
简单的场景和机器人可视化脚本

用法:
python collision_visualize.py <scene_folder> <scene_id> [robot_urdf]

示例:
python collision_visualize.py dens3 10
python collision_visualize.py dens6 5 /path/to/robot.urdf
"""

import sys
import pybullet as p
import pybullet_data
import time
from pathlib import Path

# 默认机器人URDF路径
DEFAULT_ROBOT_URDF = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/jaco_7/jaco_7s.urdf"


def load_scene_and_robot(scene_folder, scene_id, robot_urdf=None):
    """加载场景和机器人"""
    # 连接PyBullet GUI
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 加载地面
    ground_shape = p.createCollisionShape(p.GEOM_PLANE)
    ground_body = p.createMultiBody(0, ground_shape)
    p.changeVisualShape(ground_body, -1, rgbaColor=[0.8, 0.8, 0.8, 1.0])

    # 构建场景文件路径
    scene_file = f"scene_benchmarks/{scene_folder}/obstacles_{scene_id}.xml"

    if not Path(scene_file).exists():
        print(f"错误: 场景文件 {scene_file} 不存在")
        return None, None, []

    # 加载场景
    try:
        scene_objects = p.loadMJCF(scene_file)
        print(f"成功加载场景: {scene_file}")
        print(f"场景包含 {len(scene_objects)} 个物体")

        # 设置所有场景物体为静态（不受重力影响，不能被推动）
        for obj_id in scene_objects:
            p.changeDynamics(obj_id, -1, mass=0)  # 设置质量为0使其静态

    except Exception as e:
        print(f"加载场景失败: {e}")
        return None, None, []

    # 使用指定的或默认的机器人URDF
    if robot_urdf is None:
        robot_urdf = DEFAULT_ROBOT_URDF

    # 加载机器人
    try:
        robot_id = p.loadURDF(robot_urdf, [0, 0, 0], useFixedBase=True)
        print(f"成功加载机器人: {robot_urdf}")
    except Exception as e:
        print(f"加载机器人失败: {e}")
        return None, None, scene_objects

    # 设置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5],
    )

    return physics_client, robot_id, scene_objects


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) < 3:
        print(
            "用法: python collision_visualize.py <scene_folder> <scene_id> [robot_urdf]"
        )
        print("示例: python collision_visualize.py dens3 10")
        sys.exit(1)

    scene_folder = sys.argv[1]
    scene_id = sys.argv[2]
    robot_urdf = sys.argv[3] if len(sys.argv) > 3 else None

    # 加载场景和机器人
    physics_client, robot_id, scene_objects = load_scene_and_robot(
        scene_folder, scene_id, robot_urdf
    )

    if physics_client is None:
        print("初始化失败")
        sys.exit(1)

    print("\n=== 可视化已启动 ===")
    print("- 使用鼠标控制视角")
    print("- 关闭窗口退出")

    # 主循环
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)  # 240 FPS

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        p.disconnect()
        print("可视化已退出")


if __name__ == "__main__":
    main()
