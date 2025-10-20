#!/usr/bin/env python3
"""
机器人球体模型可视化工具

简单的可视化工具，用于对比URDF机器人模型与球体碰撞模型。
显示半透明球体覆盖在机器人模型上，便于直观评估球体建模的准确性。

使用示例:
    python visualize_spheres.py                    # 使用默认配置
    python visualize_spheres.py --robot ur5e       # 可视化UR5e机器人
    python visualize_spheres.py --config-type zero # 使用零配置
"""

import sys
import argparse
import numpy as np
import pybullet as p
import pybullet_data
import torch
from pathlib import Path

# 导入球体分析器
from robot_sphere_analyzer import RobotSphereAnalyzer


class SphereVisualizer:
    """球体可视化器"""

    def __init__(self, robot_name="franka", device="cuda:0"):
        """初始化可视化器

        Args:
            robot_name: 机器人名称
            device: 计算设备
        """
        self.robot_name = robot_name
        self.device = device

        # 初始化PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # 无重力

        # 相机控制参数
        self.camera_distance = 1.5
        self.camera_yaw = 45
        self.camera_pitch = -30
        self.camera_target = [0, 0, 0.5]

        # 设置初始相机位置
        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=self.camera_target,
        )

        # 创建相机控制滑块
        self.distance_slider = p.addUserDebugParameter(
            "Camera Distance", 0.5, 5.0, self.camera_distance
        )
        self.yaw_slider = p.addUserDebugParameter(
            "Camera Yaw", -180, 180, self.camera_yaw
        )
        self.pitch_slider = p.addUserDebugParameter(
            "Camera Pitch", -89, 89, self.camera_pitch
        )

        # 加载机器人
        self.robot_id = None
        self.sphere_bodies = []
        self.sphere_analyzer = None

    def load_robot(self):
        """加载机器人URDF模型"""
        # 机器人URDF路径映射
        robot_paths = {
            "franka": "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf",
            "ur5e": "/opt/ros/noetic/share/ur_description/urdf/ur5e.urdf",
        }

        robot_urdf = robot_paths.get(
            self.robot_name,
            robot_paths["franka"],  # 默认使用franka
        )

        # 检查文件是否存在
        if not Path(robot_urdf).exists():
            print(f"警告: URDF文件不存在: {robot_urdf}")
            print("使用默认路径...")
            robot_urdf = robot_paths["franka"]

        try:
            self.robot_id = p.loadURDF(
                robot_urdf,
                basePosition=[0, 0, 0],
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION,
            )
            print(f"✓ 成功加载机器人: {robot_urdf}")
            return True
        except Exception as e:
            print(f"✗ 机器人加载失败: {e}")
            return False

    def set_robot_config(self, joint_angles):
        """设置机器人关节配置

        Args:
            joint_angles: 关节角度列表
        """
        if self.robot_id is None:
            return

        num_joints = p.getNumJoints(self.robot_id)
        joint_idx = 0

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:  # 不是固定关节
                if joint_idx < len(joint_angles):
                    p.resetJointState(self.robot_id, i, joint_angles[joint_idx])
                    joint_idx += 1

    def initialize_sphere_analyzer(self):
        """初始化球体分析器"""
        try:
            self.sphere_analyzer = RobotSphereAnalyzer(self.robot_name, self.device)
            print(f"✓ 球体分析器初始化成功 ({self.robot_name})")
            return True
        except Exception as e:
            print(f"✗ 球体分析器初始化失败: {e}")
            return False

    def create_spheres(self, world_spheres, transparency=0.3):
        """创建球体可视化

        Args:
            world_spheres: 世界坐标系下的球体信息 [N, 4] (x, y, z, radius)
            transparency: 透明度 (0=完全透明, 1=完全不透明)
        """
        # 清理旧球体
        self.clear_spheres()

        for i, (x, y, z, radius) in enumerate(world_spheres):
            # 创建球体可视化形状 (半透明绿色)
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=float(radius),
                rgbaColor=[0.0, 1.0, 0.0, transparency],  # 绿色半透明
                physicsClientId=self.physics_client,
            )

            # 创建球体body (无碰撞形状，仅用于可视化)
            # baseCollisionShapeIndex=-1 表示没有碰撞检测
            sphere_body = p.createMultiBody(
                baseMass=0,  # 静态球体
                baseCollisionShapeIndex=-1,  # 无碰撞属性
                baseVisualShapeIndex=visual_shape,
                basePosition=[float(x), float(y), float(z)],
                physicsClientId=self.physics_client,
            )

            self.sphere_bodies.append(sphere_body)

        print(f"✓ 创建了 {len(self.sphere_bodies)} 个可视化球体 (无碰撞)")

    def update_sphere_positions(self):
        """更新球体位置以匹配当前机器人姿态"""
        if not self.sphere_analyzer or not self.sphere_bodies or self.robot_id is None:
            return

        try:
            # 使用PyBullet获取当前关节角度
            num_joints = p.getNumJoints(self.robot_id)
            current_angles = []
            
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                if joint_info[2] != p.JOINT_FIXED:  # 不是固定关节
                    joint_state = p.getJointState(self.robot_id, i)
                    current_angles.append(joint_state[0])  # 关节位置
            
            if not current_angles:
                return

            # 转换为张量
            joint_config = torch.tensor(
                [current_angles], dtype=torch.float32, device=torch.device(self.device)
            )

            # 获取当前姿态下的球体世界坐标
            world_spheres = self.sphere_analyzer.get_world_spheres(joint_config)

            # 更新每个球体的位置
            for i, (sphere_body, (x, y, z, radius)) in enumerate(
                zip(self.sphere_bodies, world_spheres)
            ):
                if i >= len(self.sphere_bodies):
                    break

                p.resetBasePositionAndOrientation(
                    sphere_body,
                    [float(x), float(y), float(z)],
                    [0, 0, 0, 1],  # 球体无需旋转
                    physicsClientId=self.physics_client,
                )

        except Exception:
            # 静默处理错误，避免刷屏
            pass

    def clear_spheres(self):
        """清除所有球体"""
        for sphere_body in self.sphere_bodies:
            try:
                p.removeBody(sphere_body, physicsClientId=self.physics_client)
            except Exception:
                pass
        self.sphere_bodies.clear()

    def update_camera(self):
        """根据滑块值更新相机位置"""
        # 读取滑块值
        new_distance = p.readUserDebugParameter(self.distance_slider)
        new_yaw = p.readUserDebugParameter(self.yaw_slider)
        new_pitch = p.readUserDebugParameter(self.pitch_slider)

        # 只有值发生变化时才更新相机
        if (
            abs(new_distance - self.camera_distance) > 0.01
            or abs(new_yaw - self.camera_yaw) > 0.5
            or abs(new_pitch - self.camera_pitch) > 0.5
        ):
            self.camera_distance = new_distance
            self.camera_yaw = new_yaw
            self.camera_pitch = new_pitch

            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,
                cameraYaw=self.camera_yaw,
                cameraPitch=self.camera_pitch,
                cameraTargetPosition=self.camera_target,
            )

    def visualize(self, config_type="retract"):
        """执行可视化

        Args:
            config_type: 配置类型 ("retract"=收起配置, "zero"=零配置, "custom"=自定义)
        """
        # 加载机器人
        if not self.load_robot():
            return

        # 初始化球体分析器
        if not self.initialize_sphere_analyzer():
            return

        # 获取关节配置
        if config_type == "zero":
            # 零配置
            joint_config = torch.zeros(
                (1, self.sphere_analyzer.cuda_model.get_dof()),
                dtype=torch.float32,
                device=torch.device(self.device),
            )
            print("使用零配置 (所有关节角度为0)")
        elif config_type == "custom":
            # 自定义配置 (示例: Franka的一个典型姿态)
            custom_angles = [0, -0.3, 0, -2.2, 0, 2.0, 0.79]
            joint_config = torch.tensor(
                [custom_angles], dtype=torch.float32, device=torch.device(self.device)
            )
            print(f"使用自定义配置: {custom_angles}")
        else:
            # 默认配置 (收起配置)
            joint_config, config_name = self.sphere_analyzer.get_default_joint_config()
            print(f"使用默认配置: {config_name}")

        # 设置机器人姿态
        joint_angles = joint_config.squeeze().cpu().numpy().tolist()
        self.set_robot_config(joint_angles)

        # 获取球体世界坐标
        world_spheres = self.sphere_analyzer.get_world_spheres(joint_config)
        print(f"获取到 {len(world_spheres)} 个球体")

        # 创建球体可视化
        self.create_spheres(world_spheres, transparency=0.3)

        # 打印统计信息
        radii = world_spheres[:, 3]
        print("\n=== 球体统计 ===")
        print(f"总数: {len(world_spheres)}")
        print(f"半径范围: [{radii.min():.4f}, {radii.max():.4f}] m")
        print(f"平均半径: {radii.mean():.4f} m")

        # 进入交互循环
        print("\n=== 可视化已启动 ===")
        print("提示:")
        print("  - 使用滑块控制相机位置 (Distance, Yaw, Pitch)")
        print("  - 鼠标拖动可旋转视角")
        print("  - 鼠标滚轮可缩放")
        print("  - 按 Ctrl+C 或关闭窗口退出")
        print("=" * 50)

        try:
            while True:
                # 更新相机位置
                self.update_camera()

                # 更新球体位置以匹配当前机器人姿态
                self.update_sphere_positions()

                p.stepSimulation()
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        self.clear_spheres()
        p.disconnect(physicsClientId=self.physics_client)
        print("✓ 资源已清理")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="机器人球体模型可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s                              # 使用默认配置
  %(prog)s --robot franka               # 可视化Franka机器人
  %(prog)s --config-type zero           # 使用零配置
  %(prog)s --config-type custom         # 使用自定义配置
  %(prog)s --transparency 0.5           # 设置球体透明度
        """,
    )

    parser.add_argument(
        "--robot", type=str, default="franka", help="机器人名称 (默认: franka)"
    )

    parser.add_argument(
        "--device", type=str, default="cuda:0", help="计算设备 (默认: cuda:0)"
    )

    parser.add_argument(
        "--config-type",
        type=str,
        choices=["retract", "zero", "custom"],
        default="retract",
        help="配置类型 (默认: retract)",
    )

    parser.add_argument(
        "--transparency", type=float, default=0.3, help="球体透明度 0-1 (默认: 0.3)"
    )

    args = parser.parse_args()

    try:
        # 创建可视化器
        visualizer = SphereVisualizer(args.robot, args.device)

        # 运行可视化
        visualizer.visualize(config_type=args.config_type)

    except Exception as e:
        print(f"\n✗ 可视化失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
