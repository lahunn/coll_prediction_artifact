#!/usr/bin/env python3
"""
机器人和球体结构集成可视化程序

功能：
1. 同时显示机器人模型和对应的球体结构
2. 实时计算和更新球体位姿
3. 支持关节控制和实时球体更新
4. 提供丰富的可视化控制选项
5. 支持单独显示机器人或球体结构

使用方法：
python robot_sphere_visualizer.py <robot_urdf> <sphere_yaml> [options]

示例：
# 同时显示机器人和球体
python robot_sphere_visualizer.py ../data/robots/panda/panda.urdf ../content/configs/robot/spheres/franka.yml

# 仅显示机器人
python robot_sphere_visualizer.py ../data/robots/panda/panda.urdf ../content/configs/robot/spheres/franka.yml --robot-only

# 仅显示球体
python robot_sphere_visualizer.py ../data/robots/panda/panda.urdf ../content/configs/robot/spheres/franka.yml --spheres-only
"""

import numpy as np
import pybullet as p
import pybullet_data
import time
import argparse
from pathlib import Path
from typing import List

# 导入我们之前创建的模块
from sphere_visualizer import SphereYAMLParser
from sphere_forward_kinematics import SphereForwardKinematics


class RobotSphereVisualizer:
    """机器人和球体集成可视化器"""

    def __init__(self, robot_urdf: str, sphere_yaml: str, use_gui: bool = True):
        """
        初始化可视化器

        Args:
            robot_urdf: 机器人URDF文件路径
            sphere_yaml: 球体配置YAML文件路径
            use_gui: 是否使用GUI界面
        """
        self.robot_urdf = Path(robot_urdf)
        self.sphere_yaml = Path(sphere_yaml)
        self.use_gui = use_gui

        # PyBullet相关
        self.physics_client = None
        self.robot_id = None
        self.sphere_bodies = []

        # 球体相关
        self.collision_spheres = {}
        self.sphere_fk: SphereForwardKinematics | None = None
        self.link_colors = {}

        # GUI控制
        self.joint_sliders = []
        self.control_sliders = {}

        # 可视化状态
        self.show_robot = True
        self.show_spheres = True
        self.sphere_alpha = 0.6

        # 颜色方案
        self.color_palette = [
            [1.0, 0.0, 0.0, 0.6],  # 红色
            [0.0, 1.0, 0.0, 0.6],  # 绿色
            [0.0, 0.0, 1.0, 0.6],  # 蓝色
            [1.0, 1.0, 0.0, 0.6],  # 黄色
            [1.0, 0.0, 1.0, 0.6],  # 品红
            [0.0, 1.0, 1.0, 0.6],  # 青色
            [1.0, 0.5, 0.0, 0.6],  # 橙色
            [0.5, 0.0, 1.0, 0.6],  # 紫色
            [0.8, 0.4, 0.2, 0.6],  # 棕色
            [0.2, 0.8, 0.4, 0.6],  # 浅绿
            [0.4, 0.2, 0.8, 0.6],  # 深紫
            [0.8, 0.8, 0.2, 0.6],  # 橄榄色
        ]

        self._initialize()

    def _initialize(self):
        """初始化所有组件"""
        print("=== 初始化机器人球体可视化器 ===")

        # 1. 连接PyBullet
        self._connect_pybullet()

        # 2. 加载机器人
        self._load_robot()

        # 3. 解析球体配置
        self._parse_sphere_config()

        # 4. 初始化球体正向运动学
        self._initialize_sphere_fk()

        # 5. 设置环境
        self._setup_environment()

        # 6. 初始化GUI控制
        if self.use_gui:
            self._setup_gui_controls()

        print("初始化完成!")

    def _connect_pybullet(self):
        """连接PyBullet"""
        self.physics_client = p.connect(p.GUI if self.use_gui else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print("PyBullet连接成功")

    def _load_robot(self):
        """加载机器人模型"""
        if not self.robot_urdf.exists():
            raise FileNotFoundError(f"机器人URDF文件不存在: {self.robot_urdf}")

        self.robot_id = p.loadURDF(str(self.robot_urdf), [0, 0, 0], useFixedBase=True)
        print(f"机器人加载成功: {self.robot_urdf.name}")
        print(f"机器人ID: {self.robot_id}")
        print(f"关节数量: {p.getNumJoints(self.robot_id)}")

    def _parse_sphere_config(self):
        """解析球体配置"""
        if not self.sphere_yaml.exists():
            raise FileNotFoundError(f"球体配置文件不存在: {self.sphere_yaml}")

        parser = SphereYAMLParser(str(self.sphere_yaml))
        self.collision_spheres = parser.parse()

        if not self.collision_spheres:
            raise ValueError("未找到有效的球体配置")

        parser.print_summary()

    def _initialize_sphere_fk(self):
        """初始化球体正向运动学"""
        if self.robot_id is None:
            raise ValueError("机器人未正确加载")
        self.sphere_fk = SphereForwardKinematics(self.robot_id, self.collision_spheres)
        self._assign_link_colors()

    def _assign_link_colors(self):
        """为连杆分配颜色"""
        link_names = list(self.collision_spheres.keys())
        for i, link_name in enumerate(link_names):
            color_index = i % len(self.color_palette)
            self.link_colors[link_name] = self.color_palette[color_index]

    def _setup_environment(self):
        """设置环境"""
        # 创建地面
        ground_shape = p.createCollisionShape(p.GEOM_PLANE)
        ground_body = p.createMultiBody(0, ground_shape)
        p.changeVisualShape(ground_body, -1, rgbaColor=[0.8, 0.8, 0.8, 1.0])

        # 设置初始相机位置
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5],
        )

        print("环境设置完成")

    def _setup_gui_controls(self):
        """设置GUI控制界面"""
        print("设置GUI控制界面...")

        # 相机控制
        self.control_sliders["distance"] = p.addUserDebugParameter(
            "相机距离", 0.5, 5.0, 2.0
        )
        self.control_sliders["yaw"] = p.addUserDebugParameter("水平角度", -180, 180, 45)
        self.control_sliders["pitch"] = p.addUserDebugParameter(
            "俯仰角度", -89, 89, -30
        )

        # 可视化控制
        self.control_sliders["show_robot"] = p.addUserDebugParameter(
            "显示机器人", 0, 1, 1
        )
        self.control_sliders["show_spheres"] = p.addUserDebugParameter(
            "显示球体", 0, 1, 1
        )
        self.control_sliders["robot_alpha"] = p.addUserDebugParameter(
            "机器人透明度", 0.0, 1.0, 1.0
        )
        self.control_sliders["sphere_alpha"] = p.addUserDebugParameter(
            "球体透明度", 0.0, 1.0, 0.6
        )

        # 关节控制滑块
        num_joints = p.getNumJoints(self.robot_id)
        max_joints = min(num_joints, 7)  # 最多显示7个关节

        for i in range(max_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode("utf-8")
            joint_lower = joint_info[8]
            joint_upper = joint_info[9]

            # 处理无限制关节
            if joint_lower >= joint_upper:
                joint_lower = -3.14
                joint_upper = 3.14

            slider = p.addUserDebugParameter(
                f"关节{i}({joint_name})", joint_lower, joint_upper, 0.0
            )
            self.joint_sliders.append(slider)

        print(f"创建了 {len(self.joint_sliders)} 个关节控制滑块")

    def create_sphere_visual(
        self, center: np.ndarray, radius: float, color: List[float]
    ) -> int:
        """创建球体视觉对象"""
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color
        )

        body_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=visual_shape, basePosition=center
        )

        return body_id

    def update_spheres(self):
        """更新球体位姿"""
        # 清除旧球体
        self.clear_spheres()

        if not self.show_spheres or self.sphere_fk is None:
            return

        # 计算新的球体位姿
        sphere_poses = self.sphere_fk.compute_sphere_poses()

        # 创建新球体
        for pose in sphere_poses:
            if pose.link_name in self.link_colors:
                color = self.link_colors[pose.link_name].copy()
                color[3] = self.sphere_alpha  # 设置透明度

                body_id = self.create_sphere_visual(pose.center, pose.radius, color)
                self.sphere_bodies.append(body_id)

    def clear_spheres(self):
        """清除所有球体"""
        for body_id in self.sphere_bodies:
            try:
                p.removeBody(body_id)
            except Exception:
                pass
        self.sphere_bodies.clear()

    def _update_robot_visibility(self):
        """更新机器人可见性"""
        if self.robot_id is None:
            return

        num_joints = p.getNumJoints(self.robot_id)

        if self.show_robot:
            # 显示机器人 - 恢复正常颜色
            robot_alpha = 1.0
            if self.use_gui:
                robot_alpha = p.readUserDebugParameter(
                    self.control_sliders["robot_alpha"]
                )

            # 恢复基座可见性
            p.changeVisualShape(
                self.robot_id, -1, rgbaColor=[0.7, 0.7, 0.7, robot_alpha]
            )

            # 恢复所有连杆可见性
            for i in range(num_joints):
                p.changeVisualShape(
                    self.robot_id, i, rgbaColor=[0.7, 0.7, 0.7, robot_alpha]
                )
        else:
            # 隐藏机器人 - 设置为完全透明
            # 隐藏基座
            p.changeVisualShape(self.robot_id, -1, rgbaColor=[0.0, 0.0, 0.0, 0.0])

            # 隐藏所有连杆
            for i in range(num_joints):
                p.changeVisualShape(self.robot_id, i, rgbaColor=[0.0, 0.0, 0.0, 0.0])

    def update_robot_joints(self):
        """更新机器人关节角度"""
        if not self.use_gui:
            return

        # 读取关节滑块值并设置关节角度
        for i, slider in enumerate(self.joint_sliders):
            joint_angle = p.readUserDebugParameter(slider)
            p.resetJointState(self.robot_id, i, joint_angle)

    def update_camera(self):
        """更新相机位置"""
        if not self.use_gui:
            return

        distance = p.readUserDebugParameter(self.control_sliders["distance"])
        yaw = p.readUserDebugParameter(self.control_sliders["yaw"])
        pitch = p.readUserDebugParameter(self.control_sliders["pitch"])

        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=[0, 0, 0.5],
        )

    def update_transparency(self):
        """更新透明度和显示状态"""
        if not self.use_gui:
            return

        # 读取显示控制参数
        show_robot = bool(p.readUserDebugParameter(self.control_sliders["show_robot"]))
        show_spheres = bool(
            p.readUserDebugParameter(self.control_sliders["show_spheres"])
        )

        # 更新显示状态
        if show_robot != self.show_robot:
            self.show_robot = show_robot
            self._update_robot_visibility()

        if show_spheres != self.show_spheres:
            self.show_spheres = show_spheres
            if not self.show_spheres:
                self.clear_spheres()

        # 更新机器人透明度
        if self.show_robot:
            robot_alpha = p.readUserDebugParameter(self.control_sliders["robot_alpha"])
            num_joints = p.getNumJoints(self.robot_id)

            # 设置基座透明度
            p.changeVisualShape(
                self.robot_id, -1, rgbaColor=[0.7, 0.7, 0.7, robot_alpha]
            )

            # 设置所有连杆透明度
            for i in range(num_joints):
                p.changeVisualShape(
                    self.robot_id, i, rgbaColor=[0.7, 0.7, 0.7, robot_alpha]
                )

        # 更新球体透明度
        new_sphere_alpha = p.readUserDebugParameter(
            self.control_sliders["sphere_alpha"]
        )
        if abs(new_sphere_alpha - self.sphere_alpha) > 0.01:
            self.sphere_alpha = new_sphere_alpha
            if self.show_spheres:
                self.update_spheres()  # 重新创建球体以更新透明度

    def print_current_state(self):
        """打印当前状态信息"""
        if not self.use_gui:
            return

        # 获取当前关节配置
        joint_config = []
        for i in range(len(self.joint_sliders)):
            angle = p.readUserDebugParameter(self.joint_sliders[i])
            joint_config.append(angle)

        print(f"当前关节配置: {[f'{angle:.3f}' for angle in joint_config]}")
        print(f"当前球体数量: {len(self.sphere_bodies)}")

    def run_visualization(self):
        """运行可视化循环"""
        print("\n=== 开始可视化 ===")
        print("使用滑块控制机器人关节和可视化参数")
        print("关闭窗口或按Ctrl+C退出...")

        # 初始化球体
        self.update_spheres()

        frame_count = 0
        last_print_time = time.time()

        try:
            while True:
                # 更新机器人关节
                self.update_robot_joints()

                # 更新相机
                self.update_camera()

                # 更新透明度
                self.update_transparency()

                # 每30帧更新一次球体位姿
                if frame_count % 30 == 0 and self.show_spheres:
                    self.update_spheres()

                # 每5秒打印一次状态
                current_time = time.time()
                if current_time - last_print_time > 5.0:
                    self.print_current_state()
                    last_print_time = current_time

                p.stepSimulation()
                time.sleep(1.0 / 60.0)  # 60 FPS
                frame_count += 1

        except KeyboardInterrupt:
            print("用户中断可视化")
        except Exception as e:
            print(f"可视化错误: {e}")
            import traceback

            traceback.print_exc()

    def run_static_demo(self, demo_configs: List[List[float]], hold_time: float = 3.0):
        """运行静态演示（无GUI模式）"""
        print("\n=== 开始静态演示 ===")

        for i, config in enumerate(demo_configs):
            print(
                f"\n演示配置 {i + 1}/{len(demo_configs)}: {[f'{angle:.3f}' for angle in config]}"
            )

            # 设置关节配置
            for joint_idx, angle in enumerate(config):
                if joint_idx < p.getNumJoints(self.robot_id):
                    p.resetJointState(self.robot_id, joint_idx, angle)

            # 更新球体
            self.update_spheres()

            # 等待
            time.sleep(hold_time)

            # 打印球体信息
            if self.sphere_fk is not None:
                sphere_poses = self.sphere_fk.compute_sphere_poses()
                print(f"计算了 {len(sphere_poses)} 个球体位姿")

        print("静态演示完成")

    def disconnect(self):
        """断开连接并清理资源"""
        try:
            self.clear_spheres()
            if self.physics_client is not None:
                p.disconnect(self.physics_client)
                self.physics_client = None
        except Exception as e:
            print(f"断开连接时出错: {e}")
        print("资源清理完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="机器人和球体结构集成可视化")
    parser.add_argument("robot_urdf", help="机器人URDF文件路径")
    parser.add_argument("sphere_yaml", help="球体配置YAML文件路径")
    parser.add_argument("--no-gui", action="store_true", help="无GUI模式（静态演示）")
    parser.add_argument("--demo", action="store_true", help="运行预设演示")
    parser.add_argument("--robot-only", action="store_true", help="仅显示机器人")
    parser.add_argument("--spheres-only", action="store_true", help="仅显示球体")

    args = parser.parse_args()

    # 创建可视化器
    visualizer = RobotSphereVisualizer(
        robot_urdf=args.robot_urdf,
        sphere_yaml=args.sphere_yaml,
        use_gui=not args.no_gui,
    )

    # 设置初始显示模式
    if args.robot_only:
        visualizer.show_robot = True
        visualizer.show_spheres = False
        print("模式: 仅显示机器人")
    elif args.spheres_only:
        visualizer.show_robot = False
        visualizer.show_spheres = True
        print("模式: 仅显示球体")
    else:
        print("模式: 显示机器人和球体")

    # 应用初始显示设置
    if not args.no_gui:
        visualizer._update_robot_visibility()

    try:
        if args.demo or args.no_gui:
            # 静态演示模式
            demo_configs = [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 初始姿态
                [0.5, -0.3, 0.0, -1.5, 0.0, 1.2, 0.7],  # 配置1
                [-0.5, 0.3, 0.0, -1.0, 0.0, 1.5, -0.7],  # 配置2
                [0.0, 0.8, 0.0, -2.0, 0.0, 2.5, 0.0],  # 配置3
            ]
            visualizer.run_static_demo(demo_configs)
        else:
            # 交互式可视化模式
            visualizer.run_visualization()
    finally:
        visualizer.disconnect()


if __name__ == "__main__":
    main()
