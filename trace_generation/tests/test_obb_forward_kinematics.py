# type: ignore
"""
OBB正向运动学验证测试程序

该程序用于验证预计算的OBB数据和正向运动学函数的正确性。
通过可视化机器人URDF和对应的OBB，比较它们的空间位置关系。

功能:
1. 加载预计算的机器人OBB数据 (从robot_config目录)
2. 使用OBBForwardKinematics计算OBB在不同关节配置下的位姿
3. 可视化机器人URDF和OBB进行对比验证
4. 提供关节控制和实时OBB更新
5. 验证OBB是否正确包围了对应的连杆

使用方法:
python test_obb_forward_kinematics.py <robot_name> [--links link1,link2,...]

示例:
python test_obb_forward_kinematics.py franka
python test_obb_forward_kinematics.py iiwa --links iiwa7_link_1,iiwa7_link_2

支持的机器人: franka, iiwa, iiwa_allegro, jaco7, kinova_gen3, quad_ur10e,
               simple_mimic_robot, tm12, tri_ur10e, ur5e, ur5e_robotiq_2f_140, ur10e
"""

import time
import sys
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pybullet as p

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 导入相关模块
from trace_generation.core.robot.environment import robot_urdf_mapping
from trace_generation.core.robot.obb_forward_kinematics import (
    OBBForwardKinematics,
    get_link_world_transform,
)


def load_obb_data(robot_name):
    """
    从robot_config目录加载机器人的OBB数据

    Args:
        robot_name: 机器人名称

    Returns:
        list: OBB数据列表，每个元素包含link_name, position, extents, rotation_matrix等
    """
    config_file = (
        project_root
        / "trace_generation"
        / "core"
        / "robot"
        / "robot_config"
        / f"{robot_name}_obbs.py"
    )

    if not config_file.exists():
        raise FileNotFoundError(f"找不到OBB配置文件: {config_file}")

    # 动态导入OBB数据
    spec = importlib.util.spec_from_file_location(f"{robot_name}_obbs", config_file)
    obb_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obb_module)

    # 获取OBB数据变量
    obb_var_name = f"{robot_name}_obbs"
    if not hasattr(obb_module, obb_var_name):
        raise AttributeError(f"OBB配置文件中找不到变量: {obb_var_name}")

    obbs_data = getattr(obb_module, obb_var_name)

    # 转换OBB数据格式以兼容正向运动学计算
    converted_obbs = []
    for obb in obbs_data:
        # 创建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = obb["rotation_matrix"]
        transform[:3, 3] = obb["position"]

        converted_obb = {
            "link_name": obb["link_name"],
            "transform": transform,
            "extents": obb["extents"],
            "position": obb["position"],  # 保留原始位置用于参考
            "rotation_matrix": obb["rotation_matrix"],  # 保留原始旋转矩阵用于参考
            "volume": obb.get("volume", 0.0),
        }
        converted_obbs.append(converted_obb)

    print(f"成功加载 {len(converted_obbs)} 个 OBB 数据")
    return converted_obbs


def load_robot_urdf(robot_name):
    """
    加载机器人URDF并返回robot_id

    Args:
        robot_name: 机器人名称

    Returns:
        tuple: (robot_id, physics_client_id)
    """
    # 获取URDF路径
    rel_urdf_path = robot_urdf_mapping.get(robot_name)
    if not rel_urdf_path:
        raise ValueError(f"未找到机器人 '{robot_name}' 的URDF路径")

    urdf_path = project_root / rel_urdf_path
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF文件不存在: {urdf_path}")

    # 连接PyBullet
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, 0)  # 取消重力

    # 加载机器人
    robot_id = p.loadURDF(str(urdf_path), [0, 0, 0], useFixedBase=True)

    return robot_id, physics_client


def convert_obbs_for_fk(obbs_data):
    """
    将原始OBB数据转换为正向运动学计算所需的格式

    Args:
        obbs_data: 原始OBB数据列表

    Returns:
        list: 转换后的OBB数据列表
    """
    fk_obbs = []
    for obb in obbs_data:
        fk_obb = {
            "link_name": obb["link_name"],
            "transform": obb["transform"],  # 相对于连杆的变换矩阵
            "extents": obb["extents"],
        }
        fk_obbs.append(fk_obb)
    return fk_obbs


class OBBValidationVisualizer:
    """OBB验证可视化器"""

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.obb_bodies = []
        self.link_bodies = []  # 用于标记连杆位置的辅助体
        self.obb_fk = OBBForwardKinematics(robot_id)

        # 颜色定义
        self.link_colors = [
            [1.0, 0.0, 0.0, 0.8],  # 红
            [0.0, 1.0, 0.0, 0.8],  # 绿
            [0.0, 0.0, 1.0, 0.8],  # 蓝
            [1.0, 1.0, 0.0, 0.8],  # 黄
            [1.0, 0.0, 1.0, 0.8],  # 品红
            [0.0, 1.0, 1.0, 0.8],  # 青
            [1.0, 0.5, 0.0, 0.8],  # 橙
            [0.5, 0.0, 1.0, 0.8],  # 紫
            [0.8, 0.4, 0.2, 0.8],  # 棕
            [0.2, 0.8, 0.4, 0.8],  # 浅绿
        ]

        self.obb_colors = [
            [1.0, 0.0, 0.0, 0.3],  # 半透明红
            [0.0, 1.0, 0.0, 0.3],  # 半透明绿
            [0.0, 0.0, 1.0, 0.3],  # 半透明蓝
            [1.0, 1.0, 0.0, 0.3],  # 半透明黄
            [1.0, 0.0, 1.0, 0.3],  # 半透明品红
            [0.0, 1.0, 1.0, 0.3],  # 半透明青
            [1.0, 0.5, 0.0, 0.3],  # 半透明橙
            [0.5, 0.0, 1.0, 0.3],  # 半透明紫
            [0.8, 0.4, 0.2, 0.3],  # 半透明棕
            [0.2, 0.8, 0.4, 0.3],  # 半透明浅绿
        ]

    def draw_link_markers(self, obbs_data, visible_links=None):
        """在连杆位置绘制标记点，用于验证OBB位置"""
        self.clear_link_markers()

        for i, obb in enumerate(obbs_data):
            link_name = obb["link_name"]

            if visible_links and link_name not in visible_links:
                continue

            # 获取连杆的世界变换
            link_transform = get_link_world_transform(
                self.robot_id, link_name, self.obb_fk.link_name_to_index
            )

            if link_transform is None:
                continue

            # 连杆位置
            link_position = link_transform[:3, 3]

            # 创建小球标记连杆位置
            color = self.link_colors[i % len(self.link_colors)]
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.01,  # 小球
                rgbaColor=color,
            )
            body_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=link_position,
            )
            self.link_bodies.append(body_id)

            print(
                f"标记连杆: {link_name} 位置: {np.array2string(link_position, precision=3)}"
            )

    def draw_obbs(self, obbs_data, visible_links=None):
        """绘制OBB"""
        self.clear_obbs()

        # 计算OBB位姿
        obb_poses = self.obb_fk.compute_obb_poses(convert_obbs_for_fk(obbs_data))

        for i, obb_pose in enumerate(obb_poses):
            link_name = obb_pose["link_name"]

            if visible_links and link_name not in visible_links:
                continue

            position = obb_pose["position"]
            quaternion = obb_pose["quaternion"]
            extents = obb_pose["extents"]

            color = self.obb_colors[i % len(self.obb_colors)]

            # 创建OBB可视化体
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=extents / 2.0,
                rgbaColor=color,
            )
            body_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=position,
                baseOrientation=quaternion,
            )
            self.obb_bodies.append(body_id)

            print(f"绘制OBB: {link_name}")
            print(f"  位置: {np.array2string(position, precision=3)}")
            print(f"  尺寸: {np.array2string(extents, precision=3)}")
            print(f"  姿态: {np.array2string(np.array(quaternion), precision=3)}")

    def update_visualization(self, obbs_data, visible_links=None):
        """更新可视化"""
        self.draw_link_markers(obbs_data, visible_links)
        self.draw_obbs(obbs_data, visible_links)

    def clear_link_markers(self):
        """清除连杆标记"""
        for body_id in self.link_bodies:
            p.removeBody(body_id)
        self.link_bodies.clear()

    def clear_obbs(self):
        """清除OBB"""
        for body_id in self.obb_bodies:
            p.removeBody(body_id)
        self.obb_bodies.clear()

    def clear_all(self):
        """清除所有可视化元素"""
        self.clear_link_markers()
        self.clear_obbs()


def validate_obb_coverage(robot_id, obb_visualizer, obbs_data, joint_config=None):
    """
    验证OBB是否正确覆盖了连杆

    Args:
        robot_id: 机器人ID
        obb_visualizer: OBB可视化器
        obbs_data: OBB数据
        joint_config: 关节配置（可选）
    """
    print("\n=== OBB覆盖验证 ===")

    if joint_config:
        # 设置关节位置
        num_joints = p.getNumJoints(robot_id)
        for i in range(num_joints):
            if i < len(joint_config):
                p.resetJointState(robot_id, i, joint_config[i])

    # 计算OBB位姿
    obb_poses = obb_visualizer.obb_fk.compute_obb_poses(convert_obbs_for_fk(obbs_data))

    coverage_results = []

    for obb_pose in obb_poses:
        link_name = obb_pose["link_name"]
        obb_position = obb_pose["position"]
        obb_extents = obb_pose["extents"]
        obb_transform = obb_pose["transform"]

        # 获取连杆位置作为参考
        link_transform = obb_visualizer.obb_fk.get_link_world_transform(
            robot_id, link_name
        )
        if link_transform is not None:
            link_position = link_transform[:3, 3]

            # 计算OBB中心与连杆位置的距离
            distance = np.linalg.norm(obb_position - link_position)

            # 简单的覆盖验证（OBB应该包含连杆位置）
            obb_half_extents = obb_extents / 2.0
            obb_corners = []

            # 计算OBB的8个顶点（相对于OBB中心）
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        corner_local = np.array([dx, dy, dz]) * obb_half_extents
                        # 变换到世界坐标
                        corner_world = (
                            obb_transform[:3, :3] @ corner_local + obb_position
                        )
                        obb_corners.append(corner_world)

            # 检查连杆位置是否在OBB内（简单的AABB检查）
            obb_min = np.min(obb_corners, axis=0)
            obb_max = np.max(obb_corners, axis=0)

            is_covered = np.all(link_position >= obb_min) and np.all(
                link_position <= obb_max
            )

            coverage_results.append(
                {
                    "link_name": link_name,
                    "distance": distance,
                    "is_covered": is_covered,
                    "obb_bounds": (obb_min, obb_max),
                    "link_position": link_position,
                }
            )

            print(f"{link_name}:")
            print(f"  距离: {distance:.4f}")
            print(f"  覆盖: {'✓' if is_covered else '✗'}")
            print(
                f"  OBB范围: [{np.array2string(obb_min, precision=3)}, {np.array2string(obb_max, precision=3)}]"
            )
            print(f"  连杆位置: {np.array2string(link_position, precision=3)}")

    return coverage_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OBB正向运动学验证测试")
    parser.add_argument(
        "robot_name", help="机器人名称", choices=list(robot_urdf_mapping.keys())
    )
    parser.add_argument("--links", help="要显示的连杆名称（逗号分隔）", default=None)
    parser.add_argument("--no-validation", action="store_true", help="跳过OBB覆盖验证")

    args = parser.parse_args()

    try:
        print(f"=== OBB正向运动学验证测试: {args.robot_name} ===")

        # 1. 加载机器人URDF
        print("加载机器人URDF...")
        robot_id, physics_client = load_robot_urdf(args.robot_name)

        # 2. 加载OBB数据
        print("加载OBB数据...")
        obbs_data = load_obb_data(args.robot_name)

        # 3. 初始化可视化器
        print("初始化可视化器...")
        visualizer = OBBValidationVisualizer(robot_id)

        # 4. 解析可见连杆
        visible_links = None
        if args.links:
            visible_links = [link.strip() for link in args.links.split(",")]
            print(f"显示指定连杆: {visible_links}")

        # 5. 初始可视化
        print("开始可视化...")
        visualizer.update_visualization(obbs_data, visible_links)

        # 6. 验证OBB覆盖（如果未禁用）
        if not args.no_validation:
            validate_obb_coverage(robot_id, visualizer, obbs_data, None)

        # 7. 设置GUI控制
        print("\n=== 可视化设置 ===")
        print("• 机器人将在零位姿开始")
        print("• OBB会实时显示和更新")
        print("• 按Ctrl+C退出程序")

        # 相机控制
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5],
        )

        # 8. 主循环 - 持续显示OBB验证结果
        frame_count = 0
        last_update_time = 0

        print("\n开始实时验证...")
        print("OBB将持续更新以验证正向运动学计算的正确性")
        print("程序将持续运行，显示OBB的正确位置和姿态")

        while True:
            # 机器人保持在零位姿
            # OBB会根据当前机器人状态实时更新显示

            # 实时更新OBB以验证正向运动学计算
            current_time = time.time()

            # 每秒更新一次OBB显示
            if current_time - last_update_time >= 1.0:
                visualizer.update_visualization(obbs_data, visible_links)
                last_update_time = current_time

                # 打印状态信息（可选，用于确认程序正常运行）
                if frame_count % 60 == 0:  # 每60帧（约1秒）打印一次
                    print(f"程序运行中... 帧数: {frame_count}, OBB已更新")
            p.stepSimulation()
            frame_count += 1

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        p.disconnect()
        if "visualizer" in locals():
            visualizer.clear_all()


if __name__ == "__main__":
    main()
