# type: ignore
"""
简化的机器人 OBB 可视化测试 (支持CoACD + Open3D)

这个脚本直接使用 PyBullet 来可视化机器人的 OBB，不依赖 VAMP 集成。

新功能:
- 使用CoACD进行凸分解，获得更精确的凸包
- 使用Open3D计算最小有向包围盒
- 支持collision mesh可视化
- 直接使用URDF文件路径，无需预定义机器人

依赖安装:
pip install open3d coacd yourdfpy trimesh scipy
或运行: ./install_obb_deps.sh

使用方法:
python simple_obb_visualization.py <urdf_path> [--links link1,link2,...]

示例:
python simple_obb_visualization.py ./robots/panda/panda.urdf
python simple_obb_visualization.py ./robots/panda/panda.urdf --links panda_link1,panda_link2
"""

import time
import sys
import argparse
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data

# 添加脚本目录到路径
sys.path.append(str(Path(__file__).parent))

from obb_calculator import calculate_link_obbs, check_dependencies
from obb_forward_kinematics import OBBForwardKinematics

# 检查依赖库状态
HAS_OBB_LIBS, missing_libs = check_dependencies()
if not HAS_OBB_LIBS:
    print(
        f"Warning: Some libraries not available for OBB calculation: {', '.join(missing_libs)}"
    )
    print("For full functionality, install: pip install " + " ".join(missing_libs))


def list_robot_links(urdf_path):
    """
    列出机器人URDF文件的所有连杆名称

    Args:
        urdf_path: URDF文件路径

    Returns:
        list: 连杆名称列表
    """
    # 连接到 PyBullet (无GUI模式)
    p.connect(p.DIRECT)

    try:
        # 加载机器人模型
        if not Path(urdf_path).exists():
            print(f"找不到机器人文件: {urdf_path}")
            return []

        robot_id = p.loadURDF(str(urdf_path), [0, 0, 0])
        num_joints = p.getNumJoints(robot_id)

        # 收集连杆名称
        link_names = ["panda_link0"]  # 基座连杆

        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            link_name = joint_info[12].decode("utf-8")
            link_names.append(link_name)

        return link_names
    finally:
        p.disconnect()


class RobotVisualizer:
    """处理机器人模型加载和可视化的类"""

    def __init__(self, urdf_path):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"找不到URDF文件: {urdf_path}")

        self.robot_id = None
        self.num_joints = 0
        self.link_name_to_index = {}
        self.link_color_mapping = {}
        self.link_info_list = []
        self.physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._load_environment()
        self._load_robot()
        self._get_link_info()

    def _load_environment(self):
        """加载地面等环境"""
        ground_shape = p.createCollisionShape(p.GEOM_PLANE)
        ground_body = p.createMultiBody(0, ground_shape)
        p.changeVisualShape(ground_body, -1, rgbaColor=[0.8, 0.8, 0.8, 1.0])

    def _load_robot(self):
        """加载机器人URDF模型"""
        self.robot_id = p.loadURDF(str(self.urdf_path), [0, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"加载机器人: {self.urdf_path} (基座固定)")

    def _get_link_info(self):
        """获取并存储连杆信息"""
        # 基座连杆
        base_name = p.getBodyInfo(self.robot_id)[0].decode("utf-8")
        self.link_name_to_index[base_name] = -1
        self.link_info_list.append({"name": base_name, "index": -1})

        # 其他连杆
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode("utf-8")
            self.link_name_to_index[link_name] = i
            self.link_info_list.append({"name": link_name, "index": i})

    def set_link_visibility(self, visible_links=None, show_all_links=True):
        """设置连杆的可见性和颜色"""
        # 预定义颜色
        link_colors = [
            [1.0, 0.0, 0.0, 0.8],
            [0.0, 1.0, 0.0, 0.8],
            [0.0, 0.0, 1.0, 0.8],
            [1.0, 1.0, 0.0, 0.8],
            [1.0, 0.0, 1.0, 0.8],
            [0.0, 1.0, 1.0, 0.8],
            [1.0, 0.5, 0.0, 0.8],
            [0.5, 0.0, 1.0, 0.8],
            [0.8, 0.4, 0.2, 0.8],
            [0.2, 0.8, 0.4, 0.8],
            [0.4, 0.2, 0.8, 0.8],
            [0.8, 0.8, 0.2, 0.8],
        ]
        color_names = [
            "红",
            "绿",
            "蓝",
            "黄",
            "品红",
            "青",
            "橙",
            "紫",
            "棕",
            "浅绿",
            "深紫",
            "橄榄",
        ]

        # 确定要显示的连杆
        if visible_links is None:
            visible_link_names = (
                set(info["name"] for info in self.link_info_list)
                if show_all_links
                else set()
            )
        else:
            visible_link_names = set(visible_links)

        # 设置颜色和可见性
        for i, link_info in enumerate(self.link_info_list):
            link_name = link_info["name"]
            link_index = link_info["index"]
            color_index = i % len(link_colors)
            self.link_color_mapping[link_name] = color_index

            if link_name in visible_link_names:
                p.changeVisualShape(
                    self.robot_id, link_index, rgbaColor=link_colors[color_index]
                )
                print(
                    f"显示连杆: {link_name} ({color_names[color_index % len(color_names)]}色)"
                )
            else:
                p.changeVisualShape(self.robot_id, link_index, rgbaColor=[0, 0, 0, 0])
                print(f"隐藏连杆: {link_name}")

    def get_link_world_transform(self, link_name):
        """获取连杆在世界坐标系中的4x4变换矩阵"""
        if link_name not in self.link_name_to_index:
            return np.eye(4)

        link_index = self.link_name_to_index[link_name]
        if link_index == -1:
            pos = (0, 0, 0)
            orn = (0, 0, 0, 1)
        else:
            state = p.getLinkState(self.robot_id, link_index)
            pos, orn = state[4], state[5]

        transform_matrix = np.eye(4)
        rotation = p.getMatrixFromQuaternion(orn)
        transform_matrix[:3, :3] = np.array(rotation).reshape(3, 3)
        transform_matrix[:3, 3] = pos
        return transform_matrix

    def disconnect(self):
        p.disconnect()


class OBBVisualizer:
    """处理OBB可视化的类"""

    def __init__(self):
        self.obb_bodies = []
        self.obb_fk = None  # OBB正向运动学计算器

    def initialize_obb_forward_kinematics(self, robot_id):
        """初始化OBB正向运动学计算器"""
        self.obb_fk = OBBForwardKinematics(robot_id)

    def draw_obbs(
        self, robot_visualizer, obbs_data, visible_links=None, show_all_obbs=True
    ):
        """在场景中绘制OBB - 基于机器人实时关节配置动态计算位姿"""
        obb_colors = [
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 1.0, 0.0, 0.3],
            [0.0, 0.0, 1.0, 0.3],
            [1.0, 1.0, 0.0, 0.3],
            [1.0, 0.0, 1.0, 0.3],
            [0.0, 1.0, 1.0, 0.3],
            [1.0, 0.5, 0.0, 0.3],
            [0.5, 0.0, 1.0, 0.3],
            [0.8, 0.4, 0.2, 0.3],
            [0.2, 0.8, 0.4, 0.3],
            [0.4, 0.2, 0.8, 0.3],
            [0.8, 0.8, 0.2, 0.3],
        ]
        color_names = [
            "红",
            "绿",
            "蓝",
            "黄",
            "品红",
            "青",
            "橙",
            "紫",
            "棕",
            "浅绿",
            "深紫",
            "橄榄",
        ]

        if visible_links is None:
            visible_link_names = (
                set(robot_visualizer.link_name_to_index.keys())
                if show_all_obbs
                else set()
            )
        else:
            visible_link_names = set(visible_links)

        # 获取机器人当前关节配置
        current_joint_config = self._get_current_joint_configuration(robot_visualizer)
        print(
            f"  当前关节配置: {[f'{angle:.3f}' for angle in current_joint_config[:7]]}"
        )

        # 使用obb_forward_kinematics模块计算所有OBB位姿
        if self.obb_fk is None:
            self.initialize_obb_forward_kinematics(robot_visualizer.robot_id)

        obb_poses = self.obb_fk.compute_obb_poses(obbs_data)

        for obb_pose in obb_poses:
            link_name = obb_pose["link_name"]
            if link_name not in visible_link_names:
                print(f"  隐藏 OBB: {link_name}")
                continue

            color_index = robot_visualizer.link_color_mapping.get(link_name, -1)
            obb_color = (
                obb_colors[color_index] if color_index != -1 else [0.5, 0.5, 0.5, 0.3]
            )

            # 直接使用计算好的位姿信息
            final_position = obb_pose["position"]
            final_quaternion = obb_pose["quaternion"]
            extents = obb_pose["extents"]

            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=extents / 2.0,
                rgbaColor=obb_color,
            )
            body_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=final_position,
                baseOrientation=final_quaternion,
            )
            self.obb_bodies.append(body_id)

            color_name = (
                color_names[color_index % len(color_names)]
                if color_index != -1
                else "默认"
            )
            print(
                f"  添加 OBB: {link_name} ({color_name}色, 位置: {np.array2string(final_position, precision=3)}, 尺寸: {np.array2string(extents, precision=3)})"
            )

    def _get_current_joint_configuration(self, robot_visualizer):
        """获取机器人当前的关节配置"""
        joint_config = []
        for i in range(robot_visualizer.num_joints):
            joint_state = p.getJointState(robot_visualizer.robot_id, i)
            joint_angle = joint_state[0]  # 关节角度
            joint_config.append(joint_angle)
        return joint_config

    def update_obbs(
        self, robot_visualizer, obbs_data, visible_links=None, show_all_obbs=True
    ):
        """更新OBB位姿 - 用于实时动画"""
        # 清除旧的OBB
        self.clear_obbs()

        # 重新绘制基于当前关节配置的OBB
        self.draw_obbs(robot_visualizer, obbs_data, visible_links, show_all_obbs)

    def clear_obbs(self):
        """清除所有OBB"""
        for body_id in self.obb_bodies:
            p.removeBody(body_id)
        self.obb_bodies.clear()


def main_visualization(
    urdf_path, visible_links=None, show_all_links=True, show_all_obbs=True
):
    """主可视化函数"""
    robot_vis = None
    try:
        # 1. 初始化机器人可视化
        robot_vis = RobotVisualizer(urdf_path)
        robot_vis.set_link_visibility(visible_links, show_all_links)

        # 2. 计算OBB
        if HAS_OBB_LIBS:
            print("\n计算有向包围盒...")
            obbs_data = calculate_link_obbs(str(urdf_path))
            print(f"成功计算 {len(obbs_data)} 个 OBB")

            # 3. 可视化OBB
            obb_vis = OBBVisualizer()
            obb_vis.draw_obbs(robot_vis, obbs_data, visible_links, show_all_obbs)
        else:
            obbs_data = []  # 没有OBB数据时初始化为空列表

        # 4. 设置相机和GUI控制
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5],
        )

        # 相机控制滑块
        dist_slider = p.addUserDebugParameter("距离", 0.5, 5.0, 2.0)
        yaw_slider = p.addUserDebugParameter("水平角", -180, 180, 45)
        pitch_slider = p.addUserDebugParameter("俯仰角", -89, 89, -30)

        # 关节控制滑块
        joint_sliders = []
        for i in range(min(robot_vis.num_joints, 7)):  # 最多显示7个关节
            joint_info = p.getJointInfo(robot_vis.robot_id, i)
            joint_name = joint_info[1].decode("utf-8")
            joint_lower_limit = joint_info[8]
            joint_upper_limit = joint_info[9]

            # 如果关节限制无效，使用默认范围
            if joint_lower_limit >= joint_upper_limit:
                joint_lower_limit = -3.14
                joint_upper_limit = 3.14

            slider = p.addUserDebugParameter(
                f"关节{i}({joint_name})", joint_lower_limit, joint_upper_limit, 0.0
            )
            joint_sliders.append(slider)

        # OBB更新控制
        obb_update_slider = p.addUserDebugParameter("OBB更新频率", 1, 60, 10)

        print("\n=== 可视化完成, 关闭窗口或按Ctrl+C退出... ===")
        print("提示: 使用关节滑块控制机器人姿态，OBB会实时更新")

        # 5. 运行仿真循环
        frame_count = 0
        while True:
            # 读取关节控制滑块值并设置关节角度
            for i, slider in enumerate(joint_sliders):
                joint_angle = p.readUserDebugParameter(slider)
                p.resetJointState(robot_vis.robot_id, i, joint_angle)

            # 根据更新频率更新OBB
            obb_update_freq = int(p.readUserDebugParameter(obb_update_slider))
            if HAS_OBB_LIBS and frame_count % (60 // obb_update_freq) == 0:
                obb_vis.update_obbs(robot_vis, obbs_data, visible_links, show_all_obbs)

            p.stepSimulation()
            time.sleep(1.0 / 60.0)
            frame_count += 1

            # 更新相机
            p.resetDebugVisualizerCamera(
                p.readUserDebugParameter(dist_slider),
                p.readUserDebugParameter(yaw_slider),
                p.readUserDebugParameter(pitch_slider),
                [0, 0, 0.5],
            )

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if robot_vis:
            robot_vis.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="机器人OBB可视化工具")
    parser.add_argument("urdf_path", help="URDF文件路径")
    parser.add_argument("--links", help="要显示的连杆名称（逗号分隔）", default=None)
    parser.add_argument("--no-obb", action="store_true", help="不显示OBB，只显示机器人")
    parser.add_argument(
        "--no-links", action="store_true", help="不显示机器人连杆，只显示OBB"
    )
    parser.add_argument(
        "--list-links", action="store_true", help="列出所有连杆名称并退出"
    )

    args = parser.parse_args()

    # 解析要显示的连杆
    visible_links = None
    if args.links:
        visible_links = [link.strip() for link in args.links.split(",")]

    # 如果需要列出连杆名称
    if args.list_links:
        print("=== 可用连杆名称 ===")
        available_links = list_robot_links(args.urdf_path)
        for i, link_name in enumerate(available_links):
            print(f"{i}: {link_name}")

    # 确定显示选项
    show_all_links = not args.no_links
    show_all_obbs = not args.no_obb

    # 运行可视化
    main_visualization(
        args.urdf_path,
        visible_links=visible_links,
        show_all_links=show_all_links,
        show_all_obbs=show_all_obbs,
    )
