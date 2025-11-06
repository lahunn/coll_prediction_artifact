# type: ignore
"""
OBB正向运动学计算模块

该模块提供机器人OBB（有向包围盒）的正向运动学计算功能，
支持给定关节配置下计算所有连杆OBB的世界坐标位姿。

主要功能:
1. 计算OBB在给定关节配置下的世界坐标位姿
2. 支持PyBullet仿真环境
3. 提供旋转矩阵到四元数的转换
4. 获取连杆的世界变换矩阵

核心类和函数:
- OBBForwardKinematics: 主要的OBB正向运动学计算类
- rotation_matrix_to_quaternion: 旋转矩阵转换为四元数
- get_link_world_transform: 获取连杆世界变换矩阵

使用示例:
    import pybullet as p
    from obb_forward_kinematics import OBBForwardKinematics

    # 初始化PyBullet和加载机器人
    p.connect(p.GUI)
    robot_id = p.loadURDF("robot.urdf")

    # 创建OBB正向运动学计算器
    obb_fk = OBBForwardKinematics(robot_id)

    # 计算给定关节配置下的OBB位姿
    joint_config = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    obb_poses = obb_fk.compute_obb_poses(obbs_data, joint_config)

作者: VAMP项目组
版本: 1.0.0
"""

import numpy as np
import pybullet as p
from typing import List, Dict, Optional, Tuple, Any


def rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
    """
    将3x3旋转矩阵转换为四元数(x, y, z, w)格式，兼容PyBullet

    Args:
        R: 3x3旋转矩阵

    Returns:
        四元数列表 [x, y, z, w]，符合PyBullet格式
    """
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    # 返回PyBullet格式 (x, y, z, w)
    return [x, y, z, w]


def get_link_world_transform(
    robot_id: int, link_name: str, link_name_to_index: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    获取连杆在世界坐标系中的4x4变换矩阵

    Args:
        robot_id: PyBullet机器人ID
        link_name: 连杆名称
        link_name_to_index: 连杆名称到索引的映射字典（可选，提供可提高性能）

    Returns:
        4x4变换矩阵
    """
    if robot_id is None:
        return np.eye(4)

    # 如果没有提供映射，则动态创建
    if link_name_to_index is None:
        link_name_to_index = {}

        # 基座连杆
        base_name = p.getBodyInfo(robot_id)[0].decode("utf-8")
        link_name_to_index[base_name] = -1

        # 其他连杆
        num_joints = p.getNumJoints(robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            current_link_name = joint_info[12].decode("utf-8")
            link_name_to_index[current_link_name] = i

    if link_name not in link_name_to_index:
        return np.eye(4)

    link_index = link_name_to_index[link_name]
    if link_index == -1:  # 基座连杆
        pos = (0, 0, 0)
        orn = (0, 0, 0, 1)
    else:
        state = p.getLinkState(robot_id, link_index)
        pos, orn = state[4], state[5]  # 世界坐标系中的位置和方向

    # 构建4x4变换矩阵
    transform_matrix = np.eye(4)
    rotation = p.getMatrixFromQuaternion(orn)
    transform_matrix[:3, :3] = np.array(rotation).reshape(3, 3)
    transform_matrix[:3, 3] = pos
    return transform_matrix


class OBBForwardKinematics:
    """
    OBB正向运动学计算器

    该类负责计算机器人在给定关节配置下，各连杆OBB的世界坐标位姿。
    """

    def __init__(self, robot_id: int):
        """
        初始化OBB正向运动学计算器

        Args:
            robot_id: PyBullet机器人ID
        """
        self.robot_id = robot_id
        self.num_joints = p.getNumJoints(robot_id) if robot_id is not None else 0
        self.link_name_to_index = self._build_link_mapping()

    def _build_link_mapping(self) -> Dict[str, int]:
        """
        构建连杆名称到索引的映射

        Returns:
            连杆名称到索引的映射字典
        """
        if self.robot_id is None:
            return {}

        link_name_to_index = {}

        # 基座连杆
        base_name = p.getBodyInfo(self.robot_id)[0].decode("utf-8")
        link_name_to_index[base_name] = -1

        # 其他连杆
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode("utf-8")
            link_name_to_index[link_name] = i

        return link_name_to_index

    def set_joint_configuration(self, joint_config: List[float]) -> None:
        """
        设置机器人关节配置

        Args:
            joint_config: 关节角度列表
        """
        if self.robot_id is None:
            return

        for i, angle in enumerate(joint_config):
            if i < self.num_joints:
                p.resetJointState(self.robot_id, i, angle)

    def get_current_joint_configuration(self) -> List[float]:
        """
        获取机器人当前的关节配置

        Returns:
            当前关节角度列表
        """
        if self.robot_id is None:
            return []

        joint_config = []
        for i in range(self.num_joints):
            joint_state = p.getJointState(self.robot_id, i)
            joint_angle = joint_state[0]  # 关节角度
            joint_config.append(joint_angle)
        return joint_config

    def compute_obb_poses(
        self,
        obbs_data: List[Dict[str, Any]],
        joint_config: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        计算OBB在给定关节配置下的世界坐标位姿

        Args:
            obbs_data: 原始OBB数据列表，每个元素应包含:
                - link_name: 连杆名称
                - transform: OBB相对于连杆的4x4变换矩阵
                - extents: OBB尺寸 [长, 宽, 高]
            joint_config: 关节配置，如果为None则使用当前配置

        Returns:
            OBB位姿信息列表，每个元素包含:
                - link_name: 连杆名称
                - position: 世界坐标位置 [x, y, z]
                - quaternion: 四元数姿态 [x, y, z, w]
                - extents: OBB尺寸 [长, 宽, 高]
                - transform: OBB的世界变换矩阵
                - original_obb: 原始OBB数据引用
        """
        if self.robot_id is None:
            return []

        # 如果指定了关节配置，先设置机器人关节
        if joint_config is not None:
            self.set_joint_configuration(joint_config)

        obb_poses = []

        for obb in obbs_data:
            link_name = obb["link_name"]

            # 获取连杆在当前机器人配置下的实时世界变换
            link_world_transform = get_link_world_transform(
                self.robot_id, link_name, self.link_name_to_index
            )

            # 将OBB的相对变换应用到连杆当前位姿上
            obb_world_transform = link_world_transform @ obb["transform"]

            # 提取位置和旋转
            final_position = obb_world_transform[:3, 3]
            final_rotation_matrix = obb_world_transform[:3, :3]
            final_quaternion = rotation_matrix_to_quaternion(final_rotation_matrix)

            obb_pose = {
                "link_name": link_name,
                "position": final_position,
                "quaternion": final_quaternion,
                "extents": obb["extents"],
                "transform": obb_world_transform,
                "original_obb": obb,  # 保留原始OBB数据的引用
            }

            obb_poses.append(obb_pose)

        return obb_poses

    def compute_single_obb_pose(
        self, obb_data: Dict[str, Any], joint_config: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        计算单个OBB在给定关节配置下的世界坐标位姿

        Args:
            obb_data: 单个OBB数据字典
            joint_config: 关节配置，如果为None则使用当前配置

        Returns:
            单个OBB位姿信息字典
        """
        obb_poses = self.compute_obb_poses([obb_data], joint_config)
        return obb_poses[0] if obb_poses else {}

    def get_link_names(self) -> List[str]:
        """
        获取所有连杆名称

        Returns:
            连杆名称列表
        """
        return list(self.link_name_to_index.keys())

    def get_joint_limits(self) -> List[Tuple[float, float]]:
        """
        获取所有关节的限制

        Returns:
            关节限制列表，每个元素为(下限, 上限)元组
        """
        if self.robot_id is None:
            return []

        joint_limits = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]

            # 如果关节限制无效，使用默认范围
            if lower_limit >= upper_limit:
                lower_limit = -np.pi
                upper_limit = np.pi

            joint_limits.append((lower_limit, upper_limit))

        return joint_limits


if __name__ == "__main__":
    """
    测试代码示例
    """
    print("OBB Forward Kinematics Module")
    print("这是一个模块文件，请在其他脚本中导入使用")
    print("\n使用示例:")
    print("from obb_forward_kinematics import OBBForwardKinematics")
    print("obb_fk = OBBForwardKinematics(robot_id)")
    print("obb_poses = obb_fk.compute_obb_poses(obbs_data, joint_config)")
