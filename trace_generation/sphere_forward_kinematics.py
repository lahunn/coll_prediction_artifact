#!/usr/bin/env python3
"""
球体正向运动学计算器

功能：
1. 根据机器人关节配置计算每个球体的世界坐标位置
2. 支持实时更新球体位姿
3. 提供球体变换矩阵计算
4. 与YAML球体配置文件兼容

使用方法：
from sphere_forward_kinematics import SphereForwardKinematics

sf = SphereForwardKinematics(robot_id, collision_spheres)
sphere_poses = sf.compute_sphere_poses()
"""

import numpy as np
import pybullet as p
from typing import Dict, List
from sphere_visualizer import SphereConfig


class SpherePose:
    """球体位姿数据结构"""

    def __init__(
        self,
        link_name: str,
        sphere_index: int,
        center: np.ndarray,
        radius: float,
        transform: np.ndarray,
    ):
        self.link_name = link_name
        self.sphere_index = sphere_index
        self.center = center  # 世界坐标系下的中心位置
        self.radius = radius
        self.transform = transform  # 4x4变换矩阵
        self.quaternion = self._matrix_to_quaternion(transform[:3, :3])

    def _matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """将旋转矩阵转换为四元数"""
        try:
            # 使用scipy的方法
            from scipy.spatial.transform import Rotation as R

            rotation = R.from_matrix(rotation_matrix)
            return rotation.as_quat()  # [x, y, z, w]
        except ImportError:
            # 如果没有scipy，使用简单的实现
            # 对于球体，旋转通常不重要，返回单位四元数
            return np.array([0, 0, 0, 1])

    def __repr__(self):
        return f"SpherePose({self.link_name}[{self.sphere_index}], center={self.center}, r={self.radius:.3f})"


class SphereForwardKinematics:
    """球体正向运动学计算器"""

    def __init__(self, robot_id: int, collision_spheres: Dict[str, List[SphereConfig]]):
        """
        初始化球体正向运动学计算器

        Args:
            robot_id: PyBullet机器人ID
            collision_spheres: 球体配置字典
        """
        self.robot_id = robot_id
        self.collision_spheres = collision_spheres

        # 构建连杆名称到索引的映射
        self.link_name_to_index = {}
        self._build_link_mapping()

        # 有效连杆过滤
        self.valid_links = self._filter_valid_links()

        print("球体正向运动学初始化完成:")
        print(f"  机器人ID: {robot_id}")
        print(f"  总连杆数: {len(self.collision_spheres)}")
        print(f"  有效连杆数: {len(self.valid_links)}")

    def _build_link_mapping(self):
        """构建连杆名称到PyBullet索引的映射"""
        try:
            # 获取基座连杆名称
            base_info = p.getBodyInfo(self.robot_id)
            base_name = base_info[0].decode("utf-8")
            self.link_name_to_index[base_name] = -1

            # 获取其他连杆
            num_joints = p.getNumJoints(self.robot_id)
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                link_name = joint_info[12].decode("utf-8")
                self.link_name_to_index[link_name] = i

            print(f"连杆映射构建完成: {len(self.link_name_to_index)} 个连杆")

        except Exception as e:
            print(f"构建连杆映射失败: {e}")

    def _filter_valid_links(self) -> List[str]:
        """过滤有效的连杆（既在YAML中定义又在机器人中存在）"""
        valid_links = []

        for link_name in self.collision_spheres.keys():
            if link_name in self.link_name_to_index:
                # 检查是否有有效球体
                valid_spheres = [
                    s for s in self.collision_spheres[link_name] if s.is_valid
                ]
                if valid_spheres:
                    valid_links.append(link_name)
                    print(f"  有效连杆: {link_name} ({len(valid_spheres)} 球体)")
                else:
                    print(f"  跳过连杆: {link_name} (无有效球体)")
            else:
                print(f"  跳过连杆: {link_name} (机器人中不存在)")

        return valid_links

    def get_link_transform(self, link_name: str) -> np.ndarray:
        """获取连杆的4x4变换矩阵"""
        if link_name not in self.link_name_to_index:
            print(f"警告: 连杆 {link_name} 不存在")
            return np.eye(4)

        link_index = self.link_name_to_index[link_name]

        try:
            if link_index == -1:  # 基座连杆
                pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            else:  # 其他连杆
                link_state = p.getLinkState(self.robot_id, link_index)
                pos = link_state[0]  # 连杆世界位置
                orn = link_state[1]  # 连杆世界姿态（四元数）

            # 构建4x4变换矩阵
            transform = np.eye(4)

            # 四元数转旋转矩阵
            rotation_matrix = p.getMatrixFromQuaternion(orn)
            transform[:3, :3] = np.array(rotation_matrix).reshape(3, 3)

            # 设置平移
            transform[:3, 3] = pos

            return transform

        except Exception as e:
            print(f"获取连杆 {link_name} 变换失败: {e}")
            return np.eye(4)

    def compute_sphere_world_position(
        self, link_transform: np.ndarray, sphere_center: np.ndarray
    ) -> np.ndarray:
        """计算球体在世界坐标系下的位置"""
        # 将球体中心扩展为齐次坐标
        sphere_homogeneous = np.append(sphere_center, 1.0)

        # 应用变换
        world_position_homogeneous = link_transform @ sphere_homogeneous

        # 返回3D位置
        return world_position_homogeneous[:3]

    def compute_sphere_poses(self) -> List[SpherePose]:
        """计算所有球体的世界位姿"""
        sphere_poses = []

        for link_name in self.valid_links:
            # 获取连杆变换矩阵
            link_transform = self.get_link_transform(link_name)

            # 处理该连杆的所有球体
            spheres = self.collision_spheres[link_name]
            valid_spheres = [s for s in spheres if s.is_valid]

            for sphere_index, sphere in enumerate(valid_spheres):
                # 计算球体世界位置
                world_center = self.compute_sphere_world_position(
                    link_transform, sphere.center
                )

                # 创建球体位姿对象
                sphere_pose = SpherePose(
                    link_name=link_name,
                    sphere_index=sphere_index,
                    center=world_center,
                    radius=sphere.radius,
                    transform=link_transform,
                )

                sphere_poses.append(sphere_pose)

        return sphere_poses

    def get_sphere_positions_by_link(self, link_name: str) -> List[np.ndarray]:
        """获取指定连杆的所有球体世界位置"""
        if link_name not in self.valid_links:
            return []

        link_transform = self.get_link_transform(link_name)
        spheres = self.collision_spheres[link_name]
        valid_spheres = [s for s in spheres if s.is_valid]

        positions = []
        for sphere in valid_spheres:
            world_position = self.compute_sphere_world_position(
                link_transform, sphere.center
            )
            positions.append(world_position)

        return positions

    def print_sphere_poses(self, sphere_poses: List[SpherePose]):
        """打印球体位姿信息"""
        print(f"\n=== 球体位姿信息 ({len(sphere_poses)} 个球体) ===")

        by_link = {}
        for pose in sphere_poses:
            if pose.link_name not in by_link:
                by_link[pose.link_name] = []
            by_link[pose.link_name].append(pose)

        for link_name, link_poses in by_link.items():
            print(f"\n连杆 {link_name}:")
            for pose in link_poses:
                print(
                    f"  球体{pose.sphere_index}: 位置={pose.center}, 半径={pose.radius:.3f}"
                )


def test_sphere_forward_kinematics():
    """测试球体正向运动学"""
    import pybullet_data
    from sphere_visualizer import SphereYAMLParser

    # 连接PyBullet
    p.connect(p.DIRECT)  # 无GUI模式
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    try:
        # 加载Franka机器人（需要正确的URDF路径）
        robot_urdf = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/panda/panda.urdf"
        robot_id = p.loadURDF(robot_urdf, [0, 0, 0], useFixedBase=True)

        # 解析球体配置
        yaml_file = "/home/lanh/project/robot_sim/coll_prediction_artifact/content/configs/robot/spheres/franka.yml"
        parser = SphereYAMLParser(yaml_file)
        collision_spheres = parser.parse()

        # 创建球体正向运动学计算器
        sphere_fk = SphereForwardKinematics(robot_id, collision_spheres)

        # 设置一个测试关节配置
        test_config = [0.1, -0.2, 0.0, -1.5, 0.0, 1.3, 0.7]  # 7个关节角度
        for i, angle in enumerate(test_config):
            p.resetJointState(robot_id, i, angle)

        # 计算球体位姿
        sphere_poses = sphere_fk.compute_sphere_poses()

        # 打印结果
        sphere_fk.print_sphere_poses(sphere_poses)

        print(f"\n测试完成: 成功计算 {len(sphere_poses)} 个球体的位姿")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        p.disconnect()


if __name__ == "__main__":
    test_sphere_forward_kinematics()
