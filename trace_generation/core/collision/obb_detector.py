#!/usr/bin/env python3
"""
基于几何计算的OBB碰撞检测实现

使用geometric_collision_detection模块中的几何碰撞检测函数
假设环境中只有AABB格式的障碍物，使用OBB表示机器人连杆

注意：虽然使用几何碰撞检测算法，但仍需要PyBullet来计算正向运动学
"""

import numpy as np
import time
import importlib
import pybullet as p
from typing import Optional, List, Tuple, Dict, Any

from trace_generation.core.collision.geometric_collision_detection import (
    Cuboid,
    AABB,
    cuboid_aabb,
)
from trace_generation.core.robot.obb_forward_kinematics import OBBForwardKinematics


class OBBCollisionEnv:
    """
    基于几何计算的OBB碰撞检测环境类

    不依赖PyBullet，使用纯几何算法进行碰撞检测
    使用OBB（有向包围盒）表示机器人连杆，AABB表示障碍物
    """

    def __init__(
        self,
        robot_name: str,
        config_output_file: Optional[str] = None,
        return_cycles: bool = False,
    ):
        """
        初始化OBB碰撞检测环境

        Args:
            robot_name: 机器人名称（如 'franka', 'iiwa'）
            config_output_file: 配置输出文件路径（可选）
            return_cycles: 是否返回周期数（默认False保持向后兼容）
        """
        self.robot_name = robot_name
        self.obstacle_aabbs: List[AABB] = []

        # config_output_file 相关逻辑
        self.config_output_file = config_output_file
        self.config_list = []

        # 是否返回周期数
        self.return_cycles = return_cycles

        # 初始化碰撞数据管理器（简化版）
        self.collision_check_count = 0
        self.collision_time = 0.0

        # 加载机器人OBB配置
        self.obb_data = self._load_robot_obb_config(robot_name)
        self.link_obbs: Dict[str, Cuboid] = {}  # link_name -> Cuboid

        # 机器人状态相关
        self.valid_collision_links = []  # 有效的碰撞检测连杆索引
        self._initialize_robot_links()

        # 初始化PyBullet和正向运动学（用于计算OBB位姿）
        self.physics_client = p.connect(p.DIRECT)  # 无GUI模式
        self.robot_id = self._load_robot_urdf(robot_name)
        self.obb_fk = (
            OBBForwardKinematics(self.robot_id) if self.robot_id is not None else None
        )

        if self.return_cycles:
            print("✓ [OBBCollisionEnv] 周期计数已启用")

    def _load_robot_urdf(self, robot_name: str) -> Optional[int]:
        """
        加载机器人URDF文件

        Args:
            robot_name: 机器人名称

        Returns:
            机器人ID，如果加载失败则返回None
        """
        import os

        # 机器人URDF映射（从environment.py复制）
        robot_urdf_mapping = {
            "franka": "data/robots/franka_description/franka_panda.urdf",
            "iiwa": "data/robots/iiwa_allegro_description/iiwa.urdf",
            "kinova_gen3": "data/robots/kinova/kinova_gen3_7dof.urdf",
            "ur5e": "data/robots/ur_description/ur5e.urdf",
        }

        rel_path = robot_urdf_mapping.get(robot_name)
        if rel_path is None:
            print(f"警告: 未找到机器人 {robot_name} 的URDF路径")
            return None

        # 计算URDF文件路径
        # 从当前文件位置向上找到trace_generation目录，再找项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        trace_gen_dir = os.path.abspath(os.path.join(current_dir, "../.."))
        project_root = os.path.dirname(trace_gen_dir)

        # 尝试两个可能的路径
        robot_file_trace = os.path.join(trace_gen_dir, rel_path)
        robot_file_root = os.path.join(project_root, rel_path)

        robot_file = None
        if os.path.exists(robot_file_trace):
            robot_file = robot_file_trace
        elif os.path.exists(robot_file_root):
            robot_file = robot_file_root
        else:
            print("警告: URDF文件未找到")
            print(f"  尝试路径1: {robot_file_trace}")
            print(f"  尝试路径2: {robot_file_root}")
            return None

        try:
            robot_id = p.loadURDF(
                robot_file,
                [0, 0, 0],
                [0, 0, 0, 1],
                useFixedBase=True,
                physicsClientId=self.physics_client,
            )
            return robot_id
        except Exception as e:
            print(f"警告: 加载URDF文件失败: {e}")
            return None

    def _load_robot_obb_config(self, robot_name: str) -> List[Dict[str, Any]]:
        """
        从robot_config加载机器人OBB配置

        Args:
            robot_name: 机器人名称

        Returns:
            OBB数据列表
        """
        try:
            # 动态导入机器人配置模块
            config_module = importlib.import_module(
                f"trace_generation.core.robot.robot_config.{robot_name}_obbs"
            )
            obb_data = getattr(config_module, f"{robot_name}_obbs_with_transform", None)

            if obb_data is None:
                # 尝试使用不带transform的版本
                obb_data = getattr(config_module, f"{robot_name}_obbs", None)
                if obb_data is None:
                    raise AttributeError(f"Cannot find OBB data for robot {robot_name}")

            print(f"✓ [OBBCollisionEnv] 加载机器人 {robot_name} 的OBB配置")
            return obb_data

        except ImportError as e:
            raise ImportError(f"Cannot import robot config for {robot_name}: {e}")

    def _initialize_robot_links(self):
        """初始化机器人连杆信息"""
        # 从OBB数据中提取有效的碰撞检测连杆
        self.valid_collision_links = []
        for i, obb_info in enumerate(self.obb_data):
            link_name = obb_info["link_name"]
            # 跳过base link（通常是link0），与PyBullet版本保持一致
            if link_name.endswith("link0"):
                continue
            self.valid_collision_links.append(i)

        print(
            f"✓ [OBBCollisionEnv] 初始化 {len(self.valid_collision_links)} 个碰撞检测连杆"
        )

    def load_obstacle_body_ids(self, obstacle_body_ids: List[int]):
        """
        加载障碍物体ID列表

        注意：这个方法为了保持接口一致性，但在这个几何实现中
        障碍物是通过load_obstacles方法加载的AABB数据

        Args:
            obstacle_body_ids: 障碍物体ID列表（这里用作占位符）
        """
        # 在几何实现中，这个方法主要用于接口兼容性
        # 实际的障碍物加载通过load_obstacles方法完成
        pass

    def load_obstacles(self, obstacles: List[Tuple]) -> List[int]:
        """
        加载并初始化AABB障碍物

        Args:
            obstacles: 障碍物列表，每个元素为 (halfExtents, basePosition) 元组

        Returns:
            障碍物ID列表（简单的索引列表）
        """
        self.cleanup_obstacles()
        self.obstacle_aabbs = []

        for halfExtents, basePosition in obstacles:
            # 将PyBullet格式的box转换为AABB
            # halfExtents = (hx, hy, hz), basePosition = (cx, cy, cz)
            hx, hy, hz = halfExtents
            cx, cy, cz = basePosition

            aabb = AABB(
                min_x=cx - hx,
                min_y=cy - hy,
                min_z=cz - hz,
                max_x=cx + hx,
                max_y=cy + hy,
                max_z=cz + hz,
            )
            self.obstacle_aabbs.append(aabb)

        print(f"✓ [OBBCollisionEnv] 加载 {len(self.obstacle_aabbs)} 个AABB障碍物")
        # 返回障碍物索引列表
        return list(range(len(self.obstacle_aabbs)))

    def update_obstacle_poses(self, new_obstacles: List[Tuple]):
        """
        更新障碍物位置

        Args:
            new_obstacles: 新的障碍物列表 [(halfExtents, basePosition), ...]
        """
        for i, (halfExtents, basePosition) in enumerate(new_obstacles):
            if i < len(self.obstacle_aabbs):
                hx, hy, hz = halfExtents
                cx, cy, cz = basePosition

                # 更新AABB
                self.obstacle_aabbs[i] = AABB(
                    min_x=cx - hx,
                    min_y=cy - hy,
                    min_z=cz - hz,
                    max_x=cx + hx,
                    max_y=cy + hy,
                    max_z=cz + hz,
                )

    def cleanup_obstacles(self):
        """清理障碍物"""
        self.obstacle_aabbs.clear()

    def close(self):
        """关闭碰撞检测环境"""
        self.cleanup_obstacles()
        # 断开PyBullet连接
        if hasattr(self, "physics_client") and self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass  # 忽略断开连接时的错误
        self.link_obbs.clear()

    def _get_world_transform(
        self, obb_info: Dict[str, Any], joint_config: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        获取OBB在世界坐标系中的变换矩阵

        使用PyBullet的正向运动学计算给定关节配置下OBB的世界坐标变换

        Args:
            obb_info: OBB信息字典
            joint_config: 关节配置

        Returns:
            4x4变换矩阵
        """
        # 如果没有正向运动学计算器或没有关节配置，使用默认transform
        if self.obb_fk is None or joint_config is None:
            if "transform" in obb_info:
                return obb_info["transform"].copy()

            # 否则，从position和rotation_matrix构造
            transform = np.eye(4)
            if "position" in obb_info:
                transform[:3, 3] = obb_info["position"]
            if "rotation_matrix" in obb_info:
                transform[:3, :3] = obb_info["rotation_matrix"]
            return transform

        # 使用正向运动学计算OBB的世界变换
        # 1. 设置机器人关节配置
        self.obb_fk.set_joint_configuration(joint_config.tolist())

        # 2. 计算OBB在当前配置下的世界位姿
        obb_poses = self.obb_fk.compute_obb_poses([obb_info], joint_config.tolist())

        if len(obb_poses) > 0:
            return obb_poses[0]["transform"]
        else:
            # 如果计算失败，回退到默认transform
            if "transform" in obb_info:
                return obb_info["transform"].copy()
            return np.eye(4)

    def _create_cuboid_from_obb(
        self, obb_info: Dict[str, Any], joint_config: Optional[np.ndarray] = None
    ) -> Cuboid:
        """
        从OBB信息创建Cuboid对象

        Args:
            obb_info: OBB信息字典
            joint_config: 关节配置

        Returns:
            Cuboid对象
        """
        # 获取世界变换
        world_transform = self._get_world_transform(obb_info, joint_config)

        # 提取位置
        position = world_transform[:3, 3]

        # 提取旋转矩阵的列作为轴
        rotation = world_transform[:3, :3]
        axis_1 = rotation[:, 0]  # 第一个轴
        axis_2 = rotation[:, 1]  # 第二个轴
        axis_3 = rotation[:, 2]  # 第三个轴

        # 获取半轴长
        extents = obb_info["extents"]

        # 创建Cuboid对象
        # Cuboid构造函数: x, y, z, axis_1, axis_2, axis_3
        # 其中每个axis是 (x, y, z, radius) 的元组
        cuboid = Cuboid(
            position[0],
            position[1],
            position[2],
            (axis_1[0], axis_1[1], axis_1[2], extents[0]),
            (axis_2[0], axis_2[1], axis_2[2], extents[1]),
            (axis_3[0], axis_3[1], axis_3[2], extents[2]),
        )

        return cuboid

    def _get_link_collisions(self, joint_config: Optional[np.ndarray] = None) -> Tuple:
        """
        获取各个valid link的碰撞结果

        Args:
            joint_config: 关节配置（可选）

        Returns:
            如果 self.return_cycles=False: (any_coll, link_colls)
                - any_coll: 是否有任何碰撞
                - link_colls: 各连杆碰撞状态列表（0=碰撞，1=无碰撞）
            如果 self.return_cycles=True: (any_coll, link_colls, link_cycles)
                - any_coll: 是否有任何碰撞
                - link_colls: 各连杆碰撞状态列表（0=碰撞，1=无碰撞）
                - link_cycles: 各连杆的周期数列表
        """
        any_coll = False
        link_colls = []
        link_cycles = []

        for link_idx in self.valid_collision_links:
            obb_info = self.obb_data[link_idx]

            # 创建当前关节配置下的Cuboid
            cuboid = self._create_cuboid_from_obb(obb_info, joint_config)

            # 检查与所有障碍物的碰撞
            link_collision = False
            link_total_cycles = 0
            for aabb in self.obstacle_aabbs:
                collision_result, cycles = cuboid_aabb(cuboid, aabb)
                link_total_cycles += cycles  # 累加该连杆的周期
                if collision_result == 0:  # 0表示碰撞
                    link_collision = True
                    break

            if link_collision:
                any_coll = True
                link_colls.append(0)  # 0表示碰撞
            else:
                link_colls.append(1)  # 1表示无碰撞

            link_cycles.append(link_total_cycles)

        if self.return_cycles:
            return any_coll, link_colls, link_cycles
        return any_coll, link_colls

    def check_pose(self, state: np.ndarray) -> Tuple[bool, List, List]:
        """
        检查单个配置点的碰撞数据

        Args:
            state: 配置状态

        Returns:
            (is_free, link_coords, link_colls):
                - is_free (bool): True表示无碰撞，False表示有碰撞
                - link_coords (List): 连杆坐标列表
                - link_colls (List): 连杆碰撞信息列表
        """
        start_time = time.time()
        self.collision_check_count += 1

        # 在几何实现中，我们简化状态验证（假设所有状态都有效）
        # 在实际应用中，可能需要添加关节限制检查

        # 收集Link坐标和碰撞状态
        link_coords = []
        result = self._get_link_collisions(state)

        if self.return_cycles:
            link_collision, link_colls, link_cycles = result
        else:
            link_collision, link_colls = result

        for link_idx in self.valid_collision_links:
            obb_info = self.obb_data[link_idx]
            cuboid = self._create_cuboid_from_obb(obb_info, state)
            # 使用OBB中心作为连杆坐标（简化实现）
            link_coords.append(np.array([cuboid.x, cuboid.y, cuboid.z]))

        # 判断是否无碰撞
        is_free = not link_collision

        self.collision_time += time.time() - start_time

        # 与sphere_detector保持一致，check_pose不返回cycles
        return is_free, link_coords, link_colls


    def get_collision_stats(self) -> Dict[str, Any]:
        """
        获取碰撞检测统计信息

        Returns:
            dict: 统计信息字典
        """
        return {
            "robot_name": self.robot_name,
            "num_obstacles": len(self.obstacle_aabbs),
            "num_links": len(self.valid_collision_links),
            "collision_check_count": self.collision_check_count,
            "total_collision_time": self.collision_time,
            "avg_collision_time": self.collision_time
            / max(1, self.collision_check_count),
        }
