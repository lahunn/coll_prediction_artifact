import numpy as np
import pybullet as p


class ObstacleManager:
    """障碍物管理类，负责障碍物的创建、初始化、位置管理和清理"""

    def __init__(self, physics_client):
        """
        初始化障碍物管理器

        Args:
            physics_client: PyBullet物理客户端ID
        """
        self.physics_client = physics_client
        self.obstacles = []
        self.obstacle_body_ids = []

    def create_voxel(self, halfExtents, basePosition):
        """
        创建体素障碍物

        Args:
            halfExtents: 半尺寸
            basePosition: 基础位置

        Returns:
            创建的障碍物ID
        """
        groundColId = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=halfExtents, physicsClientId=self.physics_client
        )
        groundVisID = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            rgbaColor=np.random.uniform(0, 1, size=3).tolist() + [0.8],
            specularColor=[0.4, 0.4, 0],
            halfExtents=halfExtents,
            physicsClientId=self.physics_client,
        )
        groundId = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=groundColId,
            baseVisualShapeIndex=groundVisID,
            basePosition=basePosition,
            physicsClientId=self.physics_client,
        )

        return groundId

    def init_obstacle_bodies(self, num_obstacles, initial_obstacles=None):
        """
        初始化障碍物体

        Args:
            num_obstacles: 障碍物数量
            initial_obstacles: 初始障碍物列表（可选）

        Returns:
            障碍物体ID列表
        """
        self.obstacles = initial_obstacles
        self.obstacle_body_ids = []
        for i in range(num_obstacles):
            if initial_obstacles is not None and i < len(initial_obstacles):
                halfExtents, basePosition = initial_obstacles[i]
            else:
                halfExtents = np.array([0.1, 0.1, 0.1])
                basePosition = np.array([0, 0, -10])
            body_id = self.create_voxel(halfExtents, basePosition)
            self.obstacle_body_ids.append(body_id)

        # 初始化球体障碍物（如果启用）
        return self.obstacle_body_ids

    def update_obstacle_poses(self, new_obstacles):
        """
        更新障碍物位置

        Args:
            new_obstacles: 新的障碍物位置列表
        """
        if not hasattr(self, "obstacle_body_ids"):
            raise RuntimeError("请先调用 init_obstacle_bodies() 初始化障碍物")
        for i, (_, basePosition) in enumerate(new_obstacles):
            if i < len(self.obstacle_body_ids):
                p.resetBasePositionAndOrientation(
                    self.obstacle_body_ids[i],
                    basePosition,
                    [0, 0, 0, 1],
                    physicsClientId=self.physics_client,
                )
        # 更新球体障碍物位置（如果启用）
        self.obstacles = new_obstacles

    def randomize_obstacle_poses(
        self,
        workspace_range=(-1.0, 1.0),
        safe_zone_center=(0.0, 0.0, 0.0),
        safe_zone_radius=0.3,
        max_attempts_per_obstacle=100,
    ):
        """
        随机化障碍物位置

        Args:
            workspace_range: 工作空间范围
            safe_zone_center: 安全区域中心
            safe_zone_radius: 安全区域半径
            max_attempts_per_obstacle: 每个障碍物的最大尝试次数

        Returns:
            新的障碍物位置列表
        """
        if not hasattr(self, "obstacles") or self.obstacles is None:
            raise RuntimeError("请先设置 self.obstacles")

        w_min, w_max = workspace_range
        safe_center = np.array(safe_zone_center)
        new_obstacles = []
        for halfExtents, old_position in self.obstacles:
            for _ in range(max_attempts_per_obstacle):
                new_position = np.random.uniform(w_min, w_max, size=3)

                # 确保 z 坐标不小于 0（障碍物底部不低于地面）
                if new_position[2] + halfExtents[2] < 0:
                    new_position[2] = 0

                distance_to_base = np.linalg.norm(new_position - safe_center)
                min_safe_distance = safe_zone_radius + np.max(halfExtents)
                if distance_to_base > min_safe_distance:
                    new_obstacles.append((halfExtents, new_position))
                    break
            else:
                new_obstacles.append((halfExtents, old_position))
        self.update_obstacle_poses(new_obstacles)
        return new_obstacles

    def cleanup_obstacles(self):
        """清理障碍物"""
        if hasattr(self, "obstacle_body_ids"):
            for body_id in self.obstacle_body_ids:
                try:
                    p.removeBody(body_id, physicsClientId=self.physics_client)
                except Exception:
                    pass
            self.obstacle_body_ids.clear()

    @staticmethod
    def is_overlapping(obstacle1, obstacle2):
        """
        检查两个障碍物是否重叠（基于AABB碰撞检测）

        Args:
            obstacle1: (halfExtents, position) 元组
            obstacle2: (halfExtents, position) 元组

        Returns:
            bool: 是否重叠
        """
        half1, pos1 = obstacle1
        half2, pos2 = obstacle2
        pos1, pos2 = np.array(pos1), np.array(pos2)
        half1, half2 = np.array(half1), np.array(half2)

        # 计算AABB边界
        min1 = pos1 - half1
        max1 = pos1 + half1
        min2 = pos2 - half2
        max2 = pos2 + half2

        # 检查是否在所有轴上都不重叠
        return not (
            max1[0] < min2[0]
            or max2[0] < min1[0]
            or max1[1] < min2[1]
            or max2[1] < min1[1]
            or max1[2] < min2[2]
            or max2[2] < min1[2]
        )

    @staticmethod
    def generate_random_obstacles(
        num_obstacles=10,
        workspace_range=(-1.0, 1.0),
        voxel_size_range=(0.05, 0.15),
        safe_zone_center=(0.0, 0.0, 0.0),
        safe_zone_radius=0.3,
    ):
        """
        生成随机障碍物，避开机器人基座附近的安全区域，并确保障碍物之间不重叠

        Args:
            num_obstacles: 障碍物数量
            workspace_range: 工作空间范围 (min, max)
            voxel_size_range: 体素尺寸范围 (min, max)
            safe_zone_center: 安全区域中心
            safe_zone_radius: 安全区域半径

        Returns:
            障碍物列表，每个元素为 (halfExtents, basePosition) 元组
        """
        obstacles = []
        w_min, w_max = workspace_range
        safe_center = np.array(safe_zone_center)

        for _ in range(num_obstacles):
            max_attempts = 100
            for _ in range(max_attempts):
                half_size = np.random.uniform(
                    voxel_size_range[0], voxel_size_range[1], size=3
                )
                position = np.random.uniform(w_min, w_max, size=3)

                # 确保 z 坐标不小于 0（障碍物顶部不低于地面）
                if position[2] + half_size[2] < 0:
                    continue

                distance_to_base = np.linalg.norm(position - safe_center)
                min_safe_distance = safe_zone_radius + np.max(half_size)

                if distance_to_base <= min_safe_distance:
                    continue
                new_obstacle = (half_size, position)
                obstacles.append(new_obstacle)

        return obstacles

    def load_and_init_obstacles_from_data(self, obstacles):
        """
        从障碍物数据直接加载并初始化障碍物到PyBullet环境中

        Args:
            obstacles: 障碍物列表，每个元素为 (halfExtents, basePosition) 元组

        Returns:
            创建的障碍物body ID列表
        """
        self.obstacles = obstacles

        # 清理现有障碍物
        self.cleanup_obstacles()

        self.obstacle_body_ids = []

        for halfExtents, basePosition in obstacles:
            body_id = self.create_voxel(halfExtents, basePosition)
            self.obstacle_body_ids.append(body_id)

        return self.obstacle_body_ids
