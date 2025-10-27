import numpy as np
import pybullet as p
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from sphere_as.robot_sphere_analyzer import RobotSphereAnalyzer


class SphereEnv:
    """球体碰撞检测环境类，负责球体模型相关的初始化和碰撞检测"""

    def __init__(self, robot_file="kuka_iiwa/model_0.urdf", SPH_GUI=None):
        """
        初始化球体环境

        Args:
            robot_file: 机器人URDF文件路径（用于球体分析器）
            SPH_GUI: 是否启用GUI模式
        """
        # 连接球体物理客户端
        if SPH_GUI:
            self.physics_client = p.connect(
                p.GUI,
                options="--background_color_blue=1.0 --background_color_green=1.0 --background_color_red=1.0",
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)

        # 初始化球体分析器
        self.sphere_analyzer = RobotSphereAnalyzer("franka", device="cuda:0")
        self.sphere_bodies = []
        self.sphere_obstacle_ids = []

        # 数据收集
        self.link_data = []
        self.link_coll_data = []

    def close(self):
        """关闭球体物理客户端"""
        self._cleanup_sphere_bodies()
        p.disconnect(physicsClientId=self.physics_client)

    def init_obstacle_bodies(self, num_obstacles, initial_obstacles=None):
        """初始化球体障碍物"""
        self.sphere_obstacle_ids = []
        for i in range(num_obstacles):
            if initial_obstacles is not None and i < len(initial_obstacles):
                halfExtents, basePosition = initial_obstacles[i]
            else:
                halfExtents = np.array([0.1, 0.1, 0.1])
                basePosition = np.array([0, 0, -10])

            sphere_colId = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=halfExtents,
                physicsClientId=self.physics_client,
            )
            sphere_obstId = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=sphere_colId,
                basePosition=basePosition,
                physicsClientId=self.physics_client,
            )
            self.sphere_obstacle_ids.append(sphere_obstId)

    def update_obstacle_poses(self, new_obstacles):
        """更新球体障碍物位置"""
        for i, (_, basePosition) in enumerate(new_obstacles):
            if i < len(self.sphere_obstacle_ids):
                p.resetBasePositionAndOrientation(
                    self.sphere_obstacle_ids[i],
                    basePosition,
                    [0, 0, 0, 1],
                    physicsClientId=self.physics_client,
                )

    def cleanup_obstacles(self):
        """清理球体障碍物"""
        for body_id in self.sphere_obstacle_ids:
            try:
                p.removeBody(body_id, physicsClientId=self.physics_client)
            except Exception:
                pass
        self.sphere_obstacle_ids.clear()

    def _create_sphere_bodies(self):
        """创建球体身体"""
        if self.sphere_bodies:
            return

        # 使用当前有效关节配置获取球体位置
        # 注意：这里需要从外部传入当前配置，因为SphereEnv不管理机器人
        # 暂时使用默认配置，实际使用时需要传入
        current_config = [0.0] * 7  # 假设7DOF，实际需要传入
        joint_config = torch.tensor(
            current_config, dtype=torch.float32, device=torch.device("cuda:0")
        ).unsqueeze(0)
        world_spheres = self.sphere_analyzer.get_world_spheres(joint_config)

        for x, y, z, radius in world_spheres:
            sphere_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=float(radius),
                physicsClientId=self.physics_client,
            )
            sphere_body = p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=sphere_shape,
                basePosition=[float(x), float(y), float(z)],
                physicsClientId=self.physics_client,
            )
            self.sphere_bodies.append(sphere_body)

    def _update_sphere_positions(self, state):
        """
        更新球体位置到当前关节配置并返回球体坐标
        Returns:
            list: 所有球体的中心坐标列表 [[x, y, z, r], ...]
        """
        if not self.sphere_bodies:
            self._create_sphere_bodies()

        joint_config = torch.tensor(
            state, dtype=torch.float32, device=torch.device("cuda:0")
        ).unsqueeze(0)
        world_spheres = self.sphere_analyzer.get_world_spheres(joint_config)

        sphere_coords = []
        for sphere_body, (x, y, z, radius) in zip(self.sphere_bodies, world_spheres):
            p.resetBasePositionAndOrientation(
                sphere_body,
                [float(x), float(y), float(z)],
                [0, 0, 0, 1],
                physicsClientId=self.physics_client,
            )
            sphere_coords.append([float(x), float(y), float(z), float(radius)])

        return sphere_coords

    def _check_sphere_collision(self):
        """
        检查球体与障碍物的碰撞

        Returns:
            tuple: (是否有碰撞, 各球体碰撞状态列表[0/1])
        """
        if not self.sphere_bodies or not self.sphere_obstacle_ids:
            return False, [1] * len(self.sphere_bodies)

        p.performCollisionDetection(physicsClientId=self.physics_client)

        sphere_colls = []
        any_collision = False

        for sphere_body in self.sphere_bodies:
            is_colliding = False
            for obstacle_id in self.sphere_obstacle_ids:
                contacts = p.getContactPoints(
                    bodyA=sphere_body,
                    bodyB=obstacle_id,
                    physicsClientId=self.physics_client,
                )
                if len(contacts) > 0:
                    is_colliding = True
                    any_collision = True
                    break
            sphere_colls.append(0 if is_colliding else 1)

        return any_collision, sphere_colls

    def _cleanup_sphere_bodies(self):
        """清理球体实体"""
        for sphere_body in self.sphere_bodies:
            p.removeBody(sphere_body, physicsClientId=self.physics_client)
        self.sphere_bodies.clear()

    def get_sphere_collision_data(self, state):
        """
        获取球体碰撞数据

        Args:
            state: 关节配置状态

        Returns:
            tuple: (collision, coords, colls)
        """
        coords = self._update_sphere_positions(state)
        collision, colls = self._check_sphere_collision()
        return collision, coords, colls

    def store_sphere_data(self, coords, colls, is_edge=True):
        """
        存储球体数据

        Args:
            coords: 坐标数据
            colls: 碰撞标签
            is_edge: 是否为边数据
        """
        if not coords:
            return

        if is_edge:
            self.link_data.append(coords)
            self.link_coll_data.append(colls)
        else:
            self.link_data.append([coords])
            self.link_coll_data.append([colls])
