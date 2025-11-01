import numpy as np
import pybullet as p
import torch
import sys
import os
import pickle
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from sphere_as.robot_sphere_analyzer import RobotSphereAnalyzer
from robot_as.robot_method import RobotEnv


class SphereEnv:
    """球体碰撞检测环境类，负责球体模型相关的初始化和碰撞检测"""

    def __init__(
        self,
        robot_env: RobotEnv,
        robot_name: Optional[str] = None,
        SPH_GUI=None,
    ):
        """
        初始化球体环境

        Args:
            robot_env: 复用的机器人环境实例
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
        resolved_name = robot_name or getattr(robot_env, "robot_name", None)
        if resolved_name is None:
            raise ValueError("SphereEnv requires a valid robot name.")

        self.sphere_analyzer = RobotSphereAnalyzer(resolved_name, device="cuda:0")

        # 复用传入的机器人环境（用于 link 邻接检查）
        self.robot_env = robot_env
        self.robot_name = resolved_name
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

        # 使用默认配置获取球体和link信息
        joint_config = torch.tensor(
            [0.0] * 7, dtype=torch.float32, device=torch.device("cuda:0")
        ).unsqueeze(0)
        spheres, link_ids = self.sphere_analyzer.get_world_spheres_with_links(
            joint_config
        )

        for (x, y, z, radius), link_id in zip(spheres, link_ids):
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

        # 禁用同一link和相邻link的球体之间的碰撞检测
        for i in range(len(link_ids)):
            for j in range(i + 1, len(link_ids)):
                if link_ids[i] == link_ids[j] or self.robot_env._are_links_adjacent(
                    link_ids[i], link_ids[j]
                ):
                    p.setCollisionFilterPair(
                        bodyUniqueIdA=self.sphere_bodies[i],
                        bodyUniqueIdB=self.sphere_bodies[j],
                        linkIndexA=-1,
                        linkIndexB=-1,
                        enableCollision=0,
                        physicsClientId=self.physics_client,
                    )

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

    def _get_sphere_data(self, state):
        """
        获取关节配置下的球体数据（位置、半径、link ID）

        Args:
            state: 关节配置

        Returns:
            tuple: (positions, radii, link_ids)
        """
        joint_config = torch.tensor(
            state, dtype=torch.float32, device=torch.device("cuda:0")
        ).unsqueeze(0)
        spheres, link_ids = self.sphere_analyzer.get_world_spheres_with_links(
            joint_config
        )
        positions = spheres[:, :3]  # numpy array
        radii = spheres[:, 3]  # numpy array
        return positions, radii, link_ids

    def _check_sphere_collision(self, state):
        """
        检查球体与障碍物以及球体自碰撞（通过PyBullet碰撞检测）

        Args:
            state: 关节配置

        Returns:
            tuple: (是否有碰撞, 各球体碰撞状态列表[0/1])
        """
        self._update_sphere_positions(state)
        p.performCollisionDetection(physicsClientId=self.physics_client)

        # 初始化碰撞状态，默认无碰撞
        sphere_colls = [1] * len(self.sphere_bodies)
        any_collision = False

        for i, sphere_body in enumerate(self.sphere_bodies):
            contacts = p.getContactPoints(
                bodyA=sphere_body, physicsClientId=self.physics_client
            )
            if len(contacts) > 0:
                sphere_colls[i] = 0
                any_collision = True

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
        collision, colls = self._check_sphere_collision(state)
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

    def save_collision_data(self, output_file):
        """
        保存球体碰撞数据到文件

        Args:
            output_file: 输出文件路径
        """
        with open(output_file, "wb") as f:
            pickle.dump((self.link_data, self.link_coll_data), f)
        print(f"保存球体碰撞数据到: {output_file}")
