import numpy as np
import pybullet as p
import torch
import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from sphere_as.robot_sphere_analyzer import RobotSphereAnalyzer
from robot_as.robot_method import RobotEnv


class SphereEnv:
    """球体碰撞检测环境类，负责球体模型相关的初始化和碰撞检测"""

    def __init__(
        self,
        robot_file="/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf",
        robot_name="franka",
        SPH_GUI=None,
    ):
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
        self.sphere_analyzer = RobotSphereAnalyzer(robot_name, device="cuda:0")

        # 初始化机器人环境（用于 link 邻接检查）
        self.robot_env = RobotEnv(robot_file)
        self.robot_name = robot_name
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

    def _get_sphere_data_for_collision(self, state):
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

    def _check_sphere_pair_collision(self, pos1, r1, pos2, r2):
        """
        检查两个球体是否碰撞

        Args:
            pos1: 球体1位置 [x, y, z]
            r1: 球体1半径
            pos2: 球体2位置 [x, y, z]
            r2: 球体2半径

        Returns:
            bool: 是否碰撞
        """
        distance = np.linalg.norm(pos1 - pos2)
        return distance <= (r1 + r2)

    def _check_self_collision_geometric(self, state):
        """
        基于几何算法检查机器人自碰撞

        Args:
            state: 关节配置

        Returns:
            tuple: (any_collision, sphere_colls) - 是否有自碰撞, 各球体自碰撞状态列表[0/1]
        """
        positions, radii, link_ids = self._get_sphere_data_for_collision(state)
        n_spheres = len(positions)

        # 初始化每个球体的碰撞状态，默认无碰撞
        sphere_colls = [1] * n_spheres
        any_collision = False

        for i in range(n_spheres):
            for j in range(i + 1, n_spheres):
                # 跳过同一 link 的球体
                if link_ids[i] == link_ids[j]:
                    continue
                # 跳过相邻 link 的球体
                if self.robot_env._are_links_adjacent(link_ids[i], link_ids[j]):
                    continue
                # 检查碰撞
                if self._check_sphere_pair_collision(
                    positions[i], radii[i], positions[j], radii[j]
                ):
                    sphere_colls[i] = 0
                    sphere_colls[j] = 0
                    any_collision = True

        return any_collision, sphere_colls

    def _check_sphere_obstacle_collision(self):
        """
        检查球体与障碍物的碰撞

        Returns:
            tuple: (是否有碰撞, 各球体碰撞状态列表[0/1])
        """
        if not self.sphere_bodies or not self.sphere_obstacle_ids:
            return False, [1] * len(self.sphere_bodies)

        p.performCollisionDetection(physicsClientId=self.physics_client)

        # 获取所有接触点
        contacts = p.getContactPoints(physicsClientId=self.physics_client)

        # 初始化碰撞状态，默认无碰撞
        sphere_colls = [1] * len(self.sphere_bodies)
        any_collision = False

        # 检查接触点中是否有球体与障碍物的碰撞
        for contact in contacts:
            bodyA = contact[1]
            bodyB = contact[2]

            # 检查是否是球体与障碍物的碰撞
            if bodyA in self.sphere_bodies and bodyB in self.sphere_obstacle_ids:
                idx = self.sphere_bodies.index(bodyA)
                sphere_colls[idx] = 0
                any_collision = True
            elif bodyB in self.sphere_bodies and bodyA in self.sphere_obstacle_ids:
                idx = self.sphere_bodies.index(bodyB)
                sphere_colls[idx] = 0
                any_collision = True

        return any_collision, sphere_colls

    def _check_sphere_collision(self, state):
        """
        检查球体与障碍物以及球体自碰撞

        Args:
            state: 关节配置

        Returns:
            tuple: (是否有碰撞, 各球体碰撞状态列表[0/1])
        """
        # # 检查自碰撞
        # self_any, self_colls = self._check_self_collision_geometric(state)

        # # 检查与障碍物的碰撞
        # obs_any, obs_colls = self._check_sphere_obstacle_collision()

        # # 合并结果
        # any_collision = self_any or obs_any
        # sphere_colls = [
        #     min(self_colls[i], obs_colls[i]) for i in range(len(self_colls))
        # ]

        return self._check_sphere_obstacle_collision()

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
