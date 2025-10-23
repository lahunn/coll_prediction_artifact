import numpy as np
import pybullet as p
import pickle
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from sphere_as.robot_sphere_analyzer import RobotSphereAnalyzer


class CollisionEnv:
    """完整的碰撞检测环境类,包含PyBullet初始化、机器人加载和碰撞检测功能"""

    RRT_EPS = 0.25

    def __init__(
        self,
        GUI=False,
        robot_file="kuka_iiwa/model_0.urdf",
        z_offset=0.0,
        config_output_file=None,
    ):
        """
        初始化碰撞检测环境

        Args:
            GUI: 是否启用GUI模式
            robot_file: 机器人URDF文件路径
            z_offset: Z轴偏移量
            config_output_file: 配置输出文件路径（可选）
        """
        self.robot_file = robot_file
        self.z_offset = z_offset
        self.obstacles = []
        self.obstacle_body_ids = []

        # config_output_file 相关逻辑
        self.config_output_file = config_output_file
        self.config_list = []

        # 碰撞数据收集
        self.obb_link_data = []
        self.obb_link_coll_data = []
        self.sphere_link_data = []
        self.sphere_link_coll_data = []

        # 当前边的临时数据
        self.current_edge_obb_coords = []
        self.current_edge_obb_colls = []
        self.current_edge_sphere_coords = []
        self.current_edge_sphere_colls = []

        # 连接PyBullet (主仿真器,用于link碰撞检测)
        if GUI:
            self.physics_client = p.connect(
                p.GUI,
                options="--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0",
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        # 连接第二个PyBullet实例 (用于sphere碰撞检测)
        self.sphere_physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self.sphere_physics_client)

        # 加载机器人 (主仿真器)
        self.robotId = p.loadURDF(
            robot_file,
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=True,
            physicsClientId=self.physics_client,
        )

        # 初始化球体分析器
        self.sphere_analyzer = RobotSphereAnalyzer("franka", device="cuda:0")
        self.sphere_bodies = []
        self.sphere_obstacle_ids = []

        # 找到有碰撞几何体的link
        self.valid_collision_links = self._find_valid_collision_links()

        # 获取机器人配置信息
        self.config_dim = p.getNumJoints(
            self.robotId, physicsClientId=self.physics_client
        )
        self.pose_range = [
            (
                p.getJointInfo(
                    self.robotId, jointId, physicsClientId=self.physics_client
                )[8],
                p.getJointInfo(
                    self.robotId, jointId, physicsClientId=self.physics_client
                )[9],
            )
            for jointId in range(self.config_dim)
        ]
        # 预计算正确的上下限（处理上下限可能颠倒的情况）
        self.lower_bounds = np.array([min(r[0], r[1]) for r in self.pose_range])
        self.upper_bounds = np.array([max(r[0], r[1]) for r in self.pose_range])
        self.bound = np.array(self.pose_range).T.reshape(-1)
        self.robotEndEffectorIndex = self.config_dim - 1

        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)

        # 规划相关属性
        self.init_state = [0.0] * self.config_dim
        self.goal_state = [0.0] * self.config_dim

    def close(self):
        """关闭配置输出文件句柄和PyBullet连接"""
        self._cleanup_sphere_bodies()
        p.disconnect(physicsClientId=self.physics_client)
        p.disconnect(physicsClientId=self.sphere_physics_client)

    def uniform_sample(self, n=1):
        """
        在配置空间的关节限位范围内均匀随机采样

        Args:
            n: 采样数量

        Returns:
            采样的配置，n=1时返回一维数组，否则返回二维数组
        """
        sample = np.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(n, self.config_dim),
        )
        return sample.reshape(-1) if n == 1 else sample

    def distance(self, from_state, to_state):
        """
        计算两个配置之间的欧几里得距离
        """
        to_state = np.maximum(to_state, self.lower_bounds)
        to_state = np.minimum(to_state, self.upper_bounds)
        diff = np.abs(to_state - from_state)
        return np.sqrt(np.sum(diff**2, axis=-1))

    def set_config(self, c, robotId=None):
        """
        设置机器人的关节配置

        Args:
            c: 关节配置数组
            robotId: 机器人ID（可选，默认使用self.robotId）
        """
        if robotId is None:
            robotId = self.robotId
        for i in range(p.getNumJoints(robotId, physicsClientId=self.physics_client)):
            p.resetJointState(robotId, i, c[i], physicsClientId=self.physics_client)

    def create_voxel(self, halfExtents, basePosition):
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

        # 同时在球体仿真器中创建障碍物
        sphere_colId = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=halfExtents,
            physicsClientId=self.sphere_physics_client,
        )
        sphere_obstId = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=sphere_colId,
            basePosition=basePosition,
            physicsClientId=self.sphere_physics_client,
        )
        self.sphere_obstacle_ids.append(sphere_obstId)

        return groundId

    def init_obstacle_bodies(self, num_obstacles, initial_obstacles=None):
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
        return self.obstacle_body_ids

    def update_obstacle_poses(self, new_obstacles):
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
            if i < len(self.sphere_obstacle_ids):
                p.resetBasePositionAndOrientation(
                    self.sphere_obstacle_ids[i],
                    basePosition,
                    [0, 0, 0, 1],
                    physicsClientId=self.sphere_physics_client,
                )
        self.obstacles = new_obstacles

    def randomize_obstacle_poses(
        self,
        workspace_range=(-1.0, 1.0),
        safe_zone_center=(0.0, 0.0, 0.0),
        safe_zone_radius=0.3,
        max_attempts_per_obstacle=100,
    ):
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
        if hasattr(self, "obstacle_body_ids"):
            for body_id in self.obstacle_body_ids:
                try:
                    p.removeBody(body_id, physicsClientId=self.physics_client)
                except Exception:
                    pass
            self.obstacle_body_ids.clear()

        if hasattr(self, "sphere_obstacle_ids"):
            for body_id in self.sphere_obstacle_ids:
                try:
                    p.removeBody(body_id, physicsClientId=self.sphere_physics_client)
                except Exception:
                    pass
            self.sphere_obstacle_ids.clear()

    def _valid_state(self, state):
        """检查配置是否在关节限位范围内"""
        return (state >= self.lower_bounds).all() and (state <= self.upper_bounds).all()

    def _find_valid_collision_links(self):
        """找到有碰撞几何体的link"""
        if self.robotId is None:
            return []

        valid_links = []
        num_joints = p.getNumJoints(self.robotId, physicsClientId=self.physics_client)

        # 检查base link
        collision_data = p.getCollisionShapeData(
            self.robotId, -1, physicsClientId=self.physics_client
        )
        if collision_data:
            valid_links.append(-1)

        # 检查其他link
        for i in range(num_joints):
            collision_data = p.getCollisionShapeData(
                self.robotId, i, physicsClientId=self.physics_client
            )
            if collision_data:
                valid_links.append(i)

        return valid_links

    def _get_link_pose(self, link_idx):
        """获取link的世界位姿"""
        if link_idx == -1:
            pos, orn = p.getBasePositionAndOrientation(
                self.robotId, physicsClientId=self.physics_client
            )
        else:
            link_state = p.getLinkState(
                self.robotId, link_idx, physicsClientId=self.physics_client
            )
            pos, orn = link_state[0], link_state[1]
        return list(pos) + list(orn)

    def _get_link_collisions(self):
        """获取各个valid link的碰撞结果"""
        any_coll = False
        link_colls = []
        p.performCollisionDetection(physicsClientId=self.physics_client)
        for link_idx in self.valid_collision_links:
            is_colliding = False
            for obstacle_id in self.obstacle_body_ids:
                if p.getContactPoints(
                    self.robotId,
                    obstacle_id,
                    linkIndexA=link_idx,
                    physicsClientId=self.physics_client,
                ):
                    is_colliding = True
                    any_coll = True
                    break
            link_colls.append(0 if is_colliding else 1)
        return any_coll, link_colls

    def _create_sphere_bodies(self):
        """创建用于碰撞检测的球体实体"""
        if self.sphere_bodies:
            return

        # 修复：使用当前关节配置获取球体位置
        num_joints = p.getNumJoints(self.robotId, physicsClientId=self.physics_client)
        current_angles = [
            p.getJointState(self.robotId, i, physicsClientId=self.physics_client)[0]
            for i in range(num_joints)
        ]
        joint_config = torch.tensor(
            current_angles, dtype=torch.float32, device=torch.device("cuda:0")
        ).unsqueeze(0)
        world_spheres = self.sphere_analyzer.get_world_spheres(joint_config)

        for x, y, z, radius in world_spheres:
            sphere_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=float(radius),
                physicsClientId=self.sphere_physics_client,
            )
            sphere_body = p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=sphere_shape,
                basePosition=[float(x), float(y), float(z)],
                physicsClientId=self.sphere_physics_client,
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
                physicsClientId=self.sphere_physics_client,
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

        p.performCollisionDetection(physicsClientId=self.sphere_physics_client)

        sphere_colls = []
        any_collision = False

        for sphere_body in self.sphere_bodies:
            is_colliding = False
            for obstacle_id in self.sphere_obstacle_ids:
                contacts = p.getContactPoints(
                    bodyA=sphere_body,
                    bodyB=obstacle_id,
                    physicsClientId=self.sphere_physics_client,
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
            p.removeBody(sphere_body, physicsClientId=self.sphere_physics_client)

        self.sphere_bodies.clear()

    def _point_in_free_space(self, state):
        """
        检查单个配置是否无碰撞并收集碰撞数据

        Returns:
            tuple: (is_free, link_coords, link_colls, sphere_coords, sphere_colls)
                link_coords: [[x,y,z,qx,qy,qz,qw], ...] 每个valid link的位姿
                link_colls: [0/1, ...] 每个valid link的碰撞标签
                sphere_coords: [[x,y,z,r], ...] 每个sphere的位置+半径
                sphere_colls: [0/1, ...] 每个sphere的碰撞标签
        """
        if not self._valid_state(state):
            return False, [], [], [], []

        # 设置机器人配置
        for i in range(
            p.getNumJoints(self.robotId, physicsClientId=self.physics_client)
        ):
            p.resetJointState(
                self.robotId, i, state[i], physicsClientId=self.physics_client
            )

        # 收集Link数据 (仅valid collision links)
        link_coords = []
        link_collision, link_colls = self._get_link_collisions()
        for link_idx in self.valid_collision_links:
            pose = self._get_link_pose(link_idx)
            link_coords.append(pose)

        # 收集Sphere数据
        sphere_coords = self._update_sphere_positions(state)
        sphere_collision, sphere_colls = self._check_sphere_collision()

        # 判断是否无碰撞 (两者都认为碰撞才返回碰撞)
        is_free = not (sphere_collision and link_collision)

        return is_free, link_coords, link_colls, sphere_coords, sphere_colls

    def get_collision_ratio(self):
        """计算碰撞率（obb_link_coll_data和sphere_link_coll_data中0的占比）

        Returns:
            tuple: (obb_collision_ratio, sphere_collision_ratio)
        """
        # 计算 OBB 碰撞率
        obb_ratio = 0.0
        if self.obb_link_coll_data:
            total_count = 0
            collision_count = 0
            for edge_colls in self.obb_link_coll_data:
                for pose_colls in edge_colls:
                    for coll_value in pose_colls:
                        total_count += 1
                        if coll_value == 0:
                            collision_count += 1
            obb_ratio = collision_count / total_count if total_count > 0 else 0.0

        # 计算 Sphere 碰撞率
        sphere_ratio = 0.0
        if self.sphere_link_coll_data:
            total_count = 0
            collision_count = 0
            for edge_colls in self.sphere_link_coll_data:
                for pose_colls in edge_colls:
                    for coll_value in pose_colls:
                        total_count += 1
                        if coll_value == 0:
                            collision_count += 1
            sphere_ratio = collision_count / total_count if total_count > 0 else 0.0

        return obb_ratio, sphere_ratio

    def save_collision_data(self, link_output_file, sphere_output_file):
        """保存碰撞数据到文件"""
        with open(link_output_file, "wb") as f:
            pickle.dump((self.obb_link_data, self.obb_link_coll_data), f)

        with open(sphere_output_file, "wb") as f:
            pickle.dump((self.sphere_link_data, self.sphere_link_coll_data), f)

        obb_ratio, sphere_ratio = self.get_collision_ratio()
        print(
            f"✓ 保存Link数据: {len(self.obb_link_data)} 条边, OBB碰撞率: {obb_ratio:.4f}"
        )
        print(
            f"✓ 保存Sphere数据: {len(self.sphere_link_data)} 条边, Sphere碰撞率: {sphere_ratio:.4f}"
        )

    def _state_fp(self, state):
        """检查单个状态并收集数据 (作为单条边)"""
        is_free, link_coords, link_colls, sphere_coords, sphere_colls = (
            self._point_in_free_space(state)
        )

        # 单点作为一条边 - 数据结构: [edge][pose][link][coord]
        if link_coords:
            self.obb_link_data.append([link_coords])
            self.obb_link_coll_data.append([link_colls])
            self.sphere_link_data.append([sphere_coords])
            self.sphere_link_coll_data.append([sphere_colls])

        edge_configs = [state.copy()]
        self.config_list.append(np.array(edge_configs))

        return is_free

    def _edge_fp(self, state, new_state, RRT_EPS=0.25):
        """检查边并收集数据"""
        """每次都完成一个edge中所有state的检查"""
        edge_free = True
        assert state.size == new_state.size
        if not self._state_fp(state) or not self._state_fp(new_state):
            return False

        disp = new_state - state
        d = np.linalg.norm(disp)
        K = int(d / RRT_EPS)

        edge_configs = [state.copy()]
        edge_link_coords = []  # [pose][link][coord]
        edge_link_colls = []  # [pose][link]
        edge_sphere_coords = []
        edge_sphere_colls = []

        for k in range(K):
            c = state + k * 1.0 / K * disp
            edge_configs.append(c.copy())

            is_free, link_coords, link_colls, sphere_coords, sphere_colls = (
                self._point_in_free_space(c)
            )

            if not is_free:
                edge_free = False
                # 保存当前边数据
                # if link_coords:
                #     edge_link_coords.append(link_coords)
                #     edge_link_colls.append(link_colls)
                #     edge_sphere_coords.append(sphere_coords)
                #     edge_sphere_colls.append(sphere_colls)

                #     self.obb_link_data.append(edge_link_coords)
                #     self.obb_link_coll_data.append(edge_link_colls)
                #     self.sphere_link_data.append(edge_sphere_coords)
                #     self.sphere_link_coll_data.append(edge_sphere_colls)

                # self.config_list.append(np.array(edge_configs))
                # return False

            # 收集数据
            if link_coords:
                edge_link_coords.append(link_coords)
                edge_link_colls.append(link_colls)
                edge_sphere_coords.append(sphere_coords)
                edge_sphere_colls.append(sphere_colls)

        edge_configs.append(new_state.copy())

        # 保存整条边数据
        if edge_link_coords:
            self.obb_link_data.append(edge_link_coords)
            self.obb_link_coll_data.append(edge_link_colls)
            self.sphere_link_data.append(edge_sphere_coords)
            self.sphere_link_coll_data.append(edge_sphere_colls)

        self.config_list.append(np.array(edge_configs))
        return edge_free
