import torch
import numpy as np
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.types.base import TensorDeviceType
from curobo.geom.types import WorldConfig, Cuboid


class SphereEnvCurobo:
    """基于 CuRobo 的球体碰撞检测环境类"""

    def __init__(self, robot_name="franka", world_config=None):
        """
        初始化 CuRobo 球体环境

        Args:
            robot_name: 机器人名称
            world_config: 可选的世界配置，如果为None则使用空配置
        """
        self.robot_name = robot_name
        self.tensor_args = TensorDeviceType()

        # 初始化 CuRobo RobotWorld
        if world_config is None:
            world_config = WorldConfig(cuboid=[])
        self.robot_world_config = RobotWorldConfig.load_from_config(
            f"{self.robot_name}.yml", world_config, collision_activation_distance=0.0
        )
        self.robot_world = RobotWorld(self.robot_world_config)

        # 初始化球体分析器（用于获取球体和 link 信息）
        self.sphere_analyzer = RobotSphereAnalyzer(robot_name, device="cuda:0")

        # 数据收集
        self.link_data = []
        self.link_coll_data = []

    def init_obstacle_bodies(self, num_obstacles, initial_obstacles=None):
        """
        初始化障碍物

        Args:
            num_obstacles: 障碍物数量
            initial_obstacles: 初始障碍物列表 [(half_extents, base_position), ...]
        """
        cuboids = []
        for i in range(num_obstacles):
            if initial_obstacles and i < len(initial_obstacles):
                half_extents, base_position = initial_obstacles[i]
            else:
                half_extents = [0.1, 0.1, 0.1]
                base_position = [0, 0, -10]

            # 创建立方体障碍物
            dims = [2 * h for h in half_extents]  # 转换为全尺寸
            pose = [float(p) for p in base_position] + [
                1.0,
                0.0,
                0.0,
                0.0,
            ]  # 添加四元数
            cuboid = Cuboid(
                name=f"obstacle_{i}",
                pose=pose,
                dims=dims,
                tensor_args=self.tensor_args,
            )
            cuboids.append(cuboid)

        # 更新世界配置
        world_config = WorldConfig(cuboid=cuboids)
        self.robot_world.update_world(world_config)

    def cleanup_obstacles(self):
        """清理障碍物"""
        world_config = WorldConfig(cuboid=[])
        self.robot_world.update_world(world_config)

    def _check_world_collision(self, joint_config):
        """
        检查机器人与障碍物的碰撞

        Args:
            joint_config: 关节配置张量

        Returns:
            bool: 是否有世界碰撞
        """
        d_world, _ = self.robot_world.get_world_self_collision_distance_from_joints(
            joint_config
        )
        return (d_world <= 0).item()

    def _get_spheres_with_links(self, joint_config):
        """
        获取球体及其对应的 link ID

        Args:
            joint_config: 关节配置张量

        Returns:
            tuple: (spheres, link_ids) - spheres: [n_spheres, 4], link_ids: [n_spheres]
        """
        spheres, link_ids = self.sphere_analyzer.get_world_spheres_with_links(
            joint_config
        )
        return spheres, link_ids

    def _check_pairwise_collision(self, sphere1, sphere2):
        """
        检查两个球体是否碰撞

        Args:
            sphere1: [x, y, z, radius]
            sphere2: [x, y, z, radius]

        Returns:
            bool: 是否碰撞
        """
        pos1 = np.array(sphere1[:3])
        pos2 = np.array(sphere2[:3])
        distance = np.linalg.norm(pos1 - pos2)
        return distance <= (sphere1[3] + sphere2[3])

    def _is_adjacent_link(self, link_id1, link_id2):
        """
        检查两个 link 是否相邻（简化实现：相邻 link ID 差值为1）

        Args:
            link_id1: link ID 1
            link_id2: link ID 2

        Returns:
            bool: 是否相邻
        """
        return abs(link_id1 - link_id2) <= 1  # 简化：认为相邻 link ID 相差 <=1

    def _check_self_collision_basic(self, joint_config):
        """
        基于基本算法的自碰撞检测

        Args:
            joint_config: 关节配置张量

        Returns:
            bool: 是否有自碰撞
        """
        spheres, link_ids = self._get_spheres_with_links(joint_config)

        # 逐个遍历所有球体
        for i, sphere1 in enumerate(spheres):
            link1 = link_ids[i]

            # 检查与不在相邻 link 的球体
            for j, sphere2 in enumerate(spheres):
                if i == j:
                    continue  # 跳过自己

                link2 = link_ids[j]
                if self._is_adjacent_link(link1, link2):
                    continue  # 跳过相邻 link 的球体

                # 检查碰撞
                if self._check_pairwise_collision(sphere1, sphere2):
                    return True  # 发现碰撞

        return False  # 无碰撞

    def _check_self_collision(self, joint_config):
        """
        检查机器人自碰撞（使用基本算法）

        Args:
            joint_config: 关节配置张量

        Returns:
            bool: 是否有自碰撞
        """
        return self._check_self_collision_basic(joint_config)

    def _check_sphere_collision(self, joint_config):
        """
        检查球体碰撞（世界碰撞 + 自碰撞）

        Args:
            joint_config: 关节配置张量

        Returns:
            tuple: (是否有碰撞, 球体碰撞状态列表)
        """
        world_collision = self._check_world_collision(joint_config)
        self_collision = self._check_self_collision(joint_config)

        # 合并碰撞结果
        any_collision = world_collision or self_collision

        # 简化：返回整体碰撞状态，球体级状态设为相同（假设61个球体）
        sphere_colls = [0 if any_collision else 1] * 61

        return any_collision, sphere_colls

    def get_sphere_collision_data(self, state):
        """
        获取球体碰撞数据

        Args:
            state: 关节配置

        Returns:
            tuple: (collision, coords, colls)
        """
        coords = []  # 不返回球体坐标
        joint_config = torch.tensor(
            state, dtype=torch.float32, device=self.tensor_args.device
        ).unsqueeze(0)
        collision, colls = self._check_sphere_collision(joint_config)
        return collision, coords, colls

    def store_sphere_data(self, coords, colls, is_edge=True):
        """
        存储球体数据

        Args:
            coords: 坐标数据（这里为空）
            colls: 碰撞标签
            is_edge: 是否为边数据
        """
        if not colls:
            return

        if is_edge:
            self.link_data.append(coords)
            self.link_coll_data.append(colls)
        else:
            self.link_data.append([coords])
            self.link_coll_data.append([colls])

    def close(self):
        """关闭环境"""
        pass  # CuRobo 无需显式关闭
