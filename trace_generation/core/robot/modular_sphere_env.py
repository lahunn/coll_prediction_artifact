#!/usr/bin/env python3
"""
模块化球体碰撞检测环境类

仿照modular_env.py的结构，调用sphere_detector.py中的SphereEnvGeometric实现具体功能
提供统一的接口，支持球体碰撞检测和周期成本跟踪
"""

from trace_generation.utils.problem import ProblemManager
from trace_generation.core.scene.obstacle_manager import ObstacleManager
from trace_generation.core.robot.environment import RobotEnv
from trace_generation.core.collision.sphere_detector import SphereEnvGeometric
from trace_generation.utils.planning_utils import distance


class ModularSphereEnv:
    """
    模块化球体环境类，组合各个组件提供统一的接口

    主要组件:
    - problem_manager: 问题管理器
    - obstacle_manager: 障碍物管理器
    - robot_env: 机器人环境
    - sphere_env: 球体碰撞检测环境
    """

    def __init__(
        self,
        robot_name,
        map_file=None,
        GUI=False,
        enable_self_collision=False,
        return_cycles=False,
    ):
        """
        初始化模块化球体环境

        Args:
            robot_name: 机器人名称（例如 'franka', 'iiwa'）
            map_file: 问题数据集文件路径
            GUI: 是否启用GUI模式（仅用于兼容性，球体环境无可视化）
            enable_self_collision: 是否启用自碰撞检测（默认为True）
            return_cycles: 是否返回硬件周期成本（默认为False）
        """
        # 初始化各个组件
        self.problem_manager = ProblemManager(map_file)
        self.robot_env = RobotEnv(
            robot_name, OBB_GUI=GUI, enable_self_collision=enable_self_collision
        )
        self.sphere_env = SphereEnvGeometric(
            self.robot_env, robot_name=robot_name, return_cycles=return_cycles
        )
        self.obstacle_manager = ObstacleManager(
            physics_client=self.robot_env.physics_client
        )

        # 加载障碍物到球体环境
        self.sphere_env.load_obstacles(self.obstacle_manager.obstacles)

        # 为兼容性添加属性
        self.init_state = tuple(self.robot_env.init_state)
        self.goal_state = tuple(self.robot_env.goal_state)
        self.config_dim = self.robot_env.config_dim
        self.bound = self.robot_env.bound
        self.return_cycles = return_cycles

    def init_new_problem(self, index):
        """
        初始化新问题

        Args:
            index: 问题索引

        Returns:
            dict: 问题描述
        """
        # 使用ProblemManager加载问题数据
        problem = self.problem_manager.init_new_problem(index, self.obstacle_manager)

        # 设置机器人环境的起点和终点
        self.robot_env.init_state = problem["start"]
        self.robot_env.goal_state = problem["goal"]

        # 更新兼容性属性
        self.init_state = tuple(self.robot_env.init_state)
        self.goal_state = tuple(self.robot_env.goal_state)

        # 更新球体环境中的障碍物
        self.sphere_env.load_obstacles(self.obstacle_manager.obstacles)

        # 返回问题描述
        return self.problem_manager.get_problem()

    def get_problem(self, width=15, index=None):
        """
        获取问题描述

        Args:
            width: 地图宽度
            index: 问题索引

        Returns:
            dict: 问题描述
        """
        return self.problem_manager.get_problem(width, index)

    def _state_fp(self, state):
        """
        检查状态是否在自由空间中（基于球体碰撞检测）

        Args:
            state: 配置状态

        Returns:
            bool: 是否自由（True表示自由，False表示碰撞）
        """
        result = self.sphere_env.get_sphere_collision_data(state)
        # 根据return_cycles标志处理返回值
        collision = result[0]
        return not collision  # 返回True表示自由，False表示碰撞

    def _state_fp_probe(self, state):
        """
        检查状态是否在自由空间中（带详细信息）

        Args:
            state: 配置状态

        Returns:
            tuple: (result, info, coll) - 是否自由、坐标信息、碰撞信息
        """
        if self.return_cycles:
            collision, coords, colls, cycles = (  # type: ignore[misc]
                self.sphere_env.get_sphere_collision_data(state)
            )
            return not collision, coords, colls
        else:
            collision, coords, colls = self.sphere_env.get_sphere_collision_data(state)  # type: ignore[misc]
            return not collision, coords, colls

    def _edge_fp(self, state1, state2):
        """
        检查边是否在自由空间中（基于线性插值的球体碰撞检测）

        Args:
            state1: 起点配置
            state2: 终点配置

        Returns:
            bool: 是否自由
        """
        # 使用迭代检查段落的方法
        return self._iterative_check_segment(state1, state2)

    def _edge_fp_probe(self, state1, state2, num_samples=10):
        """
        检查边是否在自由空间中（带详细信息）

        Args:
            state1: 起点配置
            state2: 终点配置
            num_samples: 采样点数量

        Returns:
            tuple: (result, info, coll) - 是否自由、坐标信息列表、碰撞信息列表
        """
        edge_coords = []
        edge_colls = []

        # 采样路径上的点
        for i in range(num_samples):
            ratio = i / (num_samples - 1) if num_samples > 1 else 0
            interpolated_state = self.robot_env.interpolate(state1, state2, ratio)

            if self.return_cycles:
                collision, coords, colls, cycles = (  # type: ignore[misc]
                    self.sphere_env.get_sphere_collision_data(interpolated_state)
                )
            else:
                collision, coords, colls = self.sphere_env.get_sphere_collision_data(  # type: ignore[misc]
                    interpolated_state
                )

            edge_coords.append(coords)
            edge_colls.append(colls)

            # 如果发现碰撞，可以提前返回
            if collision:
                return False, edge_coords, edge_colls

        return True, edge_coords, edge_colls

    def _iterative_check_segment(self, state1, state2, num_samples=10):
        """
        迭代检查路径段

        Args:
            state1: 起点配置
            state2: 终点配置
            num_samples: 采样点数量

        Returns:
            bool: 是否可行
        """
        # 检查起点
        collision1, _, _ = self.sphere_env.get_sphere_collision_data(state1)  # type: ignore[misc]
        if collision1:
            return False

        # 检查终点
        collision2, _, _ = self.sphere_env.get_sphere_collision_data(state2)  # type: ignore[misc]
        if collision2:
            return False

        # 检查中间点
        for i in range(1, num_samples - 1):
            ratio = i / (num_samples - 1)
            interpolated_state = self.robot_env.interpolate(state1, state2, ratio)
            collision, _, _ = self.sphere_env.get_sphere_collision_data(  # type: ignore[misc]
                interpolated_state
            )
            if collision:
                return False

        return True

    def in_goal_region(self, state, goal_state=None, threshold=None):
        """
        检查状态是否在目标区域（距离小于阈值且无碰撞）

        Args:
            state: 当前配置
            goal_state: 目标配置（可选，默认使用self.goal_state）
            threshold: 距离阈值（可选，默认使用RRT_EPS）

        Returns:
            bool: 是否在目标区域
        """
        if goal_state is None:
            goal_state = self.goal_state
        if threshold is None:
            threshold = 0.25

        # 检查距离和碰撞状态
        is_close = distance(state, goal_state) < threshold
        is_free = self._state_fp(state)
        return is_close and is_free

    def get_robot_points(self, config, end_point=True):
        """
        获取机器人配置下的关键点

        Args:
            config: 机器人配置
            end_point: 是否只返回末端点

        Returns:
            关键点列表
        """
        return self.robot_env.get_robot_points(config, end_point)

    def sample_n_points(self, n, need_negative=False):
        """
        采样配置点

        Args:
            n: 采样数量
            need_negative: 是否需要负样本

        Returns:
            采样点列表
        """
        return self.robot_env.sample_n_points(n, need_negative)

    def interpolate(self, from_state, to_state, ratio):
        """
        插值两个配置

        Args:
            from_state: 起始配置
            to_state: 目标配置
            ratio: 插值比例

        Returns:
            插值后的配置
        """
        return self.robot_env.interpolate(from_state, to_state, ratio)

    def obs_map(self, num):
        """
        生成障碍物地图

        Args:
            num: 网格分辨率

        Returns:
            tuple: (points_pos, points_obs)
        """
        # 从ProblemManager获取当前问题
        current_problem = self.problem_manager.get_current_problem()
        if current_problem is None:
            raise ValueError("未初始化问题，请先调用 init_new_problem()")

        obstacles = current_problem["obstacles"]

        # 使用ProblemManager的_generate_obs_map方法
        return self.problem_manager._generate_obs_map(obstacles, num)

    def plot(self, path, make_gif=False):
        """
        可视化路径

        Args:
            path: 路径点列表
            make_gif: 是否生成GIF

        Returns:
            GIF帧列表（如果make_gif=True）
        """
        # 暂时返回空列表
        return []

    def store_edge_data(self, state1, state2, num_samples=10):
        """
        存储边的球体碰撞数据

        Args:
            state1: 起点配置
            state2: 终点配置
            num_samples: 采样点数量
        """
        edge_coords = []
        edge_colls = []
        edge_cycles = [] if self.return_cycles else None

        # 采样路径上的点
        for i in range(num_samples):
            ratio = i / (num_samples - 1) if num_samples > 1 else 0
            interpolated_state = self.robot_env.interpolate(state1, state2, ratio)

            if self.return_cycles:
                collision, coords, colls, cycles = (  # type: ignore[misc]
                    self.sphere_env.get_sphere_collision_data(interpolated_state)
                )
                edge_coords.append(coords)
                edge_colls.append(colls)
                if edge_cycles is not None:
                    edge_cycles.append(cycles)
            else:
                collision, coords, colls = self.sphere_env.get_sphere_collision_data(  # type: ignore[misc]
                    interpolated_state
                )
                edge_coords.append(coords)
                edge_colls.append(colls)

        # 存储到sphere_env
        self.sphere_env.store_sphere_data(
            edge_coords, edge_colls, cycles=edge_cycles, is_edge=True
        )

    def store_state_data(self, state):
        """
        存储单个状态的球体碰撞数据

        Args:
            state: 配置状态
        """
        if self.return_cycles:
            collision, coords, colls, cycles = (  # type: ignore[misc]
                self.sphere_env.get_sphere_collision_data(state)
            )
            self.sphere_env.store_sphere_data(
                coords, colls, cycles=cycles, is_edge=False
            )
        else:
            collision, coords, colls = self.sphere_env.get_sphere_collision_data(state)  # type: ignore[misc]
            self.sphere_env.store_sphere_data(coords, colls, is_edge=False)

    def save_collision_data(self, output_file):
        """
        保存球体碰撞数据到文件

        Args:
            output_file: 输出文件路径
        """
        self.sphere_env.save_collision_data(output_file)

    def get_collision_stats(self):
        """
        获取碰撞检测统计信息

        Returns:
            dict: 统计信息字典
        """
        return self.sphere_env.get_collision_stats()

    def generate_random_obstacles(
        self,
        num_obstacles,
        workspace_range,
        voxel_size_range,
        safe_zone_center,
        safe_zone_radius,
    ):
        """
        生成随机障碍物并加载到环境中

        Args:
            num_obstacles: 障碍物数量
            workspace_range: 工作空间范围 (min, max)
            voxel_size_range: 体素尺寸范围 (min, max)
            safe_zone_center: 安全区域中心
            safe_zone_radius: 安全区域半径

        Returns:
            list: 随机生成的障碍物列表
        """
        # 生成随机障碍物
        obstacles = self.obstacle_manager.generate_random_obstacles(
            num_obstacles=num_obstacles,
            workspace_range=workspace_range,
            voxel_size_range=voxel_size_range,
            safe_zone_center=safe_zone_center,
            safe_zone_radius=safe_zone_radius,
        )

        # 加载障碍物到环境中
        self.obstacle_manager.load_obstacles(obstacles)

        # 更新球体环境中的障碍物
        self.sphere_env.load_obstacles(obstacles)

        return obstacles

    def close(self):
        """关闭环境"""
        self.sphere_env.close()
        self.robot_env.close()

    def __str__(self):
        return f"ModularSphereEnv({self.robot_env.__str__()})"
