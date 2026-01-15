from typing import Union
import numpy as np
from trace_generation.utils.problem import ProblemManager
from trace_generation.core.scene.obstacle_manager import ObstacleManager
from trace_generation.core.robot.environment import RobotEnv
from trace_generation.core.collision.collision_env import CollisionEnv


class ModularEnv:
    """
    模块化环境类，组合各个组件提供统一的接口

    支持多种碰撞检测模型（Link、Sphere等），并提供统一的接口。
    可以通过参数选择碰撞检测类型。

    主要组件:
    - problem_manager: 问题管理器
    - obstacle_manager: 障碍物管理器
    - robot_env: 机器人环境
    - collision_env: 碰撞检测环境（支持Link和Sphere模型）
    """

    # 类型注解：支持 tuple 和 np.ndarray
    init_state: Union[tuple, np.ndarray]
    goal_state: Union[tuple, np.ndarray]

    def __init__(
        self,
        robot_name,
        map_file=None,
        GUI=False,
        enable_self_collision=False,
        collision_model_type: str = "link",
        return_cycles: bool = False,
    ):
        """
        初始化模块化环境

        Args:
            robot_name: 机器人名称（例如 'franka', 'ur5e'）
            map_file: 问题数据集文件路径
            GUI: 是否启用GUI模式
            enable_self_collision: 是否启用自碰撞检测（默认为False）
            collision_model_type: 碰撞模型类型（"link" 或 "sphere"，默认 "link"）
            return_cycles: 是否返回硬件周期成本（仅Sphere模型支持，默认False）
        """
        # 初始化各个组件
        self.problem_manager = ProblemManager(map_file)
        self.robot_env = RobotEnv(
            robot_name, OBB_GUI=GUI, enable_self_collision=enable_self_collision
        )
        # 中央管理 RRT_EPS（默认为 0.25），并将其注入 CollisionEnv
        self.RRT_EPS = 0.25
        self.collision_env = CollisionEnv(
            self.robot_env,
            collision_model_type=collision_model_type,
            return_cycles=return_cycles,
            RRT_EPS=self.RRT_EPS,
        )
        self.obstacle_manager = ObstacleManager(
            physics_client=self.robot_env.physics_client
        )

        # 为兼容性添加属性
        self.init_state = tuple(self.robot_env.init_state)
        self.goal_state = tuple(self.robot_env.goal_state)
        self.config_dim = self.robot_env.config_dim
        self.dim = 3  # 假设都是3D空间
        self.bound = self.robot_env.bound
        self.collision_model_type = collision_model_type

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
        检查状态是否在自由空间中

        Args:
            state: 配置状态

        Returns:
            bool: 是否自由
        """
        return self.collision_env._state_fp(state)

    def _edge_fp(self, state1, state2):
        """
        检查边是否在自由空间中

        Args:
            state1: 起点配置
            state2: 终点配置

        Returns:
            bool: 是否自由
        """
        edge_free = self.collision_env._edge_fp(state1, state2)
        return edge_free

    def in_goal_region(self, state):
        """
        检查状态是否在目标区域

        Args:
            state: 配置状态

        Returns:
            bool: 是否在目标区域
        """
        return self.collision_env.in_goal_region(state)

    def _iterative_check_segment(self, left, right):
        """
        迭代检查路径段

        Args:
            left: 起点配置
            right: 终点配置

        Returns:
            bool: 是否可行
        """
        return self.collision_env._iterative_check_segment(left, right)

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
        negative = []
        samples = []
        # 最大循环次数：对每个需要采样的点，最多尝试 10 * n 次，以防止在高密度障碍下陷入无限循环
        max_attempts_per_sample = 10 * n if n > 0 else 20
        for i in range(n):
            attempts = 0
            while attempts < max_attempts_per_sample:
                # print(f"debug: sampling point {attempts}/{max_attempts_per_sample}")
                sample = self.robot_env.sample_n_points(1)
                if self._state_fp(sample):
                    samples.append(sample)
                    break
                elif need_negative:
                    negative.append(sample)
                attempts += 1
            else:
                # 达到最大尝试次数仍未采到自由点，跳过该采样（返回的 samples 可能少于请求的 n）
                # 不抛出异常以保持调用方的鲁棒性
                pass
        return samples, negative

    def sample_n_points_probe(self, n, need_negative=False):
        """
        采样n个自由配置点（带详细碰撞信息），使用拒绝采样

        Args:
            n: 采样数量（自由配置数量）
            need_negative: 是否收集负样本（碰撞点）

        Returns:
            如果 need_negative=False: 返回 (free, [], info_list, info_coll_list)
            如果 need_negative=True: 返回 (free, collided, info_list, info_coll_list)
        """
        negative = []
        samples = []
        info_list = []
        info_coll_list = []

        for i in range(n):
            while True:
                # 单次均匀采样
                sample = self.robot_env.sample_n_points(1)
                is_free, info, info_coll = self._state_fp_probe(sample)

                if is_free:
                    # 找到自由配置，加入正样本并退出内循环
                    samples.append(sample)
                    info_list.append(info)
                    info_coll_list.append(info_coll)
                    break
                elif need_negative:
                    # 碰撞配置，加入负样本并继续尝试
                    negative.append(sample)
                    info_list.append(info)
                    info_coll_list.append(info_coll)

        return samples, negative, info_list, info_coll_list

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
        # 这里需要实现路径可视化
        # 暂时返回空列表
        return []

    def close(self):
        """关闭环境"""
        self.robot_env.close()

    def collision_check_count(self):
        """获取碰撞检查次数"""
        return self.collision_env.detector.collision_check_count

    def collision_time(self):
        """获取碰撞检查总耗时"""
        return self.collision_env.detector.collision_time

    def _state_fp_probe(self, state):
        """
        检查状态是否在自由空间中（带详细信息）

        Args:
            state: 配置状态

        Returns:
            tuple: (result, info, coll) - 是否自由、单元坐标信息、碰撞信息
        """
        is_free, unit_coords, unit_colls = self.collision_env._point_in_free_space(
            state
        )

        # 返回格式：(result, info, coll)
        # info: unit_coords, coll: unit_colls
        return is_free, unit_coords, unit_colls

    def _edge_fp_probe(self, state1, state2):
        """
        检查边是否在自由空间中（带详细信息）

        Args:
            state1: 起点配置
            state2: 终点配置

        Returns:
            tuple: (result, info, coll) - 是否自由、单元坐标信息、碰撞信息
        """

        # 直接调用collision_env的_edge_fp获取详细信息
        edge_free = self.collision_env._edge_fp(state1, state2)

        # 从数据模型中直接获取最后一条边的数据
        unit_data = self.collision_env.data_manager.collision_data.unit_data
        unit_coll_data = self.collision_env.data_manager.collision_data.unit_coll_data

        if unit_data and unit_coll_data and edge_free is not None:
            edge_link_coords = unit_data[-1]
            edge_link_colls = unit_coll_data[-1]
        else:
            edge_link_coords = []
            edge_link_colls = []

        return edge_free, edge_link_coords, edge_link_colls

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
        self.collision_env.load_obstacles(self.obstacle_manager.obstacles)

        return obstacles

    def load_obstacles(self, obstacles):
        """
        加载指定的障碍物列表到环境中
        Args:
            obstacles: 障碍物列表
        """
        self.obstacle_manager.load_obstacles(obstacles)
        self.collision_env.load_obstacles(self.obstacle_manager.obstacles)

    def __str__(self):
        return f"ModularEnv({self.robot_env.__str__()})"
