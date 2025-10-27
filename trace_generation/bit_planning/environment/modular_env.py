from .problem import ProblemManager
from .obstacle_manager import ObstacleManager
from .robot_env import RobotEnv
from .collision_env import CollisionEnv
import numpy as np


class ModularEnv:
    """
    模块化环境类，组合各个组件提供统一的接口

    主要组件:
    - problem_manager: 问题管理器
    - obstacle_manager: 障碍物管理器
    - robot_env: 机器人环境
    - collision_env: 碰撞检测环境
    """

    def __init__(self, robot_file, map_file=None, GUI=False):
        """
        初始化模块化环境

        Args:
            robot_file: 机器人URDF文件路径
            map_file: 问题数据集文件路径
            GUI: 是否启用GUI模式
        """
        # 初始化各个组件
        self.problem_manager = ProblemManager(map_file)
        self.robot_env = RobotEnv(robot_file, OBB_GUI=GUI)
        self.collision_env = CollisionEnv(self.robot_env)
        self.obstacle_manager = ObstacleManager(
            physics_client=self.robot_env.physics_client
        )

        # 设置碰撞环境中的障碍物
        self.collision_env.load_obstacle_body_ids(
            self.obstacle_manager.obstacle_body_ids
        )

        # 为兼容性添加属性
        self.init_state = tuple(self.robot_env.init_state)
        self.goal_state = tuple(self.robot_env.goal_state)
        self.config_dim = self.robot_env.config_dim
        self.bound = self.robot_env.bound

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
        edge_free, _, _ = self.collision_env._edge_fp(state1, state2)
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
        # 这里需要实现路径可视化
        # 暂时返回空列表
        return []

    def close(self):
        """关闭环境"""
        self.robot_env.close()

    def collision_check_count(self):
        return self.collision_env.data_manager.collision_check_count

    def collision_time(self):
        return self.collision_env.data_manager.collision_time

    def _state_fp_probe(self, state):
        """
        检查状态是否在自由空间中（带详细信息）

        Args:
            state: 配置状态

        Returns:
            tuple: (result, info, coll) - 是否自由、链接坐标信息、碰撞信息
        """
        is_free, link_coords, link_colls = self.collision_env._point_in_free_space(
            state
        )
        # 返回格式：(result, info, coll)
        # info: link_coords, coll: link_colls
        return is_free, link_coords, link_colls

    def _edge_fp_probe(self, state1, state2):
        """
        检查边是否在自由空间中（带详细信息）

        Args:
            state1: 起点配置
            state2: 终点配置

        Returns:
            tuple: (result, info, coll) - 是否自由、链接坐标信息、碰撞信息
        """

        # 直接调用collision_env的_edge_fp获取详细信息
        edge_free, edge_link_coords, edge_link_colls = self.collision_env._edge_fp(
            state1, state2
        )
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
        self.obstacle_manager.load_and_init_obstacles_from_data(obstacles)

        # 更新碰撞环境中的障碍物
        self.collision_env.load_obstacle_body_ids(
            self.obstacle_manager.obstacle_body_ids
        )

        return obstacles

    def __str__(self):
        return f"ModularEnv({self.robot_env.__str__()})"
