#!/usr/bin/env python3
"""
Pose 级碰撞检测协调层

组织碰撞检测的流程：
1. 边离散化
2. 对每个 pose 调用底层 detector
3. 汇总碰撞数据
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from trace_generation.core.collision.link_collision_detector import (
    LinkCollisionDetector,
)
from trace_generation.core.collision.sphere_detector import (
    SphereEnvGeometric,
)
from trace_generation.core.collision.data_manager import CollisionDataManager
from trace_generation.utils.planning_utils import distance


class CollisionEnv:
    """
    Pose 级碰撞检测协调层

    职责：
    - 处理边的离散化
    - 为每个 pose 调用适当的碰撞检测器
    - 通过数据管理器汇总碰撞数据

    不涉及具体的碰撞检测算法实现（由下层 detector 负责）
    """

    RRT_EPS = 0.25

    def __init__(
        self,
        robot_env,
        collision_model_type: str = "link",
        config_output_file: Optional[str] = None,
        return_cycles: bool = False,
    ):
        """
        初始化碰撞检测环境

        Args:
            robot_env: RobotEnv 实例
            collision_model_type: 碰撞模型类型
                - "link": 使用 LinkCollisionDetector
                - "sphere": 使用 SphereEnvGeometric
            config_output_file: 配置输出文件路径（可选）
            return_cycles: 是否返回硬件周期成本（仅Sphere模型支持）
        """
        self.robot_env = robot_env
        self.collision_model_type = collision_model_type
        self.config_output_file = config_output_file
        self.config_list = []

        # 根据模型类型选择合适的检测器
        if collision_model_type == "link":
            self.detector = LinkCollisionDetector(
                robot_env, return_cycles=return_cycles
            )
        elif collision_model_type == "sphere":
            self.detector = SphereEnvGeometric(robot_env, return_cycles=return_cycles)
        else:
            raise ValueError(
                f"Unknown collision model type: {collision_model_type}. "
                f"Supported: 'link', 'sphere'"
            )

        # 初始化统一的数据管理器
        self.data_manager = CollisionDataManager(
            model_type=collision_model_type, return_cycles=return_cycles
        )

    def load_obstacles(self, obstacles):
        """
        加载障碍物（统一接口，委托给底层 detector）

        Args:
            obstacles:
                - Link模型：障碍物ID列表
                - Sphere模型：障碍物dict列表 [(halfExtents, basePosition), ...]
        """
        self.detector.load_obstacles(obstacles)

    def close(self):
        """关闭碰撞检测环境"""
        pass

    def _point_in_free_space(self, state) -> Tuple[bool, Dict[str, Any]]:
        """
        检查单个 pose 并收集碰撞数据

        Args:
            state: 机器人配置

        Returns:
            tuple: (is_free, collision_data)
        """
        # 调用 detector 进行碰撞检测
        is_free, collision_data = self.detector.check_pose(state)

        # 存储碰撞数据到数据管理器
        self.data_manager._store_collision_data(collision_data, is_edge=False)

        return is_free, collision_data

    def _state_fp(self, state) -> bool:
        """
        检查单个状态（作为单条边）

        Args:
            state: 机器人配置

        Returns:
            bool: 该状态是否无碰撞
        """
        is_free, collision_data = self._point_in_free_space(state)

        edge_configs = [state.copy()]
        self.config_list.append(np.array(edge_configs))

        return is_free

    def _discretize_edge(
        self,
        state: np.ndarray,
        new_state: np.ndarray,
        RRT_EPS: float = 0.25,
    ) -> List[np.ndarray]:
        """
        将边离散化为多个配置点

        Args:
            state: 起点配置
            new_state: 终点配置
            RRT_EPS: 离散化步长

        Returns:
            list: 离散化的配置列表 [起点, 中间点..., 终点]
        """
        disp = new_state - state
        d = np.linalg.norm(disp)
        K = int(d / RRT_EPS)

        edge_configs = [state.copy()]

        # 生成中间点
        for k in range(1, K + 1):
            c = state + k * 1.0 / K * disp
            edge_configs.append(c.copy())

        edge_configs.append(new_state.copy())
        return edge_configs

    def _edge_fp(
        self,
        state: np.ndarray,
        new_state: np.ndarray,
        RRT_EPS: Optional[float] = None,
    ) -> bool:
        """
        检查边并收集数据

        对边上的所有 pose 进行碰撞检测，汇总结果

        Args:
            state: 起点配置
            new_state: 终点配置
            RRT_EPS: 离散化步长

        Returns:
            bool: 整条边是否无碰撞
        """
        if RRT_EPS is None:
            RRT_EPS = self.RRT_EPS

        self.data_manager.edge_fp_call_count += 1
        assert state.size == new_state.size

        # 离散化边
        edge_configs = self._discretize_edge(state, new_state, RRT_EPS)

        # 对边上的每个 pose 进行检测
        edge_free = True
        for config in edge_configs:
            is_free, collision_data = self._point_in_free_space(config)
            if not is_free:
                edge_free = False

        self.config_list.append(np.array(edge_configs))
        return edge_free

    def in_goal_region(
        self,
        state: np.ndarray,
        goal_state: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        判断某一配置是否在目标区域（距离小于阈值且无碰撞）

        Args:
            state: 当前配置
            goal_state: 目标配置（可选，默认使用 robot_env.goal_state）
            threshold: 距离阈值（可选，默认使用 RRT_EPS）

        Returns:
            bool: 是否在目标区域
        """
        if goal_state is None:
            goal_state = self.robot_env.goal_state
        if threshold is None:
            threshold = self.RRT_EPS

        return distance(state, goal_state) < threshold and self._state_fp(state)

    def _iterative_check_segment(self, left: np.ndarray, right: np.ndarray) -> bool:
        """
        递归检查路径段的可行性（用于高精度碰撞检测）

        Args:
            left: 起点配置
            right: 终点配置

        Returns:
            bool: 路径段是否可行
        """
        edge_configs = self._discretize_edge(left, right, self.RRT_EPS)
        for config in edge_configs:
            if not self._state_fp(config):
                return False
        return True
