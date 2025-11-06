import numpy as np
import pybullet as p
import time

from core.robot.collision_data_manager import CollisionDataManager
from utils.planning_utils import distance


class CollisionEnv:
    """完整的碰撞检测环境类,包含PyBullet初始化、机器人加载和碰撞检测功能"""

    RRT_EPS = 0.25

    def __init__(
        self,
        robot_env,
        config_output_file=None,
    ):
        """
        初始化碰撞检测环境

        Args:
            robot_env: 机器人环境实例
            z_offset: Z轴偏移量
            config_output_file: 配置输出文件路径（可选）
        """
        self.robot_env = robot_env
        self.obstacle_body_ids = []

        # config_output_file 相关逻辑
        self.config_output_file = config_output_file
        self.config_list = []

        # 初始化碰撞数据管理器
        self.data_manager = CollisionDataManager()

    def load_obstacle_body_ids(self, obstacle_body_ids):
        """
        加载障碍物体ID列表

        Args:
            obstacle_body_ids: 障碍物体ID列表
        """
        self.obstacle_body_ids = obstacle_body_ids

    def close(self):
        """关闭碰撞检测环境（robot_env由外部管理）"""
        pass

    def _get_link_collisions(self):
        """获取各个valid link的碰撞结果"""
        any_coll = False
        link_colls = []
        p.performCollisionDetection(physicsClientId=self.robot_env.physics_client)
        for link_idx in self.robot_env.valid_collision_links:
            if link_idx == -1:  # 跳过base link，与KukaEnv匹配
                continue
            # 检查该link的所有接触点，包括自碰撞（匹配KukaEnv的逻辑）
            contacts = p.getContactPoints(
                self.robot_env.robotId,
                linkIndexA=link_idx,
                physicsClientId=self.robot_env.physics_client,
            )

            is_colliding = len(contacts) > 0
            if is_colliding:
                any_coll = True
                link_colls.append(0)
            else:
                link_colls.append(1)
        return any_coll, link_colls

    def _point_in_free_space(self, state):
        """
        收集单个配置点的碰撞数据

        Args:
            state: 配置状态

        Returns:
            tuple: (is_free, link_coords, link_colls)
        """
        start_time = time.time()
        self.data_manager.collision_check_count += 1

        if not self.robot_env._valid_state(state):
            self.data_manager.collision_time += time.time() - start_time
            return False, [], []

        # 设置机器人配置
        self.robot_env.set_config(state)

        # 收集Link数据 (仅valid collision links)
        link_coords = []
        link_collision, link_colls = self._get_link_collisions()
        for link_idx in self.robot_env.valid_collision_links:
            if link_idx == -1:  # 跳过base link，与KukaEnv匹配
                continue
            pose = self.robot_env._get_link_pose(link_idx)
            link_coords.append(pose)

        # 判断是否无碰撞
        is_free = not link_collision

        self.data_manager.collision_time += time.time() - start_time
        return is_free, link_coords, link_colls

    def _state_fp(self, state):
        """检查单个状态并收集数据 (作为单条边)"""
        is_free, link_coords, link_colls = self._point_in_free_space(state)

        # 单点作为一条边 - 数据结构: [edge][pose][link][coord]
        self.data_manager._store_collision_data(link_coords, link_colls, is_edge=False)

        edge_configs = [state.copy()]
        self.config_list.append(np.array(edge_configs))

        return is_free

    def _discretize_edge(self, state, new_state, RRT_EPS=0.25):
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

    def _collect_edge_collision_data(self, edge_configs):
        """
        对边上的配置点进行碰撞检查并收集数据

        Args:
            edge_configs: 配置列表

        Returns:
            tuple: (edge_free, edge_link_coords, edge_link_colls)
        """
        edge_free = True
        edge_link_coords = []  # [pose][link][coord]
        edge_link_colls = []  # [pose][link]

        for config in edge_configs:
            is_free, link_coords, link_colls = self._point_in_free_space(config)

            if not is_free:
                edge_free = False

            # 收集数据
            if link_coords:
                edge_link_coords.append(link_coords)
                edge_link_colls.append(link_colls)

        return edge_free, edge_link_coords, edge_link_colls

    def _edge_fp(self, state, new_state, RRT_EPS=RRT_EPS):
        """检查边并收集数据"""
        """每次都完成一个edge中所有state的检查"""
        self.data_manager.edge_fp_call_count += 1
        assert state.size == new_state.size
        # 离散化边
        edge_configs = self._discretize_edge(state, new_state, RRT_EPS)

        # 收集碰撞数据（包括所有配置点，以匹配 _edge_fp_probe 的需求）
        edge_free, edge_link_coords, edge_link_colls = (
            self._collect_edge_collision_data(edge_configs)
        )  # 检查所有点

        # 保存整条边数据
        if edge_link_coords:
            self.data_manager.obb_link_data.append(edge_link_coords)
            self.data_manager.obb_link_coll_data.append(edge_link_colls)

        self.config_list.append(np.array(edge_configs))
        return edge_free, edge_link_coords, edge_link_colls

    def in_goal_region(self, state, goal_state=None, threshold=None):
        """
        判断某一配置是否在目标区域（距离小于阈值且无碰撞）

        Args:
            state: 当前配置
            goal_state: 目标配置（可选，默认使用robot_env.goal_state）
            threshold: 距离阈值（可选，默认使用RRT_EPS）

        Returns:
            bool: 是否在目标区域
        """
        if goal_state is None:
            goal_state = self.robot_env.goal_state
        if threshold is None:
            threshold = self.RRT_EPS

        return distance(state, goal_state) < threshold and self._state_fp(state)

    def _iterative_check_segment(self, left, right):
        """
        递归检查路径段的可行性（用于高精度碰撞检测）

        Args:
            left: 起点配置
            right: 终点配置

        Returns:
            bool: 路径段是否可行
        """
        # 简单的离散化检查
        edge_configs = self._discretize_edge(left, right, self.RRT_EPS)
        for config in edge_configs:
            if not self._state_fp(config):
                return False
        return True
