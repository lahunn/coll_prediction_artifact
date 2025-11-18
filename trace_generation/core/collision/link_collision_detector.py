#!/usr/bin/env python3
"""
Link 级碰撞检测实现

使用 PyBullet 进行碰撞检测，检查机器人各个 Link 与环境的碰撞情况
"""

import pybullet as p
import time
from typing import Tuple, List, Dict, Any


class LinkCollisionDetector:
    """
    Link 级碰撞检测器

    职责：
    - 使用 PyBullet 进行碰撞检测
    - 收集各个 Link 的位姿数据
    - 返回标准化的碰撞检测结果

    数据格式（check_pose 返回）：
    {
        'link_coords': [...],   # List[Pose]，各 Link 的位置和姿态
        'link_colls': [...],    # List[int]，各 Link 的碰撞标签 (0=碰撞, 1=自由)
        'timestamp': float      # 检测时间戳
    }
    """

    def __init__(self, robot_env, return_cycles: bool = False):
        """
        初始化 Link 级碰撞检测器

        Args:
            robot_env: RobotEnv 实例，提供 PyBullet 接口和机器人信息
            return_cycles: 是否返回硬件周期成本（Link模型不支持，此参数仅保持接口一致性）
        """
        self.robot_env = robot_env
        self.return_cycles = return_cycles  # 保持接口一致性，但Link模型不使用此参数
        self.collision_time = 0.0
        self.collision_check_count = 0

    def check_pose(self, state) -> Tuple[bool, List, List]:
        """
        检查单个配置点的碰撞状态

        Args:
            state: numpy array，机器人配置 (DOF,)

        Returns:
            tuple: (is_free, unit_coords, unit_colls)
            - is_free (bool): 配置是否无碰撞
            - unit_coords (List[Pose]): 各 Link 的位姿
            - unit_colls (List[int]): 各 Link 的碰撞标签 (0=碰撞, 1=自由)
        """
        start_time = time.time()
        self.collision_check_count += 1

        # 验证配置合法性
        if not self.robot_env._valid_state(state):
            self.collision_time += time.time() - start_time
            return False, [], []

        # 设置机器人配置
        self.robot_env.set_config(state)

        # 获取 Link 级碰撞信息
        is_collision, link_colls = self._get_link_collisions()

        # 收集 Link 位姿数据
        link_coords = []
        for link_idx in self.robot_env.valid_collision_links:
            if link_idx == -1:  # 跳过 base link
                continue
            pose = self.robot_env._get_link_pose(link_idx)
            link_coords.append(pose)

        # 记录统计时间
        elapsed_time = time.time() - start_time
        self.collision_time += elapsed_time

        # 返回三个独立的值
        is_free = not is_collision
        return is_free, link_coords, link_colls

    def _get_link_collisions(self) -> Tuple[bool, List[int]]:
        """
        获取各个 valid link 的碰撞结果

        检查每个 link 是否与环境中的任何物体接触

        Returns:
            tuple: (any_collision, link_collision_flags)
            - any_collision (bool): 是否有任何 Link 发生碰撞
            - link_collision_flags (List[int]): 各 Link 的碰撞标志
              0 = 该 Link 与某物体接触（碰撞）
              1 = 该 Link 自由（无碰撞）
        """
        any_coll = False
        link_colls = []

        # 执行 PyBullet 碰撞检测
        p.performCollisionDetection(physicsClientId=self.robot_env.physics_client)

        # 逐个 Link 检查
        for link_idx in self.robot_env.valid_collision_links:
            if link_idx == -1:  # 跳过 base link，与 KukaEnv 匹配
                continue

            # 获取该 Link 的所有接触点（包括自碰撞）
            contacts = p.getContactPoints(
                self.robot_env.robotId,
                linkIndexA=link_idx,
                physicsClientId=self.robot_env.physics_client,
            )

            # 判断该 Link 是否碰撞
            is_colliding = len(contacts) > 0
            if is_colliding:
                any_coll = True
                link_colls.append(0)  # 碰撞标签为 0
            else:
                link_colls.append(1)  # 自由标签为 1

        return any_coll, link_colls

    def load_obstacles(self, obstacles):
        """
        加载障碍物（空方法）
        """
        pass

    def reset(self):
        """重置统计信息"""
        self.collision_time = 0.0
        self.collision_check_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        return {
            "collision_check_count": self.collision_check_count,
            "collision_time": self.collision_time,
        }
