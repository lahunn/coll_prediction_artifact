# 修订版架构 - 详细实现清单

## 概述

本文档详细说明每个文件的具体修改内容，包括代码片段和实现细节。

---

## 阶段 1：新建文件和基础结构

### 1.1 新建：collision/link_collision_detector.py

**位置**：`trace_generation/core/collision/link_collision_detector.py`

**源代码**：从 `collision_check.py` 提取

**完整代码结构**：

```python
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
    
    def __init__(self, robot_env):
        """
        初始化 Link 级碰撞检测器
        
        Args:
            robot_env: RobotEnv 实例，提供 PyBullet 接口和机器人信息
        """
        self.robot_env = robot_env
        self.collision_time = 0.0
        self.collision_check_count = 0
    
    def check_pose(self, state) -> Tuple[bool, Dict[str, Any]]:
        """
        检查单个配置点的碰撞状态
        
        Args:
            state: numpy array，机器人配置 (DOF,)
        
        Returns:
            tuple: (is_free, collision_data)
            - is_free (bool): 配置是否无碰撞
            - collision_data (dict): 包含碰撞相关数据的字典
              {
                  'link_coords': List[Pose],    # 各 Link 的位姿
                  'link_colls': List[int],      # 各 Link 的碰撞标签
                  'timestamp': float            # 检测时间戳
              }
        """
        start_time = time.time()
        self.collision_check_count += 1
        
        # 验证配置合法性
        if not self.robot_env._valid_state(state):
            self.collision_time += time.time() - start_time
            return False, {
                'link_coords': [],
                'link_colls': [],
                'timestamp': time.time(),
            }
        
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
        
        # 返回标准化的碰撞数据
        is_free = not is_collision
        collision_data = {
            'link_coords': link_coords,
            'link_colls': link_colls,
            'timestamp': time.time(),
        }
        
        return is_free, collision_data
    
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
        p.performCollisionDetection(
            physicsClientId=self.robot_env.physics_client
        )
        
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
    
    def reset(self):
        """重置统计信息"""
        self.collision_time = 0.0
        self.collision_check_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        return {
            'collision_check_count': self.collision_check_count,
            'collision_time': self.collision_time,
        }
```

**关键点**：
- `check_pose()` 是唯一的公开接口
- 返回 `(is_free, dict)` 的标准化格式
- `_get_link_collisions()` 是私有方法
- 包含统计信息（collision_check_count, collision_time）

---

### 1.2 新建：collision/collision_env.py

**位置**：`trace_generation/core/collision/collision_env.py`

**源代码**：从 `robot/collision_check.py` 移来并重构

**关键修改**：

```python
#!/usr/bin/env python3
"""
Pose 级碰撞检测协调层

组织碰撞检测的流程：
1. 边离散化
2. 对每个 pose 调用底层 detector
3. 汇总碰撞数据
"""

import numpy as np
import time
from typing import Tuple, List

from trace_generation.core.collision.link_collision_detector import LinkCollisionDetector
from trace_generation.core.collision.sphere_detector import SphereCollisionDetector
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
        config_output_file=None,
    ):
        """
        初始化碰撞检测环境
        
        Args:
            robot_env: RobotEnv 实例
            collision_model_type: 碰撞模型类型
                - "link": 使用 LinkCollisionDetector
                - "sphere": 使用 SphereCollisionDetector
            config_output_file: 配置输出文件路径（可选）
        """
        self.robot_env = robot_env
        self.collision_model_type = collision_model_type
        self.obstacle_body_ids = []
        self.config_output_file = config_output_file
        self.config_list = []
        
        # 根据模型类型选择合适的检测器
        if collision_model_type == "link":
            self.detector = LinkCollisionDetector(robot_env)
        elif collision_model_type == "sphere":
            # 假设 SphereCollisionDetector 已添加 check_pose() 接口
            self.detector = SphereCollisionDetector(robot_env)
        else:
            raise ValueError(
                f"Unknown collision model type: {collision_model_type}"
            )
        
        # 初始化统一的数据管理器
        self.data_manager = CollisionDataManager(model_type=collision_model_type)
    
    def load_obstacle_body_ids(self, obstacle_body_ids: List[int]):
        """加载障碍物体 ID 列表"""
        self.obstacle_body_ids = obstacle_body_ids
    
    def close(self):
        """关闭碰撞检测环境"""
        pass
    
    def _point_in_free_space(self, state) -> Tuple[bool, dict]:
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
        state,
        new_state,
        RRT_EPS=0.25,
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
        state,
        new_state,
        RRT_EPS=None,
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
    
    def in_goal_region(self, state, goal_state=None, threshold=None) -> bool:
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
        
        return (
            distance(state, goal_state) < threshold
            and self._state_fp(state)
        )
    
    def _iterative_check_segment(self, left, right) -> bool:
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
```

**关键修改**：
- 删除 `_get_link_collisions()` 方法（下沉到 LinkCollisionDetector）
- 创建 `self.detector` 实例（在 `__init__` 中根据 `collision_model_type` 选择）
- 修改 `_point_in_free_space()` 调用 `self.detector.check_pose()`
- 修改 `_edge_fp()` 的数据存储逻辑（通过 data_manager 而非直接访问）

---

### 1.3 重构：collision/data_manager.py

**位置**：`trace_generation/core/collision/data_manager.py`

**源代码**：从 `robot/collision_data_manager.py` 移来并重构

**新增内容**：

```python
#!/usr/bin/env python3
"""
碰撞数据管理层

支持多种碰撞模型（Link 级、Sphere 级），通过模型抽象层进行扩展
"""

import pickle
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional


class CollisionDataModel(ABC):
    """
    碰撞数据模型抽象基类
    
    定义碰撞数据存储、统计和导出的标准接口
    """
    
    @abstractmethod
    def store_collision_data(self, data: Dict, is_edge: bool = True):
        """
        存储碰撞数据
        
        Args:
            data: 碰撞数据字典（数据格式由具体模型定义）
            is_edge: 是否为边数据（True）还是单点数据（False）
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置所有数据和统计"""
        pass
    
    @abstractmethod
    def get_collision_ratio(self) -> Tuple[float, float, float]:
        """
        计算碰撞率
        
        Returns:
            tuple: (element_ratio, pose_ratio, edge_ratio)
            - element_ratio: 单个元素（Link/Sphere）的碰撞比例
            - pose_ratio: 单个姿态中有碰撞的比例
            - edge_ratio: 单条边中有碰撞的比例
        """
        pass
    
    @abstractmethod
    def save_collision_data(self, output_file: str):
        """
        保存碰撞数据到文件
        
        Args:
            output_file: 输出文件路径
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            dict: 包含统计信息的字典
        """
        pass


class LinkDataModel(CollisionDataModel):
    """
    Link 级数据模型
    
    存储 Link 位姿和碰撞标签数据
    """
    
    def __init__(self):
        """初始化 Link 数据模型"""
        self.link_data = []           # List[List[Pose]]，边-姿态-Link 三层结构
        self.link_coll_data = []      # List[List[List[int]]]，碰撞标签
    
    def store_collision_data(self, data: Dict, is_edge: bool = True):
        """
        存储 Link 碰撞数据
        
        Args:
            data: 数据字典
              {
                  'link_coords': List[Pose],
                  'link_colls': List[int],
              }
            is_edge: 是否为边数据
        """
        if not data or not data.get('link_coords'):
            return
        
        link_coords = data['link_coords']
        link_colls = data['link_colls']
        
        if is_edge:
            # 边数据：直接添加
            self.link_data.append(link_coords)
            self.link_coll_data.append(link_colls)
        else:
            # 单点数据：包装为单元素列表
            self.link_data.append([link_coords])
            self.link_coll_data.append([link_colls])
    
    def reset(self):
        """重置所有数据"""
        self.link_data.clear()
        self.link_coll_data.clear()
    
    def _calculate_collision_ratios(self) -> Tuple[float, float, float]:
        """
        计算碰撞率
        
        数据结构：
        - link_coll_data[edge_idx][pose_idx][link_idx] = 0/1
        - 0 表示碰撞，1 表示自由
        
        Returns:
            tuple: (link_ratio, pose_ratio, edge_ratio)
        """
        link_ratio = 0.0
        pose_ratio = 0.0
        edge_ratio = 0.0
        
        if not self.link_coll_data:
            return link_ratio, pose_ratio, edge_ratio
        
        total_links = 0
        collided_links = 0
        total_poses = 0
        collided_poses = 0
        total_edges = len(self.link_coll_data)
        collided_edges = 0
        
        for edge_colls in self.link_coll_data:
            is_edge_collided = False
            
            for pose_colls in edge_colls:
                total_poses += 1
                is_pose_collided = False
                
                for coll_value in pose_colls:
                    total_links += 1
                    if coll_value == 0:  # 0 表示碰撞
                        collided_links += 1
                        is_pose_collided = True
                
                if is_pose_collided:
                    collided_poses += 1
                    is_edge_collided = True
            
            if is_edge_collided:
                collided_edges += 1
        
        link_ratio = collided_links / total_links if total_links > 0 else 0.0
        pose_ratio = collided_poses / total_poses if total_poses > 0 else 0.0
        edge_ratio = collided_edges / total_edges if total_edges > 0 else 0.0
        
        return link_ratio, pose_ratio, edge_ratio
    
    def get_collision_ratio(self) -> Tuple[float, float, float]:
        """获取碰撞率"""
        return self._calculate_collision_ratios()
    
    def save_collision_data(self, output_file: str):
        """保存 Link 碰撞数据到文件"""
        data = {
            'link_data': self.link_data,
            'link_coll_data': self.link_coll_data,
        }
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 Link 数据模型的统计信息"""
        link_ratio, pose_ratio, edge_ratio = self.get_collision_ratio()
        return {
            'model_type': 'link',
            'total_edges': len(self.link_coll_data),
            'link_ratio': link_ratio,
            'pose_ratio': pose_ratio,
            'edge_ratio': edge_ratio,
        }


class SphereDataModel(CollisionDataModel):
    """
    Sphere 级数据模型
    
    存储球体位置和碰撞标签数据
    """
    
    def __init__(self, return_cycles: bool = False):
        """
        初始化 Sphere 数据模型
        
        Args:
            return_cycles: 是否返回周期信息
        """
        self.sphere_data = []          # 球体坐标数据
        self.sphere_coll_data = []     # 球体碰撞标签
        self.sphere_cycles = []        # 周期数据（可选）
        self.return_cycles = return_cycles
    
    def store_collision_data(self, data: Dict, is_edge: bool = True, cycles: Optional[int] = None):
        """
        存储 Sphere 碰撞数据
        
        Args:
            data: 数据字典
              {
                  'sphere_coords': List[Position],
                  'sphere_colls': List[int],
              }
            is_edge: 是否为边数据
            cycles: 周期数据（可选）
        """
        if not data or not data.get('sphere_coords'):
            return
        
        sphere_coords = data['sphere_coords']
        sphere_colls = data['sphere_colls']
        
        if is_edge:
            self.sphere_data.append(sphere_coords)
            self.sphere_coll_data.append(sphere_colls)
        else:
            self.sphere_data.append([sphere_coords])
            self.sphere_coll_data.append([sphere_colls])
        
        if self.return_cycles and cycles is not None:
            self.sphere_cycles.append(cycles)
    
    def reset(self):
        """重置所有数据"""
        self.sphere_data.clear()
        self.sphere_coll_data.clear()
        self.sphere_cycles.clear()
    
    def _calculate_collision_ratios(self) -> Tuple[float, float, float]:
        """计算碰撞率（逻辑与 LinkDataModel 相同）"""
        # 实现逻辑与 LinkDataModel 相同，只是处理球体数据
        link_ratio = 0.0
        pose_ratio = 0.0
        edge_ratio = 0.0
        
        if not self.sphere_coll_data:
            return link_ratio, pose_ratio, edge_ratio
        
        total_spheres = 0
        collided_spheres = 0
        total_poses = 0
        collided_poses = 0
        total_edges = len(self.sphere_coll_data)
        collided_edges = 0
        
        for edge_colls in self.sphere_coll_data:
            is_edge_collided = False
            
            for pose_colls in edge_colls:
                total_poses += 1
                is_pose_collided = False
                
                for coll_value in pose_colls:
                    total_spheres += 1
                    if coll_value == 0:  # 0 表示碰撞
                        collided_spheres += 1
                        is_pose_collided = True
                
                if is_pose_collided:
                    collided_poses += 1
                    is_edge_collided = True
            
            if is_edge_collided:
                collided_edges += 1
        
        link_ratio = collided_spheres / total_spheres if total_spheres > 0 else 0.0
        pose_ratio = collided_poses / total_poses if total_poses > 0 else 0.0
        edge_ratio = collided_edges / total_edges if total_edges > 0 else 0.0
        
        return link_ratio, pose_ratio, edge_ratio
    
    def get_collision_ratio(self) -> Tuple[float, float, float]:
        """获取碰撞率"""
        return self._calculate_collision_ratios()
    
    def save_collision_data(self, output_file: str):
        """保存 Sphere 碰撞数据到文件"""
        data = {
            'sphere_data': self.sphere_data,
            'sphere_coll_data': self.sphere_coll_data,
        }
        if self.return_cycles:
            data['sphere_cycles'] = self.sphere_cycles
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 Sphere 数据模型的统计信息"""
        link_ratio, pose_ratio, edge_ratio = self.get_collision_ratio()
        return {
            'model_type': 'sphere',
            'total_edges': len(self.sphere_coll_data),
            'sphere_ratio': link_ratio,
            'pose_ratio': pose_ratio,
            'edge_ratio': edge_ratio,
            'return_cycles': self.return_cycles,
        }


class CollisionDataManager:
    """
    统一的碰撞数据管理器
    
    职责：
    - 根据模型类型创建合适的数据模型
    - 代理数据存储、统计和导出操作
    - 提供向后兼容的属性访问
    """
    
    def __init__(self, model_type: str = "link"):
        """
        初始化碰撞数据管理器
        
        Args:
            model_type: 碰撞模型类型
                - "link": Link 级碰撞检测
                - "sphere": Sphere 级碰撞检测
        """
        self.model_type = model_type
        self.collision_data = self._create_model(model_type)
        self.edge_fp_call_count = 0
    
    def _create_model(self, model_type: str) -> CollisionDataModel:
        """
        工厂方法：创建数据模型
        
        Args:
            model_type: 模型类型
        
        Returns:
            CollisionDataModel 实例
        """
        if model_type == "link":
            return LinkDataModel()
        elif model_type == "sphere":
            return SphereDataModel(return_cycles=False)
        else:
            raise ValueError(f"Unknown collision model type: {model_type}")
    
    def _store_collision_data(self, data: Dict, is_edge: bool = True):
        """
        代理到具体模型的数据存储方法
        
        Args:
            data: 碰撞数据
            is_edge: 是否为边数据
        """
        self.collision_data.store_collision_data(data, is_edge=is_edge)
    
    def reset(self):
        """重置所有数据和统计"""
        self.collision_data.reset()
        self.edge_fp_call_count = 0
    
    def get_collision_ratio(self) -> Tuple[float, float, float]:
        """获取碰撞率"""
        return self.collision_data.get_collision_ratio()
    
    def save_collision_data(self, output_file: str):
        """保存碰撞数据到文件"""
        self.collision_data.save_collision_data(output_file)
    
    def get_collision_stats(self) -> Dict[str, Any]:
        """获取完整的碰撞统计信息"""
        stats = self.collision_data.get_stats()
        stats['edge_fp_call_count'] = self.edge_fp_call_count
        return stats
    
    # ========== 向后兼容性：属性代理 ==========
    
    @property
    def obb_link_data(self):
        """向后兼容：Link 位姿数据"""
        if isinstance(self.collision_data, LinkDataModel):
            return self.collision_data.link_data
        raise AttributeError(
            "obb_link_data is only available for 'link' model type"
        )
    
    @property
    def obb_link_coll_data(self):
        """向后兼容：Link 碰撞数据"""
        if isinstance(self.collision_data, LinkDataModel):
            return self.collision_data.link_coll_data
        raise AttributeError(
            "obb_link_coll_data is only available for 'link' model type"
        )
```

**关键特点**：
- 提供抽象基类 `CollisionDataModel`
- 两个具体实现：`LinkDataModel` 和 `SphereDataModel`
- 属性代理 `obb_link_data` 和 `obb_link_coll_data` 保持向后兼容
- 工厂方法 `_create_model()` 支持动态选择

---

## 阶段 2：修改现有文件

### 2.1 修改：collision/sphere_detector.py

**需要添加的内容**：

```python
def check_pose(self, state) -> Tuple[bool, Dict[str, Any]]:
    """
    检查单个配置点的碰撞状态（新接口）
    
    Args:
        state: 机器人配置
    
    Returns:
        tuple: (is_free, collision_data)
        - is_free (bool): 配置是否无碰撞
        - collision_data (dict): 包含碰撞相关数据的字典
          {
              'sphere_coords': List[Position],  # 各球体的位置
              'sphere_colls': List[int],        # 各球体的碰撞标签 (0=碰撞, 1=自由)
              'timestamp': float                # 检测时间戳
          }
    """
    start_time = time.time()
    self.collision_check_count += 1
    
    # 进行球体碰撞检测
    is_collision = self._check_sphere_collision(state)
    
    # 收集球体数据
    sphere_coords = self._get_sphere_coords(state)
    sphere_colls = self._get_sphere_collisions(state)
    
    self.collision_time += time.time() - start_time
    
    is_free = not is_collision
    collision_data = {
        'sphere_coords': sphere_coords,
        'sphere_colls': sphere_colls,
        'timestamp': time.time(),
    }
    
    return is_free, collision_data


def get_stats(self) -> Dict[str, Any]:
    """获取检测统计信息（新方法）"""
    return {
        'collision_check_count': self.collision_check_count,
        'collision_time': self.collision_time,
    }
```

**注意**：
- `check_pose()` 接口与 `LinkCollisionDetector` 保持一致
- 返回 dict 格式的标准化数据（包含 'sphere_coords' 而不是 'link_coords'）
- 保留现有的 `_check_sphere_collision()` 等方法，只添加新接口

---

### 2.2 修改：robot/modular_env.py

**改动位置**：导入语句和 CollisionEnv 初始化

```python
# 改前
from trace_generation.core.robot.collision_check import CollisionEnv
from trace_generation.core.robot.collision_data_manager import CollisionDataManager

# 改后
from trace_generation.core.collision.collision_env import CollisionEnv
from trace_generation.core.collision.data_manager import CollisionDataManager
```

**其他代码保持不变**（因为 API 完全兼容）

---

### 2.3 修改：robot/modular_sphere_env.py

**改动位置**：导入语句

```python
# 改前
from trace_generation.core.robot.collision_check import CollisionEnv
from trace_generation.core.robot.collision_data_manager import CollisionDataManager

# 改后
from trace_generation.core.collision.collision_env import CollisionEnv
from trace_generation.core.collision.data_manager import CollisionDataManager
```

**其他代码保持不变**

---

### 2.4 修改：robot/environment.py（如果有相关导入）

检查是否有相关导入，如果有则更新为新路径。

---

### 2.5 创建：collision/__init__.py（补充导出）

```python
"""
碰撞检测模块

提供统一的碰撞检测接口，支持多种碰撞模型
"""

from .collision_env import CollisionEnv
from .data_manager import CollisionDataManager
from .link_collision_detector import LinkCollisionDetector
from .sphere_detector import SphereCollisionDetector

__all__ = [
    'CollisionEnv',
    'CollisionDataManager',
    'LinkCollisionDetector',
    'SphereCollisionDetector',
]
```

---

### 2.6 创建：robot/__init__.py（向后兼容导出）

```python
"""
机器人相关模块

为了向后兼容，从 collision 目录重导出碰撞检测类
"""

# 向后兼容导出
from trace_generation.core.collision.collision_env import CollisionEnv
from trace_generation.core.collision.data_manager import CollisionDataManager

__all__ = [
    'CollisionEnv',
    'CollisionDataManager',
]
```

---

## 阶段 3：清理和验证

### 3.1 删除旧文件

```bash
rm trace_generation/core/robot/collision_check.py
rm trace_generation/core/robot/collision_data_manager.py
```

### 3.2 验证导入

运行脚本确保所有导入都能正常工作：

```bash
cd /home/lanh/project/robot_sim/coll_prediction_artifact
python -c "from trace_generation.core.collision import CollisionEnv; print('✓ CollisionEnv import OK')"
python -c "from trace_generation.core.collision import CollisionDataManager; print('✓ CollisionDataManager import OK')"
python -c "from trace_generation.core.collision import LinkCollisionDetector; print('✓ LinkCollisionDetector import OK')"
```

### 3.3 运行现有测试

确保现有脚本和测试仍能正常运行。

---

## 总结

| 步骤 | 操作 | 文件 | 代码量 |
|-----|------|------|--------|
| 1 | 新建 | collision/link_collision_detector.py | ~200 行 |
| 2 | 新建 | collision/collision_env.py | ~250 行（从 collision_check.py 改进） |
| 3 | 新建 | collision/data_manager.py | ~400 行（从 collision_data_manager.py 改进） |
| 4 | 修改 | collision/sphere_detector.py | +50 行（添加 check_pose() 接口） |
| 5 | 修改 | robot/modular_env.py | 2 行（导入路径） |
| 6 | 修改 | robot/modular_sphere_env.py | 2 行（导入路径） |
| 7 | 新建 | collision/__init__.py | ~10 行 |
| 8 | 新建 | robot/__init__.py | ~10 行 |

**总计新增代码**：~900 行

**优势**：
✓ 清晰的分层架构
✓ 高内聚低耦合
✓ 易于测试和扩展
✓ 充分的向后兼容性

