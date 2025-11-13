#!/usr/bin/env python3
"""
碰撞数据管理层

统一的碰撞数据管理，支持多种碰撞模型（Link 级、Sphere 级）
所有模型共享同一的数据存储和统计接口
"""

import pickle
from typing import Tuple, Dict, Any, Optional


class UnifiedCollisionDataModel:
    """
    统一的碰撞数据模型

    支持任意类型的碰撞单元（Link、Sphere等），通过统一的接口进行数据管理
    数据结构：
    - unit_data: List[List[Coords]]，边-姿态-单元坐标三层结构
    - unit_coll_data: List[List[List[int]]]，边-姿态-单元碰撞标签
    - unit_cycles: List[int]，可选的硬件周期成本数据
    """

    def __init__(self, return_cycles: bool = False):
        """
        初始化统一碰撞数据模型

        Args:
            return_cycles: 是否返回周期信息（仅 Sphere 模型使用）
        """
        self.unit_data = []  # 单元坐标数据：List[List[Coords]]
        self.unit_coll_data = []  # 单元碰撞标签：List[List[List[int]]]
        self.unit_cycles = []  # 周期数据（可选）
        self.return_cycles = return_cycles

    def store_collision_data(
        self,
        data: Dict,
        is_edge: bool = True,
        cycles: Optional[int] = None,
    ):
        """
        存储碰撞数据

        Args:
            data: 数据字典
              {
                  'unit_coords': List[Coords],
                  'unit_colls': List[int],
              }
            is_edge: 是否为边数据
            cycles: 周期数据（可选）
        """
        if not data or not data.get("unit_coords"):
            return

        unit_coords = data.get("unit_coords", [])
        unit_colls = data.get("unit_colls", [])

        if is_edge:
            # 边数据：直接添加
            self.unit_data.append(unit_coords)
            self.unit_coll_data.append(unit_colls)
        else:
            # 单点数据：包装为单元素列表
            self.unit_data.append([unit_coords])
            self.unit_coll_data.append([unit_colls])

        if self.return_cycles and cycles is not None:
            self.unit_cycles.append(cycles)

    def reset(self):
        """重置所有数据"""
        self.unit_data.clear()
        self.unit_coll_data.clear()
        self.unit_cycles.clear()

    def _calculate_collision_ratios(self) -> Tuple[float, float, float]:
        """
        计算碰撞率

        数据结构：
        - unit_coll_data[edge_idx][pose_idx][unit_idx] = 0/1
        - 0 表示碰撞，1 表示自由

        Returns:
            tuple: (unit_ratio, pose_ratio, edge_ratio)
        """
        unit_ratio = 0.0
        pose_ratio = 0.0
        edge_ratio = 0.0

        if not self.unit_coll_data:
            return unit_ratio, pose_ratio, edge_ratio

        total_units = 0
        collided_units = 0
        total_poses = 0
        collided_poses = 0
        total_edges = len(self.unit_coll_data)
        collided_edges = 0

        for edge_colls in self.unit_coll_data:
            is_edge_collided = False

            for pose_colls in edge_colls:
                total_poses += 1
                is_pose_collided = False

                for coll_value in pose_colls:
                    total_units += 1
                    if coll_value == 0:  # 0 表示碰撞
                        collided_units += 1
                        is_pose_collided = True

                if is_pose_collided:
                    collided_poses += 1
                    is_edge_collided = True

            if is_edge_collided:
                collided_edges += 1

        unit_ratio = collided_units / total_units if total_units > 0 else 0.0
        pose_ratio = collided_poses / total_poses if total_poses > 0 else 0.0
        edge_ratio = collided_edges / total_edges if total_edges > 0 else 0.0

        return unit_ratio, pose_ratio, edge_ratio

    def get_collision_ratio(self) -> Tuple[float, float, float]:
        """获取碰撞率"""
        return self._calculate_collision_ratios()

    def save_collision_data(self, output_file: str):
        """保存碰撞数据到文件"""
        data = {
            "unit_data": self.unit_data,
            "unit_coll_data": self.unit_coll_data,
        }
        if self.return_cycles:
            data["unit_cycles"] = self.unit_cycles

        with open(output_file, "wb") as f:
            pickle.dump(data, f)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        unit_ratio, pose_ratio, edge_ratio = self.get_collision_ratio()
        return {
            "model_type": "unified",
            "total_edges": len(self.unit_coll_data),
            "unit_ratio": unit_ratio,
            "pose_ratio": pose_ratio,
            "edge_ratio": edge_ratio,
            "return_cycles": self.return_cycles,
        }


# 向后兼容别名
LinkDataModel = UnifiedCollisionDataModel
SphereDataModel = UnifiedCollisionDataModel


class CollisionDataManager:
    """
    统一的碰撞数据管理器

    职责：
    - 根据模型类型创建合适的数据模型
    - 代理数据存储、统计和导出操作
    - 提供向后兼容的属性访问
    """

    def __init__(self, model_type: str = "link", return_cycles: bool = False):
        """
        初始化碰撞数据管理器

        Args:
            model_type: 碰撞模型类型
                - "link": Link 级碰撞检测
                - "sphere": Sphere 级碰撞检测
            return_cycles: 是否返回周期信息（仅 Sphere 模型使用）
        """
        self.model_type = model_type
        self.return_cycles = return_cycles
        self.collision_data = self._create_model(model_type, return_cycles)
        self.edge_fp_call_count = 0

    def _create_model(
        self, model_type: str, return_cycles: bool = False
    ) -> UnifiedCollisionDataModel:
        """
        工厂方法：创建数据模型

        Args:
            model_type: 模型类型
            return_cycles: 是否返回周期信息

        Returns:
            UnifiedCollisionDataModel 实例
        """
        if model_type == "link":
            return UnifiedCollisionDataModel(return_cycles=return_cycles)
        elif model_type == "sphere":
            return UnifiedCollisionDataModel(return_cycles=return_cycles)
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
        print(f"边检查统计: 总调用次数 {self.edge_fp_call_count}")

        self.collision_data.save_collision_data(output_file)

        ratios = self.get_collision_ratio()
        element_ratio, pose_ratio, edge_ratio = ratios

        print(
            f"✓ {self.model_type}模型 碰撞率统计: "
            f"element: {element_ratio:.4f}, pose: {pose_ratio:.4f}, edge: {edge_ratio:.4f}"
        )

    def get_collision_stats(self) -> Dict[str, Any]:
        """获取完整的碰撞统计信息"""
        stats = self.collision_data.get_stats()
        stats["edge_fp_call_count"] = self.edge_fp_call_count
        return stats
