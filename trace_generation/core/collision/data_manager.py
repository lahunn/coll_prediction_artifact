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
    
    def store_collision_data(
        self,
        data: Dict,
        is_edge: bool = True,
        cycles: Optional[int] = None,
    ):
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
        stats['edge_fp_call_count'] = self.edge_fp_call_count
        return stats
    
    # ========== 向后兼容性：属性代理 ==========
    
    @property
    def collision_check_count(self) -> int:
        """向后兼容：碰撞检查次数（来自 detector）"""
        # 注意：这个属性实际上由 detector 维护
        # 这里只是为了兼容性而提供
        if hasattr(self, '_collision_check_count'):
            return self._collision_check_count
        return 0
    
    @collision_check_count.setter
    def collision_check_count(self, value: int):
        """设置碰撞检查次数"""
        self._collision_check_count = value
    
    @property
    def collision_time(self) -> float:
        """向后兼容：碰撞检查总耗时（来自 detector）"""
        if hasattr(self, '_collision_time'):
            return self._collision_time
        return 0.0
    
    @collision_time.setter
    def collision_time(self, value: float):
        """设置碰撞检查总耗时"""
        self._collision_time = value
    
    @property
    def obb_link_data(self) -> List:
        """向后兼容：Link 位姿数据"""
        if isinstance(self.collision_data, LinkDataModel):
            return self.collision_data.link_data
        raise AttributeError(
            "obb_link_data is only available for 'link' model type"
        )
    
    @property
    def obb_link_coll_data(self) -> List:
        """向后兼容：Link 碰撞数据"""
        if isinstance(self.collision_data, LinkDataModel):
            return self.collision_data.link_coll_data
        raise AttributeError(
            "obb_link_coll_data is only available for 'link' model type"
        )
