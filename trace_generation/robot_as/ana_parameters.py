"""
碰撞检测分析参数统一管理模块

该模块统一管理OBB和Sphere碰撞检测的关键参数:
- obb_num: OBB包围盒的数量
- obb_cost: 单个OBB碰撞检测的计算成本
- sphere_num: 球体的数量
- sphere_cost: 单个球体碰撞检测的计算成本

这些参数在整个项目中被多个模块使用，统一管理便于维护和调整
"""

# OBB碰撞检测参数
obb_num = 11  # OBB包围盒数量
obb_cost = 42  # OBB碰撞检测成本

# Sphere碰撞检测参数
sphere_num = 22  # 球体数量
sphere_cost = 18  # 球体碰撞检测成本
