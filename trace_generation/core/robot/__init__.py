"""
Robot modeling and environment management.

为了向后兼容，从 collision 目录重导出碰撞检测类
"""

# 向后兼容导出
# from trace_generation.core.collision.collision_env import CollisionEnv
# from trace_generation.core.collision.data_manager import CollisionDataManager

__all__ = [
    # 'CollisionEnv',
    # 'CollisionDataManager',
]
