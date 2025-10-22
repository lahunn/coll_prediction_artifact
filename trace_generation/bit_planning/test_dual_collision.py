#!/usr/bin/env python3
"""测试双重碰撞检测功能"""
import numpy as np
from environment.collision_env import CollisionEnv

def main():
    print("=== 测试双重碰撞检测 ===")
    
    # 初始化环境
    robot_urdf = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf"
    env = CollisionEnv(GUI=False, robot_file=robot_urdf)
    
    # 创建一些障碍物
    obstacles = [
        (np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.0, 0.3])),
        (np.array([0.15, 0.15, 0.1]), np.array([-0.3, 0.4, 0.2])),
    ]
    env.init_obstacle_bodies(len(obstacles), obstacles)
    
    print(f"✓ 创建了 {len(obstacles)} 个障碍物")
    print(f"✓ 主仿真器障碍物数: {len(env.obstacle_body_ids)}")
    print(f"✓ 球体仿真器障碍物数: {len(env.sphere_obstacle_ids)}")
    
    # 测试几个配置
    test_configs = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 初始配置
        np.array([0.5, -0.5, 0.3, -1.0, 0.0, 1.5, 0.0]),  # 随机配置1
        np.array([1.0, 0.5, -0.5, -1.5, 0.5, 1.0, 0.5]),  # 随机配置2
    ]
    
    for i, config in enumerate(test_configs):
        result = env._point_in_free_space(config)
        status = "无碰撞 ✓" if result else "碰撞 ✗"
        print(f"配置 {i+1}: {status}")
    
    # 清理
    env.cleanup_obstacles()
    env.close()
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()
