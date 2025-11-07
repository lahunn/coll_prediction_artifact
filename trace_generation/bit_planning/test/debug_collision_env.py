#!/usr/bin/env python3
"""
Collision Environment Debug Script

用于调试 OBB 和 Sphere 碰撞检测的差异
可以分别在 GUI 模式下查看两个物理客户端

使用方法：
1. 运行脚本选择模式：
   python debug_collision_env.py --mode obb --gui    # 查看 OBB 碰撞检测（GUI）
   python debug_collision_env.py --mode sphere --gui # 查看 Sphere 碰撞检测（GUI）
   python debug_collision_env.py --mode both         # 同时运行两个客户端（DIRECT模式）

2. 在 GUI 模式下，可以观察：
   - 机器人配置
   - 障碍物位置
   - 碰撞检测结果
   - OBB客户端：红色背景，显示机器人link碰撞检测
   - Sphere客户端：蓝色背景，显示球体近似碰撞检测
"""

# 添加项目路径

import numpy as np
import argparse
import time
from trace_generation.robot_as.collision_check import CollisionEnv
from robot_as.robot_method import RobotEnv

def get_test_obstacles():
    """获取测试用的障碍物配置"""
    return [
        (np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.0, 0.3])),  # 障碍物1
        (np.array([0.15, 0.15, 0.1]), np.array([-0.3, 0.4, 0.2])),  # 障碍物2
        (np.array([0.08, 0.08, 0.2]), np.array([0.2, -0.3, 0.4])),  # 障碍物3
    ]

def get_test_configs():
    """获取测试用的机器人配置"""
    # Franka Panda 有14个关节：7个机器人关节 + 7个手爪关节
    # 手爪关节通常保持在中间位置或特定位置
    return [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 初始配置
        np.array([0.5, -0.5, 0.3, -1.0, 0.0, 1.5, 0.0]),  # 配置1
        np.array([1.0, 0.5, -0.5, -1.5, 0.5, 1.0, 0.5]),  # 配置2
        np.array([-0.8, 0.8, 0.2, 0.5, -0.3, -0.8, 0.1]),  # 配置3
        np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),  # 配置4
    ]

def get_sphere_debug_configs():
    """获取测试用的机器人配置"""
    # Franka Panda 有14个关节：7个机器人关节 + 7个手爪关节
    # 手爪关节通常保持在中间位置或特定位置
    return [
        np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.0, 0.00, 0.00]
        ),  # 初始配置
        np.array(
            [0.5, -0.5, 0.3, -1.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.00, 0.00, 0.0, 0.00, 0.00]
        ),  # 配置1
        np.array([1.0, 0.5, -0.5, -1.5, 0.5, 1.0, 0.5, 0.0, 0.0, 0.00, 0.00]),  # 配置2
        np.array([-0.8, 0.8, 0.2, 0.5, -0.3, -0.8, 0.1, 0.0, 0.0, 0.04, 0.04]),  # 配置3
        np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.04, 0.04]),  # 配置4
    ]

def debug_obb_collision(env, obstacles, configs, is_gui=False):
    """调试 OBB 碰撞检测"""
    print("=== 调试 OBB 碰撞检测 ===")
    print(f"机器人关节数量: {env.robot_env.config_dim}")
    print(
        f"关节限位: lower={env.robot_env.lower_bounds}, upper={env.robot_env.upper_bounds}"
    )

    # 初始化障碍物
    env.init_obstacle_bodies(len(obstacles), obstacles)

    for i, config in enumerate(configs):
        print(f"\n--- 配置 {i + 1}: {config} (长度: {len(config)}) ---")

        # 检查配置长度
        if len(config) != env.robot_env.config_dim:
            print(
                f"⚠️ 配置长度不匹配！期望 {env.robot_env.config_dim}，实际 {len(config)}"
            )
            continue

        # 设置机器人配置
        env.robot_env.set_config(config)

        # 执行碰撞检测
        link_collision, link_colls = env._get_link_collisions()

        print(f"OBB 碰撞结果: {link_collision} (各link: {link_colls})")

        if is_gui:
            print("请在GUI窗口中观察机器人和障碍物的位置，按Enter继续...")
            input()

    print("\nOBB 调试完成")

def debug_sphere_collision(env, obstacles, configs, is_gui=False):
    """调试 Sphere 碰撞检测"""
    print("=== 调试 Sphere 碰撞检测 ===")
    print(f"机器人关节数量: {env.robot_env.config_dim}")

    # 初始化障碍物
    env.init_obstacle_bodies(len(obstacles), obstacles)

    for i, config in enumerate(configs):
        print(f"\n--- 配置 {i + 1}: {config} (长度: {len(config)}) ---")

        # # 检查配置长度
        # if len(config) != env.config_dim:
        #     print(f"⚠️ 配置长度不匹配！期望 {env.config_dim}，实际 {len(config)}")
        #     continue

        # 设置机器人配置并更新球体位置
        sphere_coords = env._update_sphere_positions(config)

        # 执行碰撞检测
        sphere_collision, sphere_colls = env._check_sphere_collision()

        print(f"Sphere 碰撞结果: {sphere_collision} (各sphere: {sphere_colls})")
        print(f"球体坐标: {sphere_coords}")

        if is_gui:
            print("请在GUI窗口中观察球体和障碍物的位置，按Enter继续...")
            input()

    print("\nSphere 调试完成")

def debug_both_collision(env, obstacles, configs):
    """同时调试 OBB 和 Sphere 碰撞检测"""
    print("=== 同时调试 OBB 和 Sphere 碰撞检测 ===")
    print(f"机器人关节数量: {env.robot_env.config_dim}")

    # 初始化障碍物
    env.init_obstacle_bodies(len(obstacles), obstacles)

    for i, config in enumerate(configs):
        print(f"\n--- 配置 {i + 1}: {config} (长度: {len(config)}) ---")

        # 检查配置长度
        if len(config) != env.robot_env.config_dim:
            print(
                f"⚠️ 配置长度不匹配！期望 {env.robot_env.config_dim}，实际 {len(config)}"
            )
            continue

        # 使用 _point_in_free_space 获取完整结果
        is_free, link_coords, link_colls, sphere_coords, sphere_colls = (
            env._point_in_free_space(config)
        )

        # 分别计算碰撞结果
        link_collision = any(coll == 0 for coll in link_colls)
        sphere_collision = any(coll == 0 for coll in sphere_colls)

        print(f"OBB 碰撞: {link_collision} (各link: {link_colls})")
        print(f"Sphere 碰撞: {sphere_collision} (各sphere: {sphere_colls})")
        print(f"综合结果: is_free={is_free}")

        if link_collision != sphere_collision:
            print("⚠️  检测到不一致！")
        else:
            print("✅ 结果一致")

        print(f"Link 坐标: {link_coords}")
        print(f"Sphere 坐标: {sphere_coords}")

        time.sleep(1)  # 短暂暂停

    print("\n双重调试完成")

def main():
    parser = argparse.ArgumentParser(description="Collision Environment Debug Tool")
    parser.add_argument(
        "--mode", choices=["obb", "sphere", "both"], default="both", help="调试模式"
    )
    parser.add_argument(
        "--gui", action="store_true", help="启用GUI模式（仅对单个客户端有效）"
    )

    args = parser.parse_args()

    # 获取测试数据
    obstacles = get_test_obstacles()
    configs = get_test_configs()
    sp_configs = get_sphere_debug_configs()

    print("测试障碍物:")
    for i, (half_extents, position) in enumerate(obstacles):
        print(f"  障碍物{i + 1}: 尺寸{half_extents}, 位置{position}")

    print(f"\n测试配置数量: {len(configs)}")

    # 创建环境
    robot_urdf = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf"

    if args.mode == "obb" and args.gui:
        # OBB GUI模式
        print("启动 OBB 客户端的GUI模式...")
        robot_env = RobotEnv(robot_urdf, OBB_GUI=True)
        env = CollisionEnv(robot_env=robot_env)
    elif args.mode == "sphere" and args.gui:
        # Sphere GUI模式
        print("启动 Sphere 客户端的GUI模式...")
        robot_env = RobotEnv(
            robot_urdf, OBB_GUI=True
        )  # 使用OBB_GUI，因为RobotEnv只支持这个参数
        env = CollisionEnv(robot_env=robot_env)
    else:
        # 非GUI模式或both模式
        print("启动DIRECT模式...")
        robot_env = RobotEnv(robot_urdf, OBB_GUI=False)
        env = CollisionEnv(robot_env=robot_env)

    try:
        if args.mode == "obb":
            debug_obb_collision(
                env, obstacles, configs, args.mode == "obb" and args.gui
            )
        elif args.mode == "sphere":
            debug_sphere_collision(
                env, obstacles, configs, args.mode == "sphere" and args.gui
            )
        else:  # both
            debug_both_collision(env, obstacles, configs)

    except KeyboardInterrupt:
        print("\n用户中断调试")
    except Exception as e:
        print(f"\n调试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        robot_env.close()
        print("调试环境已清理")

if __name__ == "__main__":
    main()
