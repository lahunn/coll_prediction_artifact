#!/usr/bin/env python3
"""
OBB碰撞检测器使用示例

演示如何使用基于几何方法的OBB碰撞检测器
该检测器与collision_check.py保持相同的接口
"""
import numpy as np
import time
from trace_generation.core.collision.obb_detector import OBBCollisionEnv


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试OBB碰撞检测器基本功能")
    print("=" * 60)

    # 创建碰撞检测环境
    env = OBBCollisionEnv("franka")

    # 加载障碍物
    obstacles = [
        ((0.1, 0.1, 0.1), (0.5, 0.0, 0.3)),  # 小障碍物
        ((0.15, 0.15, 0.15), (0.0, 0.4, 0.6)),  # 中等障碍物
        ((0.05, 0.05, 0.05), (-0.3, -0.2, 0.8)),  # 小障碍物
    ]
    obstacle_ids = env.load_obstacles(obstacles)
    print(f"加载了 {len(obstacle_ids)} 个障碍物")

    # 测试多个配置
    test_configs = [
        np.zeros(7),  # 零位姿
        np.array([0.1, 0.2, -0.1, 0.3, 0.0, 0.1, -0.2]),  # 随机配置1
        np.array([-0.2, 0.1, 0.2, -0.1, 0.3, -0.1, 0.1]),  # 随机配置2
    ]

    print("\n测试不同配置的碰撞检测:")
    print("-" * 40)

    for i, config in enumerate(test_configs):
        start_time = time.time()
        is_free, link_coords, link_colls = env._point_in_free_space(config)
        elapsed = time.time() - start_time

        collision_count = sum(1 for c in link_colls if c == 0)
        print(".4f"
              f"碰撞连杆: {collision_count}/{len(link_colls)}")

    # 测试边碰撞检测
    print("\n测试边碰撞检测:")
    print("-" * 40)

    config1 = np.zeros(7)
    config2 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    start_time = time.time()
    edge_free, edge_coords, edge_colls = env._edge_fp(config1, config2)
    elapsed = time.time() - start_time

    print(f"边碰撞检测结果: {'无碰撞' if edge_free else '有碰撞'}")
    print(".4f"
    # 显示统计信息
    stats = env.get_collision_stats()
    print("\n碰撞检测统计:")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(".4f")
        else:
            print(f"{key}: {value}")

    env.close()


def test_interface_compatibility():
    """测试接口兼容性"""
    print("\n" + "=" * 60)
    print("测试接口兼容性")
    print("=" * 60)

    # 创建环境（模拟collision_check.py的使用方式）
    env = OBBCollisionEnv("franka")

    # 模拟加载障碍物（与collision_check.py相同的调用方式）
    obstacles = [
        ((0.2, 0.2, 0.2), (0.3, 0.0, 0.5)),
        ((0.1, 0.1, 0.1), (0.0, 0.4, 0.7)),
    ]
    env.load_obstacles(obstacles)

    # 测试_state_fp方法（collision_check.py中的接口）
    config = np.zeros(7)
    is_free = env._state_fp(config)
    print(f"_state_fp 测试: {'通过' if is_free else '失败'}")

    # 测试in_goal_region方法
    goal_config = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    in_goal = env.in_goal_region(config, goal_config, threshold=0.1)
    print(f"in_goal_region 测试: {'在目标区域' if in_goal else '不在目标区域'}")

    # 测试_iterative_check_segment方法
    config1 = np.zeros(7)
    config2 = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    segment_free = env._iterative_check_segment(config1, config2)
    print(f"_iterative_check_segment 测试: {'可行' if segment_free else '不可行'}")

    env.close()
    print("接口兼容性测试完成")


def benchmark_performance():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)

    env = OBBCollisionEnv("franka")

    # 加载多个障碍物
    obstacles = [((0.1, 0.1, 0.1), (np.random.uniform(-1, 1),
                                     np.random.uniform(-1, 1),
                                     np.random.uniform(0, 1)))
                 for _ in range(10)]
    env.load_obstacles(obstacles)

    # 生成随机配置进行测试
    configs = [np.random.uniform(-0.5, 0.5, 7) for _ in range(100)]

    print("测试100个随机配置...")
    start_time = time.time()

    collision_count = 0
    for config in configs:
        is_free, _, _ = env._point_in_free_space(config)
        if not is_free:
            collision_count += 1

    total_time = time.time() - start_time
    avg_time = total_time / len(configs)

    print(f"总时间: {total_time:.4f} 秒")
    print(".6f")
    print(f"碰撞配置数: {collision_count}/{len(configs)}")

    stats = env.get_collision_stats()
    print(".6f")

    env.close()


if __name__ == "__main__":
    """主测试函数"""
    print("OBB碰撞检测器完整测试")
    print("基于几何方法的机器人碰撞检测实现")
    print("-" * 60)

    try:
        test_basic_functionality()
        test_interface_compatibility()
        benchmark_performance()

        print("\n" + "=" * 60)
        print("所有测试完成！✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()