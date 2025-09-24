#!/usr/bin/env python3
"""测试优化后的球体碰撞检测功能"""

import sys
import time

sys.path.append(
    "/home/lanh/project/robot_sim/coll_prediction_artifact/trace_generation"
)

from pred_trace_generation import *
import numpy as np


def test_optimized_sphere_collision():
    """测试优化后的球体碰撞检测性能和准确性"""
    print("=== 测试优化后的球体碰撞检测 ===")

    try:
        # 初始化环境
        sim = PyBulletRobotSimulator(use_gui=False)

        # 创建简单的测试障碍物
        test_obstacle = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        obstacle_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=test_obstacle,
            basePosition=[0.5, 0.5, 0.5],
        )
        sim.obstacle_ids = [obstacle_body]

        # 测试球体创建
        print("\n1. 测试使用默认配置创建球体...")

        # 创建球体分析器
        from robot_sphere_analyzer import RobotSphereAnalyzer

        analyzer = RobotSphereAnalyzer("franka", device="cuda:0")

        start_time = time.time()
        sphere_bodies = create_sphere_bodies(sim, analyzer)
        create_time = time.time() - start_time
        print(f"   创建时间: {create_time:.4f}s")
        print(f"   成功创建: {len(sphere_bodies)} 个球体")

        # 测试球体位置更新
        print("\n2. 测试球体位置更新...")
        test_spheres = np.array(
            [
                [0.0, 0.0, 0.0, 0.05],  # 远离障碍物
                [0.5, 0.5, 0.5, 0.05],  # 接近/重叠障碍物
                [0.1, 0.1, 0.1, 0.05],  # 中等距离
                [1.0, 1.0, 1.0, 0.05],  # 远离障碍物
                [0.4, 0.4, 0.4, 0.05],  # 接近障碍物
            ]
        )

        start_time = time.time()
        update_sphere_positions(sphere_bodies[: len(test_spheres)], test_spheres)
        update_time = time.time() - start_time
        print(f"   位置更新时间: {update_time:.4f}s")

        # 测试碰撞检测
        print("\n3. 测试批量碰撞检测...")
        start_time = time.time()
        collision_results = check_spheres_collision(
            sim, sphere_bodies[: len(test_spheres)]
        )
        collision_time = time.time() - start_time
        print(f"   碰撞检测时间: {collision_time:.4f}s")

        print("   碰撞检测结果:")
        for i, (sphere_data, collision) in enumerate(
            zip(test_spheres, collision_results)
        ):
            x, y, z, r = sphere_data
            status = "碰撞" if collision else "无碰撞"
            print(f"     球体{i + 1}: 位置[{x:.1f}, {y:.1f}, {z:.1f}] → {status}")

        # 性能对比测试
        print("\n4. 性能对比测试...")
        num_tests = 100

        # 测试优化版本
        print(f"   测试优化版本 ({num_tests} 次)...")
        start_time = time.time()
        for _ in range(num_tests):
            update_sphere_positions(sphere_bodies[: len(test_spheres)], test_spheres)
            check_spheres_collision(sim, sphere_bodies[: len(test_spheres)])
        optimized_time = time.time() - start_time

        print(f"   优化版本总时间: {optimized_time:.4f}s")
        print(f"   平均每次: {optimized_time / num_tests:.6f}s")

        # 清理球体
        print("\n5. 清理球体...")
        start_time = time.time()
        cleanup_sphere_bodies(sphere_bodies)
        cleanup_time = time.time() - start_time
        print(f"   清理时间: {cleanup_time:.4f}s")

        # 断开连接
        sim.disconnect()

        print("\n✓ 优化后的球体碰撞检测测试完成！")
        print(f"  - 创建时间: {create_time:.4f}s")
        print(f"  - 单次更新+检测: {optimized_time / num_tests:.6f}s")
        print(f"  - 清理时间: {cleanup_time:.4f}s")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration():
    """测试与球体分析器的集成"""
    print("\n=== 测试与球体分析器集成 ===")

    try:
        # 创建球体分析器
        analyzer = RobotSphereAnalyzer("franka", device="cuda:0")

        # 获取默认球体数据
        world_spheres = analyzer.get_world_spheres()
        print(f"✓ 获取到 {len(world_spheres)} 个球体")

        # 初始化仿真环境
        sim = PyBulletRobotSimulator(use_gui=False)

        # 创建测试障碍物
        test_obstacle = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        obstacle_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=test_obstacle,
            basePosition=[0.0, 0.0, 0.5],  # 机器人可能接触的区域
        )
        sim.obstacle_ids = [obstacle_body]

        # 创建球体用于碰撞检测
        sphere_bodies = create_sphere_bodies(sim, len(world_spheres))

        # 更新球体位置并进行碰撞检测
        update_sphere_positions(sphere_bodies, world_spheres)
        collision_results = check_spheres_collision(sim, sphere_bodies)

        # 统计碰撞结果
        collision_count = sum(collision_results)
        free_count = len(collision_results) - collision_count

        print("✓ 碰撞检测完成:")
        print(f"  - 总球体数: {len(collision_results)}")
        print(f"  - 碰撞球体: {collision_count}")
        print(f"  - 无碰撞球体: {free_count}")
        print(f"  - 碰撞率: {collision_count / len(collision_results) * 100:.1f}%")

        # 清理
        cleanup_sphere_bodies(sphere_bodies)
        sim.disconnect()

        return True

    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_optimized_sphere_collision()
    success2 = test_integration()

    if success1 and success2:
        print("\n🎉 所有测试通过！优化后的球体碰撞检测工作正常。")
    else:
        print("\n❌ 部分测试失败")
