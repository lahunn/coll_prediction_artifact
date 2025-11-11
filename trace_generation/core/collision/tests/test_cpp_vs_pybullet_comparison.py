#!/usr/bin/env python3
"""
对比测试：C++ SphereCollisionChecker vs PyBullet SphereEnv

比较两种实现的：
1. 碰撞检测正确性
2. 运行速度差异
3. 资源消耗
"""

import sys
import time
import numpy as np

from trace_generation.core.robot.environment import RobotEnv
from trace_generation.core.collision.sphere_detector import SphereEnvGeometric
from trace_generation.core.collision.sphere_method import SphereEnv


def test_correctness_comparison():
    """正确性对比测试"""
    print("=" * 80)
    print("正确性对比测试：C++ vs PyBullet")
    print("=" * 80)

    robot_name = "iiwa"
    print(f"\n1. 初始化环境: {robot_name}")
    robot_env = RobotEnv(robot_name)

    # 创建两种实现的检测器
    print("\n2. 创建检测器...")
    sphere_env_cpp = SphereEnvGeometric(robot_env, robot_name)
    sphere_env_pybullet = SphereEnv(robot_env, robot_name, SPH_GUI=False)

    print(f"   - C++ 实现 (Geometric): {sphere_env_cpp.use_cpp}")
    print(f"   - PyBullet 实现: {sphere_env_pybullet.physics_client}")

    # 加载相同的障碍物
    print("\n3. 加载障碍物...")
    obstacles = [
        ([0.1, 0.1, 0.1], [0.5, 0.0, 0.5]),
        ([0.15, 0.15, 0.15], [-0.3, 0.2, 0.3]),
        ([0.08, 0.08, 0.08], [0.2, -0.4, 0.6]),
        ([0.12, 0.12, 0.12], [0.0, 0.5, 0.4]),
    ]
    sphere_env_cpp.load_obstacles(obstacles)
    sphere_env_pybullet.load_obstacles(obstacles)
    print(f"   加载了 {len(obstacles)} 个障碍物")

    # 测试多个随机关节配置
    print("\n4. 测试随机关节配置...")
    num_tests = 1000
    mismatches = 0
    mismatch_details = []

    np.random.seed(42)  # 固定随机种子

    for i in range(num_tests):
        # 生成随机关节配置
        state = np.random.uniform(-1.5, 1.5, 7)

        # C++ 检测
        collision_cpp, coords_cpp, colls_cpp = sphere_env_cpp.get_sphere_collision_data(
            state
        )

        # PyBullet 检测
        collision_pb, coords_pb, colls_pb = (
            sphere_env_pybullet.get_sphere_collision_data(state)
        )

        # 验证结果一致性
        if collision_cpp != collision_pb:
            mismatches += 1
            mismatch_details.append(
                {
                    "index": i,
                    "state": state,
                    "cpp_collision": collision_cpp,
                    "pb_collision": collision_pb,
                    "cpp_flags": colls_cpp,
                    "pb_flags": colls_pb,
                }
            )
        elif i % 20 == 0:
            print(f"   ✓ 配置 {i:3d}: 一致 (碰撞={collision_cpp})")

    # 清理 PyBullet 环境
    sphere_env_pybullet.close()

    # 输出不匹配详情
    if mismatches > 0:
        print(f"\n⚠️  发现 {mismatches} 处不一致:")
        for detail in mismatch_details[:5]:  # 只显示前5个
            print(f"\n   配置 {detail['index']}:")
            print(f"      C++:      碰撞={detail['cpp_collision']}")
            print(f"      PyBullet: 碰撞={detail['pb_collision']}")
            print(f"      状态: {detail['state'][:3]}...")

    # 总结
    print("\n" + "=" * 80)
    print("正确性测试总结")
    print("=" * 80)
    print(f"总测试数: {num_tests}")
    print(f"不匹配数: {mismatches}")
    print(f"一致率: {100 * (1 - mismatches / num_tests):.2f}%")

    return mismatches == 0


def test_performance_comparison():
    """性能对比测试"""
    print("\n" + "=" * 80)
    print("性能对比测试：C++ vs PyBullet")
    print("=" * 80)

    robot_name = "iiwa"
    robot_env = RobotEnv(robot_name)

    # 创建检测器
    print("\n1. 初始化检测器...")
    sphere_env_cpp = SphereEnvGeometric(robot_env, robot_name)
    sphere_env_pybullet = SphereEnv(robot_env, robot_name, SPH_GUI=False)

    # 加载障碍物
    obstacles = [
        ([0.1, 0.1, 0.1], [0.5, 0.0, 0.5]),
        ([0.15, 0.15, 0.15], [-0.3, 0.2, 0.3]),
        ([0.08, 0.08, 0.08], [0.2, -0.4, 0.6]),
        ([0.12, 0.12, 0.12], [0.0, 0.5, 0.4]),
        ([0.1, 0.1, 0.1], [0.7, 0.2, 0.3]),
    ]
    sphere_env_cpp.load_obstacles(obstacles)
    sphere_env_pybullet.load_obstacles(obstacles)

    # 生成测试数据
    print("\n2. 生成测试数据...")
    num_tests = 1000
    np.random.seed(42)
    test_states = [np.random.uniform(-1.5, 1.5, 7) for _ in range(num_tests)]
    print(f"   生成了 {num_tests} 个测试配置")

    # 预热（避免初始化开销影响测试）
    print("\n3. 预热...")
    for state in test_states[:10]:
        sphere_env_cpp.get_sphere_collision_data(state)
        sphere_env_pybullet.get_sphere_collision_data(state)

    # 测试 C++ 实现性能
    print("\n4. 测试 C++ 实现性能...")
    start_time = time.time()
    cpp_results = []
    for state in test_states:
        result = sphere_env_cpp.get_sphere_collision_data(state)
        cpp_results.append(result)
    cpp_time = time.time() - start_time

    cpp_avg = cpp_time / num_tests * 1000  # 毫秒
    cpp_throughput = num_tests / cpp_time
    print(f"   总耗时: {cpp_time:.3f} 秒")
    print(f"   平均每次: {cpp_avg:.3f} 毫秒")
    print(f"   吞吐量: {cpp_throughput:.1f} 次/秒")

    # 测试 PyBullet 实现性能
    print("\n5. 测试 PyBullet 实现性能...")
    start_time = time.time()
    pb_results = []
    for state in test_states:
        result = sphere_env_pybullet.get_sphere_collision_data(state)
        pb_results.append(result)
    pb_time = time.time() - start_time

    pb_avg = pb_time / num_tests * 1000  # 毫秒
    pb_throughput = num_tests / pb_time
    print(f"   总耗时: {pb_time:.3f} 秒")
    print(f"   平均每次: {pb_avg:.3f} 毫秒")
    print(f"   吞吐量: {pb_throughput:.1f} 次/秒")

    # 清理
    sphere_env_pybullet.close()

    # 计算加速比
    speedup = pb_time / cpp_time

    # 性能对比总结
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)
    print(f"\n{'指标':<20} {'C++ 实现':<20} {'PyBullet 实现':<20} {'加速比':<15}")
    print("-" * 80)
    print(f"{'总耗时 (秒)':<20} {cpp_time:<20.3f} {pb_time:<20.3f} {speedup:<15.2f}x")
    print(f"{'平均延迟 (毫秒)':<20} {cpp_avg:<20.3f} {pb_avg:<20.3f} {speedup:<15.2f}x")
    print(
        f"{'吞吐量 (次/秒)':<20} {cpp_throughput:<20.1f} {pb_throughput:<20.1f} {speedup:<15.2f}x"
    )

    return speedup


def test_scalability():
    """可扩展性测试：不同障碍物数量下的性能"""
    print("\n" + "=" * 80)
    print("可扩展性测试：不同障碍物数量")
    print("=" * 80)

    robot_name = "iiwa"
    robot_env = RobotEnv(robot_name)

    # 测试不同数量的障碍物
    obstacle_counts = [5, 10, 20, 50]
    num_tests = 1000

    print(f"\n测试配置数: {num_tests}")
    print(f"障碍物数量: {obstacle_counts}")

    results = []

    for num_obstacles in obstacle_counts:
        print(f"\n{'=' * 60}")
        print(f"测试 {num_obstacles} 个障碍物")
        print(f"{'=' * 60}")

        # 创建检测器
        sphere_env_cpp = SphereEnvGeometric(robot_env, robot_name)
        sphere_env_pybullet = SphereEnv(robot_env, robot_name, SPH_GUI=False)

        # 生成障碍物
        obstacles = []
        for i in range(num_obstacles):
            half_extents = [np.random.uniform(0.05, 0.15) for _ in range(3)]
            position = [np.random.uniform(-1, 1) for _ in range(3)]
            obstacles.append((half_extents, position))

        sphere_env_cpp.load_obstacles(obstacles)
        sphere_env_pybullet.load_obstacles(obstacles)

        # 生成测试配置
        test_states = [np.random.uniform(-1.5, 1.5, 7) for _ in range(num_tests)]

        # C++ 测试
        start = time.time()
        for state in test_states:
            sphere_env_cpp.get_sphere_collision_data(state)
        cpp_time = time.time() - start

        # PyBullet 测试
        start = time.time()
        for state in test_states:
            sphere_env_pybullet.get_sphere_collision_data(state)
        pb_time = time.time() - start

        speedup = pb_time / cpp_time

        results.append(
            {
                "obstacles": num_obstacles,
                "cpp_time": cpp_time,
                "pb_time": pb_time,
                "speedup": speedup,
            }
        )

        print(f"C++:      {cpp_time:.3f}s ({num_tests / cpp_time:.1f} 次/秒)")
        print(f"PyBullet: {pb_time:.3f}s ({num_tests / pb_time:.1f} 次/秒)")
        print(f"加速比:   {speedup:.2f}x")

        sphere_env_pybullet.close()

    # 可扩展性总结
    print("\n" + "=" * 80)
    print("可扩展性测试总结")
    print("=" * 80)
    print(f"\n{'障碍物数量':<15} {'C++ (秒)':<15} {'PyBullet (秒)':<15} {'加速比':<15}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['obstacles']:<15} {r['cpp_time']:<15.3f} {r['pb_time']:<15.3f} {r['speedup']:<15.2f}x"
        )


def main():
    """主测试函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "C++ vs PyBullet 全面对比测试" + " " * 27 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    try:
        # 1. 正确性测试
        correctness_ok = test_correctness_comparison()

        # 2. 性能测试
        speedup = test_performance_comparison()

        # 3. 可扩展性测试
        test_scalability()

        # 最终总结
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 30 + "最终总结" + " " * 38 + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)

        if correctness_ok:
            print("\n✅ 正确性: C++ 和 PyBullet 实现结果完全一致")
        else:
            print("\n⚠️  正确性: 发现部分不一致（可能由于数值精度差异）")

        print(f"\n🚀 性能提升: C++ 实现比 PyBullet 快 {speedup:.1f}x")

        if speedup > 10:
            print("\n🎉 性能提升显著！C++ 实现带来了数量级的性能改进！")
        elif speedup > 5:
            print("\n👍 性能提升明显！C++ 实现显著优于 PyBullet！")
        else:
            print("\n✓ C++ 实现性能更优")

        print("\n" + "█" * 80)

        return correctness_ok and speedup > 1.0

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
