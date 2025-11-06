#!/usr/bin/env python3
"""
性能基准测试：对比 SphereEnv (PyBullet) 和 SphereEnvGeometric (纯几何) 的耗时

测量不同场景下两种方法的碰撞检测性能
"""

import sys
import os
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from sphere_as.sphere_method import SphereEnv
from sphere_as.sphere_method_geometric import SphereEnvGeometric
from robot_as.robot_method import RobotEnv


def generate_random_obstacles(num_obstacles, seed=42):
    """生成随机障碍物"""
    np.random.seed(seed)
    obstacles = []
    for _ in range(num_obstacles):
        half_extents = [
            np.random.uniform(0.1, 0.5),
            np.random.uniform(0.1, 0.5),
            np.random.uniform(0.1, 0.5),
        ]
        base_position = [
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
        ]
        obstacles.append((half_extents, base_position))
    return obstacles


def generate_random_poses(num_poses, dof=7, seed=42):
    """生成随机关节配置"""
    np.random.seed(seed)
    poses = []
    for _ in range(num_poses):
        pose = [np.random.uniform(-np.pi, np.pi) for _ in range(dof)]
        poses.append(pose)
    return poses


def benchmark_method(sphere_env, poses, method_name):
    """
    基准测试单个方法

    Args:
        sphere_env: 球体环境实例
        poses: 关节配置列表
        method_name: 方法名称

    Returns:
        dict: 性能统计信息
    """
    print(f"\n{'=' * 70}")
    print(f"测试 {method_name}")
    print(f"{'=' * 70}")

    # 预热（排除初始化开销）
    warmup_poses = poses[:5]
    print("预热运行...")
    for pose in warmup_poses:
        sphere_env.get_sphere_collision_data(pose)

    # 正式测试
    print(f"正式测试 {len(poses)} 个配置...")
    times = []
    collision_count = 0

    for i, pose in enumerate(poses):
        start_time = time.perf_counter()
        collision, coords, colls = sphere_env.get_sphere_collision_data(pose)
        end_time = time.perf_counter()

        elapsed = (end_time - start_time) * 1000  # 转换为毫秒
        times.append(elapsed)

        if collision:
            collision_count += 1

        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{len(poses)} - 当前耗时: {elapsed:.3f}ms")

    # 统计结果
    times = np.array(times)
    stats = {
        "method": method_name,
        "total_tests": len(poses),
        "collision_count": collision_count,
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "median_time_ms": np.median(times),
        "total_time_s": np.sum(times) / 1000.0,
        "times": times,
    }

    return stats


def print_comparison(stats_pb, stats_geo):
    """打印对比结果"""
    print("\n" + "=" * 70)
    print("性能对比结果")
    print("=" * 70)

    print(f"\n{'指标':<25} {'PyBullet':<20} {'Geometric':<20} {'加速比':<15}")
    print("-" * 80)

    # 平均耗时
    speedup = stats_pb["mean_time_ms"] / stats_geo["mean_time_ms"]
    print(
        f"{'平均耗时 (ms)':<25} {stats_pb['mean_time_ms']:>18.3f} {stats_geo['mean_time_ms']:>18.3f} {speedup:>13.2f}x"
    )

    # 中位数耗时
    speedup = stats_pb["median_time_ms"] / stats_geo["median_time_ms"]
    print(
        f"{'中位数耗时 (ms)':<25} {stats_pb['median_time_ms']:>18.3f} {stats_geo['median_time_ms']:>18.3f} {speedup:>13.2f}x"
    )

    # 最小耗时
    speedup = stats_pb["min_time_ms"] / stats_geo["min_time_ms"]
    print(
        f"{'最小耗时 (ms)':<25} {stats_pb['min_time_ms']:>18.3f} {stats_geo['min_time_ms']:>18.3f} {speedup:>13.2f}x"
    )

    # 最大耗时
    speedup = stats_pb["max_time_ms"] / stats_geo["max_time_ms"]
    print(
        f"{'最大耗时 (ms)':<25} {stats_pb['max_time_ms']:>18.3f} {stats_geo['max_time_ms']:>18.3f} {speedup:>13.2f}x"
    )

    # 标准差
    print(
        f"{'标准差 (ms)':<25} {stats_pb['std_time_ms']:>18.3f} {stats_geo['std_time_ms']:>18.3f} {'-':>15}"
    )

    # 总耗时
    speedup = stats_pb["total_time_s"] / stats_geo["total_time_s"]
    print(
        f"{'总耗时 (s)':<25} {stats_pb['total_time_s']:>18.3f} {stats_geo['total_time_s']:>18.3f} {speedup:>13.2f}x"
    )

    # 碰撞检测数
    print(
        f"{'检测到碰撞数':<25} {stats_pb['collision_count']:>18} {stats_geo['collision_count']:>18} {'-':>15}"
    )

    print("-" * 80)

    # 百分位数
    print("\n耗时分布 (ms):")
    print(f"{'百分位':<15} {'PyBullet':<20} {'Geometric':<20}")
    print("-" * 55)
    for p in [25, 50, 75, 90, 95, 99]:
        pb_val = np.percentile(stats_pb["times"], p)
        geo_val = np.percentile(stats_geo["times"], p)
        print(f"{'P' + str(p):<15} {pb_val:>18.3f} {geo_val:>18.3f}")

    print("\n" + "=" * 70)

    # 总结
    avg_speedup = stats_pb["mean_time_ms"] / stats_geo["mean_time_ms"]
    if avg_speedup > 1:
        print(f"✓ Geometric方法平均快 {avg_speedup:.2f}x")
    else:
        print(f"✗ PyBullet方法平均快 {1 / avg_speedup:.2f}x")

    time_saved = stats_pb["total_time_s"] - stats_geo["total_time_s"]
    if time_saved > 0:
        pct_saved = (time_saved / stats_pb["total_time_s"]) * 100
        print(f"✓ 总共节省时间: {time_saved:.3f}s ({pct_saved:.1f}%)")
    else:
        pct_lost = (abs(time_saved) / stats_pb["total_time_s"]) * 100
        print(f"✗ 总共增加时间: {abs(time_saved):.3f}s ({pct_lost:.1f}%)")

    print("=" * 70)


def run_benchmark(robot_name="franka", num_obstacles=5, num_poses=100):
    """
    运行完整的性能基准测试

    Args:
        robot_name: 机器人名称
        num_obstacles: 障碍物数量
        num_poses: 测试的关节配置数量
    """
    print("=" * 70)
    print("球体碰撞检测性能基准测试")
    print("=" * 70)
    print(f"机器人: {robot_name}")
    print(f"障碍物数量: {num_obstacles}")
    print(f"测试配置数: {num_poses}")
    print("=" * 70)

    # 初始化环境
    print("\n初始化机器人环境...")
    robot_env = RobotEnv(robot_name=robot_name)

    print("初始化 SphereEnv (PyBullet)...")
    sphere_env_pb = SphereEnv(robot_env, robot_name=robot_name, SPH_GUI=False)

    print("初始化 SphereEnvGeometric (纯几何)...")
    sphere_env_geo = SphereEnvGeometric(robot_env, robot_name=robot_name)

    # 生成测试数据
    print(f"\n生成 {num_obstacles} 个随机障碍物...")
    obstacles = generate_random_obstacles(num_obstacles, seed=42)

    print(f"生成 {num_poses} 个随机关节配置...")
    poses = generate_random_poses(num_poses, dof=7, seed=42)

    # 加载障碍物
    print("\n加载障碍物到两个环境...")
    sphere_env_pb.load_obstacles(obstacles)
    sphere_env_geo.load_obstacles(obstacles)

    # 基准测试 PyBullet
    stats_pb = benchmark_method(sphere_env_pb, poses, "PyBullet")

    # 基准测试 Geometric
    stats_geo = benchmark_method(sphere_env_geo, poses, "Geometric")

    # 打印对比结果
    print_comparison(stats_pb, stats_geo)

    # 清理
    print("\n清理环境...")
    sphere_env_pb.close()
    sphere_env_geo.close()
    robot_env.close()

    return stats_pb, stats_geo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="球体碰撞检测性能基准测试")
    parser.add_argument("--robot", type=str, default="franka", help="机器人名称")
    parser.add_argument("--obstacles", type=int, default=5, help="障碍物数量")
    parser.add_argument("--poses", type=int, default=100, help="测试的关节配置数量")

    args = parser.parse_args()

    try:
        stats_pb, stats_geo = run_benchmark(
            robot_name=args.robot, num_obstacles=args.obstacles, num_poses=args.poses
        )

        print("\n测试完成!")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
