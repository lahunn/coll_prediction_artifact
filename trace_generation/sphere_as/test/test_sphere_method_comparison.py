#!/usr/bin/env python3
"""
对比测试 SphereEnv (PyBullet) 和 SphereEnvGeometric (纯几何)

在相同的机器人、障碍物环境和关节配置下，对比两种实现的碰撞检测结果是否一致
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from sphere_as.sphere_method import SphereEnv
from sphere_as.sphere_method_geometric import SphereEnvGeometric
from robot_as.robot_method import RobotEnv


def generate_random_obstacles(num_obstacles, seed=42):
    """
    生成随机障碍物

    Args:
        num_obstacles: 障碍物数量
        seed: 随机种子

    Returns:
        障碍物列表 [(halfExtents, basePosition), ...]
    """
    np.random.seed(seed)
    obstacles = []

    for _ in range(num_obstacles):
        # 随机半边长 (0.1-0.5)
        half_extents = [
            np.random.uniform(0.1, 0.5),
            np.random.uniform(0.1, 0.5),
            np.random.uniform(0.1, 0.5),
        ]

        # 随机位置 (-2 到 2 范围内)
        base_position = [
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
        ]

        obstacles.append((half_extents, base_position))

    return obstacles


def generate_random_poses(num_poses, dof=7, seed=42):
    """
    生成随机关节配置

    Args:
        num_poses: 配置数量
        dof: 自由度数量
        seed: 随机种子

    Returns:
        关节配置列表
    """
    np.random.seed(seed)
    poses = []

    for _ in range(num_poses):
        # 生成 [-pi, pi] 范围内的随机关节角度
        pose = [np.random.uniform(-np.pi, np.pi) for _ in range(dof)]
        poses.append(pose)

    return poses


def compare_collision_results(result_pybullet, result_geometric, pose_idx):
    """
    比较两种方法的碰撞检测结果

    Args:
        result_pybullet: PyBullet版本结果 (collision, coords, colls)
        result_geometric: 几何版本结果 (collision, coords, colls)
        pose_idx: 当前测试的pose索引

    Returns:
        bool: 结果是否一致
    """
    collision_pb, coords_pb, colls_pb = result_pybullet
    collision_geo, coords_geo, colls_geo = result_geometric

    # 比较总体碰撞状态
    if collision_pb != collision_geo:
        print(f"  ✗ Pose {pose_idx}: 总体碰撞状态不一致")
        print(f"    PyBullet: {collision_pb}, Geometric: {collision_geo}")
        return False

    # 比较各球体碰撞状态
    if len(colls_pb) != len(colls_geo):
        print(f"  ✗ Pose {pose_idx}: 球体数量不一致")
        print(f"    PyBullet: {len(colls_pb)}, Geometric: {len(colls_geo)}")
        return False

    # 检查每个球体的碰撞状态
    mismatch_count = 0
    for i, (coll_pb, coll_geo) in enumerate(zip(colls_pb, colls_geo)):
        if coll_pb != coll_geo:
            mismatch_count += 1
            if mismatch_count <= 3:  # 只打印前3个不匹配
                print(
                    f"  ⚠ Pose {pose_idx}, 球体 {i}: PyBullet={coll_pb}, Geometric={coll_geo}"
                )

    if mismatch_count > 0:
        print(
            f"  ✗ Pose {pose_idx}: {mismatch_count}/{len(colls_pb)} 个球体碰撞状态不一致"
        )
        return False

    # 比较球体坐标 (允许小误差)
    coords_diff = 0.0
    for coord_pb, coord_geo in zip(coords_pb, coords_geo):
        for val_pb, val_geo in zip(coord_pb, coord_geo):
            coords_diff += abs(val_pb - val_geo)

    avg_diff = coords_diff / (len(coords_pb) * 4) if coords_pb else 0.0

    if avg_diff > 1e-5:
        print(f"  ⚠ Pose {pose_idx}: 坐标差异较大 (平均: {avg_diff:.6f})")

    return True


def run_comparison_test(robot_name="panda", num_obstacles=5, num_poses=20):
    """
    运行对比测试

    Args:
        robot_name: 机器人名称
        num_obstacles: 障碍物数量
        num_poses: 测试的关节配置数量
    """
    print("=" * 70)
    print("SphereEnv vs SphereEnvGeometric 对比测试")
    print("=" * 70)
    print(f"机器人: {robot_name}")
    print(f"障碍物数量: {num_obstacles}")
    print(f"测试配置数: {num_poses}")
    print("=" * 70)

    # 初始化机器人环境
    print("\n[1/5] 初始化机器人环境...")
    robot_env = RobotEnv(robot_name=robot_name)

    # 初始化两个球体环境
    print("[2/5] 初始化 SphereEnv (PyBullet)...")
    sphere_env_pb = SphereEnv(robot_env, robot_name=robot_name, SPH_GUI=False)

    print("[3/5] 初始化 SphereEnvGeometric (纯几何)...")
    sphere_env_geo = SphereEnvGeometric(robot_env, robot_name=robot_name)

    # 生成相同的随机障碍物
    print(f"[4/5] 生成 {num_obstacles} 个随机障碍物...")
    obstacles = generate_random_obstacles(num_obstacles, seed=42)

    print("障碍物信息:")
    for i, (half_ext, pos) in enumerate(obstacles):
        print(f"  障碍物 {i}: size={half_ext}, pos={pos}")

    # 加载障碍物到两个环境
    sphere_env_pb.load_obstacles(obstacles)
    sphere_env_geo.load_obstacles(obstacles)

    # 生成随机关节配置
    print(f"\n[5/5] 生成 {num_poses} 个随机关节配置并进行对比...")
    poses = generate_random_poses(num_poses, dof=7, seed=42)

    # 对比测试
    consistent_count = 0
    inconsistent_count = 0

    print("\n开始碰撞检测对比:")
    print("-" * 70)

    for i, pose in enumerate(poses):
        # PyBullet版本
        result_pb = sphere_env_pb.get_sphere_collision_data(pose)

        # 几何版本
        result_geo = sphere_env_geo.get_sphere_collision_data(pose)

        # 比较结果
        is_consistent = compare_collision_results(result_pb, result_geo, i)

        if is_consistent:
            consistent_count += 1
            collision_status = "碰撞" if result_pb[0] else "无碰撞"
            print(f"  ✓ Pose {i:2d}: 一致 ({collision_status})")
        else:
            inconsistent_count += 1

    # 输出统计结果
    print("-" * 70)
    print("\n测试结果统计:")
    print(
        f"  一致: {consistent_count}/{num_poses} ({consistent_count / num_poses * 100:.1f}%)"
    )
    print(
        f"  不一致: {inconsistent_count}/{num_poses} ({inconsistent_count / num_poses * 100:.1f}%)"
    )

    # 清理环境
    print("\n清理环境...")
    sphere_env_pb.close()
    sphere_env_geo.close()
    robot_env.close()

    print("=" * 70)
    if inconsistent_count == 0:
        print("✓✓✓ 所有测试通过! 两种实现完全一致 ✓✓✓")
        print("=" * 70)
        return True
    else:
        print(f"✗✗✗ 发现 {inconsistent_count} 处不一致 ✗✗✗")
        print("=" * 70)
        return False


def run_detailed_single_test():
    """运行单个详细测试用例，便于调试"""
    print("=" * 70)
    print("详细单例测试")
    print("=" * 70)

    robot_name = "franka"
    robot_env = RobotEnv(robot_name=robot_name)

    sphere_env_pb = SphereEnv(robot_env, robot_name=robot_name, SPH_GUI=False)
    sphere_env_geo = SphereEnvGeometric(robot_env, robot_name=robot_name)

    # 创建一个简单障碍物
    obstacles = [
        ([0.5, 0.5, 0.5], [1.0, 0.0, 0.5])  # 单个障碍物
    ]

    sphere_env_pb.load_obstacles(obstacles)
    sphere_env_geo.load_obstacles(obstacles)

    # 测试零位姿
    pose = [0.0] * 7

    print(f"\n测试关节配置: {pose}")
    print(f"障碍物: size={obstacles[0][0]}, pos={obstacles[0][1]}")

    result_pb = sphere_env_pb.get_sphere_collision_data(pose)
    result_geo = sphere_env_geo.get_sphere_collision_data(pose)

    collision_pb, coords_pb, colls_pb = result_pb
    collision_geo, coords_geo, colls_geo = result_geo

    print("\nPyBullet 结果:")
    print(f"  总体碰撞: {collision_pb}")
    print(f"  球体数量: {len(coords_pb)}")
    print(f"  碰撞球体数: {colls_pb.count(0)}")

    print("\nGeometric 结果:")
    print(f"  总体碰撞: {collision_geo}")
    print(f"  球体数量: {len(coords_geo)}")
    print(f"  碰撞球体数: {colls_geo.count(0)}")

    print("\n详细对比:")
    for i in range(min(len(colls_pb), len(colls_geo))):
        status = "✓" if colls_pb[i] == colls_geo[i] else "✗"
        print(f"  {status} 球体 {i}: PyBullet={colls_pb[i]}, Geometric={colls_geo[i]}")

    sphere_env_pb.close()
    sphere_env_geo.close()
    robot_env.close()

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="对比测试 SphereEnv 和 SphereEnvGeometric"
    )
    parser.add_argument("--robot", type=str, default="franka", help="机器人名称")
    parser.add_argument("--obstacles", type=int, default=5, help="障碍物数量")
    parser.add_argument("--poses", type=int, default=20, help="测试的关节配置数量")
    parser.add_argument("--detailed", action="store_true", help="运行详细单例测试")

    args = parser.parse_args()

    try:
        if args.detailed:
            run_detailed_single_test()
        else:
            success = run_comparison_test(
                robot_name=args.robot,
                num_obstacles=args.obstacles,
                num_poses=args.poses,
            )
            sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
