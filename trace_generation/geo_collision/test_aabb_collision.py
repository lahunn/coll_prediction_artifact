#!/usr/bin/env python3
"""
测试AABB碰撞检测函数
"""

import sys

sys.path.insert(
    0,
    "/home/lanh/project/robot_sim/coll_prediction_artifact/trace_generation/geo_collision",
)

from geometric_collision_detection import (
    Sphere,
    AABB,
    Cuboid,
    sphere_aabb,
    cuboid_aabb,
)
import numpy as np


def test_sphere_aabb():
    """测试球-AABB碰撞检测"""
    print("=" * 60)
    print("测试 sphere_aabb 函数")
    print("=" * 60)

    # 测试1: AABB在原点，球体在远处（不碰撞）
    aabb = AABB(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)  # 中心在原点，边长2
    sphere = Sphere(5.0, 0.0, 0.0, 1.0)  # 球心在(5,0,0)，半径1
    result, cycles = sphere_aabb(sphere, aabb)
    print(f"测试1 - 远距离分离: {result} (期望: 1), cycles: {cycles}")
    assert result == 1, "应该不碰撞"

    # 测试2: 球体与AABB碰撞
    sphere = Sphere(1.5, 0.0, 0.0, 1.0)  # 球心在(1.5,0,0)，半径1
    result, cycles = sphere_aabb(sphere, aabb)
    print(f"测试2 - 碰撞: {result} (期望: 0), cycles: {cycles}")
    assert result == 0, "应该碰撞"

    # 测试3: 球体刚好接触AABB表面
    sphere = Sphere(2.0, 0.0, 0.0, 1.0)  # 球心在(2,0,0)，半径1，刚好接触
    result, cycles = sphere_aabb(sphere, aabb)
    print(f"测试3 - 刚好接触: {result} (期望: 0), cycles: {cycles}")
    assert result == 0, "刚好接触算碰撞"

    # 测试4: 球体完全包含AABB
    sphere = Sphere(0.0, 0.0, 0.0, 5.0)  # 球心在原点，半径5
    result, cycles = sphere_aabb(sphere, aabb)
    print(f"测试4 - 球体包含AABB: {result} (期望: 0), cycles: {cycles}")
    assert result == 0, "应该碰撞"

    # 测试5: 球体在AABB内部
    sphere = Sphere(0.0, 0.0, 0.0, 0.5)  # 球心在原点，半径0.5
    result, cycles = sphere_aabb(sphere, aabb)
    print(f"测试5 - 球体在AABB内部: {result} (期望: 0), cycles: {cycles}")
    assert result == 0, "应该碰撞"

    # 测试6: 球体在角落附近（测试多轴超出）
    sphere = Sphere(2.0, 2.0, 2.0, 1.0)  # 球心在(2,2,2)，半径1
    result, cycles = sphere_aabb(sphere, aabb)
    print(f"测试6 - 角落附近: {result} (期望: 1), cycles: {cycles}")
    # 距离 = sqrt((2-1)^2 + (2-1)^2 + (2-1)^2) = sqrt(3) ≈ 1.732 > 1，不碰撞
    assert result == 1, "应该不碰撞"

    # 测试7: 球体更接近角落
    sphere = Sphere(1.8, 1.8, 1.8, 1.0)  # 球心在(1.8,1.8,1.8)，半径1
    result, cycles = sphere_aabb(sphere, aabb)
    print(f"测试7 - 更接近角落: {result} (期望: 1), cycles: {cycles}")
    # 距离 = sqrt((1.8-1)^2 * 3) = sqrt(1.92) ≈ 1.386 > 1，但接近
    assert result == 1, "应该不碰撞"

    # 测试8: 球体与角落碰撞
    sphere = Sphere(1.5, 1.5, 1.5, 1.0)  # 球心在(1.5,1.5,1.5)，半径1
    result, cycles = sphere_aabb(sphere, aabb)
    print(f"测试8 - 与角落碰撞: {result} (期望: 0), cycles: {cycles}")
    # 距离 = sqrt((1.5-1)^2 * 3) = sqrt(0.75) ≈ 0.866 < 1，碰撞
    assert result == 0, "应该碰撞"

    print("\n✓ 所有 sphere_aabb 测试通过!\n")


def test_aabb_cuboid():
    """测试AABB-OBB碰撞检测"""
    print("=" * 60)
    print("测试 aabb_cuboid 函数")
    print("=" * 60)

    # 测试1: 两个轴对齐的盒子，分离
    aabb = AABB(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    # OBB也是轴对齐的，中心在(5,0,0)
    cuboid = Cuboid(
        5.0,
        0.0,
        0.0,
        (1.0, 0.0, 0.0, 1.0),  # X轴，半长1
        (0.0, 1.0, 0.0, 1.0),  # Y轴，半长1
        (0.0, 0.0, 1.0, 1.0),
    )  # Z轴，半长1
    result, cycles = cuboid_aabb(cuboid, aabb)
    print(f"测试1 - 轴对齐分离: {result} (期望: 1), cycles: {cycles}")
    assert result == 1, "应该不碰撞"

    # 测试2: 两个轴对齐的盒子，碰撞
    cuboid = Cuboid(
        1.5, 0.0, 0.0, (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    )
    result, cycles = cuboid_aabb(cuboid, aabb)
    print(f"测试2 - 轴对齐碰撞: {result} (期望: 0), cycles: {cycles}")
    assert result == 0, "应该碰撞"

    # 测试3: OBB旋转45度（绕Z轴）
    cos45 = np.cos(np.pi / 4)
    sin45 = np.sin(np.pi / 4)
    cuboid = Cuboid(
        0.0,
        0.0,
        0.0,
        (cos45, sin45, 0.0, 1.0),  # 旋转后的X轴
        (-sin45, cos45, 0.0, 1.0),  # 旋转后的Y轴
        (0.0, 0.0, 1.0, 1.0),
    )  # Z轴不变
    result, cycles = cuboid_aabb(cuboid, aabb)
    print(f"测试3 - OBB旋转45度（重叠）: {result} (期望: 0), cycles: {cycles}")
    assert result == 0, "应该碰撞"

    # 测试4: OBB旋转后分离
    cuboid = Cuboid(
        3.0,
        3.0,
        0.0,
        (cos45, sin45, 0.0, 0.5),
        (-sin45, cos45, 0.0, 0.5),
        (0.0, 0.0, 1.0, 0.5),
    )
    result, cycles = cuboid_aabb(cuboid, aabb)
    print(f"测试4 - OBB旋转后分离: {result} (期望: 1), cycles: {cycles}")
    assert result == 1, "应该不碰撞"

    # 测试5: 测试wrapper函数
    result_wrapper, cycles_wrapper = cuboid_aabb(cuboid, aabb)
    print(f"测试5 - wrapper函数: {result_wrapper} (期望: 1), cycles: {cycles_wrapper}")
    assert result_wrapper == result, "wrapper应该返回相同结果"

    # 测试6: OBB完全包含AABB
    cuboid = Cuboid(
        0.0, 0.0, 0.0, (1.0, 0.0, 0.0, 3.0), (0.0, 1.0, 0.0, 3.0), (0.0, 0.0, 1.0, 3.0)
    )
    result, cycles = cuboid_aabb(cuboid, aabb)
    print(f"测试6 - OBB包含AABB: {result} (期望: 0), cycles: {cycles}")
    assert result == 0, "应该碰撞"

    # 测试7: 边缘接触
    cuboid = Cuboid(
        2.0, 0.0, 0.0, (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    )
    result, cycles = cuboid_aabb(cuboid, aabb)
    print(f"测试7 - 边缘接触: {result} (期望: 0), cycles: {cycles}")
    assert result == 0, "边缘接触算碰撞"

    print("\n✓ 所有 aabb_cuboid 测试通过!\n")


if __name__ == "__main__":
    test_sphere_aabb()
    test_aabb_cuboid()
    print("=" * 60)
    print("✓✓✓ 所有AABB碰撞检测测试通过! ✓✓✓")
    print("=" * 60)
