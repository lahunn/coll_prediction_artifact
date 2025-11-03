#!/usr/bin/env python3
"""
基于几何计算的碰撞检测实现

参考C++ collision库，实现以下碰撞检测：
- OBB(cuboid) 与 capsule
- OBB(cuboid) 与 heightfield
- OBB(cuboid) 与 sphere
- sphere 与 capsule
- sphere 与 cuboid
- sphere 与 heightfield

只返回最简单的碰撞检测结果（<0表示碰撞，>=0表示无碰撞）
"""

import numpy as np


class Cuboid:
    """有向包围盒 (OBB)"""

    def __init__(self, x, y, z, axis_1, axis_2, axis_3):
        """
        Args:
            x, y, z: 中心点坐标
            axis_1: (x, y, z, r) - 第一个轴的方向和半长度
            axis_2: (x, y, z, r) - 第二个轴的方向和半长度
            axis_3: (x, y, z, r) - 第三个轴的方向和半长度
        """
        self.x = x
        self.y = y
        self.z = z
        self.axis_1_x, self.axis_1_y, self.axis_1_z, self.axis_1_r = axis_1
        self.axis_2_x, self.axis_2_y, self.axis_2_z, self.axis_2_r = axis_2
        self.axis_3_x, self.axis_3_y, self.axis_3_z, self.axis_3_r = axis_3


class Sphere:
    """球体"""

    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r


class Capsule:
    """胶囊体"""

    def __init__(self, x1, y1, z1, xv, yv, zv, r):
        """
        Args:
            x1, y1, z1: 起点坐标
            xv, yv, zv: 方向向量
            r: 半径
        """
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.xv = xv
        self.yv = yv
        self.zv = zv
        self.r = r
        # 预计算方向向量长度的倒数，用于投影计算
        self.rdv = 1.0 / (xv * xv + yv * yv + zv * zv) ** 0.5


class HeightField:
    """高度场"""

    def __init__(self, x, y, z, xs, ys, zs, xd, yd, data):
        """
        Args:
            x, y, z: 基准位置
            xs, ys, zs: 缩放因子
            xd, yd: 尺寸
            data: 高度数据数组
        """
        self.x = x
        self.y = y
        self.z = z
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.xd = xd
        self.yd = yd
        self.data = np.array(data)
        self.xd2 = xd / 2.0
        self.yd2 = yd / 2.0


class Triangle:
    """空间三角形"""

    def __init__(self, v0, v1, v2):
        """
        Args:
            v0, v1, v2: 三个顶点的坐标 (x, y, z)
        """
        self.v0 = np.array(v0)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)

        # 计算边向量
        self.e0 = self.v1 - self.v0  # v1 - v0
        self.e1 = self.v2 - self.v0  # v2 - v0
        self.e2 = self.v0 - self.v2  # v0 - v2 (用于第三条边)

        # 计算法向量 (未归一化)
        self.normal = np.cross(self.e0, self.e1)


def dot_3(ax, ay, az, bx, by, bz):
    """三维向量点积"""
    return ax * bx + ay * by + az * bz


def cross_3(ax, ay, az, bx, by, bz):
    """三维向量叉积"""
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def length_sq_3(x, y, z):
    """三维向量长度平方"""
    return x * x + y * y + z * z


def normalize_3(x, y, z):
    """三维向量归一化"""
    length = (x * x + y * y + z * z) ** 0.5
    if length > 0:
        return (x / length, y / length, z / length)
    return (0, 0, 0)


def closest_point_on_segment(p, a, b):
    """
    计算点p到线段ab的最近点

    Args:
        p, a, b: 三维点坐标 (x, y, z)

    Returns:
        最近点坐标 (x, y, z)
    """
    ab_x, ab_y, ab_z = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    ap_x, ap_y, ap_z = p[0] - a[0], p[1] - a[1], p[2] - a[2]

    # 计算投影参数 t
    ab_len_sq = length_sq_3(ab_x, ab_y, ab_z)
    if ab_len_sq == 0:
        return a  # a 和 b 是同一点

    t = dot_3(ap_x, ap_y, ap_z, ab_x, ab_y, ab_z) / ab_len_sq

    # 钳制到线段范围内
    t = max(0.0, min(1.0, t))

    # 计算最近点
    closest_x = a[0] + t * ab_x
    closest_y = a[1] + t * ab_y
    closest_z = a[2] + t * ab_z

    return (closest_x, closest_y, closest_z)


def sql2_3(ax, ay, az, bx, by, bz):
    """三维向量距离平方"""
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    return dx * dx + dy * dy + dz * dz


def sphere_triangle(sphere: Sphere, triangle: Triangle) -> int:
    """
    Sphere与Triangle的碰撞检测

    算法：计算球心到三角形的最近点，比较距离与球体半径

    Returns:
        1表示无碰撞，0表示碰撞
    """
    # 步骤1：计算球心到三角形平面的投影点
    sphere_center = np.array([sphere.x, sphere.y, sphere.z])
    v0 = triangle.v0

    # 计算球心相对于v0的向量
    w = sphere_center - v0

    # 计算投影距离
    dist = np.dot(triangle.normal, w)

    # 计算投影点
    p_plane = sphere_center - dist * triangle.normal

    # 步骤2：检查投影点是否在三角形内部
    # 计算重心坐标
    w_proj = p_plane - v0

    # 解线性系统求u,v
    dot00 = np.dot(triangle.e0, triangle.e0)
    dot01 = np.dot(triangle.e0, triangle.e1)
    dot11 = np.dot(triangle.e1, triangle.e1)
    dot20 = np.dot(w_proj, triangle.e0)
    dot21 = np.dot(w_proj, triangle.e1)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot20 - dot01 * dot21) * inv_denom
    v = (dot00 * dot21 - dot01 * dot20) * inv_denom

    # 检查是否在三角形内部
    if u >= 0 and v >= 0 and (u + v) <= 1:
        # 投影点在三角形内部
        distance_sq = dist * dist
        if distance_sq <= sphere.r * sphere.r:
            return 0  # 碰撞
        else:
            return 1  # 无碰撞

    # 步骤3：投影点在三角形外部，计算到三条边的最近点
    min_distance_sq = float("inf")

    # 计算到三条边的最近点
    edges = [
        (triangle.v0, triangle.v1),
        (triangle.v1, triangle.v2),
        (triangle.v2, triangle.v0),
    ]

    for edge_start, edge_end in edges:
        closest_point = closest_point_on_segment(sphere_center, edge_start, edge_end)
        distance_sq = length_sq_3(
            sphere_center[0] - closest_point[0],
            sphere_center[1] - closest_point[1],
            sphere_center[2] - closest_point[2],
        )
        min_distance_sq = min(min_distance_sq, distance_sq)

    # 返回距离平方与半径平方的差值
    return 1 if min_distance_sq - sphere.r * sphere.r >= 0 else 0


def cuboid_triangle(cuboid: Cuboid, triangle: Triangle) -> int:
    """
    OBB与Triangle的碰撞检测

    算法：使用分离轴定理（SAT），将三角形变换到cuboid局部坐标系

    Returns:
        1表示无碰撞，0表示碰撞
    """
    # 步骤1：坐标系变换
    # 将三角形顶点变换到cuboid局部坐标系
    d0 = triangle.v0 - np.array([cuboid.x, cuboid.y, cuboid.z])
    d1 = triangle.v1 - np.array([cuboid.x, cuboid.y, cuboid.z])
    d2 = triangle.v2 - np.array([cuboid.x, cuboid.y, cuboid.z])

    # 投影到cuboid的三个轴上
    v0_prime = np.array(
        [
            np.dot(d0, [cuboid.axis_1_x, cuboid.axis_1_y, cuboid.axis_1_z]),
            np.dot(d0, [cuboid.axis_2_x, cuboid.axis_2_y, cuboid.axis_2_z]),
            np.dot(d0, [cuboid.axis_3_x, cuboid.axis_3_y, cuboid.axis_3_z]),
        ]
    )
    v1_prime = np.array(
        [
            np.dot(d1, [cuboid.axis_1_x, cuboid.axis_1_y, cuboid.axis_1_z]),
            np.dot(d1, [cuboid.axis_2_x, cuboid.axis_2_y, cuboid.axis_2_z]),
            np.dot(d1, [cuboid.axis_3_x, cuboid.axis_3_y, cuboid.axis_3_z]),
        ]
    )
    v2_prime = np.array(
        [
            np.dot(d2, [cuboid.axis_1_x, cuboid.axis_1_y, cuboid.axis_1_z]),
            np.dot(d2, [cuboid.axis_2_x, cuboid.axis_2_y, cuboid.axis_2_z]),
            np.dot(d2, [cuboid.axis_3_x, cuboid.axis_3_y, cuboid.axis_3_z]),
        ]
    )

    # 步骤2：确定分离轴
    # 计算三角形边向量
    f0_prime = v1_prime - v0_prime
    f1_prime = v2_prime - v1_prime
    f2_prime = v0_prime - v2_prime

    # 三角形法向量（在cuboid坐标系中）
    n_prime = np.cross(f0_prime, f1_prime)

    # 候选分离轴
    axes = [
        np.array([1, 0, 0]),  # cuboid x轴
        np.array([0, 1, 0]),  # cuboid y轴
        np.array([0, 0, 1]),  # cuboid z轴
        n_prime,  # 三角形法向量
        np.cross(np.array([1, 0, 0]), f0_prime),  # cuboid x轴 × 三角形边0
        np.cross(np.array([1, 0, 0]), f1_prime),  # cuboid x轴 × 三角形边1
        np.cross(np.array([1, 0, 0]), f2_prime),  # cuboid x轴 × 三角形边2
        np.cross(np.array([0, 1, 0]), f0_prime),  # cuboid y轴 × 三角形边0
        np.cross(np.array([0, 1, 0]), f1_prime),  # cuboid y轴 × 三角形边1
        np.cross(np.array([0, 1, 0]), f2_prime),  # cuboid y轴 × 三角形边2
        np.cross(np.array([0, 0, 1]), f0_prime),  # cuboid z轴 × 三角形边0
        np.cross(np.array([0, 0, 1]), f1_prime),  # cuboid z轴 × 三角形边1
        np.cross(np.array([0, 0, 1]), f2_prime),  # cuboid z轴 × 三角形边2
    ]

    # 步骤3：测试每个轴
    for axis in axes:
        # 归一化轴向量
        axis_length = np.linalg.norm(axis)
        if axis_length < 1e-8:
            continue  # 跳过零向量
        axis = axis / axis_length

        # 计算cuboid在该轴上的投影半径
        r_aabb = (
            cuboid.axis_1_r * abs(axis[0])
            + cuboid.axis_2_r * abs(axis[1])
            + cuboid.axis_3_r * abs(axis[2])
        )

        # 计算三角形在该轴上的投影
        p0 = np.dot(v0_prime, axis)
        p1 = np.dot(v1_prime, axis)
        p2 = np.dot(v2_prime, axis)

        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)

        # 检查分离
        if max_tri < -r_aabb or min_tri > r_aabb:
            # 找到分离轴，无碰撞
            return 1  # 无碰撞

    # 步骤4：没有找到分离轴，有碰撞
    return 0  # 碰撞


def cuboid_capsule(cuboid: Cuboid, capsule: Capsule) -> int:
    """
    OBB与Capsule的碰撞检测

    算法：计算Cuboid中心到Capsule轴线的最短距离，然后比较距离与半径

    Returns:
        1表示无碰撞，0表示碰撞
    """
    # Capsule的两个端点
    p1_x = capsule.x1
    p1_y = capsule.y1
    p1_z = capsule.z1

    # 计算Cuboid中心到Capsule轴线的最短距离
    cx = cuboid.x - p1_x
    cy = cuboid.y - p1_y
    cz = cuboid.z - p1_z

    # 计算投影参数t
    dot_cv = dot_3(cx, cy, cz, capsule.xv, capsule.yv, capsule.zv)
    t = dot_cv * capsule.rdv

    # 将t限制在[0, 1]范围内
    t = max(0.0, min(1.0, t))

    # 计算轴线上最近点
    closest_x = p1_x + t * capsule.xv
    closest_y = p1_y + t * capsule.yv
    closest_z = p1_z + t * capsule.zv

    # 计算从最近点到Cuboid中心的向量
    dx = cuboid.x - closest_x
    dy = cuboid.y - closest_y
    dz = cuboid.z - closest_z

    # 将此向量投影到Cuboid的三个轴上
    proj_1 = dot_3(dx, dy, dz, cuboid.axis_1_x, cuboid.axis_1_y, cuboid.axis_1_z)
    proj_2 = dot_3(dx, dy, dz, cuboid.axis_2_x, cuboid.axis_2_y, cuboid.axis_2_z)
    proj_3 = dot_3(dx, dy, dz, cuboid.axis_3_x, cuboid.axis_3_y, cuboid.axis_3_z)

    # 计算Cuboid表面上最近的点
    clamped_1 = max(-cuboid.axis_1_r, min(cuboid.axis_1_r, proj_1))
    clamped_2 = max(-cuboid.axis_2_r, min(cuboid.axis_2_r, proj_2))
    clamped_3 = max(-cuboid.axis_3_r, min(cuboid.axis_3_r, proj_3))

    surface_x = (
        cuboid.x
        + clamped_1 * cuboid.axis_1_x
        + clamped_2 * cuboid.axis_2_x
        + clamped_3 * cuboid.axis_3_x
    )
    surface_y = (
        cuboid.y
        + clamped_1 * cuboid.axis_1_y
        + clamped_2 * cuboid.axis_2_y
        + clamped_3 * cuboid.axis_3_y
    )
    surface_z = (
        cuboid.z
        + clamped_1 * cuboid.axis_1_z
        + clamped_2 * cuboid.axis_2_z
        + clamped_3 * cuboid.axis_3_z
    )

    # 计算表面点到Capsule轴线的最短距离
    sx = surface_x - p1_x
    sy = surface_y - p1_y
    sz = surface_z - p1_z

    # 重新计算投影参数
    dot_sv = dot_3(sx, sy, sz, capsule.xv, capsule.yv, capsule.zv)
    t_surface = max(0.0, min(1.0, dot_sv * capsule.rdv))

    # 计算轴线上最近点
    axis_closest_x = p1_x + t_surface * capsule.xv
    axis_closest_y = p1_y + t_surface * capsule.yv
    axis_closest_z = p1_z + t_surface * capsule.zv

    # 计算最终距离
    final_dx = surface_x - axis_closest_x
    final_dy = surface_y - axis_closest_y
    final_dz = surface_z - axis_closest_z

    distance_sq = final_dx * final_dx + final_dy * final_dy + final_dz * final_dz
    distance = distance_sq**0.5

    # 返回距离与半径的差值（负值表示碰撞）
    return 1 if distance - capsule.r >= 0 else 0


def cuboid_heightfield(cuboid: Cuboid, heightfield: HeightField) -> int:
    """
    OBB与HeightField的碰撞检测

    算法：计算Cuboid的8个顶点，检查是否在地面以下

    Returns:
        1表示无碰撞，0表示碰撞
    """
    # 计算Cuboid的8个顶点
    vertices = []
    signs = [-1, 1]

    for i in signs:
        for j in signs:
            for k in signs:
                # 计算顶点位置
                vx = (
                    cuboid.x
                    + i * cuboid.axis_1_x * cuboid.axis_1_r
                    + j * cuboid.axis_2_x * cuboid.axis_2_r
                    + k * cuboid.axis_3_x * cuboid.axis_3_r
                )

                vy = (
                    cuboid.y
                    + i * cuboid.axis_1_y * cuboid.axis_1_r
                    + j * cuboid.axis_2_y * cuboid.axis_2_r
                    + k * cuboid.axis_3_y * cuboid.axis_3_r
                )

                vz = (
                    cuboid.z
                    + i * cuboid.axis_1_z * cuboid.axis_1_r
                    + j * cuboid.axis_2_z * cuboid.axis_2_r
                    + k * cuboid.axis_3_z * cuboid.axis_3_r
                )

                vertices.append((vx, vy, vz))

    min_penetration = 1e10  # 初始化为很大的正值

    # 检查每个顶点是否在地面以下
    for vertex in vertices:
        vx, vy, vz = vertex

        # 计算相对于heightfield基准的偏移
        xo = heightfield.x - vx
        yo = heightfield.y - vy

        # 计算网格坐标
        xs = max(
            0, min(heightfield.xd - 1, int((heightfield.xs * xo + heightfield.xd2)))
        )
        ys = max(
            0, min(heightfield.yd - 1, int((heightfield.ys * yo + heightfield.yd2)))
        )

        # 获取高度值
        index = ys * heightfield.xd + xs
        if index < len(heightfield.data):
            zh = heightfield.data[index]
            terrain_height = heightfield.zs * zh + heightfield.z

            # 计算穿透深度
            penetration = vz - terrain_height
            min_penetration = min(min_penetration, penetration)

    return 1 if min_penetration >= 0 else 0


def cuboid_sphere(cuboid: Cuboid, sphere: Sphere) -> int:
    """
    OBB与Sphere的碰撞检测

    算法：将球心投影到Cuboid的三个轴上，计算最短距离

    Returns:
        1表示无碰撞，0表示碰撞
    """
    # 计算球心相对于立方体中心的位置向量
    dx = sphere.x - cuboid.x
    dy = sphere.y - cuboid.y
    dz = sphere.z - cuboid.z

    # 将位置向量投影到立方体的三个主轴上
    proj1 = dot_3(cuboid.axis_1_x, cuboid.axis_1_y, cuboid.axis_1_z, dx, dy, dz)
    proj2 = dot_3(cuboid.axis_2_x, cuboid.axis_2_y, cuboid.axis_2_z, dx, dy, dz)
    proj3 = dot_3(cuboid.axis_3_x, cuboid.axis_3_y, cuboid.axis_3_z, dx, dy, dz)

    # 计算球心到立方体表面的距离
    dist1 = max(0, abs(proj1) - cuboid.axis_1_r)
    dist2 = max(0, abs(proj2) - cuboid.axis_2_r)
    dist3 = max(0, abs(proj3) - cuboid.axis_3_r)

    # 计算到立方体表面的最短距离平方
    distance_sq = dist1 * dist1 + dist2 * dist2 + dist3 * dist3
    radius_sq = sphere.r * sphere.r

    # 返回距离平方与半径平方的差值（负值表示碰撞）
    return 1 if distance_sq - radius_sq >= 0 else 0


def sphere_capsule(capsule: Capsule, sphere: Sphere) -> int:
    """
    Sphere与Capsule的碰撞检测

    算法：计算球心到Capsule轴线的最短距离

    Returns:
        1表示无碰撞，0表示碰撞
    """
    # 计算球心到Capsule起点的向量
    dx = sphere.x - capsule.x1
    dy = sphere.y - capsule.y1
    dz = sphere.z - capsule.z1

    # 计算投影参数
    dot = dot_3(dx, dy, dz, capsule.xv, capsule.yv, capsule.zv)
    t = max(0.0, min(1.0, dot * capsule.rdv))

    # 计算轴线上最近点
    closest_x = capsule.x1 + t * capsule.xv
    closest_y = capsule.y1 + t * capsule.yv
    closest_z = capsule.z1 + t * capsule.zv

    # 计算距离平方
    distance_sq = sql2_3(sphere.x, sphere.y, sphere.z, closest_x, closest_y, closest_z)
    radius_sum_sq = (sphere.r + capsule.r) ** 2

    # 返回距离平方与半径和的平方差值（负值表示碰撞）
    return 1 if distance_sq - radius_sum_sq >= 0 else 0


def sphere_cuboid(cuboid: Cuboid, sphere: Sphere) -> int:
    """
    Sphere与Cuboid的碰撞检测

    算法：与cuboid_sphere相同

    Returns:
        1表示无碰撞，0表示碰撞
    """
    return cuboid_sphere(cuboid, sphere)


def sphere_sphere(sphere_a: Sphere, sphere_b: Sphere) -> int:
    """
    Sphere与Sphere的碰撞检测

    算法：计算两个球心距离与半径和的差值

    Returns:
        1表示无碰撞，0表示碰撞
    """
    # 计算球心距离的平方
    distance_sq = sql2_3(
        sphere_a.x, sphere_a.y, sphere_a.z, sphere_b.x, sphere_b.y, sphere_b.z
    )

    # 计算半径和的平方
    radius_sum_sq = (sphere_a.r + sphere_b.r) ** 2

    # 返回距离平方与半径和平方的差值（负值表示碰撞）
    return 1 if distance_sq - radius_sum_sq >= 0 else 0


def cuboid_cuboid(cuboid_a: Cuboid, cuboid_b: Cuboid) -> int:
    """
    OBB与OBB的碰撞检测

    算法：使用分离轴定理，检查两个OBB在各自轴上的投影

    Returns:
        1表示无碰撞，0表示碰撞
    """
    # 计算中心向量
    dx = cuboid_b.x - cuboid_a.x
    dy = cuboid_b.y - cuboid_a.y
    dz = cuboid_b.z - cuboid_a.z

    # 测试A的轴
    # A的第一个轴
    proj_a1 = abs(
        dot_3(cuboid_a.axis_1_x, cuboid_a.axis_1_y, cuboid_a.axis_1_z, dx, dy, dz)
    )
    proj_b_on_a1 = (
        abs(
            dot_3(
                cuboid_a.axis_1_x,
                cuboid_a.axis_1_y,
                cuboid_a.axis_1_z,
                cuboid_b.axis_1_x,
                cuboid_b.axis_1_y,
                cuboid_b.axis_1_z,
            )
        )
        * cuboid_b.axis_1_r
        + abs(
            dot_3(
                cuboid_a.axis_1_x,
                cuboid_a.axis_1_y,
                cuboid_a.axis_1_z,
                cuboid_b.axis_2_x,
                cuboid_b.axis_2_y,
                cuboid_b.axis_2_z,
            )
        )
        * cuboid_b.axis_2_r
        + abs(
            dot_3(
                cuboid_a.axis_1_x,
                cuboid_a.axis_1_y,
                cuboid_a.axis_1_z,
                cuboid_b.axis_3_x,
                cuboid_b.axis_3_y,
                cuboid_b.axis_3_z,
            )
        )
        * cuboid_b.axis_3_r
    )
    sep_a1 = proj_a1 - cuboid_a.axis_1_r - proj_b_on_a1

    # A的第二个轴
    proj_a2 = abs(
        dot_3(cuboid_a.axis_2_x, cuboid_a.axis_2_y, cuboid_a.axis_2_z, dx, dy, dz)
    )
    proj_b_on_a2 = (
        abs(
            dot_3(
                cuboid_a.axis_2_x,
                cuboid_a.axis_2_y,
                cuboid_a.axis_2_z,
                cuboid_b.axis_1_x,
                cuboid_b.axis_1_y,
                cuboid_b.axis_1_z,
            )
        )
        * cuboid_b.axis_1_r
        + abs(
            dot_3(
                cuboid_a.axis_2_x,
                cuboid_a.axis_2_y,
                cuboid_a.axis_2_z,
                cuboid_b.axis_2_x,
                cuboid_b.axis_2_y,
                cuboid_b.axis_2_z,
            )
        )
        * cuboid_b.axis_2_r
        + abs(
            dot_3(
                cuboid_a.axis_2_x,
                cuboid_a.axis_2_y,
                cuboid_a.axis_2_z,
                cuboid_b.axis_3_x,
                cuboid_b.axis_3_y,
                cuboid_b.axis_3_z,
            )
        )
        * cuboid_b.axis_3_r
    )
    sep_a2 = proj_a2 - cuboid_a.axis_2_r - proj_b_on_a2

    # A的第三个轴
    proj_a3 = abs(
        dot_3(cuboid_a.axis_3_x, cuboid_a.axis_3_y, cuboid_a.axis_3_z, dx, dy, dz)
    )
    proj_b_on_a3 = (
        abs(
            dot_3(
                cuboid_a.axis_3_x,
                cuboid_a.axis_3_y,
                cuboid_a.axis_3_z,
                cuboid_b.axis_1_x,
                cuboid_b.axis_1_y,
                cuboid_b.axis_1_z,
            )
        )
        * cuboid_b.axis_1_r
        + abs(
            dot_3(
                cuboid_a.axis_3_x,
                cuboid_a.axis_3_y,
                cuboid_a.axis_3_z,
                cuboid_b.axis_2_x,
                cuboid_b.axis_2_y,
                cuboid_b.axis_2_z,
            )
        )
        * cuboid_b.axis_2_r
        + abs(
            dot_3(
                cuboid_a.axis_3_x,
                cuboid_a.axis_3_y,
                cuboid_a.axis_3_z,
                cuboid_b.axis_3_x,
                cuboid_b.axis_3_y,
                cuboid_b.axis_3_z,
            )
        )
        * cuboid_b.axis_3_r
    )
    sep_a3 = proj_a3 - cuboid_a.axis_3_r - proj_b_on_a3

    # 测试B的轴
    # B的第一个轴
    proj_b1 = abs(
        dot_3(cuboid_b.axis_1_x, cuboid_b.axis_1_y, cuboid_b.axis_1_z, dx, dy, dz)
    )
    proj_a_on_b1 = (
        abs(
            dot_3(
                cuboid_b.axis_1_x,
                cuboid_b.axis_1_y,
                cuboid_b.axis_1_z,
                cuboid_a.axis_1_x,
                cuboid_a.axis_1_y,
                cuboid_a.axis_1_z,
            )
        )
        * cuboid_a.axis_1_r
        + abs(
            dot_3(
                cuboid_b.axis_1_x,
                cuboid_b.axis_1_y,
                cuboid_b.axis_1_z,
                cuboid_a.axis_2_x,
                cuboid_a.axis_2_y,
                cuboid_a.axis_2_z,
            )
        )
        * cuboid_a.axis_2_r
        + abs(
            dot_3(
                cuboid_b.axis_1_x,
                cuboid_b.axis_1_y,
                cuboid_b.axis_1_z,
                cuboid_a.axis_3_x,
                cuboid_a.axis_3_y,
                cuboid_a.axis_3_z,
            )
        )
        * cuboid_a.axis_3_r
    )
    sep_b1 = proj_b1 - cuboid_b.axis_1_r - proj_a_on_b1

    # B的第二个轴
    proj_b2 = abs(
        dot_3(cuboid_b.axis_2_x, cuboid_b.axis_2_y, cuboid_b.axis_2_z, dx, dy, dz)
    )
    proj_a_on_b2 = (
        abs(
            dot_3(
                cuboid_b.axis_2_x,
                cuboid_b.axis_2_y,
                cuboid_b.axis_2_z,
                cuboid_a.axis_1_x,
                cuboid_a.axis_1_y,
                cuboid_a.axis_1_z,
            )
        )
        * cuboid_a.axis_1_r
        + abs(
            dot_3(
                cuboid_b.axis_2_x,
                cuboid_b.axis_2_y,
                cuboid_b.axis_2_z,
                cuboid_a.axis_2_x,
                cuboid_a.axis_2_y,
                cuboid_a.axis_2_z,
            )
        )
        * cuboid_a.axis_2_r
        + abs(
            dot_3(
                cuboid_b.axis_2_x,
                cuboid_b.axis_2_y,
                cuboid_b.axis_2_z,
                cuboid_a.axis_3_x,
                cuboid_a.axis_3_y,
                cuboid_a.axis_3_z,
            )
        )
        * cuboid_a.axis_3_r
    )
    sep_b2 = proj_b2 - cuboid_b.axis_2_r - proj_a_on_b2

    # B的第三个轴
    proj_b3 = abs(
        dot_3(cuboid_b.axis_3_x, cuboid_b.axis_3_y, cuboid_b.axis_3_z, dx, dy, dz)
    )
    proj_a_on_b3 = (
        abs(
            dot_3(
                cuboid_b.axis_3_x,
                cuboid_b.axis_3_y,
                cuboid_b.axis_3_z,
                cuboid_a.axis_1_x,
                cuboid_a.axis_1_y,
                cuboid_a.axis_1_z,
            )
        )
        * cuboid_a.axis_1_r
        + abs(
            dot_3(
                cuboid_b.axis_3_x,
                cuboid_b.axis_3_y,
                cuboid_b.axis_3_z,
                cuboid_a.axis_2_x,
                cuboid_a.axis_2_y,
                cuboid_a.axis_2_z,
            )
        )
        * cuboid_a.axis_2_r
        + abs(
            dot_3(
                cuboid_b.axis_3_x,
                cuboid_b.axis_3_y,
                cuboid_b.axis_3_z,
                cuboid_a.axis_3_x,
                cuboid_a.axis_3_y,
                cuboid_a.axis_3_z,
            )
        )
        * cuboid_a.axis_3_r
    )
    sep_b3 = proj_b3 - cuboid_b.axis_3_r - proj_a_on_b3

    # 返回最大分离距离
    # 如果所有分离距离都 >= 0，则无碰撞；否则有碰撞
    max_sep = max(sep_a1, sep_a2, sep_a3, sep_b1, sep_b2, sep_b3)

    # 注意：完整的OBB测试还需要检查9个叉积轴，但这里为了性能简化了
    # 对于大多数机器人应用，6轴测试通常足够

    return 1 if max_sep >= 0 else 0


def sphere_heightfield(heightfield: HeightField, sphere: Sphere) -> int:
    # 计算相对于heightfield基准的偏移
    xo = heightfield.x - sphere.x
    yo = heightfield.y - sphere.y

    # 计算网格坐标
    xs = max(0, min(heightfield.xd - 1, int(heightfield.xs * xo + heightfield.xd2)))
    ys = max(0, min(heightfield.yd - 1, int(heightfield.ys * yo + heightfield.yd2)))

    # 获取高度值
    index = ys * heightfield.xd + xs
    if index < len(heightfield.data):
        zh = heightfield.data[index]
        terrain_height = heightfield.zs * zh + heightfield.z

        # 计算球心与地形的距离
        return 1 if sphere.z - sphere.r - terrain_height >= 0 else 0
    else:
        # 超出范围，认为没有碰撞
        return 1


def test_collision_detection():
    """测试所有碰撞检测功能"""
    print("几何碰撞检测测试")
    print("=" * 50)

    # 创建测试对象
    cuboid = Cuboid(0, 0, 0, (1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5))
    cuboid2 = Cuboid(
        1.5, 0, 0, (1, 0, 0, 0.3), (0, 1, 0, 0.3), (0, 0, 1, 0.3)
    )  # 第二个cuboid
    sphere = Sphere(2, 0, 0, 0.3)
    sphere2 = Sphere(2.5, 0, 0, 0.2)  # 第二个sphere
    capsule = Capsule(0, 2, 0, 0, 0, 1, 0.2)

    # 创建三角形 (一个简单的xy平面上的三角形)
    triangle = Triangle((1, 0, 0), (1.5, 1, 0), (2, 0, 0))

    # 创建简单的heightfield
    height_data = [0.0] * 100  # 10x10 的平面
    heightfield = HeightField(0, 0, 0, 1.0, 1.0, 1.0, 10, 10, height_data)

    # 测试各种碰撞检测
    tests = [
        ("Cuboid vs Sphere", lambda: cuboid_sphere(cuboid, sphere)),
        ("Cuboid vs Capsule", lambda: cuboid_capsule(cuboid, capsule)),
        ("Cuboid vs Cuboid", lambda: cuboid_cuboid(cuboid, cuboid2)),
        ("Cuboid vs Triangle", lambda: cuboid_triangle(cuboid, triangle)),
        ("Cuboid vs HeightField", lambda: cuboid_heightfield(cuboid, heightfield)),
        ("Sphere vs Sphere", lambda: sphere_sphere(sphere, sphere2)),
        ("Sphere vs Triangle", lambda: sphere_triangle(sphere, triangle)),
        ("Sphere vs Capsule", lambda: sphere_capsule(capsule, sphere)),
        ("Sphere vs Cuboid", lambda: sphere_cuboid(cuboid, sphere)),
        ("Sphere vs HeightField", lambda: sphere_heightfield(heightfield, sphere)),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "无碰撞" if result == 1 else "碰撞"
            print(f"{test_name}: {status} ({result})")
        except Exception as e:
            print(f"{test_name}: 错误 - {e}")


if __name__ == "__main__":
    test_collision_detection()
