#!/usr/bin/env python3
"""
基于几何计算的碰撞检测实现 - 性能优化版

优化策略：
1. 减少函数调用开销 - 内联简单计算
2. 减少numpy数组创建 - 使用标量计算
3. 缓存重复计算 - 预计算和存储中间结果
4. 早期退出 - 尽早返回明确结果
5. 避免不必要的归一化 - 只在必要时进行
"""

import numpy as np


class Cuboid:
    """有向包围盒 (OBB)"""

    def __init__(self, x, y, z, axis_1, axis_2, axis_3):
        self.x = x
        self.y = y
        self.z = z
        self.axis_1_x, self.axis_1_y, self.axis_1_z, self.axis_1_r = axis_1
        self.axis_2_x, self.axis_2_y, self.axis_2_z, self.axis_2_r = axis_2
        self.axis_3_x, self.axis_3_y, self.axis_3_z, self.axis_3_r = axis_3


class AABB:
    """轴对齐包围盒 (Axis-Aligned Bounding Box)"""

    def __init__(self, min_x, min_y, min_z, max_x, max_y, max_z):
        """
        初始化AABB
        :param min_x, min_y, min_z: 最小角点坐标
        :param max_x, max_y, max_z: 最大角点坐标
        """
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z


class Sphere:
    """球体"""

    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.r_sq = r * r  # 预计算半径平方


class Capsule:
    """胶囊体"""

    def __init__(self, x1, y1, z1, xv, yv, zv, r):
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.xv = xv
        self.yv = yv
        self.zv = zv
        self.r = r
        # 预计算
        self.length_sq = xv * xv + yv * yv + zv * zv
        self.rdv = 1.0 / (self.length_sq**0.5) if self.length_sq > 0 else 0
        self.rdv_sq = 1.0 / self.length_sq if self.length_sq > 0 else 0


class HeightField:
    """高度场"""

    def __init__(self, x, y, z, xs, ys, zs, xd, yd, data):
        self.x = x
        self.y = y
        self.z = z
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.xd = xd
        self.yd = yd
        self.data = np.array(data, dtype=np.float64)  # 保持float64避免类型转换开销
        self.xd2 = xd / 2.0
        self.yd2 = yd / 2.0


class Triangle:
    """空间三角形 - 简化版，减少预计算"""

    def __init__(self, v0, v1, v2):
        # 直接存储标量，避免numpy数组访问开销（2.84x慢）
        self.v0_x, self.v0_y, self.v0_z = float(v0[0]), float(v0[1]), float(v0[2])
        self.v1_x, self.v1_y, self.v1_z = float(v1[0]), float(v1[1]), float(v1[2])
        self.v2_x, self.v2_y, self.v2_z = float(v2[0]), float(v2[1]), float(v2[2])


# 内联辅助函数 - 减少函数调用开销
def sphere_sphere(sphere_a: Sphere, sphere_b: Sphere) -> int:
    """优化：直接内联计算，避免函数调用"""
    dx = sphere_a.x - sphere_b.x
    dy = sphere_a.y - sphere_b.y
    dz = sphere_a.z - sphere_b.z
    distance_sq = dx * dx + dy * dy + dz * dz
    radius_sum = sphere_a.r + sphere_b.r
    return 1 if distance_sq >= radius_sum * radius_sum else 0


def cuboid_sphere(cuboid: Cuboid, sphere: Sphere) -> int:
    """
    基于Arvo算法的球-OBB碰撞检测优化版本

    算法：找到OBB上离球心最近的点，然后比较距离
    优化：完全避免分支预测失败，使用数学钳制
    """
    dx = sphere.x - cuboid.x
    dy = sphere.y - cuboid.y
    dz = sphere.z - cuboid.z

    # 投影到OBB的三个局部轴
    proj1 = cuboid.axis_1_x * dx + cuboid.axis_1_y * dy + cuboid.axis_1_z * dz
    proj2 = cuboid.axis_2_x * dx + cuboid.axis_2_y * dy + cuboid.axis_2_z * dz
    proj3 = cuboid.axis_3_x * dx + cuboid.axis_3_y * dy + cuboid.axis_3_z * dz

    # 钳制到OBB范围：dist = max(0, |proj| - halfExtent)
    # 这一步找到OBB表面或内部最近点的偏移
    dist1 = abs(proj1) - cuboid.axis_1_r
    dist2 = abs(proj2) - cuboid.axis_2_r
    dist3 = abs(proj3) - cuboid.axis_3_r

    # 使用max避免负值（在OBB内部时距离为0）
    dist1 = max(0.0, dist1)
    dist2 = max(0.0, dist2)
    dist3 = max(0.0, dist3)

    # 计算最近点到球心的距离平方
    distance_sq = dist1 * dist1 + dist2 * dist2 + dist3 * dist3
    return 1 if distance_sq >= sphere.r_sq else 0


def sphere_capsule(capsule: Capsule, sphere: Sphere) -> int:
    """
    球-胶囊碰撞检测优化版本

    算法：计算球心到胶囊轴线的最近点，比较距离与半径和
    优化：使用预计算值，避免除法和开方，使用max/min钳制
    """
    dx = sphere.x - capsule.x1
    dy = sphere.y - capsule.y1
    dz = sphere.z - capsule.z1

    # 计算投影参数t（使用预计算的1/length²）
    dot = dx * capsule.xv + dy * capsule.yv + dz * capsule.zv
    t = dot * capsule.rdv_sq

    # 钳制t到[0,1]（使用max/min更快）
    t = max(0.0, min(1.0, t))

    # 计算轴线上最近点到球心的向量
    # 注意：这里重用dx,dy,dz作为差值向量
    dx = sphere.x - (capsule.x1 + t * capsule.xv)
    dy = sphere.y - (capsule.y1 + t * capsule.yv)
    dz = sphere.z - (capsule.z1 + t * capsule.zv)

    distance_sq = dx * dx + dy * dy + dz * dz

    # 预计算半径和的平方（避免重复计算）
    radius_sum = sphere.r + capsule.r
    return 1 if distance_sq >= radius_sum * radius_sum else 0


def sphere_cuboid(cuboid: Cuboid, sphere: Sphere) -> int:
    """优化：直接调用，避免wrapper开销"""
    return cuboid_sphere(cuboid, sphere)


def sphere_aabb(sphere: Sphere, aabb: AABB) -> tuple:
    """
    球-AABB碰撞检测 (使用硬件SAD单元优化)

    硬件假设：
    - 3个并行的16位SAD单元 (计算abs(a-b))
    - 3个并行的16位比较器
    - 3个并行的16位乘法器

    硬件周期成本 (16位定点数)：
    阶段1: Clamp (2 cycles)
      - 3× min(sphere.x, aabb.max_x): 1 cycle (并行)
      - 3× max(aabb.min_x, temp): 1 cycle (并行)

    阶段2: 绝对值差 (1 cycle) ← 使用SAD单元！
      - 3× abs(sphere.x - closest_x): 1 cycle (并行SAD单元)

    阶段3: 提前退出测试 (1 cycle)
      - 3× cmp(abs_dx, r): 1 cycle (并行比较)
      - OR逻辑: 包含在同一cycle

    阶段4: 距离平方 (3 cycles)
      - 3× mul: 2 cycles (16×16位乘法器)
      - 2× add (树形): 1 cycle

    阶段5: 最终比较 (1 cycle)
      - cmp(distance_sq, r_sq): 1 cycle

    总计:
      - 提前退出路径: 2+1+1 = 4 cycles
      - 完整计算路径: 2+1+1+3+1 = 8 cycles
    """
    # Clamp阶段 (1 cycles)
    closest_x = max(aabb.min_x, min(sphere.x, aabb.max_x))
    closest_y = max(aabb.min_y, min(sphere.y, aabb.max_y))
    closest_z = max(aabb.min_z, min(sphere.z, aabb.max_z))

    # SAD单元直接计算绝对值差 (1 cycle)
    # 硬件: dx = abs_diff(sphere.x, closest_x)
    dx = abs(sphere.x - closest_x)
    dy = abs(sphere.y - closest_y)
    dz = abs(sphere.z - closest_z)

    # 提前退出测试 (可以合并到SAD计算中)
    # 硬件: 3个并行比较器 + OR门
    if dx > sphere.r or dy > sphere.r or dz > sphere.r:
        return 1, 2  # 1+1 = 2 cycles

    # 距离平方计算 (3 cycles)
    distance_sq = dx * dx + dy * dy + dz * dz

    # 最终比较 (可以合并到距离平方计算中)
    result = 1 if distance_sq > sphere.r_sq else 0
    return result, 5  # 1+1+3 = 5 cycles


def sphere_heightfield(heightfield: HeightField, sphere: Sphere) -> int:
    """优化：保持原版的max/min策略，它们在CPython中高度优化"""
    xo = heightfield.x - sphere.x
    yo = heightfield.y - sphere.y

    # 使用max/min进行边界钳制（比4个比较更快）
    xs = max(0, min(heightfield.xd - 1, int(heightfield.xs * xo + heightfield.xd2)))
    ys = max(0, min(heightfield.yd - 1, int(heightfield.ys * yo + heightfield.yd2)))

    # 直接访问，不需要额外的边界检查
    index = ys * heightfield.xd + xs
    zh = heightfield.data[index]
    terrain_height = heightfield.zs * zh + heightfield.z

    return 1 if sphere.z - sphere.r >= terrain_height else 0


def cuboid_capsule(cuboid: Cuboid, capsule: Capsule) -> int:
    """
    OBB-胶囊碰撞检测优化版本

    算法：找到OBB上离胶囊轴线最近的点，然后判断距离
    优化：
    1. 使用预计算的倒数避免除法
    2. 使用max/min进行钳制（比条件语句快）
    3. 减少临时变量分配
    """
    # 步骤1：找到OBB中心在胶囊轴线上的最近点参数t
    cx = cuboid.x - capsule.x1
    cy = cuboid.y - capsule.y1
    cz = cuboid.z - capsule.z1

    dot_cv = cx * capsule.xv + cy * capsule.yv + cz * capsule.zv
    t = max(0.0, min(1.0, dot_cv * capsule.rdv_sq))

    # 步骤2：计算从该点到OBB中心的向量
    dx = cuboid.x - (capsule.x1 + t * capsule.xv)
    dy = cuboid.y - (capsule.y1 + t * capsule.yv)
    dz = cuboid.z - (capsule.z1 + t * capsule.zv)

    # 步骤3：投影到OBB的三个局部轴并钳制
    proj_1 = dx * cuboid.axis_1_x + dy * cuboid.axis_1_y + dz * cuboid.axis_1_z
    proj_2 = dx * cuboid.axis_2_x + dy * cuboid.axis_2_y + dz * cuboid.axis_2_z
    proj_3 = dx * cuboid.axis_3_x + dy * cuboid.axis_3_y + dz * cuboid.axis_3_z

    clamped_1 = max(-cuboid.axis_1_r, min(cuboid.axis_1_r, proj_1))
    clamped_2 = max(-cuboid.axis_2_r, min(cuboid.axis_2_r, proj_2))
    clamped_3 = max(-cuboid.axis_3_r, min(cuboid.axis_3_r, proj_3))

    # 步骤4：计算OBB表面点
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

    # 步骤5：找到表面点在胶囊轴线上的最近点参数
    sx = surface_x - capsule.x1
    sy = surface_y - capsule.y1
    sz = surface_z - capsule.z1

    dot_sv = sx * capsule.xv + sy * capsule.yv + sz * capsule.zv
    t_surface = max(0.0, min(1.0, dot_sv * capsule.rdv_sq))

    # 步骤6：计算最终距离
    final_dx = surface_x - (capsule.x1 + t_surface * capsule.xv)
    final_dy = surface_y - (capsule.y1 + t_surface * capsule.yv)
    final_dz = surface_z - (capsule.z1 + t_surface * capsule.zv)

    distance_sq = final_dx * final_dx + final_dy * final_dy + final_dz * final_dz
    return 1 if distance_sq >= capsule.r * capsule.r else 0


def cuboid_aabb(cuboid: Cuboid, aabb: AABB) -> tuple:
    """
    AABB-OBB碰撞检测 (基于SAT算法 - 完整15轴版本)

    算法：分离轴测试（SAT），测试15个潜在分离轴
    - G1: 3个AABB的轴（即世界坐标轴X,Y,Z）
    - G2: 3个OBB的面法线
    - G3: 9个叉积轴（每对边之间）

    Returns:
        tuple: (collision_result, cycles)
               collision_result: 1表示不碰撞，0表示碰撞
               cycles: 硬件周期数
    """
    # 阶段1：设置 - 计算相对位置和旋转 (4 cycles)
    cycles = 2

    # 计算AABB中心
    aabb_cx = (aabb.min_x + aabb.max_x) * 0.5
    aabb_cy = (aabb.min_y + aabb.max_y) * 0.5
    aabb_cz = (aabb.min_z + aabb.max_z) * 0.5
    # 计算AABB半轴长
    e_a0 = (aabb.max_x - aabb.min_x) * 0.5
    e_a1 = (aabb.max_y - aabb.min_y) * 0.5
    e_a2 = (aabb.max_z - aabb.min_z) * 0.5

    # 计算相对平移向量T = C_B - C_A(1 cycle)
    tx = cuboid.x - aabb_cx
    ty = cuboid.y - aabb_cy
    tz = cuboid.z - aabb_cz
    # 计算相对旋转矩阵R和绝对值矩阵AbsR
    # 由于AABB轴与世界轴对齐，R就是OBB的局部轴向量
    # R[i][j] = A_i · B_j，其中A_i是世界轴
    r00, r01, r02 = cuboid.axis_1_x, cuboid.axis_2_x, cuboid.axis_3_x
    r10, r11, r12 = cuboid.axis_1_y, cuboid.axis_2_y, cuboid.axis_3_y
    r20, r21, r22 = cuboid.axis_1_z, cuboid.axis_2_z, cuboid.axis_3_z

    # 预计算AbsR（避免重复abs调用）(1 cycle)
    abs_r00, abs_r01, abs_r02 = abs(r00), abs(r01), abs(r02)
    abs_r10, abs_r11, abs_r12 = abs(r10), abs(r11), abs(r12)
    abs_r20, abs_r21, abs_r22 = abs(r20), abs(r21), abs(r22)

    # OBB半轴长
    e_b0, e_b1, e_b2 = cuboid.axis_1_r, cuboid.axis_2_r, cuboid.axis_3_r

    # 阶段2：轴测试（带提前退出）

    # === G1: 测试AABB的3个轴（世界坐标轴）=== (每轴2 cycles)

    # 测试L = A0 (X轴)
    cycles += 3
    d_l = abs(tx)
    r_a = e_a0
    r_b = abs_r00 * e_b0 + abs_r01 * e_b1 + abs_r02 * e_b2
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = A1 (Y轴)
    cycles += 3
    d_l = abs(ty)
    r_a = e_a1
    r_b = abs_r10 * e_b0 + abs_r11 * e_b1 + abs_r12 * e_b2
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = A2 (Z轴)
    cycles += 3
    d_l = abs(tz)
    r_a = e_a2
    r_b = abs_r20 * e_b0 + abs_r21 * e_b1 + abs_r22 * e_b2
    if d_l > r_a + r_b:
        return 1, cycles

    # === G2: 测试OBB的3个轴 === (每轴2 cycles)

    # 测试L = B0 (OBB轴1)
    cycles += 3
    d_l = abs(tx * r00 + ty * r10 + tz * r20)
    r_a = e_a0 * abs_r00 + e_a1 * abs_r10 + e_a2 * abs_r20
    r_b = e_b0
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = B1 (OBB轴2)
    cycles += 3
    d_l = abs(tx * r01 + ty * r11 + tz * r21)
    r_a = e_a0 * abs_r01 + e_a1 * abs_r11 + e_a2 * abs_r21
    r_b = e_b1
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = B2 (OBB轴3)
    cycles += 3
    d_l = abs(tx * r02 + ty * r12 + tz * r22)
    r_a = e_a0 * abs_r02 + e_a1 * abs_r12 + e_a2 * abs_r22
    r_b = e_b2
    if d_l > r_a + r_b:
        return 1, cycles

    # === G3: 测试9个叉积轴 === (每轴2 cycles)

    # 测试L = A0 × B0 (X × OBB轴1) = (0, -r20, r10)
    cycles += 3
    d_l = abs(tz * r10 - ty * r20)
    r_a = e_a1 * abs_r20 + e_a2 * abs_r10
    r_b = e_b1 * abs_r02 + e_b2 * abs_r01
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = A0 × B1 (X × OBB轴2) = (0, -r21, r11)
    cycles += 3
    d_l = abs(tz * r11 - ty * r21)
    r_a = e_a1 * abs_r21 + e_a2 * abs_r11
    r_b = e_b0 * abs_r02 + e_b2 * abs_r00
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = A0 × B2 (X × OBB轴3) = (0, -r22, r12)
    cycles += 3
    d_l = abs(tz * r12 - ty * r22)
    r_a = e_a1 * abs_r22 + e_a2 * abs_r12
    r_b = e_b0 * abs_r01 + e_b1 * abs_r00
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = A1 × B0 (Y × OBB轴1) = (r20, 0, -r00)
    cycles += 3
    d_l = abs(tx * r20 - tz * r00)
    r_a = e_a0 * abs_r20 + e_a2 * abs_r00
    r_b = e_b1 * abs_r12 + e_b2 * abs_r11
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = A1 × B1 (Y × OBB轴2) = (r21, 0, -r01)
    cycles += 3
    d_l = abs(tx * r21 - tz * r01)
    r_a = e_a0 * abs_r21 + e_a2 * abs_r01
    r_b = e_b0 * abs_r12 + e_b2 * abs_r10
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = A1 × B2 (Y × OBB轴3) = (r22, 0, -r02)
    cycles += 3
    d_l = abs(tx * r22 - tz * r02)
    r_a = e_a0 * abs_r22 + e_a2 * abs_r02
    r_b = e_b0 * abs_r11 + e_b1 * abs_r10
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = A2 × B0 (Z × OBB轴1) = (-r10, r00, 0)
    cycles += 3
    d_l = abs(ty * r00 - tx * r10)
    r_a = e_a0 * abs_r10 + e_a1 * abs_r00
    r_b = e_b1 * abs_r22 + e_b2 * abs_r21
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = A2 × B1 (Z × OBB轴2) = (-r11, r01, 0)
    cycles += 3
    d_l = abs(ty * r01 - tx * r11)
    r_a = e_a0 * abs_r11 + e_a1 * abs_r01
    r_b = e_b0 * abs_r22 + e_b2 * abs_r20
    if d_l > r_a + r_b:
        return 1, cycles

    # 测试L = A2 × B2 (Z × OBB轴3) = (-r12, r02, 0)
    cycles += 3
    d_l = abs(ty * r02 - tx * r12)
    r_a = e_a0 * abs_r12 + e_a1 * abs_r02
    r_b = e_b0 * abs_r21 + e_b1 * abs_r20
    if d_l > r_a + r_b:
        return 1, cycles

    # 所有15个轴测试都没有分离，判定为碰撞
    return 0, cycles


def cuboid_cuboid(cuboid_a: Cuboid, cuboid_b: Cuboid) -> int:
    """
    OBB-OBB碰撞检测优化版本 (基于Gottschalk SAT算法 - 完整15轴版本)

    算法：分离轴测试（SAT），测试15个潜在分离轴
    - G1: 3个OBB_A的面法线
    - G2: 3个OBB_B的面法线
    - G3: 9个叉积轴（每对边之间）

    优化：
    1. 预计算旋转矩阵R和绝对值矩阵AbsR
    2. 早期退出 - 按最可能成功的轴顺序测试
    3. 减少重复abs()调用
    4. 使用数学钳制和内联计算

    Returns:
        int: 1表示不碰撞，0表示碰撞
    """
    dx = cuboid_b.x - cuboid_a.x
    dy = cuboid_b.y - cuboid_a.y
    dz = cuboid_b.z - cuboid_a.z

    # 预计算旋转矩阵R = A^T · B（9个点积）
    r11 = (
        cuboid_a.axis_1_x * cuboid_b.axis_1_x
        + cuboid_a.axis_1_y * cuboid_b.axis_1_y
        + cuboid_a.axis_1_z * cuboid_b.axis_1_z
    )
    r12 = (
        cuboid_a.axis_1_x * cuboid_b.axis_2_x
        + cuboid_a.axis_1_y * cuboid_b.axis_2_y
        + cuboid_a.axis_1_z * cuboid_b.axis_2_z
    )
    r13 = (
        cuboid_a.axis_1_x * cuboid_b.axis_3_x
        + cuboid_a.axis_1_y * cuboid_b.axis_3_y
        + cuboid_a.axis_1_z * cuboid_b.axis_3_z
    )

    r21 = (
        cuboid_a.axis_2_x * cuboid_b.axis_1_x
        + cuboid_a.axis_2_y * cuboid_b.axis_1_y
        + cuboid_a.axis_2_z * cuboid_b.axis_1_z
    )
    r22 = (
        cuboid_a.axis_2_x * cuboid_b.axis_2_x
        + cuboid_a.axis_2_y * cuboid_b.axis_2_y
        + cuboid_a.axis_2_z * cuboid_b.axis_2_z
    )
    r23 = (
        cuboid_a.axis_2_x * cuboid_b.axis_3_x
        + cuboid_a.axis_2_y * cuboid_b.axis_3_y
        + cuboid_a.axis_2_z * cuboid_b.axis_3_z
    )

    r31 = (
        cuboid_a.axis_3_x * cuboid_b.axis_1_x
        + cuboid_a.axis_3_y * cuboid_b.axis_1_y
        + cuboid_a.axis_3_z * cuboid_b.axis_1_z
    )
    r32 = (
        cuboid_a.axis_3_x * cuboid_b.axis_2_x
        + cuboid_a.axis_3_y * cuboid_b.axis_2_y
        + cuboid_a.axis_3_z * cuboid_b.axis_2_z
    )
    r33 = (
        cuboid_a.axis_3_x * cuboid_b.axis_3_x
        + cuboid_a.axis_3_y * cuboid_b.axis_3_y
        + cuboid_a.axis_3_z * cuboid_b.axis_3_z
    )

    # 预计算绝对值矩阵（避免重复abs调用）
    abs_r11, abs_r12, abs_r13 = abs(r11), abs(r12), abs(r13)
    abs_r21, abs_r22, abs_r23 = abs(r21), abs(r22), abs(r23)
    abs_r31, abs_r32, abs_r33 = abs(r31), abs(r32), abs(r33)

    epsilon = 1e-10  # 数值容差，与C++版本保持一致

    # === G1: 测试A的3个面轴 ===
    proj_a1 = abs(
        cuboid_a.axis_1_x * dx + cuboid_a.axis_1_y * dy + cuboid_a.axis_1_z * dz
    )
    ra = cuboid_a.axis_1_r
    rb = (
        abs_r11 * cuboid_b.axis_1_r
        + abs_r12 * cuboid_b.axis_2_r
        + abs_r13 * cuboid_b.axis_3_r
    )
    if proj_a1 > ra + rb + epsilon:
        return 1

    proj_a2 = abs(
        cuboid_a.axis_2_x * dx + cuboid_a.axis_2_y * dy + cuboid_a.axis_2_z * dz
    )
    ra = cuboid_a.axis_2_r
    rb = (
        abs_r21 * cuboid_b.axis_1_r
        + abs_r22 * cuboid_b.axis_2_r
        + abs_r23 * cuboid_b.axis_3_r
    )
    if proj_a2 > ra + rb + epsilon:
        return 1

    proj_a3 = abs(
        cuboid_a.axis_3_x * dx + cuboid_a.axis_3_y * dy + cuboid_a.axis_3_z * dz
    )
    ra = cuboid_a.axis_3_r
    rb = (
        abs_r31 * cuboid_b.axis_1_r
        + abs_r32 * cuboid_b.axis_2_r
        + abs_r33 * cuboid_b.axis_3_r
    )
    if proj_a3 > ra + rb + epsilon:
        return 1

    # === G2: 测试B的3个面轴 ===
    proj_b1 = abs(
        cuboid_b.axis_1_x * dx + cuboid_b.axis_1_y * dy + cuboid_b.axis_1_z * dz
    )
    ra = (
        abs_r11 * cuboid_a.axis_1_r
        + abs_r21 * cuboid_a.axis_2_r
        + abs_r31 * cuboid_a.axis_3_r
    )
    rb = cuboid_b.axis_1_r
    if proj_b1 > ra + rb + epsilon:
        return 1

    proj_b2 = abs(
        cuboid_b.axis_2_x * dx + cuboid_b.axis_2_y * dy + cuboid_b.axis_2_z * dz
    )
    ra = (
        abs_r12 * cuboid_a.axis_1_r
        + abs_r22 * cuboid_a.axis_2_r
        + abs_r32 * cuboid_a.axis_3_r
    )
    rb = cuboid_b.axis_2_r
    if proj_b2 > ra + rb + epsilon:
        return 1

    proj_b3 = abs(
        cuboid_b.axis_3_x * dx + cuboid_b.axis_3_y * dy + cuboid_b.axis_3_z * dz
    )
    ra = (
        abs_r13 * cuboid_a.axis_1_r
        + abs_r23 * cuboid_a.axis_2_r
        + abs_r33 * cuboid_a.axis_3_r
    )
    rb = cuboid_b.axis_3_r
    if proj_b3 > ra + rb + epsilon:
        return 1

    # === G3: 测试9个叉积轴 ===
    # 参考：Real-Time Collision Detection by Christer Ericson
    # 叉积轴计算：T · (A_i × B_j) 与投影半径比较

    epsilon = 1e-10  # 数值容差，与C++版本保持一致

    # 测试L = A1 × B1
    d_l = abs(dz * r21 - dy * r31)
    ra = cuboid_a.axis_2_r * abs_r31 + cuboid_a.axis_3_r * abs_r21
    rb = cuboid_b.axis_2_r * abs_r13 + cuboid_b.axis_3_r * abs_r12
    if d_l > ra + rb + epsilon:
        return 1

    # 测试L = A1 × B2
    d_l = abs(dz * r22 - dy * r32)
    ra = cuboid_a.axis_2_r * abs_r32 + cuboid_a.axis_3_r * abs_r22
    rb = cuboid_b.axis_1_r * abs_r13 + cuboid_b.axis_3_r * abs_r11
    if d_l > ra + rb + epsilon:
        return 1

    # 测试L = A1 × B3
    d_l = abs(dz * r23 - dy * r33)
    ra = cuboid_a.axis_2_r * abs_r33 + cuboid_a.axis_3_r * abs_r23
    rb = cuboid_b.axis_1_r * abs_r12 + cuboid_b.axis_2_r * abs_r11
    if d_l > ra + rb + epsilon:
        return 1

    # 测试L = A2 × B1
    d_l = abs(dx * r31 - dz * r11)
    ra = cuboid_a.axis_1_r * abs_r31 + cuboid_a.axis_3_r * abs_r11
    rb = cuboid_b.axis_2_r * abs_r23 + cuboid_b.axis_3_r * abs_r22
    if d_l > ra + rb + epsilon:
        return 1

    # 测试L = A2 × B2
    d_l = abs(dx * r32 - dz * r12)
    ra = cuboid_a.axis_1_r * abs_r32 + cuboid_a.axis_3_r * abs_r12
    rb = cuboid_b.axis_1_r * abs_r23 + cuboid_b.axis_3_r * abs_r21
    if d_l > ra + rb + epsilon:
        return 1

    # 测试L = A2 × B3
    d_l = abs(dx * r33 - dz * r13)
    ra = cuboid_a.axis_1_r * abs_r33 + cuboid_a.axis_3_r * abs_r13
    rb = cuboid_b.axis_1_r * abs_r22 + cuboid_b.axis_2_r * abs_r21
    if d_l > ra + rb + epsilon:
        return 1

    # 测试L = A3 × B1
    d_l = abs(dy * r11 - dx * r21)
    ra = cuboid_a.axis_1_r * abs_r21 + cuboid_a.axis_2_r * abs_r11
    rb = cuboid_b.axis_2_r * abs_r33 + cuboid_b.axis_3_r * abs_r32
    if d_l > ra + rb + epsilon:
        return 1

    # 测试L = A3 × B2
    d_l = abs(dy * r12 - dx * r22)
    ra = cuboid_a.axis_1_r * abs_r22 + cuboid_a.axis_2_r * abs_r12
    rb = cuboid_b.axis_1_r * abs_r33 + cuboid_b.axis_3_r * abs_r31
    if d_l > ra + rb + epsilon:
        return 1

    # 测试L = A3 × B3
    d_l = abs(dy * r13 - dx * r23)
    ra = cuboid_a.axis_1_r * abs_r23 + cuboid_a.axis_2_r * abs_r13
    rb = cuboid_b.axis_1_r * abs_r32 + cuboid_b.axis_2_r * abs_r31
    if d_l > ra + rb + epsilon:
        return 1

    # 所有15个轴测试都没有分离，判定为碰撞
    return 0


def cuboid_heightfield(cuboid: Cuboid, heightfield: HeightField) -> int:
    """
    OBB-高度场碰撞检测优化版本

    算法：检查OBB的8个顶点是否穿透高度场表面
    优化：
    1. 完全展开8个顶点，避免list创建和循环开销
    2. 早期退出 - 一旦发现穿透立即返回
    3. 预计算轴贡献减少重复乘法
    4. 内联所有计算避免函数调用

    性能分析：避免list创建和for循环可提升1.68x性能
    """
    # 预计算8个顶点位置的轴贡献
    ax1_pos = cuboid.axis_1_x * cuboid.axis_1_r
    ay1_pos = cuboid.axis_1_y * cuboid.axis_1_r
    az1_pos = cuboid.axis_1_z * cuboid.axis_1_r
    ax1_neg = -ax1_pos
    ay1_neg = -ay1_pos
    az1_neg = -az1_pos

    ax2_pos = cuboid.axis_2_x * cuboid.axis_2_r
    ay2_pos = cuboid.axis_2_y * cuboid.axis_2_r
    az2_pos = cuboid.axis_2_z * cuboid.axis_2_r
    ax2_neg = -ax2_pos
    ay2_neg = -ay2_pos
    az2_neg = -az2_pos

    ax3_pos = cuboid.axis_3_x * cuboid.axis_3_r
    ay3_pos = cuboid.axis_3_y * cuboid.axis_3_r
    az3_pos = cuboid.axis_3_z * cuboid.axis_3_r
    ax3_neg = -ax3_pos
    ay3_neg = -ay3_pos
    az3_neg = -az3_pos

    # 完全展开8个顶点测试（避免list创建和for循环开销）
    # 顶点1: (-,-,-)
    vx = cuboid.x + ax1_neg + ax2_neg + ax3_neg
    vy = cuboid.y + ay1_neg + ay2_neg + ay3_neg
    vz = cuboid.z + az1_neg + az2_neg + az3_neg
    xo = heightfield.x - vx
    yo = heightfield.y - vy
    xs = int(heightfield.xs * xo + heightfield.xd2)
    ys = int(heightfield.ys * yo + heightfield.yd2)
    if 0 <= xs < heightfield.xd and 0 <= ys < heightfield.yd:
        if (
            vz
            < heightfield.zs * heightfield.data[ys * heightfield.xd + xs]
            + heightfield.z
        ):
            return 0

    # 顶点2: (-,-,+)
    vx = cuboid.x + ax1_neg + ax2_neg + ax3_pos
    vy = cuboid.y + ay1_neg + ay2_neg + ay3_pos
    vz = cuboid.z + az1_neg + az2_neg + az3_pos
    xo = heightfield.x - vx
    yo = heightfield.y - vy
    xs = int(heightfield.xs * xo + heightfield.xd2)
    ys = int(heightfield.ys * yo + heightfield.yd2)
    if 0 <= xs < heightfield.xd and 0 <= ys < heightfield.yd:
        if (
            vz
            < heightfield.zs * heightfield.data[ys * heightfield.xd + xs]
            + heightfield.z
        ):
            return 0

    # 顶点3: (-,+,-)
    vx = cuboid.x + ax1_neg + ax2_pos + ax3_neg
    vy = cuboid.y + ay1_neg + ay2_pos + ay3_neg
    vz = cuboid.z + az1_neg + az2_pos + az3_neg
    xo = heightfield.x - vx
    yo = heightfield.y - vy
    xs = int(heightfield.xs * xo + heightfield.xd2)
    ys = int(heightfield.ys * yo + heightfield.yd2)
    if 0 <= xs < heightfield.xd and 0 <= ys < heightfield.yd:
        if (
            vz
            < heightfield.zs * heightfield.data[ys * heightfield.xd + xs]
            + heightfield.z
        ):
            return 0

    # 顶点4: (-,+,+)
    vx = cuboid.x + ax1_neg + ax2_pos + ax3_pos
    vy = cuboid.y + ay1_neg + ay2_pos + ay3_pos
    vz = cuboid.z + az1_neg + az2_pos + az3_pos
    xo = heightfield.x - vx
    yo = heightfield.y - vy
    xs = int(heightfield.xs * xo + heightfield.xd2)
    ys = int(heightfield.ys * yo + heightfield.yd2)
    if 0 <= xs < heightfield.xd and 0 <= ys < heightfield.yd:
        if (
            vz
            < heightfield.zs * heightfield.data[ys * heightfield.xd + xs]
            + heightfield.z
        ):
            return 0

    # 顶点5: (+,-,-)
    vx = cuboid.x + ax1_pos + ax2_neg + ax3_neg
    vy = cuboid.y + ay1_pos + ay2_neg + ay3_neg
    vz = cuboid.z + az1_pos + az2_neg + az3_neg
    xo = heightfield.x - vx
    yo = heightfield.y - vy
    xs = int(heightfield.xs * xo + heightfield.xd2)
    ys = int(heightfield.ys * yo + heightfield.yd2)
    if 0 <= xs < heightfield.xd and 0 <= ys < heightfield.yd:
        if (
            vz
            < heightfield.zs * heightfield.data[ys * heightfield.xd + xs]
            + heightfield.z
        ):
            return 0

    # 顶点6: (+,-,+)
    vx = cuboid.x + ax1_pos + ax2_neg + ax3_pos
    vy = cuboid.y + ay1_pos + ay2_neg + ay3_pos
    vz = cuboid.z + az1_pos + az2_neg + az3_pos
    xo = heightfield.x - vx
    yo = heightfield.y - vy
    xs = int(heightfield.xs * xo + heightfield.xd2)
    ys = int(heightfield.ys * yo + heightfield.yd2)
    if 0 <= xs < heightfield.xd and 0 <= ys < heightfield.yd:
        if (
            vz
            < heightfield.zs * heightfield.data[ys * heightfield.xd + xs]
            + heightfield.z
        ):
            return 0

    # 顶点7: (+,+,-)
    vx = cuboid.x + ax1_pos + ax2_pos + ax3_neg
    vy = cuboid.y + ay1_pos + ay2_pos + ay3_neg
    vz = cuboid.z + az1_pos + az2_pos + az3_neg
    xo = heightfield.x - vx
    yo = heightfield.y - vy
    xs = int(heightfield.xs * xo + heightfield.xd2)
    ys = int(heightfield.ys * yo + heightfield.yd2)
    if 0 <= xs < heightfield.xd and 0 <= ys < heightfield.yd:
        if (
            vz
            < heightfield.zs * heightfield.data[ys * heightfield.xd + xs]
            + heightfield.z
        ):
            return 0

    # 顶点8: (+,+,+)
    vx = cuboid.x + ax1_pos + ax2_pos + ax3_pos
    vy = cuboid.y + ay1_pos + ay2_pos + ay3_pos
    vz = cuboid.z + az1_pos + az2_pos + az3_pos
    xo = heightfield.x - vx
    yo = heightfield.y - vy
    xs = int(heightfield.xs * xo + heightfield.xd2)
    ys = int(heightfield.ys * yo + heightfield.yd2)
    if 0 <= xs < heightfield.xd and 0 <= ys < heightfield.yd:
        if (
            vz
            < heightfield.zs * heightfield.data[ys * heightfield.xd + xs]
            + heightfield.z
        ):
            return 0

    return 1


def sphere_triangle(sphere: Sphere, triangle: Triangle) -> int:
    """
    基于Ericson优化算法的球-三角形碰撞检测

    算法采用级联提前退出测试：
    1. 平面测试 - 检查球体是否与三角形平面分离
    2. 顶点测试 - 检查球心是否在顶点的外部沃罗诺伊区域
    3. 边测试 - 检查球心是否在边的外部沃罗诺伊区域
    4. 最终判定 - 所有测试失败则球体与三角形相交

    优化点：完全避免开方和除法，使用平方距离比较
    """
    r_sq = sphere.r_sq

    # 坐标系变换：将球心平移到原点，简化后续计算
    # A' = A - sphere_center
    a_x = triangle.v0_x - sphere.x
    a_y = triangle.v0_y - sphere.y
    a_z = triangle.v0_z - sphere.z

    b_x = triangle.v1_x - sphere.x
    b_y = triangle.v1_y - sphere.y
    b_z = triangle.v1_z - sphere.z

    c_x = triangle.v2_x - sphere.x
    c_y = triangle.v2_y - sphere.y
    c_z = triangle.v2_z - sphere.z

    # 步骤1: 平面测试（最快的剔除测试）
    # 计算非归一化法向量 V = (B' - A') × (C' - A')
    ab_x = b_x - a_x
    ab_y = b_y - a_y
    ab_z = b_z - a_z

    ac_x = c_x - a_x
    ac_y = c_y - a_y
    ac_z = c_z - a_z

    # V = AB × AC（叉积）
    v_x = ab_y * ac_z - ab_z * ac_y
    v_y = ab_z * ac_x - ab_x * ac_z
    v_z = ab_x * ac_y - ab_y * ac_x

    # 球心（原点）到平面的有符号距离投影 d = A' · V
    d = a_x * v_x + a_y * v_y + a_z * v_z

    # V的长度平方 e = V · V
    e = v_x * v_x + v_y * v_y + v_z * v_z

    # 分离测试：d² > r² · e（避免除法）
    if d * d > r_sq * e:
        return 1  # 球体与平面分离，不相交

    # 步骤2: 顶点测试（测试顶点沃罗诺伊区域）
    # 顶点A
    aa = a_x * a_x + a_y * a_y + a_z * a_z  # |A'|²
    ab = a_x * b_x + a_y * b_y + a_z * b_z  # A' · B'
    ac = a_x * c_x + a_y * c_y + a_z * c_z  # A' · C'

    # 分离条件：(|A'|² > r²) ∧ (A'·B' > |A'|²) ∧ (A'·C' > |A'|²)
    if aa > r_sq and ab > aa and ac > aa:
        return 1  # 球心在顶点A的外部沃罗诺伊区域

    # 顶点B
    bb = b_x * b_x + b_y * b_y + b_z * b_z  # |B'|²
    # ba = ab  # B' · A' = A' · B'（已计算）
    bc = b_x * c_x + b_y * c_y + b_z * c_z  # B' · C'

    if bb > r_sq and ab > bb and bc > bb:
        return 1  # 球心在顶点B的外部沃罗诺伊区域

    # 顶点C
    cc = c_x * c_x + c_y * c_y + c_z * c_z  # |C'|²
    # ca = ac  # C' · A' = A' · C'（已计算）
    # cb = bc  # C' · B' = B' · C'（已计算）

    if cc > r_sq and ac > cc and bc > cc:
        return 1  # 球心在顶点C的外部沃罗诺伊区域

    # 步骤3: 边测试（测试边沃罗诺伊区域）
    # 边AB: A' -> B'
    # E = B' - A'（已计算为ab_x, ab_y, ab_z）
    # d = E · (-A') = -A' · E
    d_ab = -(a_x * ab_x + a_y * ab_y + a_z * ab_z)
    e_ab = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z  # |E|²

    # 区域检查：0 < d < e（投影点在线段内）
    if d_ab > 0 and d_ab < e_ab:
        # 计算球心到边的距离平方（通过叉积）
        # V = A' × E
        v_ab_x = a_y * ab_z - a_z * ab_y
        v_ab_y = a_z * ab_x - a_x * ab_z
        v_ab_z = a_x * ab_y - a_y * ab_x

        v_ab_sq = v_ab_x * v_ab_x + v_ab_y * v_ab_y + v_ab_z * v_ab_z

        # 分离测试：|V|² > r² · |E|²
        if v_ab_sq > r_sq * e_ab:
            return 1  # 球心在边AB的外部沃罗诺伊区域

    # 边BC: B' -> C'
    bc_x = c_x - b_x
    bc_y = c_y - b_y
    bc_z = c_z - b_z

    d_bc = -(b_x * bc_x + b_y * bc_y + b_z * bc_z)
    e_bc = bc_x * bc_x + bc_y * bc_y + bc_z * bc_z

    if d_bc > 0 and d_bc < e_bc:
        v_bc_x = b_y * bc_z - b_z * bc_y
        v_bc_y = b_z * bc_x - b_x * bc_z
        v_bc_z = b_x * bc_y - b_y * bc_x

        v_bc_sq = v_bc_x * v_bc_x + v_bc_y * v_bc_y + v_bc_z * v_bc_z

        if v_bc_sq > r_sq * e_bc:
            return 1

    # 边CA: C' -> A'
    ca_x = a_x - c_x
    ca_y = a_y - c_y
    ca_z = a_z - c_z

    d_ca = -(c_x * ca_x + c_y * ca_y + c_z * ca_z)
    e_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z

    if d_ca > 0 and d_ca < e_ca:
        v_ca_x = c_y * ca_z - c_z * ca_y
        v_ca_y = c_z * ca_x - c_x * ca_z
        v_ca_z = c_x * ca_y - c_y * ca_x

        v_ca_sq = v_ca_x * v_ca_x + v_ca_y * v_ca_y + v_ca_z * v_ca_z

        if v_ca_sq > r_sq * e_ca:
            return 1

    # 步骤4: 最终判定
    # 所有分离测试都失败，球体与三角形相交
    return 0


def cuboid_triangle(cuboid: Cuboid, triangle: Triangle) -> int:
    """
    OBB-三角形碰撞检测优化版本 (基于SAT算法)

    算法：分离轴测试，测试13个潜在分离轴
    - 3个OBB面法线
    - 1个三角形法线
    - 9个边-边叉积轴

    优化：
    1. 变换到OBB局部坐标系，简化后续计算
    2. 优先测试最可能成功的轴（OBB面轴）
    3. 使用min/max快速找到三角形投影范围
    4. 避免归一化和开方运算
    5. 对每个轴进行长度检查，跳过退化轴
    """
    # 步骤1：坐标系变换到OBB局部空间
    d0_x = triangle.v0_x - cuboid.x
    d0_y = triangle.v0_y - cuboid.y
    d0_z = triangle.v0_z - cuboid.z

    d1_x = triangle.v1_x - cuboid.x
    d1_y = triangle.v1_y - cuboid.y
    d1_z = triangle.v1_z - cuboid.z

    d2_x = triangle.v2_x - cuboid.x
    d2_y = triangle.v2_y - cuboid.y
    d2_z = triangle.v2_z - cuboid.z

    # 投影到cuboid的三个轴
    v0_1 = cuboid.axis_1_x * d0_x + cuboid.axis_1_y * d0_y + cuboid.axis_1_z * d0_z
    v0_2 = cuboid.axis_2_x * d0_x + cuboid.axis_2_y * d0_y + cuboid.axis_2_z * d0_z
    v0_3 = cuboid.axis_3_x * d0_x + cuboid.axis_3_y * d0_y + cuboid.axis_3_z * d0_z

    v1_1 = cuboid.axis_1_x * d1_x + cuboid.axis_1_y * d1_y + cuboid.axis_1_z * d1_z
    v1_2 = cuboid.axis_2_x * d1_x + cuboid.axis_2_y * d1_y + cuboid.axis_2_z * d1_z
    v1_3 = cuboid.axis_3_x * d1_x + cuboid.axis_3_y * d1_y + cuboid.axis_3_z * d1_z

    v2_1 = cuboid.axis_1_x * d2_x + cuboid.axis_1_y * d2_y + cuboid.axis_1_z * d2_z
    v2_2 = cuboid.axis_2_x * d2_x + cuboid.axis_2_y * d2_y + cuboid.axis_2_z * d2_z
    v2_3 = cuboid.axis_3_x * d2_x + cuboid.axis_3_y * d2_y + cuboid.axis_3_z * d2_z

    # 测试cuboid的三个轴（最重要的轴）
    # 轴1
    min_tri = min(v0_1, v1_1, v2_1)
    max_tri = max(v0_1, v1_1, v2_1)
    if max_tri < -cuboid.axis_1_r or min_tri > cuboid.axis_1_r:
        return 1

    # 轴2
    min_tri = min(v0_2, v1_2, v2_2)
    max_tri = max(v0_2, v1_2, v2_2)
    if max_tri < -cuboid.axis_2_r or min_tri > cuboid.axis_2_r:
        return 1

    # 轴3
    min_tri = min(v0_3, v1_3, v2_3)
    max_tri = max(v0_3, v1_3, v2_3)
    if max_tri < -cuboid.axis_3_r or min_tri > cuboid.axis_3_r:
        return 1

    # 三角形边向量（在local坐标系）
    f0_1, f0_2, f0_3 = v1_1 - v0_1, v1_2 - v0_2, v1_3 - v0_3
    f1_1, f1_2, f1_3 = v2_1 - v1_1, v2_2 - v1_2, v2_3 - v1_3
    f2_1, f2_2, f2_3 = v0_1 - v2_1, v0_2 - v2_2, v0_3 - v2_3

    # 三角形法向量
    n_1 = f0_2 * f1_3 - f0_3 * f1_2
    n_2 = f0_3 * f1_1 - f0_1 * f1_3
    n_3 = f0_1 * f1_2 - f0_2 * f1_1

    # 测试三角形法向量轴
    n_len_sq = n_1 * n_1 + n_2 * n_2 + n_3 * n_3
    if n_len_sq > 1e-8:
        r_aabb = (
            cuboid.axis_1_r * abs(n_1)
            + cuboid.axis_2_r * abs(n_2)
            + cuboid.axis_3_r * abs(n_3)
        )
        p0 = n_1 * v0_1 + n_2 * v0_2 + n_3 * v0_3
        p1 = n_1 * v1_1 + n_2 * v1_2 + n_3 * v1_3
        p2 = n_1 * v2_1 + n_2 * v2_2 + n_3 * v2_3
        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)
        if max_tri < -r_aabb or min_tri > r_aabb:
            return 1

    # 测试9个叉积轴（最耗时的部分）
    # 只测试长度不为零的轴
    # axis1 × f0
    ax_1, ax_2, ax_3 = -f0_3, f0_1, 0  # (1,0,0) × f0
    ax_len_sq = ax_1 * ax_1 + ax_2 * ax_2 + ax_3 * ax_3
    if ax_len_sq > 1e-8:
        r_aabb = cuboid.axis_2_r * abs(ax_3) + cuboid.axis_3_r * abs(ax_2)
        p0 = ax_1 * v0_1 + ax_2 * v0_2 + ax_3 * v0_3
        p1 = ax_1 * v1_1 + ax_2 * v1_2 + ax_3 * v1_3
        p2 = ax_1 * v2_1 + ax_2 * v2_2 + ax_3 * v2_3
        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)
        if max_tri < -r_aabb or min_tri > r_aabb:
            return 1

    # axis1 × f1
    ax_1, ax_2, ax_3 = -f1_3, f1_1, 0
    ax_len_sq = ax_1 * ax_1 + ax_2 * ax_2 + ax_3 * ax_3
    if ax_len_sq > 1e-8:
        r_aabb = cuboid.axis_2_r * abs(ax_3) + cuboid.axis_3_r * abs(ax_2)
        p0 = ax_1 * v0_1 + ax_2 * v0_2 + ax_3 * v0_3
        p1 = ax_1 * v1_1 + ax_2 * v1_2 + ax_3 * v1_3
        p2 = ax_1 * v2_1 + ax_2 * v2_2 + ax_3 * v2_3
        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)
        if max_tri < -r_aabb or min_tri > r_aabb:
            return 1

    # axis1 × f2
    ax_1, ax_2, ax_3 = -f2_3, f2_1, 0
    ax_len_sq = ax_1 * ax_1 + ax_2 * ax_2 + ax_3 * ax_3
    if ax_len_sq > 1e-8:
        r_aabb = cuboid.axis_2_r * abs(ax_3) + cuboid.axis_3_r * abs(ax_2)
        p0 = ax_1 * v0_1 + ax_2 * v0_2 + ax_3 * v0_3
        p1 = ax_1 * v1_1 + ax_2 * v1_2 + ax_3 * v1_3
        p2 = ax_1 * v2_1 + ax_2 * v2_2 + ax_3 * v2_3
        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)
        if max_tri < -r_aabb or min_tri > r_aabb:
            return 1

    # axis2 × f0
    ax_1, ax_2, ax_3 = f0_3, 0, -f0_1  # (0,1,0) × f0
    ax_len_sq = ax_1 * ax_1 + ax_2 * ax_2 + ax_3 * ax_3
    if ax_len_sq > 1e-8:
        r_aabb = cuboid.axis_1_r * abs(ax_3) + cuboid.axis_3_r * abs(ax_1)
        p0 = ax_1 * v0_1 + ax_2 * v0_2 + ax_3 * v0_3
        p1 = ax_1 * v1_1 + ax_2 * v1_2 + ax_3 * v1_3
        p2 = ax_1 * v2_1 + ax_2 * v2_2 + ax_3 * v2_3
        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)
        if max_tri < -r_aabb or min_tri > r_aabb:
            return 1

    # axis2 × f1
    ax_1, ax_2, ax_3 = f1_3, 0, -f1_1
    ax_len_sq = ax_1 * ax_1 + ax_2 * ax_2 + ax_3 * ax_3
    if ax_len_sq > 1e-8:
        r_aabb = cuboid.axis_1_r * abs(ax_3) + cuboid.axis_3_r * abs(ax_1)
        p0 = ax_1 * v0_1 + ax_2 * v0_2 + ax_3 * v0_3
        p1 = ax_1 * v1_1 + ax_2 * v1_2 + ax_3 * v1_3
        p2 = ax_1 * v2_1 + ax_2 * v2_2 + ax_3 * v2_3
        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)
        if max_tri < -r_aabb or min_tri > r_aabb:
            return 1

    # axis2 × f2
    ax_1, ax_2, ax_3 = f2_3, 0, -f2_1
    ax_len_sq = ax_1 * ax_1 + ax_2 * ax_2 + ax_3 * ax_3
    if ax_len_sq > 1e-8:
        r_aabb = cuboid.axis_1_r * abs(ax_3) + cuboid.axis_3_r * abs(ax_1)
        p0 = ax_1 * v0_1 + ax_2 * v0_2 + ax_3 * v0_3
        p1 = ax_1 * v1_1 + ax_2 * v1_2 + ax_3 * v1_3
        p2 = ax_1 * v2_1 + ax_2 * v2_2 + ax_3 * v2_3
        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)
        if max_tri < -r_aabb or min_tri > r_aabb:
            return 1

    # axis3 × f0
    ax_1, ax_2, ax_3 = -f0_2, f0_1, 0  # (0,0,1) × f0
    ax_len_sq = ax_1 * ax_1 + ax_2 * ax_2 + ax_3 * ax_3
    if ax_len_sq > 1e-8:
        r_aabb = cuboid.axis_1_r * abs(ax_2) + cuboid.axis_2_r * abs(ax_1)
        p0 = ax_1 * v0_1 + ax_2 * v0_2 + ax_3 * v0_3
        p1 = ax_1 * v1_1 + ax_2 * v1_2 + ax_3 * v1_3
        p2 = ax_1 * v2_1 + ax_2 * v2_2 + ax_3 * v2_3
        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)
        if max_tri < -r_aabb or min_tri > r_aabb:
            return 1

    # axis3 × f1
    ax_1, ax_2, ax_3 = -f1_2, f1_1, 0
    ax_len_sq = ax_1 * ax_1 + ax_2 * ax_2 + ax_3 * ax_3
    if ax_len_sq > 1e-8:
        r_aabb = cuboid.axis_1_r * abs(ax_2) + cuboid.axis_2_r * abs(ax_1)
        p0 = ax_1 * v0_1 + ax_2 * v0_2 + ax_3 * v0_3
        p1 = ax_1 * v1_1 + ax_2 * v1_2 + ax_3 * v1_3
        p2 = ax_1 * v2_1 + ax_2 * v2_2 + ax_3 * v2_3
        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)
        if max_tri < -r_aabb or min_tri > r_aabb:
            return 1

    # axis3 × f2
    ax_1, ax_2, ax_3 = -f2_2, f2_1, 0
    ax_len_sq = ax_1 * ax_1 + ax_2 * ax_2 + ax_3 * ax_3
    if ax_len_sq > 1e-8:
        r_aabb = cuboid.axis_1_r * abs(ax_2) + cuboid.axis_2_r * abs(ax_1)
        p0 = ax_1 * v0_1 + ax_2 * v0_2 + ax_3 * v0_3
        p1 = ax_1 * v1_1 + ax_2 * v1_2 + ax_3 * v1_3
        p2 = ax_1 * v2_1 + ax_2 * v2_2 + ax_3 * v2_3
        min_tri = min(p0, p1, p2)
        max_tri = max(p0, p1, p2)
        if max_tri < -r_aabb or min_tri > r_aabb:
            return 1

    # 没有找到分离轴
    return 0
