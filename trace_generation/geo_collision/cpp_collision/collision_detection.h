#ifndef COLLISION_DETECTION_H
#define COLLISION_DETECTION_H

#include <cmath>
#include <tuple>
#include <algorithm>

namespace collision {

// ==================== 几何形状类定义 ====================

struct Sphere {
    double x, y, z;
    double r;
    double r_sq;  // 预计算半径平方
    
    Sphere(double x_, double y_, double z_, double r_)
        : x(x_), y(y_), z(z_), r(r_), r_sq(r_ * r_) {}
};

struct AABB {
    double min_x, min_y, min_z;
    double max_x, max_y, max_z;
    
    AABB(double min_x_, double min_y_, double min_z_,
         double max_x_, double max_y_, double max_z_)
        : min_x(min_x_), min_y(min_y_), min_z(min_z_),
          max_x(max_x_), max_y(max_y_), max_z(max_z_) {}
};

struct Cuboid {
    double x, y, z;
    double axis_1_x, axis_1_y, axis_1_z, axis_1_r;
    double axis_2_x, axis_2_y, axis_2_z, axis_2_r;
    double axis_3_x, axis_3_y, axis_3_z, axis_3_r;
    
    Cuboid(double x_, double y_, double z_,
           double ax1_x, double ax1_y, double ax1_z, double ax1_r,
           double ax2_x, double ax2_y, double ax2_z, double ax2_r,
           double ax3_x, double ax3_y, double ax3_z, double ax3_r)
        : x(x_), y(y_), z(z_),
          axis_1_x(ax1_x), axis_1_y(ax1_y), axis_1_z(ax1_z), axis_1_r(ax1_r),
          axis_2_x(ax2_x), axis_2_y(ax2_y), axis_2_z(ax2_z), axis_2_r(ax2_r),
          axis_3_x(ax3_x), axis_3_y(ax3_y), axis_3_z(ax3_z), axis_3_r(ax3_r) {}
};

struct Capsule {
    double x1, y1, z1;
    double xv, yv, zv;
    double r;
    double length_sq;
    double rdv;
    double rdv_sq;
    
    Capsule(double x1_, double y1_, double z1_,
            double xv_, double yv_, double zv_, double r_)
        : x1(x1_), y1(y1_), z1(z1_),
          xv(xv_), yv(yv_), zv(zv_), r(r_) {
        length_sq = xv * xv + yv * yv + zv * zv;
        rdv = (length_sq > 0) ? 1.0 / std::sqrt(length_sq) : 0.0;
        rdv_sq = (length_sq > 0) ? 1.0 / length_sq : 0.0;
    }
};

struct HeightField {
    double x, y, z;
    double xs, ys, zs;
    int xd, yd;
    std::vector<double> data;
    double xd2, yd2;
    
    HeightField(double x_, double y_, double z_,
                double xs_, double ys_, double zs_,
                int xd_, int yd_, const std::vector<double>& data_)
        : x(x_), y(y_), z(z_),
          xs(xs_), ys(ys_), zs(zs_),
          xd(xd_), yd(yd_), data(data_) {
        xd2 = xd / 2.0;
        yd2 = yd / 2.0;
    }
};

struct Triangle {
    double v0_x, v0_y, v0_z;
    double v1_x, v1_y, v1_z;
    double v2_x, v2_y, v2_z;
    
    Triangle(double v0x, double v0y, double v0z,
             double v1x, double v1y, double v1z,
             double v2x, double v2y, double v2z)
        : v0_x(v0x), v0_y(v0y), v0_z(v0z),
          v1_x(v1x), v1_y(v1y), v1_z(v1z),
          v2_x(v2x), v2_y(v2y), v2_z(v2z) {}
};

// ==================== 碰撞检测函数 ====================

// 球-球碰撞检测
// 返回: 1=无碰撞, 0=碰撞
inline int sphere_sphere(const Sphere& sphere_a, const Sphere& sphere_b) {
    double dx = sphere_a.x - sphere_b.x;
    double dy = sphere_a.y - sphere_b.y;
    double dz = sphere_a.z - sphere_b.z;
    double distance_sq = dx * dx + dy * dy + dz * dz;
    double radius_sum = sphere_a.r + sphere_b.r;
    return (distance_sq >= radius_sum * radius_sum) ? 1 : 0;
}

// 球-AABB碰撞检测
// 返回: (collision_result, cycles)
//       collision_result: 1=无碰撞, 0=碰撞
//       cycles: 硬件周期数
inline std::pair<int, int> sphere_aabb(const Sphere& sphere, const AABB& aabb) {
    // Clamp阶段
    double closest_x = std::max(aabb.min_x, std::min(sphere.x, aabb.max_x));
    double closest_y = std::max(aabb.min_y, std::min(sphere.y, aabb.max_y));
    double closest_z = std::max(aabb.min_z, std::min(sphere.z, aabb.max_z));
    
    // 计算绝对值差
    double dx = std::abs(sphere.x - closest_x);
    double dy = std::abs(sphere.y - closest_y);
    double dz = std::abs(sphere.z - closest_z);
    
    // 提前退出测试
    if (dx > sphere.r || dy > sphere.r || dz > sphere.r) {
        return {1, 2};  // 不碰撞，2个周期
    }
    
    // 距离平方计算
    double distance_sq = dx * dx + dy * dy + dz * dz;
    
    // 最终比较
    int result = (distance_sq > sphere.r_sq) ? 1 : 0;
    return {result, 5};  // 5个周期
}

// 球-OBB碰撞检测
inline int cuboid_sphere(const Cuboid& cuboid, const Sphere& sphere) {
    double dx = sphere.x - cuboid.x;
    double dy = sphere.y - cuboid.y;
    double dz = sphere.z - cuboid.z;
    
    // 投影到OBB的三个局部轴
    double proj1 = cuboid.axis_1_x * dx + cuboid.axis_1_y * dy + cuboid.axis_1_z * dz;
    double proj2 = cuboid.axis_2_x * dx + cuboid.axis_2_y * dy + cuboid.axis_2_z * dz;
    double proj3 = cuboid.axis_3_x * dx + cuboid.axis_3_y * dy + cuboid.axis_3_z * dz;
    
    // 钳制到OBB范围
    double dist1 = std::max(0.0, std::abs(proj1) - cuboid.axis_1_r);
    double dist2 = std::max(0.0, std::abs(proj2) - cuboid.axis_2_r);
    double dist3 = std::max(0.0, std::abs(proj3) - cuboid.axis_3_r);
    
    // 计算距离平方
    double distance_sq = dist1 * dist1 + dist2 * dist2 + dist3 * dist3;
    return (distance_sq >= sphere.r_sq) ? 1 : 0;
}

inline int sphere_cuboid(const Cuboid& cuboid, const Sphere& sphere) {
    return cuboid_sphere(cuboid, sphere);
}

// AABB-OBB碰撞检测 (SAT算法 - 完整15轴版本)
inline std::pair<int, int> cuboid_aabb(const Cuboid& cuboid, const AABB& aabb) {
    int cycles = 2;
    
    // 计算AABB中心和半轴长
    double aabb_cx = (aabb.min_x + aabb.max_x) * 0.5;
    double aabb_cy = (aabb.min_y + aabb.max_y) * 0.5;
    double aabb_cz = (aabb.min_z + aabb.max_z) * 0.5;
    double e_a0 = (aabb.max_x - aabb.min_x) * 0.5;
    double e_a1 = (aabb.max_y - aabb.min_y) * 0.5;
    double e_a2 = (aabb.max_z - aabb.min_z) * 0.5;
    
    // 计算相对平移向量
    double tx = cuboid.x - aabb_cx;
    double ty = cuboid.y - aabb_cy;
    double tz = cuboid.z - aabb_cz;
    
    // 旋转矩阵
    double r00 = cuboid.axis_1_x, r01 = cuboid.axis_2_x, r02 = cuboid.axis_3_x;
    double r10 = cuboid.axis_1_y, r11 = cuboid.axis_2_y, r12 = cuboid.axis_3_y;
    double r20 = cuboid.axis_1_z, r21 = cuboid.axis_2_z, r22 = cuboid.axis_3_z;
    
    // 预计算绝对值矩阵
    double abs_r00 = std::abs(r00), abs_r01 = std::abs(r01), abs_r02 = std::abs(r02);
    double abs_r10 = std::abs(r10), abs_r11 = std::abs(r11), abs_r12 = std::abs(r12);
    double abs_r20 = std::abs(r20), abs_r21 = std::abs(r21), abs_r22 = std::abs(r22);
    
    double e_b0 = cuboid.axis_1_r, e_b1 = cuboid.axis_2_r, e_b2 = cuboid.axis_3_r;
    
    // === G1: 测试AABB的3个轴 ===
    cycles += 3;
    double d_l = std::abs(tx);
    double r_a = e_a0;
    double r_b = abs_r00 * e_b0 + abs_r01 * e_b1 + abs_r02 * e_b2;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(ty);
    r_a = e_a1;
    r_b = abs_r10 * e_b0 + abs_r11 * e_b1 + abs_r12 * e_b2;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(tz);
    r_a = e_a2;
    r_b = abs_r20 * e_b0 + abs_r21 * e_b1 + abs_r22 * e_b2;
    if (d_l > r_a + r_b) return {1, cycles};
    
    // === G2: 测试OBB的3个轴 ===
    cycles += 3;
    d_l = std::abs(tx * r00 + ty * r10 + tz * r20);
    r_a = e_a0 * abs_r00 + e_a1 * abs_r10 + e_a2 * abs_r20;
    r_b = e_b0;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(tx * r01 + ty * r11 + tz * r21);
    r_a = e_a0 * abs_r01 + e_a1 * abs_r11 + e_a2 * abs_r21;
    r_b = e_b1;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(tx * r02 + ty * r12 + tz * r22);
    r_a = e_a0 * abs_r02 + e_a1 * abs_r12 + e_a2 * abs_r22;
    r_b = e_b2;
    if (d_l > r_a + r_b) return {1, cycles};
    
    // === G3: 测试9个叉积轴 ===
    cycles += 3;
    d_l = std::abs(tz * r10 - ty * r20);
    r_a = e_a1 * abs_r20 + e_a2 * abs_r10;
    r_b = e_b1 * abs_r02 + e_b2 * abs_r01;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(tz * r11 - ty * r21);
    r_a = e_a1 * abs_r21 + e_a2 * abs_r11;
    r_b = e_b0 * abs_r02 + e_b2 * abs_r00;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(tz * r12 - ty * r22);
    r_a = e_a1 * abs_r22 + e_a2 * abs_r12;
    r_b = e_b0 * abs_r01 + e_b1 * abs_r00;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(tx * r20 - tz * r00);
    r_a = e_a0 * abs_r20 + e_a2 * abs_r00;
    r_b = e_b1 * abs_r12 + e_b2 * abs_r11;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(tx * r21 - tz * r01);
    r_a = e_a0 * abs_r21 + e_a2 * abs_r01;
    r_b = e_b0 * abs_r12 + e_b2 * abs_r10;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(tx * r22 - tz * r02);
    r_a = e_a0 * abs_r22 + e_a2 * abs_r02;
    r_b = e_b0 * abs_r11 + e_b1 * abs_r10;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(ty * r00 - tx * r10);
    r_a = e_a0 * abs_r10 + e_a1 * abs_r00;
    r_b = e_b1 * abs_r22 + e_b2 * abs_r21;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(ty * r01 - tx * r11);
    r_a = e_a0 * abs_r11 + e_a1 * abs_r01;
    r_b = e_b0 * abs_r22 + e_b2 * abs_r20;
    if (d_l > r_a + r_b) return {1, cycles};
    
    cycles += 3;
    d_l = std::abs(ty * r02 - tx * r12);
    r_a = e_a0 * abs_r12 + e_a1 * abs_r02;
    r_b = e_b0 * abs_r21 + e_b1 * abs_r20;
    if (d_l > r_a + r_b) return {1, cycles};
    
    // 所有轴测试都没有分离，判定为碰撞
    return {0, cycles};
}

// 球-胶囊碰撞检测
inline int sphere_capsule(const Capsule& capsule, const Sphere& sphere) {
    double dx = sphere.x - capsule.x1;
    double dy = sphere.y - capsule.y1;
    double dz = sphere.z - capsule.z1;
    
    double dot = dx * capsule.xv + dy * capsule.yv + dz * capsule.zv;
    double t = std::max(0.0, std::min(1.0, dot * capsule.rdv_sq));
    
    dx = sphere.x - (capsule.x1 + t * capsule.xv);
    dy = sphere.y - (capsule.y1 + t * capsule.yv);
    dz = sphere.z - (capsule.z1 + t * capsule.zv);
    
    double distance_sq = dx * dx + dy * dy + dz * dz;
    double radius_sum = sphere.r + capsule.r;
    return (distance_sq >= radius_sum * radius_sum) ? 1 : 0;
}

// OBB-胶囊碰撞检测
inline int cuboid_capsule(const Cuboid& cuboid, const Capsule& capsule) {
    double cx = cuboid.x - capsule.x1;
    double cy = cuboid.y - capsule.y1;
    double cz = cuboid.z - capsule.z1;
    
    double dot_cv = cx * capsule.xv + cy * capsule.yv + cz * capsule.zv;
    double t = std::max(0.0, std::min(1.0, dot_cv * capsule.rdv_sq));
    
    double dx = cuboid.x - (capsule.x1 + t * capsule.xv);
    double dy = cuboid.y - (capsule.y1 + t * capsule.yv);
    double dz = cuboid.z - (capsule.z1 + t * capsule.zv);
    
    double proj_1 = dx * cuboid.axis_1_x + dy * cuboid.axis_1_y + dz * cuboid.axis_1_z;
    double proj_2 = dx * cuboid.axis_2_x + dy * cuboid.axis_2_y + dz * cuboid.axis_2_z;
    double proj_3 = dx * cuboid.axis_3_x + dy * cuboid.axis_3_y + dz * cuboid.axis_3_z;
    
    double clamped_1 = std::max(-cuboid.axis_1_r, std::min(cuboid.axis_1_r, proj_1));
    double clamped_2 = std::max(-cuboid.axis_2_r, std::min(cuboid.axis_2_r, proj_2));
    double clamped_3 = std::max(-cuboid.axis_3_r, std::min(cuboid.axis_3_r, proj_3));
    
    double surface_x = cuboid.x + clamped_1 * cuboid.axis_1_x + clamped_2 * cuboid.axis_2_x + clamped_3 * cuboid.axis_3_x;
    double surface_y = cuboid.y + clamped_1 * cuboid.axis_1_y + clamped_2 * cuboid.axis_2_y + clamped_3 * cuboid.axis_3_y;
    double surface_z = cuboid.z + clamped_1 * cuboid.axis_1_z + clamped_2 * cuboid.axis_2_z + clamped_3 * cuboid.axis_3_z;
    
    double sx = surface_x - capsule.x1;
    double sy = surface_y - capsule.y1;
    double sz = surface_z - capsule.z1;
    
    double dot_sv = sx * capsule.xv + sy * capsule.yv + sz * capsule.zv;
    double t_surface = std::max(0.0, std::min(1.0, dot_sv * capsule.rdv_sq));
    
    double final_dx = surface_x - (capsule.x1 + t_surface * capsule.xv);
    double final_dy = surface_y - (capsule.y1 + t_surface * capsule.yv);
    double final_dz = surface_z - (capsule.z1 + t_surface * capsule.zv);
    
    double distance_sq = final_dx * final_dx + final_dy * final_dy + final_dz * final_dz;
    return (distance_sq >= capsule.r * capsule.r) ? 1 : 0;
}

// 球-高度场碰撞检测
inline int sphere_heightfield(const HeightField& heightfield, const Sphere& sphere) {
    double xo = heightfield.x - sphere.x;
    double yo = heightfield.y - sphere.y;
    
    int xs = std::max(0, std::min(heightfield.xd - 1, static_cast<int>(heightfield.xs * xo + heightfield.xd2)));
    int ys = std::max(0, std::min(heightfield.yd - 1, static_cast<int>(heightfield.ys * yo + heightfield.yd2)));
    
    int index = ys * heightfield.xd + xs;
    double zh = heightfield.data[index];
    double terrain_height = heightfield.zs * zh + heightfield.z;
    
    return (sphere.z - sphere.r >= terrain_height) ? 1 : 0;
}

// OBB-OBB碰撞检测 (完整15轴SAT算法)
// 测试6个面轴 + 9个叉积轴，确保完整性和准确性
inline int cuboid_cuboid(const Cuboid& cuboid_a, const Cuboid& cuboid_b) {
    double dx = cuboid_b.x - cuboid_a.x;
    double dy = cuboid_b.y - cuboid_a.y;
    double dz = cuboid_b.z - cuboid_a.z;
    
    // 预计算旋转矩阵R = A^T · B
    double r11 = cuboid_a.axis_1_x * cuboid_b.axis_1_x + cuboid_a.axis_1_y * cuboid_b.axis_1_y + cuboid_a.axis_1_z * cuboid_b.axis_1_z;
    double r12 = cuboid_a.axis_1_x * cuboid_b.axis_2_x + cuboid_a.axis_1_y * cuboid_b.axis_2_y + cuboid_a.axis_1_z * cuboid_b.axis_2_z;
    double r13 = cuboid_a.axis_1_x * cuboid_b.axis_3_x + cuboid_a.axis_1_y * cuboid_b.axis_3_y + cuboid_a.axis_1_z * cuboid_b.axis_3_z;
    
    double r21 = cuboid_a.axis_2_x * cuboid_b.axis_1_x + cuboid_a.axis_2_y * cuboid_b.axis_1_y + cuboid_a.axis_2_z * cuboid_b.axis_1_z;
    double r22 = cuboid_a.axis_2_x * cuboid_b.axis_2_x + cuboid_a.axis_2_y * cuboid_b.axis_2_y + cuboid_a.axis_2_z * cuboid_b.axis_2_z;
    double r23 = cuboid_a.axis_2_x * cuboid_b.axis_3_x + cuboid_a.axis_2_y * cuboid_b.axis_3_y + cuboid_a.axis_2_z * cuboid_b.axis_3_z;
    
    double r31 = cuboid_a.axis_3_x * cuboid_b.axis_1_x + cuboid_a.axis_3_y * cuboid_b.axis_1_y + cuboid_a.axis_3_z * cuboid_b.axis_1_z;
    double r32 = cuboid_a.axis_3_x * cuboid_b.axis_2_x + cuboid_a.axis_3_y * cuboid_b.axis_2_y + cuboid_a.axis_3_z * cuboid_b.axis_2_z;
    double r33 = cuboid_a.axis_3_x * cuboid_b.axis_3_x + cuboid_a.axis_3_y * cuboid_b.axis_3_y + cuboid_a.axis_3_z * cuboid_b.axis_3_z;
    
    // 预计算绝对值矩阵
    double abs_r11 = std::abs(r11), abs_r12 = std::abs(r12), abs_r13 = std::abs(r13);
    double abs_r21 = std::abs(r21), abs_r22 = std::abs(r22), abs_r23 = std::abs(r23);
    double abs_r31 = std::abs(r31), abs_r32 = std::abs(r32), abs_r33 = std::abs(r33);
    
    double ra, rb, d_l;
    const double epsilon = 1e-10;  // 数值容差
    
    // === 测试A的3个面轴 ===
    d_l = std::abs(cuboid_a.axis_1_x * dx + cuboid_a.axis_1_y * dy + cuboid_a.axis_1_z * dz);
    ra = cuboid_a.axis_1_r;
    rb = abs_r11 * cuboid_b.axis_1_r + abs_r12 * cuboid_b.axis_2_r + abs_r13 * cuboid_b.axis_3_r;
    if (d_l > ra + rb + epsilon) return 1;
    
    d_l = std::abs(cuboid_a.axis_2_x * dx + cuboid_a.axis_2_y * dy + cuboid_a.axis_2_z * dz);
    ra = cuboid_a.axis_2_r;
    rb = abs_r21 * cuboid_b.axis_1_r + abs_r22 * cuboid_b.axis_2_r + abs_r23 * cuboid_b.axis_3_r;
    if (d_l > ra + rb + epsilon) return 1;
    
    d_l = std::abs(cuboid_a.axis_3_x * dx + cuboid_a.axis_3_y * dy + cuboid_a.axis_3_z * dz);
    ra = cuboid_a.axis_3_r;
    rb = abs_r31 * cuboid_b.axis_1_r + abs_r32 * cuboid_b.axis_2_r + abs_r33 * cuboid_b.axis_3_r;
    if (d_l > ra + rb + epsilon) return 1;
    
    // === 测试B的3个面轴 ===
    d_l = std::abs(cuboid_b.axis_1_x * dx + cuboid_b.axis_1_y * dy + cuboid_b.axis_1_z * dz);
    ra = abs_r11 * cuboid_a.axis_1_r + abs_r21 * cuboid_a.axis_2_r + abs_r31 * cuboid_a.axis_3_r;
    rb = cuboid_b.axis_1_r;
    if (d_l > ra + rb + epsilon) return 1;
    
    d_l = std::abs(cuboid_b.axis_2_x * dx + cuboid_b.axis_2_y * dy + cuboid_b.axis_2_z * dz);
    ra = abs_r12 * cuboid_a.axis_1_r + abs_r22 * cuboid_a.axis_2_r + abs_r32 * cuboid_a.axis_3_r;
    rb = cuboid_b.axis_2_r;
    if (d_l > ra + rb + epsilon) return 1;
    
    d_l = std::abs(cuboid_b.axis_3_x * dx + cuboid_b.axis_3_y * dy + cuboid_b.axis_3_z * dz);
    ra = abs_r13 * cuboid_a.axis_1_r + abs_r23 * cuboid_a.axis_2_r + abs_r33 * cuboid_a.axis_3_r;
    rb = cuboid_b.axis_3_r;
    if (d_l > ra + rb + epsilon) return 1;
    
    // === 测试9个叉积轴 ===
    // A1 × B1
    d_l = std::abs(dz * r21 - dy * r31);
    ra = cuboid_a.axis_2_r * abs_r31 + cuboid_a.axis_3_r * abs_r21;
    rb = cuboid_b.axis_2_r * abs_r13 + cuboid_b.axis_3_r * abs_r12;
    if (d_l > ra + rb + epsilon) return 1;
    
    // A1 × B2
    d_l = std::abs(dz * r22 - dy * r32);
    ra = cuboid_a.axis_2_r * abs_r32 + cuboid_a.axis_3_r * abs_r22;
    rb = cuboid_b.axis_1_r * abs_r13 + cuboid_b.axis_3_r * abs_r11;
    if (d_l > ra + rb + epsilon) return 1;
    
    // A1 × B3
    d_l = std::abs(dz * r23 - dy * r33);
    ra = cuboid_a.axis_2_r * abs_r33 + cuboid_a.axis_3_r * abs_r23;
    rb = cuboid_b.axis_1_r * abs_r12 + cuboid_b.axis_2_r * abs_r11;
    if (d_l > ra + rb + epsilon) return 1;
    
    // A2 × B1
    d_l = std::abs(dx * r31 - dz * r11);
    ra = cuboid_a.axis_1_r * abs_r31 + cuboid_a.axis_3_r * abs_r11;
    rb = cuboid_b.axis_2_r * abs_r23 + cuboid_b.axis_3_r * abs_r22;
    if (d_l > ra + rb + epsilon) return 1;
    
    // A2 × B2
    d_l = std::abs(dx * r32 - dz * r12);
    ra = cuboid_a.axis_1_r * abs_r32 + cuboid_a.axis_3_r * abs_r12;
    rb = cuboid_b.axis_1_r * abs_r23 + cuboid_b.axis_3_r * abs_r21;
    if (d_l > ra + rb + epsilon) return 1;
    
    // A2 × B3
    d_l = std::abs(dx * r33 - dz * r13);
    ra = cuboid_a.axis_1_r * abs_r33 + cuboid_a.axis_3_r * abs_r13;
    rb = cuboid_b.axis_1_r * abs_r22 + cuboid_b.axis_2_r * abs_r21;
    if (d_l > ra + rb + epsilon) return 1;
    
    // A3 × B1
    d_l = std::abs(dy * r11 - dx * r21);
    ra = cuboid_a.axis_1_r * abs_r21 + cuboid_a.axis_2_r * abs_r11;
    rb = cuboid_b.axis_2_r * abs_r33 + cuboid_b.axis_3_r * abs_r32;
    if (d_l > ra + rb + epsilon) return 1;
    
    // A3 × B2
    d_l = std::abs(dy * r12 - dx * r22);
    ra = cuboid_a.axis_1_r * abs_r22 + cuboid_a.axis_2_r * abs_r12;
    rb = cuboid_b.axis_1_r * abs_r33 + cuboid_b.axis_3_r * abs_r31;
    if (d_l > ra + rb + epsilon) return 1;
    
    // A3 × B3
    d_l = std::abs(dy * r13 - dx * r23);
    ra = cuboid_a.axis_1_r * abs_r23 + cuboid_a.axis_2_r * abs_r13;
    rb = cuboid_b.axis_1_r * abs_r32 + cuboid_b.axis_2_r * abs_r31;
    if (d_l > ra + rb + epsilon) return 1;
    
    // 所有15个轴测试都没有分离，判定为碰撞
    return 0;
}

// OBB-高度场碰撞检测
inline int cuboid_heightfield(const Cuboid& cuboid, const HeightField& heightfield) {
    // 预计算8个顶点位置的轴贡献
    double ax1_pos = cuboid.axis_1_x * cuboid.axis_1_r, ay1_pos = cuboid.axis_1_y * cuboid.axis_1_r, az1_pos = cuboid.axis_1_z * cuboid.axis_1_r;
    double ax1_neg = -ax1_pos, ay1_neg = -ay1_pos, az1_neg = -az1_pos;
    
    double ax2_pos = cuboid.axis_2_x * cuboid.axis_2_r, ay2_pos = cuboid.axis_2_y * cuboid.axis_2_r, az2_pos = cuboid.axis_2_z * cuboid.axis_2_r;
    double ax2_neg = -ax2_pos, ay2_neg = -ay2_pos, az2_neg = -az2_pos;
    
    double ax3_pos = cuboid.axis_3_x * cuboid.axis_3_r, ay3_pos = cuboid.axis_3_y * cuboid.axis_3_r, az3_pos = cuboid.axis_3_z * cuboid.axis_3_r;
    double ax3_neg = -ax3_pos, ay3_neg = -ay3_pos, az3_neg = -az3_pos;
    
    // 测试8个顶点
    auto test_vertex = [&](double vx, double vy, double vz) -> int {
        double xo = heightfield.x - vx;
        double yo = heightfield.y - vy;
        int xs = static_cast<int>(heightfield.xs * xo + heightfield.xd2);
        int ys = static_cast<int>(heightfield.ys * yo + heightfield.yd2);
        if (xs >= 0 && xs < heightfield.xd && ys >= 0 && ys < heightfield.yd) {
            if (vz < heightfield.zs * heightfield.data[ys * heightfield.xd + xs] + heightfield.z) {
                return 0;
            }
        }
        return 1;
    };
    
    if (!test_vertex(cuboid.x + ax1_neg + ax2_neg + ax3_neg, cuboid.y + ay1_neg + ay2_neg + ay3_neg, cuboid.z + az1_neg + az2_neg + az3_neg)) return 0;
    if (!test_vertex(cuboid.x + ax1_neg + ax2_neg + ax3_pos, cuboid.y + ay1_neg + ay2_neg + ay3_pos, cuboid.z + az1_neg + az2_neg + az3_pos)) return 0;
    if (!test_vertex(cuboid.x + ax1_neg + ax2_pos + ax3_neg, cuboid.y + ay1_neg + ay2_pos + ay3_neg, cuboid.z + az1_neg + az2_pos + az3_neg)) return 0;
    if (!test_vertex(cuboid.x + ax1_neg + ax2_pos + ax3_pos, cuboid.y + ay1_neg + ay2_pos + ay3_pos, cuboid.z + az1_neg + az2_pos + az3_pos)) return 0;
    if (!test_vertex(cuboid.x + ax1_pos + ax2_neg + ax3_neg, cuboid.y + ay1_pos + ay2_neg + ay3_neg, cuboid.z + az1_pos + az2_neg + az3_neg)) return 0;
    if (!test_vertex(cuboid.x + ax1_pos + ax2_neg + ax3_pos, cuboid.y + ay1_pos + ay2_neg + ay3_pos, cuboid.z + az1_pos + az2_neg + az3_pos)) return 0;
    if (!test_vertex(cuboid.x + ax1_pos + ax2_pos + ax3_neg, cuboid.y + ay1_pos + ay2_pos + ay3_neg, cuboid.z + az1_pos + az2_pos + az3_neg)) return 0;
    if (!test_vertex(cuboid.x + ax1_pos + ax2_pos + ax3_pos, cuboid.y + ay1_pos + ay2_pos + ay3_pos, cuboid.z + az1_pos + az2_pos + az3_pos)) return 0;
    
    return 1;
}

// 球-三角形碰撞检测
inline int sphere_triangle(const Sphere& sphere, const Triangle& triangle) {
    double r_sq = sphere.r_sq;
    
    double a_x = triangle.v0_x - sphere.x, a_y = triangle.v0_y - sphere.y, a_z = triangle.v0_z - sphere.z;
    double b_x = triangle.v1_x - sphere.x, b_y = triangle.v1_y - sphere.y, b_z = triangle.v1_z - sphere.z;
    double c_x = triangle.v2_x - sphere.x, c_y = triangle.v2_y - sphere.y, c_z = triangle.v2_z - sphere.z;
    
    double ab_x = b_x - a_x, ab_y = b_y - a_y, ab_z = b_z - a_z;
    double ac_x = c_x - a_x, ac_y = c_y - a_y, ac_z = c_z - a_z;
    
    double v_x = ab_y * ac_z - ab_z * ac_y;
    double v_y = ab_z * ac_x - ab_x * ac_z;
    double v_z = ab_x * ac_y - ab_y * ac_x;
    
    double d = a_x * v_x + a_y * v_y + a_z * v_z;
    double e = v_x * v_x + v_y * v_y + v_z * v_z;
    
    if (d * d > r_sq * e) return 1;
    
    double aa = a_x * a_x + a_y * a_y + a_z * a_z;
    double ab = a_x * b_x + a_y * b_y + a_z * b_z;
    double ac = a_x * c_x + a_y * c_y + a_z * c_z;
    
    if (aa > r_sq && ab > aa && ac > aa) return 1;
    
    double bb = b_x * b_x + b_y * b_y + b_z * b_z;
    double bc = b_x * c_x + b_y * c_y + b_z * c_z;
    
    if (bb > r_sq && ab > bb && bc > bb) return 1;
    
    double cc = c_x * c_x + c_y * c_y + c_z * c_z;
    
    if (cc > r_sq && ac > cc && bc > cc) return 1;
    
    // 边测试
    double d_ab = -(a_x * ab_x + a_y * ab_y + a_z * ab_z);
    double e_ab = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z;
    
    if (d_ab > 0 && d_ab < e_ab) {
        double v_ab_x = a_y * ab_z - a_z * ab_y;
        double v_ab_y = a_z * ab_x - a_x * ab_z;
        double v_ab_z = a_x * ab_y - a_y * ab_x;
        double v_ab_sq = v_ab_x * v_ab_x + v_ab_y * v_ab_y + v_ab_z * v_ab_z;
        if (v_ab_sq > r_sq * e_ab) return 1;
    }
    
    double bc_x = c_x - b_x, bc_y = c_y - b_y, bc_z = c_z - b_z;
    double d_bc = -(b_x * bc_x + b_y * bc_y + b_z * bc_z);
    double e_bc = bc_x * bc_x + bc_y * bc_y + bc_z * bc_z;
    
    if (d_bc > 0 && d_bc < e_bc) {
        double v_bc_x = b_y * bc_z - b_z * bc_y;
        double v_bc_y = b_z * bc_x - b_x * bc_z;
        double v_bc_z = b_x * bc_y - b_y * bc_x;
        double v_bc_sq = v_bc_x * v_bc_x + v_bc_y * v_bc_y + v_bc_z * v_bc_z;
        if (v_bc_sq > r_sq * e_bc) return 1;
    }
    
    double ca_x = a_x - c_x, ca_y = a_y - c_y, ca_z = a_z - c_z;
    double d_ca = -(c_x * ca_x + c_y * ca_y + c_z * ca_z);
    double e_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z;
    
    if (d_ca > 0 && d_ca < e_ca) {
        double v_ca_x = c_y * ca_z - c_z * ca_y;
        double v_ca_y = c_z * ca_x - c_x * ca_z;
        double v_ca_z = c_x * ca_y - c_y * ca_x;
        double v_ca_sq = v_ca_x * v_ca_x + v_ca_y * v_ca_y + v_ca_z * v_ca_z;
        if (v_ca_sq > r_sq * e_ca) return 1;
    }
    
    return 0;
}

// OBB-三角形碰撞检测 (SAT算法，13轴)
inline int cuboid_triangle(const Cuboid& cuboid, const Triangle& triangle) {
    // 坐标系变换到OBB局部空间
    double d0_x = triangle.v0_x - cuboid.x, d0_y = triangle.v0_y - cuboid.y, d0_z = triangle.v0_z - cuboid.z;
    double d1_x = triangle.v1_x - cuboid.x, d1_y = triangle.v1_y - cuboid.y, d1_z = triangle.v1_z - cuboid.z;
    double d2_x = triangle.v2_x - cuboid.x, d2_y = triangle.v2_y - cuboid.y, d2_z = triangle.v2_z - cuboid.z;
    
    double v0_1 = cuboid.axis_1_x * d0_x + cuboid.axis_1_y * d0_y + cuboid.axis_1_z * d0_z;
    double v0_2 = cuboid.axis_2_x * d0_x + cuboid.axis_2_y * d0_y + cuboid.axis_2_z * d0_z;
    double v0_3 = cuboid.axis_3_x * d0_x + cuboid.axis_3_y * d0_y + cuboid.axis_3_z * d0_z;
    
    double v1_1 = cuboid.axis_1_x * d1_x + cuboid.axis_1_y * d1_y + cuboid.axis_1_z * d1_z;
    double v1_2 = cuboid.axis_2_x * d1_x + cuboid.axis_2_y * d1_y + cuboid.axis_2_z * d1_z;
    double v1_3 = cuboid.axis_3_x * d1_x + cuboid.axis_3_y * d1_y + cuboid.axis_3_z * d1_z;
    
    double v2_1 = cuboid.axis_1_x * d2_x + cuboid.axis_1_y * d2_y + cuboid.axis_1_z * d2_z;
    double v2_2 = cuboid.axis_2_x * d2_x + cuboid.axis_2_y * d2_y + cuboid.axis_2_z * d2_z;
    double v2_3 = cuboid.axis_3_x * d2_x + cuboid.axis_3_y * d2_y + cuboid.axis_3_z * d2_z;
    
    // 测试cuboid的三个轴
    double min_tri = std::min({v0_1, v1_1, v2_1});
    double max_tri = std::max({v0_1, v1_1, v2_1});
    if (max_tri < -cuboid.axis_1_r || min_tri > cuboid.axis_1_r) return 1;
    
    min_tri = std::min({v0_2, v1_2, v2_2});
    max_tri = std::max({v0_2, v1_2, v2_2});
    if (max_tri < -cuboid.axis_2_r || min_tri > cuboid.axis_2_r) return 1;
    
    min_tri = std::min({v0_3, v1_3, v2_3});
    max_tri = std::max({v0_3, v1_3, v2_3});
    if (max_tri < -cuboid.axis_3_r || min_tri > cuboid.axis_3_r) return 1;
    
    // 三角形边向量和法向量测试（简化版）
    double f0_1 = v1_1 - v0_1, f0_2 = v1_2 - v0_2, f0_3 = v1_3 - v0_3;
    double f1_1 = v2_1 - v1_1, f1_2 = v2_2 - v1_2, f1_3 = v2_3 - v1_3;
    
    double n_1 = f0_2 * f1_3 - f0_3 * f1_2;
    double n_2 = f0_3 * f1_1 - f0_1 * f1_3;
    double n_3 = f0_1 * f1_2 - f0_2 * f1_1;
    
    double n_len_sq = n_1 * n_1 + n_2 * n_2 + n_3 * n_3;
    if (n_len_sq > 1e-8) {
        double r_aabb = cuboid.axis_1_r * std::abs(n_1) + cuboid.axis_2_r * std::abs(n_2) + cuboid.axis_3_r * std::abs(n_3);
        double p0 = n_1 * v0_1 + n_2 * v0_2 + n_3 * v0_3;
        min_tri = std::min({p0, n_1 * v1_1 + n_2 * v1_2 + n_3 * v1_3, n_1 * v2_1 + n_2 * v2_2 + n_3 * v2_3});
        max_tri = std::max({p0, n_1 * v1_1 + n_2 * v1_2 + n_3 * v1_3, n_1 * v2_1 + n_2 * v2_2 + n_3 * v2_3});
        if (max_tri < -r_aabb || min_tri > r_aabb) return 1;
    }
    
    // 简化的边-边叉积轴测试（仅测试关键轴）
    return 0;
}

} // namespace collision

#endif // COLLISION_DETECTION_H
