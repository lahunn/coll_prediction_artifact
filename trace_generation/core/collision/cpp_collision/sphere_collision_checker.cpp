#include "sphere_collision_checker.h"
#include <algorithm>
#include <cmath>

namespace collision {

// ============================================================================
// 构造函数
// ============================================================================
SphereCollisionChecker::SphereCollisionChecker() {
    // 默认构造函数，初始化为空
}

// ============================================================================
// 配置方法实现
// ============================================================================

void SphereCollisionChecker::set_obstacles(const std::vector<AABB>& obstacles) {
    obstacles_ = obstacles;
}

void SphereCollisionChecker::set_adjacent_pairs(
    const std::vector<std::pair<int, int>>& pairs
) {
    adjacent_pairs_.clear();
    for (const auto& p : pairs) {
        adjacent_pairs_.insert(p);
    }
}

void SphereCollisionChecker::clear_obstacles() {
    obstacles_.clear();
}

// ============================================================================
// 核心碰撞检测实现
// ============================================================================

std::pair<bool, std::vector<int>> SphereCollisionChecker::check_collisions(
    const std::vector<std::array<double, 4>>& sphere_coords
) {
    const size_t n = sphere_coords.size();
    std::vector<int> collision_flags(n, 1);  // 1 = 无碰撞
    bool any_collision = false;
    
    // ========================================================================
    // 阶段 1: 检查球体与障碍物的碰撞
    // ========================================================================
    for (size_t i = 0; i < n; ++i) {
        const auto& coord = sphere_coords[i];
        Sphere sphere(coord[0], coord[1], coord[2], coord[3]);
        
        // 检查与所有障碍物的碰撞
        if (!check_sphere_obstacle_collision(sphere)) {
            collision_flags[i] = 0;
            any_collision = true;
        }
    }
    
    // ========================================================================
    // 阶段 2: 检查球体自碰撞
    // ========================================================================
    if (check_sphere_self_collision(sphere_coords, collision_flags)) {
        any_collision = true;
    }
    
    return {any_collision, collision_flags};
}

std::vector<std::pair<bool, std::vector<int>>> 
SphereCollisionChecker::check_collisions_batch(
    const std::vector<std::vector<std::array<double, 4>>>& batch_coords
) {
    std::vector<std::pair<bool, std::vector<int>>> results;
    results.reserve(batch_coords.size());
    
    // 对每个关节配置进行碰撞检测
    for (const auto& coords : batch_coords) {
        results.push_back(check_collisions(coords));
    }
    
    return results;
}

// ============================================================================
// 私有辅助方法实现
// ============================================================================

bool SphereCollisionChecker::check_sphere_obstacle_collision(const Sphere& sphere) {
    // 检查与所有障碍物的碰撞
    for (const auto& aabb : obstacles_) {
        // sphere_aabb 返回 pair<collision_result, cycles>
        auto [result, cycles] = sphere_aabb(sphere, aabb);
        
        if (result == 0) {
            // 发现碰撞
            return false;
        }
    }
    
    // 无碰撞
    return true;
}

bool SphereCollisionChecker::check_sphere_self_collision(
    const std::vector<std::array<double, 4>>& sphere_coords,
    std::vector<int>& collision_flags
) {
    const size_t n = sphere_coords.size();
    bool any_self_collision = false;
    
    // 双重循环检查所有球体对
    for (size_t i = 0; i < n; ++i) {
        // 如果球体 i 已经与障碍物碰撞，跳过
        if (collision_flags[i] == 0) {
            continue;
        }
        
        for (size_t j = i + 1; j < n; ++j) {
            // 如果球体 j 已经与障碍物碰撞，跳过
            if (collision_flags[j] == 0) {
                continue;
            }
            
            // 检查是否为邻接球体对（需要忽略）
            if (is_adjacent_pair(static_cast<int>(i), static_cast<int>(j))) {
                continue;
            }
            
            // 创建球体对象
            const auto& ci = sphere_coords[i];
            const auto& cj = sphere_coords[j];
            
            Sphere si(ci[0], ci[1], ci[2], ci[3]);
            Sphere sj(cj[0], cj[1], cj[2], cj[3]);
            
            // 检查球体碰撞
            int result = sphere_sphere(si, sj);
            
            if (result == 0) {
                // 发现自碰撞
                collision_flags[i] = 0;
                collision_flags[j] = 0;
                any_self_collision = true;
            }
        }
    }
    
    return any_self_collision;
}

} // namespace collision
