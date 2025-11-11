#ifndef SPHERE_COLLISION_CHECKER_H
#define SPHERE_COLLISION_CHECKER_H

#include "collision_detection.h"
#include <vector>
#include <array>
#include <unordered_set>
#include <utility>

namespace collision {

// ============================================================================
// 哈希函数：用于 std::pair<int, int> 在 unordered_set 中使用
// ============================================================================
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// ============================================================================
// 球体碰撞检测器类
// ============================================================================
/**
 * SphereCollisionChecker - 高性能球体碰撞检测器
 * 
 * 功能：
 * 1. 球体与 AABB 障碍物的碰撞检测
 * 2. 球体自碰撞检测（忽略相邻 link 的球体对）
 * 3. 批量碰撞检测支持
 * 
 * 性能：相比 Python 实现，预期加速 10-50x
 */
class SphereCollisionChecker {
public:
    // 构造函数
    SphereCollisionChecker();
    
    // 析构函数
    ~SphereCollisionChecker() = default;
    
    // ========================================================================
    // 配置方法
    // ========================================================================
    
    /**
     * 设置障碍物列表
     * @param obstacles AABB 障碍物列表
     */
    void set_obstacles(const std::vector<AABB>& obstacles);
    
    /**
     * 设置需要忽略碰撞的相邻球体对
     * @param pairs 球体索引对列表 [(i1, j1), (i2, j2), ...]
     */
    void set_adjacent_pairs(const std::vector<std::pair<int, int>>& pairs);
    
    /**
     * 清空障碍物
     */
    void clear_obstacles();
    
    // ========================================================================
    // 碰撞检测方法
    // ========================================================================
    
    /**
     * 检查单个关节配置的球体碰撞
     * @param sphere_coords 球体坐标列表，每个球体为 [x, y, z, radius]
     * @return tuple<bool, vector<int>, vector<int>>
     *         - first: 是否存在碰撞
     *         - second: 碰撞标志数组 (1=无碰撞, 0=有碰撞)
     *         - third: 每个球体的周期数列表
     */
    std::tuple<bool, std::vector<int>, std::vector<int>> check_collisions(
        const std::vector<std::array<double, 4>>& sphere_coords
    );
    
    /**
     * 批量检查多个关节配置的球体碰撞
     * @param batch_coords 多个关节配置的球体坐标
     * @return 每个配置的碰撞检测结果 (碰撞状态, 碰撞标志数组, 每个球体周期数)
     */
    std::vector<std::tuple<bool, std::vector<int>, std::vector<int>>> check_collisions_batch(
        const std::vector<std::vector<std::array<double, 4>>>& batch_coords
    );
    
    // ========================================================================
    // 查询方法
    // ========================================================================
    
    /**
     * 获取当前障碍物数量
     */
    size_t get_obstacle_count() const { return obstacles_.size(); }
    
    /**
     * 获取当前邻接对数量
     */
    size_t get_adjacent_pairs_count() const { return adjacent_pairs_.size(); }

private:
    // ========================================================================
    // 私有成员变量
    // ========================================================================
    
    std::vector<AABB> obstacles_;                                     // 障碍物列表
    std::unordered_set<std::pair<int, int>, PairHash> adjacent_pairs_; // 邻接球体对
    
    // ========================================================================
    // 私有辅助方法
    // ========================================================================
    
    /**
     * 判断两个球体索引是否为邻接对
     * @param i 球体索引 i
     * @param j 球体索引 j
     * @return true 如果是邻接对
     */
    inline bool is_adjacent_pair(int i, int j) const {
        return adjacent_pairs_.count({i, j}) > 0 || 
               adjacent_pairs_.count({j, i}) > 0;
    }
};

} // namespace collision

#endif // SPHERE_COLLISION_CHECKER_H
