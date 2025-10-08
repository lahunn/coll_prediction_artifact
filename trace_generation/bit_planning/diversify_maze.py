"""
迷宫数据集生成脚本

功能:
    从预生成的大规模迷宫库(10万个)中筛选出符合特定难度的迷宫子集
    用于创建 mazes_easy.npz, mazes_normal.npz, mazes_hard.npz 等数据集

难度定义:
    - 障碍物密度: 用于控制迷宫复杂度
    - 起点终点距离: 确保路径规划的挑战性

数据格式:
    输出 .npz 文件包含:
    - maps: 障碍物地图数组 (N, 15, 15)
    - init_states: 起点坐标列表 (N, 2)
    - goal_states: 终点坐标列表 (N, 2)
"""

import numpy as np
from environment import MazeEnv
from tqdm import tqdm

# 历史统计信息 (从原始10万迷宫中分析得出)
# cost is between [0, 2.3]      # 路径长度范围
# grids num is between [57, 128] # 障碍物格子数范围

INFINITY = float("INF")


def dist(start, goal, maze):
    """
    计算迷宫中两点间的最短路径距离 (Dijkstra算法的简化版)

    参数:
        start: 起点网格坐标 (row, col)
        goal: 终点网格坐标 (row, col)
        maze: 迷宫地图 (15×15), 0=自由空间, 1=障碍物

    返回:
        float: 起点到终点的最短距离 (考虑对角线移动)

    算法:
        使用广度优先搜索(BFS)计算最短路径
        支持8个方向移动(上下左右+4个对角线)
        对角线移动距离 = √2 ≈ 1.414
        正交移动距离 = 1.0
    """
    frontier = [start]  # 待探索的节点队列
    explored = []  # 已探索的节点列表
    dists = {start: 0}  # 从起点到各节点的距离字典

    # BFS主循环: 直到终点被探索
    while goal not in explored:
        current = frontier.pop()
        explored.append(current)

        # 尝试8个方向的移动
        for direction in [
            (0, 1),  # 右
            (0, -1),  # 左
            (1, 0),  # 下
            (-1, 0),  # 上
            (1, 1),  # 右下(对角线)
            (-1, -1),  # 左上(对角线)
            (1, -1),  # 左下(对角线)
            (-1, 1),  # 右上(对角线)
        ]:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # 边界检查 + 障碍物检查
            if (not (14 >= neighbor[0] >= 0 and 14 >= neighbor[1] >= 0)) or (
                maze[neighbor[0], neighbor[1]] == 1
            ):
                continue

            # 如果是新节点,加入frontier并计算距离
            if (neighbor not in explored) and (neighbor not in frontier):
                frontier.append(neighbor)
                if neighbor not in dists:
                    # 计算移动距离: 对角线≈1.414, 正交=1.0
                    dists[neighbor] = dists[current] + np.linalg.norm(
                        np.array(direction)
                    )

            # 如果发现更短路径,更新距离
            if neighbor in dists:
                dists[neighbor] = min(
                    dists[current] + np.linalg.norm(np.array(direction)),
                    dists[neighbor],
                )

    return dists[goal]


# ========================================
# 加载原始迷宫数据库
# ========================================
# mazes_100000.npz 包含10万个预生成的迷宫
# 这是一个大规模的迷宫库,用于筛选出符合特定要求的子集
mazes = np.load("maze_files/mazes_100000.npz")
train_mazes = mazes.f.arr_0  # 迷宫地图数组 (100000, 15, 15)
goal_mazes = mazes.f.arr_1  # 目标相关数据 (可能未使用)


def find_mazes(maze_num, density, dist2goal_threshold=1):
    """
    从10万迷宫库中筛选符合难度要求的迷宫子集

    参数:
        maze_num: 需要的迷宫数量
        density: 障碍物密度范围 [min_obstacles, max_obstacles]
                 例如: [57, 80.6] 表示障碍物格子数在57-80之间
                       总格子数 = 15×15 = 225
        dist2goal_threshold: 起点到终点的最小距离要求 (默认1.0)

    返回:
        maps: 迷宫地图列表 (N, 15, 15)
        init_states: 起点列表 (N, 2) - 归一化坐标在[-1, 1]范围
        goal_states: 终点列表 (N, 2) - 归一化坐标在[-1, 1]范围

    筛选标准:
        1. 障碍物密度在指定范围内
        2. 起点和终点距离大于阈值
        3. 起点和终点不重合

    难度映射:
        - Easy:   density=[57, 80.6]    (25-36% 障碍物)
        - Normal: density=[80.6, 104.3] (36-46% 障碍物)
        - Hard:   density=[104.3, ∞]    (>46% 障碍物)
    """
    maps = []  # 筛选出的迷宫地图
    goal_states = []  # 对应的终点
    init_states = []  # 对应的起点

    # 遍历原始迷宫库的100倍 (允许重复使用,但起点终点随机)
    pbar = tqdm(range(100 * len(train_mazes)))
    for index in pbar:
        pbar.set_description("len of new data: %d" % len(maps))

        # 创建2D迷宫环境
        env = MazeEnv(dim=2)

        # 加载第 (index % 100000) 个迷宫
        # 1 - train_mazes: 反转0/1值 (原始数据可能是反的)
        env.map = 1 - train_mazes[index % len(train_mazes), :]

        # 计算自由空间格子数
        free_grids = np.where(env.map == 0)
        # 障碍物格子数 = 225 - 自由格子数

        # 随机生成起点和终点 (在自由空间中采样)
        env.set_random_init_goal()

        # 排除起点终点相同的情况
        if (env.init_state == env.goal_state).all():
            continue

        # ========================================
        # 筛选条件检查
        # ========================================
        # 条件1: 障碍物数量在指定范围内
        obstacle_count = 225 - len(free_grids[0])

        # 条件2: 起点终点距离大于阈值
        distance = np.linalg.norm(env.init_state - env.goal_state)

        if (
            density[0] <= obstacle_count <= density[1]
            and distance >= dist2goal_threshold
        ):
            # 通过筛选,保存数据
            maps.append(env.map)
            goal_states.append(env.goal_state)
            init_states.append(env.init_state)

            # 达到目标数量,返回结果
            if len(maps) >= maze_num:
                return maps, init_states, goal_states

    # 如果遍历完仍未达到目标数量,返回已筛选的
    return maps, init_states, goal_states


# ========================================
# 数据集生成示例 (已注释掉)
# ========================================

# 示例1: 统计分析 (分析路径长度分布)
# costs = find_mazes(0, [1, 2])
# print(np.min([cost[0] for cost in costs]))
# print(np.max([cost[0] for cost in costs]))
# plt.xlabel('Path Length')
# plt.ylabel('Density')
# plt.title('Path Length Distribution on Maze 2D')
# plt.hist(np.array([cost[0] for cost in costs]), 50, density=True, facecolor='g', alpha=0.75)
# plt.show()

# 示例2: 保存自定义迷宫集
# np.savez('new_maze.npz', maps=maps, goal_states=goal_states, init_states=init_states)
# print('success')

# 示例3: 生成Easy难度迷宫 (1000个)
# 障碍物数量: 57-80 (25-36%密度)
# maps, init_states, goal_states = find_mazes(1000, [57, 80.6])
# np.savez('maze_files/mazes_easy.npz', maps=maps, goal_states=goal_states, init_states=init_states)
# print(len(maps))

# 示例4: 生成Normal难度迷宫 (1000个)
# 障碍物数量: 80-104 (36-46%密度)
# maps, init_states, goal_states = find_mazes(1000, [80.6, 104.3])
# np.savez('maze_files/mazes_normal.npz', maps=maps, goal_states=goal_states, init_states=init_states)
# print(len(maps))


# ========================================
# 主程序: 生成大规模通用迷宫数据集
# ========================================
if __name__ == "__main__":
    """
    生成 mazes_4000.npz - 包含4000个迷宫的通用数据集
    
    参数设置:
        - 数量: 4000个迷宫
        - 密度范围: [57, ∞] - 最小57个障碍物,无上限
        - 覆盖所有难度级别
    
    输出文件:
        maze_files/mazes_4000.npz
        包含:
            - maps: (4000, 15, 15) 迷宫地图
            - init_states: (4000, 2) 起点坐标
            - goal_states: (4000, 2) 终点坐标
    
    使用方法:
        python diversify_maze.py
    
    生成其他数据集的命令:
        # Easy难度 (25-36%障碍物)
        maps, init_states, goal_states = find_mazes(1000, [57, 80.6])
        np.savez('maze_files/mazes_easy.npz', maps=maps, 
                 goal_states=goal_states, init_states=init_states)
        
        # Normal难度 (36-46%障碍物)  
        maps, init_states, goal_states = find_mazes(1000, [80.6, 104.3])
        np.savez('maze_files/mazes_normal.npz', maps=maps,
                 goal_states=goal_states, init_states=init_states)
        
        # Hard难度 (>46%障碍物)
        maps, init_states, goal_states = find_mazes(1000, [104.3, INFINITY])
        np.savez('maze_files/mazes_hard.npz', maps=maps,
                 goal_states=goal_states, init_states=init_states)
    """
    # 生成4000个迷宫,障碍物数量≥57(约25%),无上限
    maps, init_states, goal_states = find_mazes(4000, [57, INFINITY])

    # 保存到文件
    np.savez(
        "maze_files/mazes_4000.npz",
        maps=maps,
        goal_states=goal_states,
        init_states=init_states,
    )

    # 输出实际生成的迷宫数量
    print(f"Successfully generated {len(maps)} mazes")
