"""
BIT*和LazySP算法评估脚本

功能:
    1. eval_bit(): 评估BIT* (Batch Informed Trees*) 算法性能
    2. eval_lazysp(): 评估LazySP (Lazy Shortest Path) 算法性能

算法特点:
    - BIT*: 批量采样的最优路径规划算法,支持增量搜索
    - LazySP: 延迟碰撞检测的最短路径算法,提高效率

输出:
    (成功率, 碰撞检测次数, 运行时间, 路径成本, 总时间, 路径列表)
"""

import numpy as np
from config import set_random_seed
import torch
from torch_geometric.nn import knn_graph
from collections import defaultdict
from time import time
from tqdm import tqdm
from torch_sparse import coalesce
from algorithm.bit_star import BITStar
from algorithm.lazy_sp import LazySP
from eval_gnn import path_cost

INFINITY = float("inf")


def construct_graph(env, points, check_collision=True):
    """
    构建k近邻图用于路径规划

    参数:
        env: 环境对象,提供碰撞检测接口
        points: 配置空间中的采样点列表
        check_collision: 是否进行碰撞检测 (默认True)

    返回:
        edge_cost: 边成本字典 {节点: [邻居1成本, 邻居2成本, ...]}
        neighbors: 邻居列表字典 {节点: [邻居1, 邻居2, ...]}
        edge_index: 边索引数组 [[起点, 终点], ...]
        edge_free: 边是否无碰撞的布尔列表
    """
    # 构建k=6的近邻图 (每个点连接到最近的6个点)
    edge_index = knn_graph(torch.FloatTensor(points), k=6, loop=True)

    # 添加反向边 (使图变为无向图)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)

    # 合并重复边
    edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
    edge_index = edge_index_torch.data.cpu().numpy().T

    # 初始化数据结构
    edge_cost = defaultdict(list)  # 边的成本
    edge_free = []  # 边是否自由(无碰撞)
    neighbors = defaultdict(list)  # 邻居列表

    # 遍历所有边,进行碰撞检测并计算成本
    for i, edge in enumerate(edge_index):
        if env._edge_fp(points[edge[0]], points[edge[1]]):
            # 边无碰撞,计算欧氏距离作为成本
            edge_cost[edge[1]].append(np.linalg.norm(points[edge[1]] - points[edge[0]]))
            edge_free.append(True)
        else:
            # 边有碰撞,设置无穷大成本
            edge_cost[edge[1]].append(INFINITY)
            edge_free.append(False)
        neighbors[edge[1]].append(edge[0])

    return edge_cost, neighbors, edge_index, edge_free


def min_dist(q, dist):
    """
    从候选集合中找到距离最小的节点

    参数:
        q: 候选节点集合
        dist: 距离字典 {节点: 距离值}

    返回:
        min_node: 距离最小的节点

    说明:
        用于Dijkstra算法的辅助函数,保持主算法代码简洁
    """
    min_node = None
    for node in q:
        if min_node is None:
            min_node = node
        elif dist[node] < dist[min_node]:
            min_node = node

    return min_node


def dijkstra(nodes, edges, costs, source):
    """
    Dijkstra最短路径算法

    参数:
        nodes: 所有节点的集合
        edges: 边字典 {节点: [邻居节点列表]}
        costs: 成本字典 {节点: [到邻居的成本列表]}
        source: 起点节点

    返回:
        dist: 从起点到各节点的最短距离 {节点: 距离}
        prev: 最短路径树中各节点的前驱 {节点: 前驱节点}
    """
    q = set()  # 待访问节点集合
    dist = {}  # 最短距离字典
    prev = {}  # 前驱节点字典

    # 初始化所有节点
    for v in nodes:
        dist[v] = INFINITY  # 初始距离设为无穷大
        prev[v] = INFINITY  # 前驱节点未知
        q.add(v)  # 加入待访问集合

    # 起点到自身的距离为0
    dist[source] = 0

    # 主循环: 逐步确定每个节点的最短路径
    while q:
        # 选择距离最小的未访问节点
        u = min_dist(q, dist)

        q.remove(u)  # 标记为已访问

        # 更新所有邻居节点的距离
        for index, v in enumerate(edges[u]):
            alt = dist[u] + costs[u][index]  # 经过u到v的距离
            if alt < dist[v]:
                # 发现更短路径,更新距离和前驱
                dist[v] = alt
                prev[v] = u

    return dist, prev


def eval_bit(str, seed, env, indexes, use_tqdm=False, batch=50, t_max=1000, **kwargs):
    """
    评估BIT* (Batch Informed Trees*) 算法性能

    参数:
        str: 环境字符串描述
        seed: 随机种子
        env: 环境对象
        indexes: 要测试的问题索引列表
        use_tqdm: 是否显示进度条 (默认False)
        batch: 批量采样大小 (默认50)
        t_max: 最大采样点数 (默认1000)
        **kwargs: 其他参数

    返回:
        n_success: 成功求解的问题数
        collision: 平均碰撞检测次数
        running_time: 平均运行时间 (仅统计成功案例)
        solution_cost: 平均路径成本 (仅统计成功案例)
        total_time: 总时间
        paths: 所有问题的最优路径列表
    """
    set_random_seed(seed)

    time0 = time()  # 记录总体开始时间
    solutions = []  # 存储所有问题的求解结果
    paths = []  # 存储所有问题的最优路径

    # 根据use_tqdm决定是否显示进度条
    pbar = tqdm(indexes) if use_tqdm else indexes

    # 遍历所有问题索引
    for problem_index in pbar:
        # 初始化新问题 (设置起点、终点、障碍物等)
        env.init_new_problem(problem_index)
        print("going to BITStar")
        print(problem_index)

        # 创建BIT*规划器实例
        bit = BITStar(env, batch_size=batch, T=t_max, sampling=None)

        # 执行规划
        # time_budget=300: 最大规划时间300秒
        # refine_time_budget=0: 不进行路径优化
        solution = bit.plan(
            INFINITY, problemindex=problem_index, time_budget=300, refine_time_budget=0
        )
        solutions.append((solution))
        paths.append(bit.get_best_path())

    # ========================================
    # 统计性能指标
    # ========================================
    # solution格式: (samples, edges, collision_checks, path_cost, T, time)
    # s[-3]是路径成本, s[2]是碰撞检测次数, s[-1]是运行时间

    # 成功率: 路径成本不为无穷大的问题数
    n_success = sum([s[-3] != INFINITY for s in solutions])

    # 平均碰撞检测次数
    collision = np.mean([s[2] for s in solutions])

    # 平均运行时间 (仅统计成功案例)
    running_time = np.mean([s[-1] for s in solutions if s[-3] != INFINITY])

    # 平均路径成本 (仅统计成功案例)
    solution_cost = (
        float(sum([s[-3] for s in solutions if s[-3] != INFINITY])) / n_success
    )

    # 总运行时间
    total_time = sum([s[-1] for s in solutions])

    # 输出统计结果
    print("success rate: %d" % n_success)
    print("collision check: %.2f" % collision)
    print("running time: %.2f" % running_time)
    print("path cost: %.2f" % solution_cost)
    print("total time: %.2f" % total_time)
    print("")

    return n_success, collision, running_time, solution_cost, total_time, paths


def eval_lazysp(
    str, seed, env, indexes, use_tqdm=False, batch=50, t_max=1000, **kwargs
):
    """
    评估LazySP (Lazy Shortest Path) 算法性能

    算法特点:
        延迟碰撞检测 - 先假设所有边无碰撞计算最短路径,
        再逐步验证路径上的边,发现碰撞后重新规划

    参数:
        str: 环境字符串描述
        seed: 随机种子
        env: 环境对象
        indexes: 要测试的问题索引列表
        use_tqdm: 是否显示进度条 (默认False)
        batch: 批量采样大小 (默认50)
        t_max: 最大采样点数 (默认1000)
        **kwargs: 其他参数

    返回:
        n_success: 成功求解的问题数
        collision: 平均碰撞检测次数
        running_time: 平均运行时间 (仅统计成功案例)
        solution_cost: 平均路径成本 (仅统计成功案例)
        total_time: 总时间
        paths: 所有问题的最优路径列表
    """
    set_random_seed(seed)

    time0 = time()  # 记录总体开始时间
    solutions = []  # 存储所有问题的求解结果
    paths = []  # 存储所有问题的路径

    # 根据use_tqdm决定是否显示进度条
    pbar = tqdm(indexes) if use_tqdm else indexes

    # 遍历所有问题索引
    for problem_index in pbar:
        # 初始化新问题
        env.init_new_problem(problem_index)

        # 创建LazySP规划器实例
        lazy_sp = LazySP(env, batch_size=batch, T=t_max)

        # 执行规划
        solution = lazy_sp.plan()
        solutions.append((solution))

        # solution[2]是规划得到的路径
        paths.append(solution[2])

    # ========================================
    # 统计性能指标
    # ========================================
    # solution格式: (start_node, collision_checks, path, n_samples, time)
    # s[1]是碰撞检测次数, s[2]是路径, s[4]是运行时间

    # 成功率: 路径非空的问题数
    n_success = sum([len(p) != 0 for p in paths])

    # 平均碰撞检测次数
    collision = np.mean([s[1] for s in solutions])

    # 平均运行时间 (仅统计成功案例)
    running_time = np.mean([s[4] for s in solutions if len(s[2]) != 0])

    # 平均路径成本 (仅统计成功案例)
    # 使用path_cost函数计算每条路径的总长度
    solution_cost = float(sum([path_cost(p) for p in paths if len(p) != 0])) / n_success

    # 总运行时间
    total_time = sum([s[4] for s in solutions])

    # 输出统计结果
    print("success rate: %d" % n_success)
    print("collision check: %.2f" % collision)
    print("running time: %.2f" % running_time)
    print("path cost: %.2f" % solution_cost)
    print("total time: %.2f" % total_time)
    print("")

    return n_success, collision, running_time, solution_cost, total_time, paths
