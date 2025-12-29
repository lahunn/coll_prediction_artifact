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
from trace_generation.bit_planning.algorithm.config import set_random_seed
from tqdm import tqdm
from algorithm.bit_star import BITStar

INFINITY = float("inf")


def eval_bit(str, seed, env, indexes, use_tqdm=False, batch=50, t_max=1000, **kwargs):
    """
    评估BIT* (Batch Informed Trees*) 算法性能

    参数:
        str: 环境字符串描述
        seed: 随机种子
        env: 环境对象 (ModularEnv实例)
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

    solutions = []  # 存储所有问题的求解结果
    paths = []  # 存储所有问题的最优路径

    # 根据use_tqdm决定是否显示进度条
    pbar = tqdm(indexes) if use_tqdm else indexes

    # 遍历所有问题索引
    for problem_index in pbar:
        # 初始化新问题 (设置起点、终点、障碍物等)
        env.init_new_problem(problem_index)

        # 创建BIT*规划器实例
        bit = BITStar(
            env,
            batch_size=batch,
            T=t_max,
            sampling=None,
        )

        # 执行规划
        # time_budget=300: 最大规划时间300秒
        # refine_time_budget=0: 不进行路径优化
        solution = bit.plan(
            INFINITY, problemindex=problem_index, time_budget=300, refine_time_budget=0
        )
        solutions.append(solution)
        paths.append(bit.get_best_path())

    # ========================================
    # 统计性能指标（solution 现在是 dict）
    # ========================================
    # solution 字典包含: cost, collision_checks, total_time, success, path
    n_success = sum([1 for s in solutions if s.get("cost", INFINITY) != INFINITY])

    collision = np.mean([s.get("collision_checks", 0) for s in solutions]) if solutions else 0.0

    if n_success > 0:
        running_time = np.mean([s.get("total_time", 0.0) for s in solutions if s.get("cost", INFINITY) != INFINITY])
    else:
        running_time = 0.0

    if n_success > 0:
        solution_cost = float(sum([s.get("cost", INFINITY) for s in solutions if s.get("cost", INFINITY) != INFINITY])) / n_success
    else:
        solution_cost = 0.0

    total_time = sum([s.get("total_time", 0.0) for s in solutions])

    # 输出统计结果
    print("success rate: %d" % n_success)
    print("collision check: %.2f" % collision)
    print("running time: %.2f" % running_time)
    print("path cost: %.2f" % solution_cost)
    print("total time: %.2f" % total_time)
    print("")

    return n_success, collision, running_time, solution_cost, total_time, paths
