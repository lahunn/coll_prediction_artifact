"""
多算法运动规划性能评估脚本

功能:
    在多种环境下批量评估不同路径规划算法的性能
    支持的算法: GNN, NEXT, BIT*, RRT*, LazySP
    支持的环境: 2D/3D迷宫, Kuka机械臂(7D/13D/14D)

使用方法:
    python eval_all.py <起始索引> <算法名称>
    例如: python eval_all.py 2000 BIT*

输出:
    data/result.p - 包含所有评估结果的pickle文件
"""

from eval_gnn import eval_gnn
from eval_next import eval_next
from eval_bit import eval_bit, eval_lazysp
from eval_rrt import eval_rrt
import numpy as np
from environment import MazeEnv, KukaEnv, Kuka2Env
import pickle
import sys

# ========================================
# 环境配置: 定义所有测试环境
# ========================================
env_names = [
    "Maze_2D_Easy",  # 2D迷宫 - 简单难度
    "Maze_2D_Normal",  # 2D迷宫 - 中等难度
    "Maze_2D_Hard",  # 2D迷宫 - 困难难度
    "Maze_3D",  # 3D迷宫
    "Kuka_7D",  # Kuka机械臂 - 7自由度
    "Kuka_13D",  # Kuka机械臂 - 13自由度
    "Kuka_14D",  # 双臂Kuka机械臂 - 14自由度
]

# 创建对应的环境实例
envs = [
    MazeEnv(dim=2, map_file="maze_files/mazes_easy.npz"),
    MazeEnv(dim=2, map_file="maze_files/mazes_normal.npz"),
    MazeEnv(dim=2, map_file="maze_files/mazes_hard.npz"),
    MazeEnv(dim=3, map_file="maze_files/mazes_hard_3.npz"),
    KukaEnv(),  # 默认7DOF Kuka
    KukaEnv(
        kuka_file="kuka_iiwa/model_3.urdf", map_file="maze_files/kukas_13_3000.pkl"
    ),  # 13DOF Kuka
    Kuka2Env(),  # 双臂14DOF Kuka
]

# ========================================
# 问题索引范围配置
# ========================================
# 默认索引范围 (迷宫环境用0-999, 机器人环境用2000-2999)
indexeses = [
    np.arange(1000),  # Maze_2D_Easy
    np.arange(1000),  # Maze_2D_Normal
    np.arange(1000),  # Maze_2D_Hard
    np.arange(2000, 3000),  # Maze_3D
    np.arange(2000, 3000),  # Kuka_7D
    np.arange(2000, 3000),  # Kuka_13D
    np.arange(2000, 3000),  # Kuka_14D
]

# 从命令行参数获取起始索引,用于Kuka_7D环境
# 允许灵活指定要测试的问题范围 (例如: 2000-2199)
innn = int(sys.argv[1])
indexeses = [
    np.arange(1000),  # Maze_2D_Easy
    np.arange(1000),  # Maze_2D_Normal
    np.arange(1000),  # Maze_2D_Hard
    np.arange(2000, 3000),  # Maze_3D
    np.arange(innn, innn + 200),  # Kuka_7D - 可配置范围
    np.arange(2000, 3000),  # Kuka_13D
    np.arange(2000, 3000),  # Kuka_14D
]

# ========================================
# 算法配置
# ========================================
# 随机种子列表 (可扩展为多个种子以获得统计显著性)
seeds = [1234]  # , 2341, 3412, 4123]

# 所有可用的评估算法
methods = [eval_gnn, eval_next, eval_bit, eval_rrt, eval_lazysp]
method_names = ["GNN", "NEXT", "BIT*", "RRT*", "LazySP"]

# 存储所有评估结果的字典
# 格式: {(env_name, method_name, seed): (success, collision, time, cost, total_time)}
result_total = {}

# ========================================
# 环境和算法过滤配置
# ========================================
# 跳过的环境列表 (这些环境不参与当前评估)
skim_env = [
    "Maze_2D_Easy",
    "Maze_2D_Normal",
    "Kuka_13D",
    "Kuka_14D",
    "Maze_3D",
    "Maze_2D_Hard",
]
# 说明: 当前配置只评估 Kuka_7D 环境

# 跳过的算法列表 (预留用于过滤,当前未使用)
skim_methods = ["GNN", "NEXT", "LazySP", "BIT*"]

# ========================================
# 主评估循环
# ========================================
for env_name, env, indexes in zip(env_names, envs, indexeses):
    # 跳过不需要评估的环境
    if env_name in skim_env:
        continue

    # 遍历所有算法
    for method_name, method in zip(method_names, methods):
        # 只运行命令行指定的算法
        # 例如: python eval_all.py 2000 BIT* 只会运行BIT*算法
        if not (method_name == sys.argv[2]):
            continue

        results = []  # 存储当前算法在当前环境下的所有结果

        # 使用不同随机种子运行多次 (当前只有一个种子)
        for seed in seeds:
            # 调用对应算法的评估函数
            # 返回: (成功率, 碰撞检测次数, 运行时间, 路径成本, 总时间, 路径列表)
            result = method(
                str=str(env), seed=seed, env=env, indexes=indexes, use_tqdm=False
            )
            results.append(result)

            # 保存单次运行结果
            result_total[env_name, method_name, str(seed)] = result

            # 实时保存结果到文件 (防止长时间运行后数据丢失)
            pickle.dump(result_total, open("data/result.p", "wb"))

        # ========================================
        # 结果统计和输出 (当前被禁用)
        # ========================================
        if False:  # 设置为True可启用统计输出
            print(env_name, method_name, "Avg")
            print("success rate:", np.mean([result[0] for result in results]))
            print("collision check: %.2f" % np.mean([result[1] for result in results]))
            print("running time: %.2f" % np.mean([result[2] for result in results]))
            print("path cost: %.2f" % np.mean([result[3] for result in results]))
            print("total time: %.2f" % np.mean([result[4] for result in results]))
            print("")

        # 计算平均性能指标
        # 汇总多个种子的结果 (当前只有1个种子,所以平均值就是单次结果)
        result_total[env_name, method_name, "Avg"] = tuple(
            [np.mean([result[i] for result in results]) for i in range(5)]
        )

        # 保存更新后的完整结果
        pickle.dump(result_total, open("data/result.p", "wb"))
