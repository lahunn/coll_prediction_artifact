from eval_bit import eval_bit
import numpy as np
from trace_generation.robot_as.modular_env import ModularEnv
import pickle
import sys

# 环境定义（使用ModularEnv）
env_names = ["Kuka_7D", "Kuka_13D", "Kuka_14D"]
envs = [
    ModularEnv(
        robot_file="kuka_iiwa/model_0.urdf", map_file="maze_files/kukas_7_3000.pkl"
    ),
    ModularEnv(
        robot_file="kuka_iiwa/model_3.urdf", map_file="maze_files/kukas_13_3000.pkl"
    ),
    ModularEnv(
        robot_file="kuka_iiwa/model_0.urdf", map_file="maze_files/kukas_14_3000.pkl"
    ),
]
indexeses = [
    np.arange(2000, 3000),
    np.arange(2000, 3000),
    np.arange(2000, 3000),
]

# 动态设置Kuka_7D索引（如果提供参数）
if len(sys.argv) > 1:
    innn = int(sys.argv[1])
    indexeses[1] = np.arange(innn, innn + 200)

seeds = [1234]  # 可以使用多个种子

# 只评估BIT*
method_name = "BIT*"
method = eval_bit

result_total = {}

# 跳过某些环境（可选）
skim_env = [
    "Maze_2D_Easy",
    "Maze_2D_Normal",
    "Maze_3D",
    "Maze_2D_Hard",
    "Kuka_13D",
    "Kuka_14D",
]

# 遍历环境，评估BIT*算法
for env_name, env, indexes in zip(env_names, envs, indexeses):
    if env_name in skim_env:
        continue
    print(f"Evaluating {method_name} on {env_name}")
    results = []
    for seed in seeds:
        print(f"  Seed: {seed}")
        # 调用eval_bit函数：它接收问题索引列表，为每个索引初始化环境，
        # 创建BITStar实例，调用plan()方法运行BIT*规划算法，
        # 返回成功率、碰撞检查次数、运行时间等统计数据
        result = method(
            str=str(env), seed=seed, env=env, indexes=indexes, use_tqdm=False
        )
        results.append(result)
        result_total[env_name, method_name, str(seed)] = result
        pickle.dump(result_total, open("data/result_bit_only.p", "wb"))

    # 计算平均
    result_total[env_name, method_name, "Avg"] = tuple(
        [np.mean([r[i] for r in results]) for i in range(5)]
    )
    pickle.dump(result_total, open("data/result_bit_only.p", "wb"))

    # 打印平均结果
    print(f"{env_name} {method_name} Avg")
    print("success rate:", np.mean([r[0] for r in results]))
    print("collision check: %.2f" % np.mean([r[1] for r in results]))
    print("running time: %.2f" % np.mean([r[2] for r in results]))
    print("path cost: %.2f" % np.mean([r[3] for r in results]))
    print("total time: %.2f" % np.mean([r[4] for r in results]))
    print("")

print("Evaluation complete. Results saved to data/result_bit_only.p")
