import random
from tqdm import tqdm


def find_sim_cost(R, C, A, N):
    """
    通过蒙特卡洛模拟计算预期的计算成本（运行次数）。

    参数:
    R (float): 真实碰撞率。
    C (float): 预测器的覆盖率（召回率）。
    A (float): 预测器的准确率（精确率）。
    N (int): 检查的总样本数。

    返回:
    float: 10000次模拟运行的平均成本。
    """
    all_runs = 0
    # 运行10000次模拟以获得稳定的平均值
    for ex in tqdm(range(0, 10000)):
        pred = []  # 存储预测为会碰撞的样本
        non_pred = []  # 存储预测为不会碰撞的样本

        # 根据预测器触发的概率，将N个样本分类
        for i in range(0, N):
            # 预测器触发（预测为碰撞）的概率为 R*C/A
            if random.random() <= R * C / A:
                pred.append(1)
            else:
                non_pred.append(1)

        coll = 0  # 标志位，指示是否检测到碰撞
        runs = 0  # 本次模拟的运行次数（成本）

        # 首先检查被预测为会碰撞的样本
        for i in pred:
            runs += 1
            # 如果预测为碰撞，实际发生碰撞的概率为A（准确率）
            if random.random() <= A:
                coll = 1
                break

        # 如果在预测碰撞的样本中未发现碰撞，则继续检查未被预测的样本
        if coll == 0:
            for i in non_pred:
                runs += 1
                # 如果预测为不碰撞，实际发生碰撞的概率为 R*(1-C/A)
                if random.random() <= (R * (1 - C / A)):
                    coll = 1
                    break
        all_runs += runs
    # 返回所有模拟的平均运行次数
    return all_runs / 10000
