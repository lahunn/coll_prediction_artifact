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
                if random.random() <= (R * (1 - C) / (1 - C * R / A)):
                    coll = 1
                    break
        all_runs += runs
    # 返回所有模拟的平均运行次数
    return all_runs / 10000

def calculate_expected_checks(R, C, A, N):
    """
    使用精确的封闭形式公式计算期望的碰撞检测次数。

    参数:
    R (float): 真实碰撞率 (0 <= R <= 1)。
    C (float): 预测器的覆盖率（召回率）(0 <= C <= 1)。
    A (float): 预测器的准确率（精确率）(0 <= A <= 1)。
    N (int): 检查的总样本数。

    返回:
    float: 执行的碰撞检测任务总次数的精确期望。
    """

    # --- 输入验证 ---
    if not (0 <= R <= 1 and 0 <= A <= 1 and 0 <= C <= 1):
        raise ValueError("概率值 R, A, C 必须在 [0, 1] 范围内。")
    if N <= 0:
        raise ValueError("任务总数 N 必须为正整数。")
    if A == 0 and C * R > 0:
        raise ValueError("当 A=0 时, C*R 必须也为0。")
    # P(Y) = C*R/A 必须小于等于1
    if C * R > A + 1e-9: # 加上一个小的容差避免浮点数问题
        raise ValueError(f"参数组合无效: C*R ({C*R}) 不能大于 A ({A})。")
        
    # --- 边界情况处理 ---
    # 如果实际碰撞概率为0，则永远不会碰撞，必须执行完所有N次检测。
    if R == 0:
        return float(N)
    
    # 如果精确率为0 (且C*R=0)，则所有预测为碰撞的都不是碰撞。
    # 此时组1为空或无用，相当于无策略。
    if A == 0:
        return (1 - (1 - R)**N) / R

    # --- 计算中间变量 ---
    
    # 任务被预测为"碰撞"的概率 P(Y)
    prob_predicted_positive = (C * R) / A

    # 组2内任务实际为碰撞的概率 P2
    if abs(prob_predicted_positive - 1.0) < 1e-9:
        # 如果所有任务都被预测为碰撞，则组2为空，P2无意义，且第二项为0。
        P2 = 0 # 设为0以避免除零错误
    else:
        P2 = ((1 - C) * R) / (1 - prob_predicted_positive)

    # --- 计算精确期望 ---
    
    # E = (1 - (1 - CR)^N)/A + ((1 - CR)^N - (1-R)^N)/P2
    
    term1 = (1 - (1 - C * R)**N) / A
    
    # (1 - CR)^N
    term_1_minus_cr_pow_n = (1 - C * R)**N
    # (1 - R)^N
    term_1_minus_r_pow_n = (1 - R)**N
    
    numerator_term2 = term_1_minus_cr_pow_n - term_1_minus_r_pow_n
    
    if abs(P2) < 1e-9:
        # 如果P2为0，意味着组2中没有碰撞。
        # 此时需要检查分子是否也为0。 (1-CR)^N - (1-R)^N 只有在C=1或R=0时为0。
        # 如果分子不为0，P2=0意味着发散，但这在物理上不可能。
        # 在我们的模型中，P2=0意味着(1-C)R=0。如果R>0，则C=1。
        # 当C=1时，(1-CR)^N - (1-R)^N = 0，所以第二项是 0/0 的形式。
        # 此时，所有碰撞都在组1，组2没有碰撞，所以第二项贡献为0。
        term2 = 0.0
    else:
        term2 = numerator_term2 / P2
    return term1 + term2

def calculate_baseline_expectation(N: int, R: float) -> float:
    """
    计算在无排序策略下，执行的碰撞检测任务总次数的期望。

    该模型假设每个检测任务都是独立的，且具有相同的碰撞概率。
    一旦检测到碰撞，整个过程即停止。

    参数:
    N (int): 总的碰撞检测任务数量。必须是正整数。
    P (float): 任何一个任务实际为碰撞的先验概率。必须在 [0.0, 1.0] 范围内。

    返回:
    float: 执行的碰撞检测任务总次数的期望值。
    """
    
    # --- 输入验证 ---
    if not 0.0 <= R <= 1.0:
        raise ValueError("概率 P 必须在 [0.0, 1.0] 范围内。")
        
    # --- 边界情况处理 ---
    # 如果碰撞概率为 0，那么永远不会发生碰撞，
    # 必须执行完所有 N 次检测才能结束。
    if R == 0.0:
        return float(N)
        
    # --- 应用主公式 ---
    # E = (1 - (1-P)^N) / P
    expected_value = (1 - (1 - R)**N) / R
    
    return expected_value