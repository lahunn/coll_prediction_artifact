"""
碰撞预测策略模块
提供两种不同的碰撞预测策略：
1. 固定阈值策略 (Fixed Threshold Strategy)
2. 自适应阈值策略 (Adaptive Threshold Strategy)
"""

import random


class CollisionPredictionStrategy:
    """碰撞预测策略基类"""

    def __init__(self, update_prob=0.5, max_count=255):
        """
        初始化预测策略

        Args:
            update_prob: 自由样本的更新概率
            max_count: 计数器最大值（默认255，对应8-bit存储）
        """
        self.colldict = {}  # 碰撞字典 {hash_key: [碰撞次数, 自由次数]}
        self.update_prob = update_prob
        self.max_count = max_count  # SRAM位宽限制（如8bit = 255）

        # 统计变量
        self.all_zerozero = 0  # True Positives
        self.all_onezero = 0  # False Positives
        self.all_total_colliding = 0  # 总碰撞数

    def predict_collision(self, keyy):
        """
        预测给定哈希键是否会碰撞

        Args:
            keyy: 配置的哈希键

        Returns:
            bool: True表示预测碰撞，False表示预测自由
        """
        raise NotImplementedError("子类必须实现此方法")

    def update_history(self, keyy, label):
        """
        更新碰撞历史字典
        模拟SRAM有限位宽存储：当计数器达到max_count时，两个计数器同时除以2

        Args:
            keyy: 配置的哈希键
            label: 真实标签 (0=碰撞, 1=自由)
        """
        if keyy in self.colldict:
            # 碰撞样本总是更新；自由样本按概率采样更新
            if (label > 0.5 and random.random() <= self.update_prob) or label < 0.5:
                # 处理可能的数组类型
                try:
                    label_int = int(label.item())
                except (AttributeError, TypeError):
                    label_int = int(label)

                # 检查是否达到计数器上限
                if self.colldict[keyy][label_int] >= self.max_count:
                    # 饱和计数器：两个计数器同时除以2（右移1位）
                    self.colldict[keyy][0] = self.colldict[keyy][0] // 2
                    self.colldict[keyy][1] = self.colldict[keyy][1] // 2

                # 增加计数
                self.colldict[keyy][label_int] += 1
        else:
            # 新键：初始化统计
            if (label > 0.5 and random.random() <= self.update_prob) or label < 0.5:
                self.colldict[keyy] = [0, 0]
                try:
                    label_int = int(label.item())
                except (AttributeError, TypeError):
                    label_int = int(label)
                self.colldict[keyy][label_int] += 1

    def reset_collision_history(self):
        """
        重置碰撞历史字典
        """
        self.colldict.clear()

    def update_statistics(self, predicted, true_label):
        """
        更新统计变量

        Args:
            predicted: 预测结果 (0=碰撞, 1=自由)
            true_label: 真实标签 (0=碰撞, 1=自由)
        """
        if true_label < 0.5:  # 真实为碰撞
            self.all_total_colliding += 1
            if predicted == 0:  # 预测也为碰撞 (TP)
                self.all_zerozero += 1
        elif predicted == 0:  # 真实为自由但预测为碰撞 (FP)
            self.all_onezero += 1

    def reset_statistics(self):
        """重置统计变量"""
        self.all_zerozero = 0
        self.all_onezero = 0
        self.all_total_colliding = 0

    def get_metrics(self):
        """
        计算并返回评估指标

        Returns:
            tuple: (precision, recall)
        """
        precision = (
            self.all_zerozero * 100 / (self.all_zerozero + self.all_onezero)
            if (self.all_zerozero + self.all_onezero) > 0
            else 0.0
        )
        recall = (
            self.all_zerozero * 100 / self.all_total_colliding
            if self.all_total_colliding > 0
            else 0.0
        )
        return precision, recall

    def get_collision_ratio(self):
        """
        计算colldict中真实的碰撞比率
        基于所有记录的样本（碰撞+自由）计算碰撞样本的比例

        Returns:
            float: 碰撞比率 (0.0 到 1.0)
        """
        if len(self.colldict) == 0:
            return 0.0

        total_collision_samples = 0
        total_free_samples = 0

        for counts in self.colldict.values():
            total_collision_samples += counts[0]  # 碰撞次数
            total_free_samples += counts[1]  # 自由次数

        total_samples = total_collision_samples + total_free_samples

        if total_samples == 0:
            return 0.0

        return total_collision_samples / total_samples


class FixedThresholdStrategy(CollisionPredictionStrategy):
    """
    固定阈值策略
    使用固定的敏感度阈值进行碰撞预测
    """

    def __init__(self, threshold=0.1, update_prob=0.5, max_count=255):
        """
        初始化固定阈值策略

        Args:
            threshold: 碰撞预测阈值 (S值)
            update_prob: 自由样本的更新概率
            max_count: 计数器最大值（默认255，对应8-bit存储）
        """
        super().__init__(update_prob, max_count)
        self.threshold = threshold

    def predict_collision(self, keyy):
        """
        使用固定阈值预测碰撞

        预测规则：如果 碰撞次数 > threshold × 自由次数，则预测为碰撞

        Args:
            keyy: 配置的哈希键

        Returns:
            bool: True表示预测碰撞，False表示预测自由
        """
        if keyy not in self.colldict:
            return False  # 未见过的配置，默认预测为自由

        coll_count = self.colldict[keyy][0]
        free_count = self.colldict[keyy][1]

        # 预测逻辑：碰撞计数 > 阈值 × 自由计数
        return coll_count > self.threshold * free_count

    def __str__(self):
        return f"FixedThresholdStrategy(threshold={self.threshold})"


class AdaptiveThresholdStrategy(CollisionPredictionStrategy):
    """
    自适应阈值策略
    根据colldict中碰撞占优的条目比例动态调整敏感度阈值
    """

    def __init__(self, s_min=0.01, s_max=1.0, update_prob=0.5, max_count=255):
        """
        初始化自适应阈值策略

        Args:
            s_min: 最小敏感度（高碰撞倾向时使用）
            s_max: 最大敏感度（低碰撞倾向时使用）
            update_prob: 自由样本的更新概率
            max_count: 计数器最大值（默认255，对应8-bit存储）
        """
        super().__init__(update_prob, max_count)
        self.s_min = s_min
        self.s_max = s_max
        self.current_threshold = s_max  # 初始使用最大敏感度

    def update_threshold(self):
        """
        根据colldict中的统计数据动态更新阈值
        计算碰撞数大于自由数的条目比例作为碰撞倾向指标
        """
        if len(self.colldict) == 0:
            self.current_threshold = self.s_max
            return

        # 统计碰撞占优的条目数
        collision_dominant_count = sum(
            1 for counts in self.colldict.values() if counts[0] > counts[1]
        )

        # 计算碰撞倾向比例
        collision_dominant_ratio = collision_dominant_count / len(self.colldict)

        # 线性插值计算当前敏感度
        # 碰撞倾向比例越高，阈值越低（越容易预测为碰撞）
        self.current_threshold = (
            self.s_max - (self.s_max - self.s_min) * collision_dominant_ratio
        )

    def predict_collision(self, keyy):
        """
        使用自适应阈值预测碰撞

        Args:
            keyy: 配置的哈希键

        Returns:
            bool: True表示预测碰撞，False表示预测自由
        """
        # 每次预测前更新阈值
        self.update_threshold()

        if keyy not in self.colldict:
            return False  # 未见过的配置，默认预测为自由

        coll_count = self.colldict[keyy][0]
        free_count = self.colldict[keyy][1]

        # 使用动态调整的阈值进行预测
        return coll_count > self.current_threshold * free_count

    def get_current_threshold(self):
        """获取当前的敏感度阈值"""
        return self.current_threshold

    def get_collision_dominant_ratio(self):
        """获取当前colldict中碰撞占优的条目比例"""
        if len(self.colldict) == 0:
            return 0.0
        collision_dominant_count = sum(
            1 for counts in self.colldict.values() if counts[0] > counts[1]
        )
        return collision_dominant_count / len(self.colldict)

    def __str__(self):
        return (
            f"AdaptiveThresholdStrategy(s_min={self.s_min}, s_max={self.s_max}, "
            f"current_threshold={self.current_threshold:.4f}, "
            f"collision_dominant_ratio={self.get_collision_dominant_ratio():.4f})"
        )


def generate_hash_key(code_quant, bitsize):
    """
    生成配置的哈希键（通用方法，用于向后兼容）

    Args:
        code_quant: 量化后的配置向量
        bitsize: 哈希码的位数

    Returns:
        str: 哈希键字符串
    """
    keyy = ""
    for j in range(bitsize):
        keyy += str(code_quant[j]).zfill(2)
    return keyy


def generate_coordinate_hash_key(code_quant, bitsize, consider_dir=False, dirr=None):
    """
    生成基于机器人关节坐标的哈希键（coord_hashing方法）
    用于机器人配置空间的碰撞检测

    Args:
        code_quant: 量化后的关节坐标配置向量 [N维]
        bitsize: 坐标的维度数
        consider_dir: 是否考虑运动方向（默认False）
        dirr: 运动方向字符串（仅当consider_dir=True时使用）

    Returns:
        str: 哈希键字符串，格式为 "0101020304..." (每个坐标用2位数字表示)

    示例:
        >>> code_quant = [1, 5, 10, 23]
        >>> generate_coordinate_hash_key(code_quant, 4)
        '01051023'
        >>> generate_coordinate_hash_key(code_quant, 4, consider_dir=True, dirr='L')
        '01051023L'
    """
    keyy = ""
    for j in range(bitsize):
        if code_quant[j] < 10:
            keyy += "0"
        keyy += str(code_quant[j])

    # 如果考虑方向，将方向信息添加到键中
    if consider_dir and dirr is not None:
        keyy += dirr

    return keyy


def generate_sphere_hash_key(position_quant, radius_quant=None, consider_radius=False):
    """
    生成基于球体位置和半径的哈希键（coord_hashing_sphere方法）
    用于球体障碍物的碰撞检测

    Args:
        position_quant: 量化后的球体位置 [x, y, z]（3维向量）
        radius_quant: 量化后的球体半径（标量）
        consider_radius: 是否在哈希键中包含半径信息（默认False）

    Returns:
        str: 哈希键字符串
            - 仅位置: "010203" (x,y,z各2位数字，共6位)
            - 位置+半径: "01020305" (x,y,z各2位，半径2位，共8位)

    示例:
        >>> position_quant = [1, 5, 10]
        >>> generate_sphere_hash_key(position_quant)
        '010510'
        >>> generate_sphere_hash_key(position_quant, radius_quant=15, consider_radius=True)
        '01051015'
    """
    keyy = ""

    # 添加球体位置信息 (x, y, z)
    for coord in position_quant:
        if coord < 10:
            keyy += "0"
        keyy += str(coord)

    # 根据参数决定是否添加球体半径信息
    if consider_radius and radius_quant is not None:
        if radius_quant < 10:
            keyy += "0"
        keyy += str(radius_quant)

    return keyy


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
    for ex in range(0, 10000):
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


def evaluate_strategy_on_trajectory(
    strategy, code_pred_quant, label_pred, group_size=11
):
    """
    在一条轨迹上评估碰撞预测策略

    Args:
        strategy: 碰撞预测策略实例
        code_pred_quant: 量化后的配置数据
        label_pred: 真实标签
        group_size: 分组大小（默认11，对应11个关节）

    Returns:
        tuple: (预测正确数, 总碰撞数)
    """
    bitsize = len(code_pred_quant[0])
    correct_predictions = 0
    total_collisions = 0

    # 以group_size为步长遍历所有状态
    for bini in range(0, len(code_pred_quant), group_size):
        predicted = 1  # 默认预测为非碰撞
        true_ans = 1  # 默认真实为非碰撞

        # 检查一个运动轨迹中的group_size个连续状态点
        for i in range(bini, min(bini + group_size, len(code_pred_quant))):
            # 生成哈希键
            keyy = generate_hash_key(code_pred_quant[i], bitsize)

            # 使用策略进行预测
            if strategy.predict_collision(keyy):
                predicted = 0  # 预测为碰撞

            # 更新策略的历史记录
            strategy.update_history(keyy, label_pred[i])

            # 检查真实标签
            if label_pred[i] < 0.5:
                true_ans = 0  # 真实为碰撞
                if predicted == 0:
                    correct_predictions += 1
                    break  # 提前退出

        # 更新统计
        strategy.update_statistics(predicted, true_ans)

        if true_ans == 0:
            total_collisions += 1

    return correct_predictions, total_collisions


def evaluate_strategy_on_spheres(
    strategy, position_quant, radius_quant, label_pred, consider_radius=False
):
    """
    在球体数据上评估碰撞预测策略

    Args:
        strategy: 碰撞预测策略实例
        position_quant: 量化后的球体位置数据 [N, 3] (x, y, z)
        radius_quant: 量化后的球体半径数据 [N] 或 [N, 1]
        label_pred: 真实标签 [N]
        consider_radius: 是否在哈希键中包含半径信息（默认False）

    Returns:
        tuple: (预测正确数, 总碰撞数)
    """
    correct_predictions = 0
    total_collisions = 0

    # 确保 radius_quant 是一维数组
    if len(radius_quant.shape) > 1:
        radius_quant = radius_quant.flatten()

    # 遍历所有球体样本
    for i in range(len(position_quant)):
        predicted = 1  # 默认预测为非碰撞

        # 获取真实标签
        try:
            true_ans = int(label_pred[i].item())
        except (AttributeError, TypeError):
            true_ans = int(label_pred[i])

        # 生成球体哈希键
        keyy = generate_sphere_hash_key(
            position_quant[i],
            radius_quant[i] if consider_radius else None,
            consider_radius=consider_radius,
        )

        # 使用策略进行预测
        if strategy.predict_collision(keyy):
            predicted = 0  # 预测为碰撞

        # 更新策略的历史记录
        strategy.update_history(keyy, label_pred[i])

        # 检查是否正确预测
        if true_ans == 0:  # 真实为碰撞
            total_collisions += 1
            if predicted == 0:  # 预测也为碰撞
                correct_predictions += 1

        # 更新统计
        strategy.update_statistics(predicted, true_ans)

    return correct_predictions, total_collisions


# 使用示例
if __name__ == "__main__":
    print("=" * 70)
    print("碰撞预测策略模块使用示例")
    print("=" * 70)

    # ========== 示例1: 两种预测策略的比较 ==========
    print("\n【示例1】两种预测策略的比较\n")

    # 创建策略实例
    fixed_strategy = FixedThresholdStrategy(threshold=0.1, update_prob=0.5)
    adaptive_strategy = AdaptiveThresholdStrategy(
        s_min=0.01, s_max=1.0, update_prob=0.5
    )

    print(f"固定阈值策略: {fixed_strategy}")
    print(f"自适应阈值策略: {adaptive_strategy}")

    # ========== 示例2: 机器人关节坐标哈希键生成 ==========
    print("\n" + "=" * 70)
    print("【示例2】机器人关节坐标哈希键生成（coord_hashing方法）\n")

    # 模拟机器人7自由度关节配置的量化值
    joint_config_quant = [1, 5, 10, 23, 8, 15, 3]

    # 不考虑运动方向
    key1 = generate_coordinate_hash_key(joint_config_quant, len(joint_config_quant))
    print(f"关节配置: {joint_config_quant}")
    print(f"哈希键（仅位置）: {key1}")

    # 考虑运动方向
    key2 = generate_coordinate_hash_key(
        joint_config_quant, len(joint_config_quant), consider_dir=True, dirr="L"
    )
    print(f"哈希键（位置+方向）: {key2}")

    # ========== 示例3: 球体障碍物哈希键生成 ==========
    print("\n" + "=" * 70)
    print("【示例3】球体障碍物哈希键生成（coord_hashing_sphere方法）\n")

    # 模拟球体位置和半径的量化值
    sphere_position_quant = [8, 15, 22]  # x, y, z坐标
    sphere_radius_quant = 5  # 半径

    # 仅使用位置信息
    key3 = generate_sphere_hash_key(sphere_position_quant, consider_radius=False)
    print(f"球体位置: {sphere_position_quant}, 半径: {sphere_radius_quant}")
    print(f"哈希键（仅位置）: {key3}")

    # 使用位置和半径信息
    key4 = generate_sphere_hash_key(
        sphere_position_quant, sphere_radius_quant, consider_radius=True
    )
    print(f"哈希键（位置+半径）: {key4}")

    # ========== 示例4: 键值生成方法的对比 ==========
    print("\n" + "=" * 70)
    print("【示例4】两种键值生成方法的区别\n")

    print("方法1 - 机器人关节坐标（coord_hashing）:")
    print("  - 应用场景: 机器人配置空间碰撞检测")
    print("  - 输入数据: 多自由度关节角度/位置")
    print("  - 可选信息: 运动方向")
    print("  - 键长度: 灵活（取决于自由度数量）")
    print(f"  - 示例键: {key1}")

    print("\n方法2 - 球体障碍物（coord_hashing_sphere）:")
    print("  - 应用场景: 球体障碍物碰撞检测")
    print("  - 输入数据: 3D空间位置坐标 (x,y,z)")
    print("  - 可选信息: 球体半径")
    print("  - 键长度: 固定（6位或8位）")
    print(f"  - 示例键: {key4}")

    print("\n" + "=" * 70)
