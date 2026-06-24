# 碰撞预测算法评估模块 (prediction_approaches)

该文件夹包含了用于机器人运动规划中碰撞预测的多种哈希 (Hashing) 方法的实现、评估、自适应优化和可视化的相关代码。这些方法旨在通过快速查询历史碰撞数据来预测给定的机器人状态是否会发生碰撞，从而避免昂贵且耗时的几何/物理碰撞检测 (CDU) 计算。

## 📁 目录结构与文件分布

```
prediction_approaches/
├── collision_prediction_strategies.py  # 核心：碰撞预测策略基类与派生类 (固定/自适应阈值)
├── pose_hashing.py                     # 策略 1：基于关节姿态 (Pose) 原始数据的哈希
├── coord_hashing.py                    # 策略 2：基于端点/连杆三维坐标 (Coordinate) 原始数据的哈希
├── coord_hashing_sphere.py             # 策略 3：基于多球包络 (Sphere) 位置和半径的哈希
├── enpose_hashing.py                   # 策略 4：使用自编码器神经网络压缩关节姿态后再进行哈希 (GPU 版)
├── enpose_hashing_cpu.py               # 策略 4.1：使用预先计算压缩数据的姿态哈希 (CPU 运行版)
├── encoord_hashing.py                  # 策略 5：结合自编码器降维后的三维坐标哈希
├── models_new.py                       # 神经网络模型：定义用于提取低维特征的 ResNet 结构
├── optimize_s_parameters.py            # OBB 版本的 S 参数 (敏感度阈值) 搜索优化脚本
├── optimize_s_parameters_sphere.py     # 球体包络版本的 S 参数搜索优化脚本
├── launch_adaptive_evaluation.py       # 自适应阈值预测评估入口脚本
├── utils/
│   ├── utils.py                        # 数学工具：包含蒙特卡洛预测成本模拟和精确解析期望计算公式
│   └── plot_analyze_expected_checks.py # 分析和绘制预测执行期望次数的脚本
├── bash_script/                        # 运行与复现脚本目录 (包含 fig9.sh, fig13.sh, fig14.sh 等)
├── plot/                               # 绘图脚本目录 (包含 plot_fig9.py, plot_fig13.py 等)
└── result_files/                       # 存放运行后生成的 .csv 实验结果数据
```

---

## 核心组件与关键代码解析

### 1. 碰撞预测策略实现 (`collision_prediction_strategies.py`)

它是碰撞预测单元 (COPU) 行为逻辑的核心抽象，定义了如何将真实标签记录至历史表，以及如何依据历史频次预测未知状态。

*   **`CollisionPredictionStrategy` (基类)**：
    *   `colldict`：一个核心哈希表，其键 (`keyy`) 是离散状态的哈希字符串，值是 `[collision_count, free_count]` (碰撞次数与自由次数)。
    *   `update_history(keyy, label)`：更新哈希表。为了模拟硬件有限位宽的 SRAM（如 8-bit SRAM 最大计数 255），它实现了一个**饱和计数器机制**。当某项计数达到 `max_count` 时，碰撞和自由次数同时除以 2（右移 1 位）。为节省内存和带宽，自由样本 (`label=1`) 会按照 `update_prob` 概率随机采样更新，而碰撞样本总是更新。
    *   `inherit_collision_history(source_strategy, rate)`：从另一个策略对象中继承碰撞历史表。支持按比例 `rate`（0.0 ~ 1.0）衰减，用于在连续的规划任务中作为**热启动 (Warm-start)** 机制，降低旧经验的权重。
*   **`FixedThresholdStrategy` (派生类)**：
    *   实现固定敏感度阈值 $S$ 策略。
    *   预测规则：如果 `collision_count > S * free_count`，则预测为碰撞（返回 `True`）；否则预测为自由（返回 `False`）。对于全新未遇到的哈希键，默认预测为自由。
*   **`AdaptiveThresholdStrategy` (派生类)**：
    *   实现自适应敏感度阈值策略。
    *   阈值动态调整：每次调用预测前，根据当前 `colldict` 中碰撞计数大于自由计数的条目比例 $R_{dominant}$，在预设的 $[s_{min}, s_{max}]$ 范围进行线性插值得到当前的 $S$。当场景中碰撞倾向较高时，阈值被自动拉低，使系统更容易预测为碰撞，从而偏向安全和召回率。

### 2. 状态哈希策略的实现

机器人状态的空间结构不同，对应的哈希离散化方式也不同：

*   **姿态哈希 (`pose_hashing.py`)**：
    将机器人的关节角度（7自由度 KUKA 机器人有7个关节角，在 $[-1.0, 1.0]$ 区间内）使用 `np.digitize` 划分为 $B$ 个区间（如 16 或 32 个 bins）。把量化后的关节角度拼接，通过组合方式生成哈希键。
*   **坐标哈希 (`coord_hashing.py` / `coord_hashing_sphere.py`)**：
    *   `coord_hashing.py` 使用机器人的连杆端点或末端执行器的三维笛卡尔坐标来进行离散化。
    *   `coord_hashing_sphere.py` 是用于多球包络 (Multi-Sphere) 碰撞检查的版本。对机器人每个几何碰撞球体的三维空间位置坐标进行等距离散分桶，并可选择是否将量化后的半径信息 (`consider_radius=True`) 组合到哈希键中（如 `x_quant + y_quant + z_quant + radius_quant`）。
*   **编码哈希 (`enpose_hashing.py` / `encoord_hashing.py`)**：
    由于机器人的关节角 and 连杆坐标存在极强的物理耦合度，直接哈希会面临严重的“维度灾难”导致哈希表极稀疏。因此，这些脚本使用 `models_new.py` 中的 `ResNet` 自动编码器 (Autoencoder)，在 GPU 上将高维的机器人状态压缩到极低维（例如 2维或 3维的潜空间向量），然后再将这个低维向量进行离散化哈希。

### 3. S参数 (敏感度阈值) 搜索与优化

在哈希预测中，不同的 $S$ 阈值直接影响预测的**精确率 (Precision)** 和**召回率 (Recall)**。为了找到在特定障碍物密度下计算成本最低的最优 $S$：

*   **`optimize_s_parameters.py` (OBB 版本) & `optimize_s_parameters_sphere.py` (球体版本)**：
    *   以碰撞检查的**预期总开销**为优化目标函数。
    *   在特定密度级别（如 `low`/`mid`/`high`）的数据集上，从 $0.0$ 到 $1.0$（或更大范围）以一定步长扫描固定阈值 $S$。
    *   在每一轮扫描中，通过加载 benchmark 数据并运行策略，统计姿态和元素级别的指标（精确率、召回率、碰撞率等）。
    *   利用 `utils.py` 中的性能估算模型，计算在当前指标下系统预期的碰撞检测总次数，并导出最优的 $S$ 参数写入 csv 记录文件。

### 4. 性能估算数学工具 (`utils/utils.py`)

提供系统级别的碰撞检测开销预估模型，包含以下两个函数：
*   **`find_sim_cost(R, C, A, N)`**：通过 10000 次蒙特卡洛模拟运行，计算预测器在真实碰撞率为 $R$、召回率（覆盖率）为 $C$、精确率为 $A$、任务数为 $N$ 时的平均碰撞检测开销。
*   **`calculate_expected_checks(R, C, A, N)`**：利用**封闭形式 (Closed-form) 的数学期望公式**进行精确求解，无需重复模拟，速度更快且结果确定。该期望计算考虑了在检测到真实碰撞时，后续检测可以“提早终止 (Early Stopping)”的机制。

---

## 📊 实验与图表复现指南

本模块包含多套现成的 Shell 脚本，帮助复现论文中的数据指标。脚本统一放置在 `bash_script/` 目录中，绘图结果保存在 `plot/figs/` 目录中。

> [!NOTE]
> 运行以下脚本前，请确保在根目录下运行了 `bash download.sh` 和 `bash install.sh`，从而获取了 Trace 数据集并完成了模块的本地安装。

### 1. 综合哈希方法对比实验 (Figure 9)
对比 Pose、EnPose、Coord 和 EnCoord 哈希方法在不同障碍物密度下的精确率与召回率：
```bash
cd prediction_approaches/bash_script
bash fig9.sh
```
*   **对应绘图**: 调用 `prediction_approaches/plot/plot_fig9.py`。
*   **复现结果**: 在 `plot/figs/` 下生成 Figure 9 相关的精确率和召回率对比柱状图。

### 2. 坐标哈希 S 参数敏感度评估 (Figure 13)
分析坐标哈希算法在低、中、高密度下随着敏感度阈值 $S$ 变化时，其召回率、精确率以及期望计算成本的变化曲线：
```bash
cd prediction_approaches/bash_script
bash fig13.sh
```
*   **对应绘图**: 调用 `prediction_approaches/plot/plot_fig13.py`。
*   **复现结果**: 在 `plot/figs/` 下生成 Figure 13 对应的三轴敏感度曲线 PDF 图表。

### 3. CHT 更新频率 (Update Frequency) 影响评估 (Figure 14)
评估不同的自由样本更新率对哈希表状态、碰撞预测性能的影响：
```bash
cd prediction_approaches/bash_script
bash fig14.sh
```
*   **对应绘图**: 调用 `prediction_approaches/plot/plot_fig14.py`。
*   **复现结果**: 生成 CHT 更新频率对比折线图。

### 4. 自适应阈值性能评估
运行自适应阈值在测试轨迹上的表现测试：
```bash
cd prediction_approaches
python launch_adaptive_evaluation.py
```
*   评估并对比自适应敏感度策略与最佳固定敏感度策略在不同测试场景下的召回率表现以及对冷启动/动态环境的适应能力。
