# 碰撞检测轨迹与数据集生成模块 (trace_generation)

该模块负责生成运动规划及碰撞预测研究所需的所有数据集（Traces），包括随机碰撞检测场景数据集与实际运动规划路径轨迹数据集。它还提供了基于 Python 和 C++（基于 pybind11 绑定）的双版本高效几何碰撞检测算法库，以及多种运动规划器（如 BIT*、RRT、Lazy SP、GNN 路径规划器）的基准测试框架。

> [!NOTE]
> 该模块已完成了目录重构，形成了清晰的高内聚、低耦合架构。所有的导入应当通过统一的模块化包路径（例如 `from trace_generation.core.collision.geometric_collision_detection import sphere_aabb`）。

---

## 📁 目录结构与模块说明

```
trace_generation/
├── core/                       # 核心算法库
│   ├── collision/              # 几何碰撞检测引擎
│   │   ├── geometric_collision_detection.py # Python 手写高度优化的基本几何体碰撞求交算法
│   │   ├── obb_detector.py     # OBB (有向包围盒) 碰撞检测器
│   │   ├── sphere_detector.py  # Sphere (球体包络) 碰撞检测器
│   │   ├── sphere_method.py    # 球体相交检测辅助实现
│   │   ├── cpp_collision/      # C++ 加速碰撞检测模块 (提速 10-50 倍)
│   │   │   ├── CMakeLists.txt  # C++ 编译配置文件
│   │   │   ├── bindings.cpp    # pybind11 Python 接口绑定代码
│   │   │   ├── collision_detection.h # 核心有向包围盒与几何体相交算法
│   │   │   └── build.sh        # 一键编译安装脚本
│   │   └── link_collision_detector.py # 连杆级碰撞检测器接口
│   ├── robot/                  # 机器人运动学与几何建模
│   │   ├── environment.py      # 机器人 PyBullet 仿真环境与正运动学
│   │   ├── modular_env.py      # 机器人场景与环境的模块化管理
│   │   ├── obb_calculator.py   # 计算连杆在世界坐标系下 OBB 姿态与尺寸
│   │   ├── obb_forward_kinematics.py # 前向运动学求 OBB
│   │   ├── sphere_analyzer.py  # 分析并将连杆分解为多球体包络的模型
│   │   └── robot_config/       # 不同型号机器人 (Franka Panda, KUKA IIWA) 的几何参数
│   └── scene/                  # 环境场景控制
│       ├── scene_generator.py  # 随机障碍物场景生成器
│       └── obstacle_manager.py # 场景中障碍物管理与冲突检测
├── scripts/                    # 轨迹生成与性能对比可执行脚本
│   ├── pred_trace_generation.py# 批量采样生成哈希预测所需的 OBB 碰撞 Trace
│   ├── generate_sphere_data.py # 批量生成球体碰撞 Trace 数据的脚本
│   ├── compare_collision_cost.py# 分析并对比 OBB 检测与 Sphere 检测在 CPU 上的耗时
│   ├── launch_pred.sh          # 批量生成 scene_benchmarks 数据的入口脚本
│   └── generate_sphere_data.sh # 批量生成球体测试数据的脚本
├── config/                     # 参数配置
│   └── ana_parameters.py       # 存放机器人几何参数、包围盒数量及仿真成本常数
├── workspace_bound/            # 存放机器人可达关节空间的边界配置 (JSON)
│   ├── franka_panda_workspace.json
│   └── iiwa_workspace.json
├── visualization/              # 基于 PyBullet 的可视化工具
│   ├── collision_visualizer.py # 场景与碰撞的可视化展示
│   ├── robot_sphere_visualizer.py # 机器人多球体包络模型的可视化
│   └── simple_obb_visualization.py # 机器人 OBB 连杆包迹的可视化
├── algorithm_evaluation/       # 运动规划算法与神经网络规划器的训练与评估
│   ├── algorithm/              # 运动规划算法实现
│   │   ├── bit_star.py         # BIT* 算法实现 (用于生成真实规划轨迹)
│   │   ├── gnnmp.py            # GNNMP (图神经网络运动规划) 规划器
│   │   ├── next_planner.py     # Neural Explorer 规划器
│   │   └── lazy_sp.py          # Lazy SP 规划器
│   ├── generate_problem_dataset.py # 批量生成运动规划问题集 (Start, Goal, Obstacles)
│   ├── train_explorer.py       # 训练神经探索器模型
│   ├── train_next.py           # 训练 NEXT 神经运动规划器
│   ├── eval_bit.py             # 评估并提取 BIT* 算法运行过程中的碰撞检测 Trace
│   └── PKL_FORMAT.md           # 轨迹问题数据集存储格式说明文档
└── requirements.txt            # 该模块特有的 Python 依赖包列表
```

---

## 核心算法设计与关键代码解析

### 1. 几何碰撞检测库 (`core/collision/`)

该模块负责判定机器人与障碍物之间是否发生接触，提供了一套轻量级、无庞大第三方物理引擎依赖的高效碰撞检测求解器。

*   **Python 版 (`geometric_collision_detection.py`)**：
    手写的高性能 3D 几何相交相交检查算子（AABB vs AABB, OBB vs OBB, OBB vs Sphere, Capsule vs Capsule, OBB vs Capsule 等）。
    *   **优化策略**：内联简单计算、减少 numpy 数组创建以降低开销，使用标量直接进行分离轴定理（Separating Axis Theorem, SAT）计算，对可能相交的情况进行**早期退出（Early Exit）**判断。
*   **C++ 版 (`cpp_collision/`)**：
    当面对高频率碰撞查询时，纯 Python 耗时过高。因此开发了 C++ 实现的 `collision_detection.h`。通过 `bindings.cpp` 将 C++ 函数绑定为 Python 模块。
    *   **性能提升**：对比 Python 实现，C++ 版本能够提供 **10 至 50 倍的碰撞检查加速**。
    *   **编译与安装**：
        ```bash
        cd trace_generation/core/collision/cpp_collision/
        bash build.sh
        ```
        编译生成的可共享库（`.so` 文件）将被链接到 Python 环境中自动载入。

### 2. 机器人运动学与几何表征 (`core/robot/`)

用于将关节空间角（Joint Angles）映射到笛卡尔三维空间中：
*   **有向包围盒 (OBB) 表达**：`obb_calculator.py` 依据关节角度，通过正运动学（Forward Kinematics）推算出机器人每个 Link 在空间中的旋转矩阵与中心位置，形成有向立方体包围盒（Cuboid），每个 Link 通常对应 1 个 OBB。
*   **多球体包络 (Multi-Sphere) 表达**：`sphere_analyzer.py` 通过一组重叠分布的球体（Sphere）来逼近机器人 Link 的复杂几何外轮廓。在碰撞预测中，球体相比 OBB 碰撞判定逻辑简单得多（只需计算中心距离与半径之和），因此硬件开销更小。

### 3. 数据生成脚本 (`scripts/` 与 `algorithm_evaluation/`)

这些脚本用于制造整个项目实验的输入数据（Traces）：

*   **随机姿态 Trace（Scene Benchmarks）**：
    `pred_trace_generation.py` 随机产生不同障碍物数量（如 6 个、9 个、12 个 cuboids 对应低、中、高密度）的场景，并在机器人关节空间内进行随机姿态采样。调用 OBB 碰撞检测器进行计算，生成 `obstacles_{benchid}_coord.pkl`，包含：（三维坐标, 运动方向, 碰撞标签 [0-碰撞, 1-自由]）。该数据直接用于评估 `prediction_approaches` 下的各种哈希算法。
*   **实际运动规划轨迹 Trace**：
    在 `algorithm_evaluation/` 中，使用 BIT* 规划器对 7-DOF 机器人解决点对点规划问题。在求解路径过程中，规划器会不断在图节点和边上发起大量的碰撞查询。`eval_bit.py` 会将这些查询及其实时碰撞标签录制为日志 Trace（保存在 `logfiles_BIT_link/` 目录下），用于微架构仿真器运行真实时序仿真。

---

## 🚀 轨迹数据生成指南

您可以按照以下步骤生成个性化轨迹数据集：

### 1. 批量生成随机场景与位姿碰撞 Trace (用于哈希方法对比)
运行内置的 launch 脚本，在指定路径下生成 3 种障碍物密度下的随机场景测试数据集：
```bash
cd trace_generation
# 启动场景生成脚本
bash launch_pred.sh
```
该脚本会自动调用 `scripts/pred_trace_generation.py`，并将生成的多维度二进制 pickle 文件存入 `../trace_files/scene_benchmarks/` 目录下。

### 2. 生成多球体碰撞数据集
在完成了 OBB 数据的生成后，提取相对应机器人在相同位姿下的多球体包迹的中心点坐标和半径数据：
```bash
cd trace_generation/scripts
bash generate_sphere_data.sh
```
这将在 `scene_benchmarks` 的相应子目录下生成后缀为 `_sphere.pkl` 的球体碰撞 Trace，主要用于 `coord_hashing_sphere.py` 和球体微架构仿真。

### 3. 运行运动规划闭环日志生成 (用于微架构仿真)
使用标准算法规划轨迹并导出 CDU 执行记录：
```bash
cd trace_generation/algorithm_evaluation
# 生成标准运动规划测试问题集
bash generate_standard_dataset.sh
# 运行 BIT* 规划器并录制碰撞 Trace 日志
bash launch_bit_trace.sh
```
录制完成的 Trace 将生成在 `trace_generation/algorithm_evaluation/logfiles_BIT_link` 目录，以供硬件仿真器读取。
