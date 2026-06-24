# MPAccel_sim: 机器人加速碰撞预测与运动规划仿真平台

本仓库包含了用于论文 **"Accelerated Collision-Prediction for Autonomous Robots"** (ISCA Paper) 的研究、实验和基准测试代码。

该项目提出了一种基于哈希算法的碰撞预测单元 (COPU)，用于加速自主机器人在运动规划过程中的碰撞检测。通过哈希离散化机器人的状态（关节空间或笛卡尔空间坐标），并在哈希表中查找历史预测结果，从而大幅减少昂贵且耗时的几何/物理碰撞检测 (CDU) 计算。

---

## 📁 1. 目录结构与各文件夹功能

项目主要由以下三大核心研究模块和若干辅助数据/文档文件夹组成：

### 核心实验与仿真模块
1.  **`prediction_approaches/` (碰撞预测算法评估)**
    *   **功能**: 对比和评估不同的碰撞预测哈希策略（在不引入具体运动规划器的情况下进行独立测试）。
    *   **核心策略文件**:
        *   `pose_hashing.py`: 基于机器人原始关节角（Pose）的姿态哈希。
        *   `coord_hashing.py`: 基于机器人各连杆或端点三维空间笛卡尔坐标的坐标哈希。
        *   `enpose_hashing.py` / `enpose_hashing_cpu.py`: 使用自编码器 (Autoencoders) 等神经网络进行状态压缩降维后再进行哈希的编码哈希策略。
        *   `models_new.py`: 包含用于高维数据特征提取的 ResNet 等模型结构。
    *   **功能脚本**:
        *   `bash_script/`: 包含各种对比运行脚本，例如 `run_sphere_cost_analysis.sh` (球体包络) 和 `run_coord_cost_analysis.sh` (OBB包络) 分析。
        *   `plot/`: 包含专门用于提取结果并绘图的脚本，如 `plot_comparison_results.py`，负责生成论文中的绝大多数核心性能图（精确率、召回率、计算开销等）。
        *   `result_files/`: 存储运行产生的实验数据（CSV格式）。

2.  **`motion_planning_prediction/` (运动规划集成仿真与微架构模拟)**
    *   **功能**: 评估将“碰撞预测单元 (COPU) + 碰撞检测单元 (CDU)”双硬件架构集成进具体运动规划算法（如 BIT*、MPNet 等）时的系统级性能表现。
    *   **核心脚本**:
        *   `prediction_simulation_2D.py` / `prediction_simulation_2D_full.py`: 二维空间的碰撞预测与规划仿真。
        *   `CSP_simulation_nDOF.py`: 高自由度 (n-DOF) 机器人规划仿真。
    *   **功能脚本**:
        *   `strategy_evaluation/`: 包含用于评估不同硬件缓存和流水线缓冲机制的仿真运行脚本（例如双缓冲机制 Ping-Pong Buffer 的效果对比）。
        *   `plots/`: 包含分析硬件时钟周期开销、集中式存储冲突率等的绘图脚本。

3.  **`trace_generation/` (碰撞检测轨迹生成)**
    *   **功能**: 负责生成用于前述各个评测模块的机器人运动轨迹、工作空间碰撞状态等标准数据集 (Traces)。该模块在 2025-11-06 进行了目录结构重构，使关注点更分离，引入了模块化导入方式。
    *   **核心结构**:
        *   `core/`: 核心计算库，包含几何碰撞检测 (`core/collision/`)、机器人连杆建模与正运动学环境 (`core/robot/`)。其中 `core/collision/cpp_collision/` 包含了 **C++ 加速的碰撞检测扩展**，可比纯 Python 版本提速 **10-50 倍**。
        *   `scripts/`: 具体的运行脚本（如场景和轨迹批量生成脚本 `launch_pred.sh`、球体数据提取脚本 `generate_sphere_data.sh` 等）。
        *   `config/`: 存放各种机器人模型和系统参数配置（如 `config/ana_parameters.py`）。
        *   `data/`: 存储工作空间范围限制 (`workspace_bounds/`) 等元数据。
        *   `bit_planning/`: 实现 BIT* 运动规划算法，用于生成 KUKA 机器人规划过程中的真实运动轨迹。

### 数据、图表与文档文件夹
*   **`trace_files/` (轨迹数据集目录)**
    *   **功能**: 存放碰撞检测仿真和实验所需的各种输入/输出轨迹和难度测试数据集。
    *   **结构**: 包含 `scene_benchmarks/`（场景基准数据）、`bit_traces/`（运动规划轨迹）、`problems/`（问题集）等子目录。可以直接从下方的数据下载指南中获取预先生成的数据集。
*   **`data/` (场景与机器人模型元数据)**
    *   **功能**: 存储 PyBullet 仿真环境中使用的机器人模型（URDF格式）、障碍物和场景的三维网格/几何文件（OBJ、OFF、XML格式）。
    *   **结构**: 包含 `robots/`、`objects/`、`terrains/` 等目录。
*   **`docs/` (设计规范与文档)**
    *   **功能**: 包含项目的详细架构规格说明（PDF、Markdown 格式），便于开发者深入了解预测算法原理。
*   **`figure/` (图表与可视化输出)**
    *   **功能**: 存放论文发表、报告所使用的各类实验分析图、系统硬件微架构原理图、顶层数据流图，以及部分特殊图像的生成脚本（在 `figure/script/` 中）。

---

## 📥 2. 数据文件下载与管理

由于三维模型网格文件 (`data/`) 和大规模仿真轨迹数据 (`trace_files/`) 文件量大，项目代码库不直接附带，而是托管在云端。在首次运行任何实验前，**必须下载并解压数据文件**。

### 自动下载步骤
在项目根目录下，执行内置的一键下载脚本即可：
```bash
bash download.sh
```

### `download.sh` 内部逻辑与下载链接
该脚本利用命令行工具 `gdown` 从 Google Drive 下载对应的 `.zip` 包并自动解压缩：
1.  **安装 `gdown`**:
    ```bash
    python -m pip install gdown
    ```
2.  **下载并解压轨迹数据 `trace_files.zip`**:
    *   **Google Drive ID**: `1qIPjpnaVPdTzAlKZsiLHd5mtVsJSYLSR`
    *   **解压路径**: 解压为根目录下的 `trace_files/` 文件夹。
3.  **下载并解压网格和模型数据 `data.zip`**:
    *   **Google Drive ID**: `1gHkn0fdEqVJ8-UEq14pc4OOFmqqCXI10`
    *   **解压路径**: 解压为根目录下的 `data/` 文件夹。

```bash
# download.sh 内部核心代码：
gdown 1qIPjpnaVPdTzAlKZsiLHd5mtVsJSYLSR
gdown 1gHkn0fdEqVJ8-UEq14pc4OOFmqqCXI10 
unzip trace_files.zip
unzip data.zip
```

---

## ⚙️ 3. 环境配置与安装

### 推荐环境
*   **Python**: `python==3.6.12` (论文开发所用) 或者是 `python>=3.7` (重构后的兼容环境)
*   **系统**: 建议使用 Linux / macOS

### 快捷安装方式
项目根目录下提供了快捷安装脚本 `install.sh`，可自动完成本地包注册，解决各种相对导入路径报错问题。
```bash
bash install.sh
```
在运行该脚本时：
1.  您可以选择是否自动在当前路径下创建隔离的 `venv` 虚拟环境。
2.  脚本会自动以 **开发模式 (Developer Mode)** 安装此项目：
    ```bash
    pip3 install -e .
    ```
    此命令使得在任何子目录下都可以安全地以 `import trace_generation`、`import prediction_approaches` 等标准方式导入核心库，避免因执行脚本的目录层级不同导致的路径查找错误。

### 手动安装与扩展依赖
您也可以手动安装，首先安装基础依赖：
```bash
python -m pip install -r requirements.txt
```

根据您需要运行的模块，可以使用 `setup.py` 中预设的可选依赖：
```bash
# 1. 安装 OBB 碰撞检测相关的几何与三维处理库 (open3d, coacd, trimesh 等)
pip install -e ".[obb]"

# 2. 安装 CUDA 加速以及神经网络支持库 (PyTorch, CuRobo 等)
pip install -e ".[cuda]"

# 3. 安装开发、格式检查及测试工具 (pytest, black, flake8 等)
pip install -e ".[dev]"
```

---

## 📊 4. 核心实验与图表复现

项目提供了一组开箱即用的 Shell 脚本，可以让您在短时间内复现论文中提出的各项核心数据和趋势图表。

### 实验 1: 比较不同的碰撞预测策略 (复现 Figure 9, 13, 14)
该实验通过评估机器人在随机障碍物场景下的不同 Pose，统计传统基于 OBB 的 Link 哈希和基于 Sphere 包络的哈希策略在不同采样和阈值参数下的精确率、召回率和计算开销占比。
```bash
cd prediction_approaches

# 运行所有 Figure 9 相关的实验与数据导出
bash fig9.sh

# 运行 Figure 13 相关实验
bash fig13.sh

# 运行 Figure 14 相关实验
bash fig14.sh
```
*实验完成后，图表结果将保存在 `prediction_approaches/plot/figs/` 或相关输出目录中。*

### 实验 2: 规划仿真与时钟周期分析 (复现 Figure 15, 16)
该部分仿真将预测单元与真实的运动规划闭环系统集成（使用 BIT* 规划器对 KUKA 机械臂运行轨迹仿真），通过微架构仿真器导出在不同规划难度级别下采用不同缓冲、流水线设计时的硬件周期消耗和冲突率。
```bash
cd motion_planning_prediction

# 运行并复现 Figure 15 (例如 Link 策略与 Sphere 策略集成后的周期对比)
bash fig15.sh

# 运行并复现 Figure 16
bash fig16.sh
```

---

## 🔄 5. 轨迹数据生成 (Trace Generation)

如果您不想直接使用预下载的轨迹数据集，而是希望生成属于自己的全新数据集和运动路径，可以运行以下命令：

### A. 生成用于哈希策略对比的随机轨迹 (Scene Benchmarks)
此脚本会自动生成 400 个包含不同随机障碍物分布的场景，并在每个场景下采样 1000 个机器人的姿态，生成对应的数据存储至 `trace_generation/scene_benchmark`：
```bash
cd trace_generation
# 激活 python 3.7+ 的新环境 (例如 new_env)
conda activate new_env
python -m pip install -r requirements.txt
# 启动批量场景预测轨迹生成
bash launch_pred.sh
```

### B. 生成用于微架构仿真的运动路径轨迹 (BIT* 运动规划)
此脚本会使用 BIT* 规划器对 KUKA-7自由度机械臂运行运动规划，并将解算路径和过程中的碰撞查询保存到 `coll_prediction_artifact/trace_generation/logfiles_BIT_link` 目录，以作为预测仿真的输入数据。
```bash
cd trace_generation/bit_planning
# 为规划器创建专用的 conda 环境 (支持 Ubuntu 18.04 / macOS)
conda env create -f environment.yml
conda activate myenv
# 启动轨迹生成
bash launch_bit_trace.sh
```

---

## 📝 开发者贡献与代码风格

1.  **重构架构导入规范**:
    在当前版本中，请务必使用重构后的包级相对或绝对导入：
    ```python
    # 推荐使用模块化统一导入
    from trace_generation.core.collision.geometric_collision_detection import sphere_aabb
    from trace_generation.core.robot.environment import RobotEnv
    ```
2.  **本地测试**:
    如需验证环境配置和重构后代码的逻辑正确性，可直接运行 pytest：
    ```bash
    pytest trace_generation/tests/
    ```
