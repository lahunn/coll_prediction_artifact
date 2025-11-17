# 硬件架构文档：用于机器人加速器的碰撞预测

(基于: Shah & Aamodt, "Collision Prediction for Robotics Accelerators", ISCA 2024)

## 1. 概述

本文档描述了一种用于机器人运动规划的硬件加速架构。该架构的核心目的是通过**预测**哪些碰撞检测查询 (Collision Detection Queries, CDQs) 可能会失败（即检测到碰撞），来_减少_冗余的计算。

通过优先执行那些_预测会碰撞_的查询，系统可以更快地丢弃无效的机器人路径，从而提高整体运动规划的吞吐量和能效。

该架构在基线碰撞检测加速器（由多个 `CDU` 组成）的基础上，引入了一个顶层的**碰撞检测查询调度器 (Collision Detection Query Scheduler)** 和多个并行的**碰撞预测单元 (Collision Prediction Unit, COPU)** 模块。

## 2. 顶层系统架构

整个加速器系统按层次结构组织，如图 12 (左) 所示：

![alt text](image.png)

1. **Controller (控制器)**:
    
    - 顶层的主控单元。
        
    - 负责与运动规划算法（例如 MPNet）的采样和路径搜索模块通信。
        
    - 向 `Collision Detection Query Scheduler` 发送运动规划任务（例如，要检查的一系列机器人姿态/运动）和环境占用信息。
        
2. **Collision Detection Query Scheduler (碰撞检测查询调度器)**:
    
    - **用户重点关注组件**。
        
    - 这是顶层的调度单元，位于 `Controller` 之下。
        
    - 它管理着**多个**并行的 `COPU + CDUs` 处理模块。
        
    - **职责**:
        
        - 接收来自 `Controller` 的环境信息和机器人姿态/运动。
            
        - 将这些任务分派给一个或多个可用的 `COPU + CDUs` 模块。
            
        - 从各个模块收集最终的碰撞结果 (Collision result)，并将其汇总后报告给 `Controller`。
            
3. **COPU + CDUs 模块 (并行处理块)**:
    
    - 系统包含多个这样的模块，它们并行工作。
        
    - **每个模块都是一个自包含的处理单元**，内部包含一个 COPU（及其相关逻辑）和一组（例如 6 或 8 个）`Collision Detection Units (CDUs)`。
        
    - **关键的共享模型**: **一个**全局的 `Collision History Table (CHT)` 由系统内**所有**并行的 `COPU` 模块共享访问。
        
    - **澄清**: `CHT` 是一个**中央资源**。每个 `COPU` 模块（及其内部的预测器和更新单元）都会读取和写入这_同一个_ `CHT`。
        

## 3. COPU (碰撞预测单元) 模块微架构

每个 `COPU + CDUs` 模块（如图 12 (右) 所示）包含以下关键组件：
![alt text](image-1.png)

### 3.1. OBB Generation Unit (OBB 生成单元)

- **输入**: 机器人的姿态 (Robot's pose)（来自顶层调度器）。
    
- **功能**:
    
    1. 执行正向运动学 (Forward Kinematics) 计算。
        
    2. 为机器人的每个连杆 (link) 生成一个或多个几何基元（例如，Oriented Bounding Boxes, OBBs）。
        
- **输出**: OBBs（包含其中心坐标 `OBB.c`）。
    

### 3.2. Collision Predictor (碰撞预测器)

- **输入**: 来自 `OBB Generation Unit` 的 OBB（特别是其中心坐标 `OBB.c`）。
    
- **功能**:
    
    1. **哈希生成**: 使用 `COORD` 哈希算法 (见第 4 节) 计算 `OBB.c` 的哈希码 (hash code)。
        
    2. **CHT 访问**: 以此哈希码为地址，从**共享的** `Collision History Table (CHT)` 中读取相应的条目。
        
    3. **读取计数器**: 获取两个值：`COLL` (碰撞计数) 和 `NONCOLL` (无碰撞计数)。
        
    4. **预测逻辑**: 执行比较：`if (COLL > S * NONCOLL)`。
        
- **输出**: 预测结果（`True` = 预测碰撞, `False` = 预测无碰撞）。
    

### 3.3. Collision History Table (CHT, 碰撞历史表)

- **实现**: SRAM 存储器。
    
- **共享模型**: 这是一个**全局共享**的SRAM。它被系统内**所有**的 `COPU` 模块（及其内部的 `Collision Predictor` 和 `Query Update Unit`）并发访问。
    
- **条目结构**:
    
    - 每个条目对应一个哈希桶。
        
    - 标准配置下，每个条目包含**两个 4-bit 饱和计数器**:
        
        - `COLL`: 记录映射到此条目的 CDQ 过去_导致碰撞_的次数。
            
        - `NONCOLL`: 记录映射到此条目的 CDQ 过去_未导致碰撞_的次数。
            
    - _注_: 当 `S=0` (最激进策略) 时，`NONCOLL` 计数器可以被优化掉，每个条目仅需 1-bit (`COLL > 0`?)。
        
- **大小**: 典型值为 4096 条目 (对应 12-bit 哈希码)。
    
- **生命周期**: CHT 在**每次新**的运动规划查询开始时（即环境可能已更新）被**重置为零**。
    

### 3.4. Queues (优先级队列)

- **QCOLL (碰撞预测队列)**:
    
    - 存储被 `Collision Predictor` 预测为**会碰撞**的 OBBs。
        
    - **高优先级**。
        
    - 尺寸较小 (例如 8-entry)。
        
- **QNONCOLL (无碰撞预测队列)**:
    
    - 存储被 `Collision Predictor` 预测为**不会碰撞**的 OBBs。
        
    - **低优先级**。
        
    - 尺寸较大 (例如 56-entry)，用于缓冲。
        

### 3.5. Query Dispatcher (查询分派器)

- **功能**: 管理 OBBs 到_本模块内_的并行 `CDUs` 的分派。
    
- **核心调度逻辑**:
    
    1. **优先处理 QCOLL**: 只要 `QCOLL` 队列**非空**，并且有**空闲的 `CDU`**，就立即从 `QCOLL` 取出 OBB 并发给 `CDU`。
        
    2. **处理 QNONCOLL**: **仅当 `QCOLL` 队列为空**时，才考虑从 `QNONCOLL` 分派。
        
    3. **QNONCOLL 分派触发条件**: 即使 `QCOLL` 为空，从 `QNONCOLL` 分派也必须满足以下**任一**条件：
        
        - a) `QNONCOLL` 队列**已满** (必须清空以接收新 OBB)。
            
        - b) `OBB Generation Unit` 已处理完当前姿态/运动的**所有 OBBs** (表示没有新的 OBB 会进入 `QCOLL`，必须开始处理 `QNONCOLL` 中的剩余工作)。
            

### 3.6. Collision Detection Units (CDUs, 碰撞检测单元)

- **功能**: 基线的硬件单元，执行实际的、计算密集型的碰撞检测。
    
- **输入**: 一个 OBB（来自 `Query Dispatcher`）和环境占用数据。
    
- **输出**: `CDQ result` (布尔值：`True` = 发生碰撞, `False` = 无碰撞)。
    

### 3.7. Query Update Unit (查询更新单元)

- **功能**: 在 CDQ 执行_后_，将结果写回**共享的 CHT**。
    
- **输入**: `CDQ result` (来自 `CDU`) 和该 OBB 的哈希码 (来自 `Collision Predictor` 或随 OBB 传递)。
    
- **更新逻辑**:
    
    1. 读取**共享 CHT** 中对应哈希码的条目。
        
    2. 如果 `CDQ result` 为 `True`，则**递增 `COLL` 计数器**。
        
    3. 如果 `CDQ result` 为 `False`，则**根据 `U` 参数决定是否递增 `NONCOLL`** (见 6.2 节)。
        
    4. 将更新后的计数器写回 CHT。（需要处理并发写入）
        

### 3.8. Result Collector (结果收集器)

- **功能**: 聚合来自_本模块内_所有 `CDUs` 的结果。
    
- **逻辑**:
    
    - 对所有 `CDQ result` 执行逻辑 "OR" 操作。
        
    - **提前退出 (Early Exit)**: **一旦任何一个 `CDU`** 返回 `True` (发生碰撞)，`Result Collector` 立即停止等待其他 `CDUs`，并向_顶层的 `Collision Detection Query Scheduler`_ 报告 `Collision result = True`。
        

## 4. 核心算法: COORD 哈希函数

COORD 是用于将机器人连杆的物理位置映射到 CHT 条目的哈希函数 (如图 10)。

- **输入**: 机器人连杆（或其 OBB）的**物理空间**中心笛卡尔坐标 (x, y, z)。
    
- **算法**:
    
    1. 获取每个坐标 (x, y, z) 的 16-bit 定点表示。
        
    2. **提取**: 仅保留每个坐标的最高有效位 (MSBs)，例如 4-bit MSBs。
        
    3. **串联**: 将这三组 MSBs **串联** (Concatenate) 起来，形成最终的 CHT 地址。
        
- **示例**:
    
    - `link.c[0]` (x) -> `MSBs(x, 4-bit)`
        
    - `link.c[1]` (y) -> `MSBs(y, 4-bit)`
        
    - `link.c[2]` (z) -> `MSBs(z, 4-bit)`
        
    - `Hash Code (12-bit)` = `MSBs(x) | MSBs(y) | MSBs(z)`
        
- **原理**: 这种哈希方式利用了**物理空间局部性**。物理上彼此靠近的连杆中心将被映射到 CHT 中的相同或邻近的条目，从而共享碰撞历史。
    

## 5. 关键数据流与逻辑 (用于代码设计)

以下是处理单个机器人姿态 (Pose) 的完整数据流：

1. **生成 (Generate)**: `OBB Generation Unit` 接收一个 `Pose`，并为该姿态的所有 N 个连杆 OBBs。
    
2. **循环处理 OBB (Loop OBBs)**: 对于 N 个 OBBs 中的_每一个_ OBB：
    
3. **哈希 (Hash)**: `Collision Predictor` 计算该 OBB 中心的 `COORD` 哈希码。
    
4. **读取 (Read)**: `Collision Predictor` 使用哈希码从**共享 CHT** 读取 `[COLL, NONCOLL]`。
    
5. **预测 (Predict)**: `Collision Predictor` 计算 `prediction = (COLL > S * NONCOLL)`。
    
6. **入队 (Enqueue)**:
    
    - 如果 `prediction == True`，OBB 进入 `QCOLL`。
        
    - 如果 `prediction == False`，OBB 进入 `QNONCOLL`。
        
7. **分派 (Dispatch)**: `Query Dispatcher` 并行地：
    
    - a) 检查 `QCOLL`，如果非空且有空闲 `CDU`，则分派。
        
    - b) 检查 `QCOLL` 是否为空，如果为空，则检查 `QNONCOLL` 的分派条件（队列满或工作完成），满足条件且有空闲 `CDU` 时，才从 `QNONCOLL` 分派。
        
8. **执行 (Execute)**: `CDU` 接收 OBB，执行碰撞检测，返回 `CDQ result` (True/False)。
    
9. **更新 (Update)**: `Query Update Unit` 接收 `CDQ result` 和哈希码。
    
    - 如果 `result == True`，`CHT[hash].COLL++`。（注意：此操作需保证原子性或在共享资源上正确处理并发）
        
    - 如果 `result == False`
        

且 random() < U，CHT[hash].NONCOLL++。（同上）

10. 收集 (Collect): Result Collector 监听所有 CDU。

* 如果任何 CDQ result == True: 立即向顶层报告 Collision = True。

* 如果所有 CDQs 完成且均为 False: 向顶层报告 Collision = False。

## 6. 关键设计参数 (用于配置)

在设计实现时，以下参数是可配置的：

- **`S` (预测策略阈值)**:
    
    - 控制预测器的"激进"程度。
        
    - `S=0`: 最激进。只要 `COLL > 0` 就预测碰撞。
        
    - `S=1/2` 或 `S=1`: 平衡策略。
        
    - `S=2`: 保守策略（需要更多碰撞证据才预测碰撞）。
        
- **`U` (无碰撞更新频率)**:
    
    - 一个 `[0.0, 1.0]` 之间的浮点数 (或定点数)。
        
    - `U=1.0`: 每次无碰撞都更新 `NONCOLL`。
        
    - `U=0.125`: 只有 12.5% 的无碰撞结果会更新 `NONCOLL`，以减少 CHT 的写流量。
        
- **CHT Size (CHT 大小)**:
    
    - 例如：`4096` 条目 (12-bit 哈希)。
        
- **CHT Entry Width (CHT 条目位宽)**:
    
    - 例如：`8-bit` (4-bit `COLL` + 4-bit `NONCOLL`)。
        
- **Queue Sizes (队列大小)**:
    
    - `QCOLL_SIZE`: e.g., `8`
        
    - `QNONCOLL_SIZE`: e.g., `56`
        
- **CDU Grouping (CDU 分组)**:
    
    - 每个 COPU 模块管理的 CDU 数量 (e.g., `N_CDUS_PER_COPU = 6`)。