# CSP (Coarse-step Policy) 策略详解

**文档来源分析：** _Shah 等 - 2023 - Energy-efficient realtime motion planning_

## 1. 概述 (Introduction)

**CSP (Coarse-step Policy，粗步长策略)** 是一种用于机器人运动规划（Motion Planning）中碰撞检测的调度算法。

在基于采样的运动规划（如 MPNet, RRT* 等）中，碰撞检测占据了超过 90% 的计算时间。CSP 的核心目标是解决**动作内并行（Intra-motion parallelism）**时的计算冗余问题，通过一种“由粗到细”的跳跃式检测顺序，在保证检测准确性的同时，最大程度地减少不必要的计算，从而提升能效和速度。

## 2. 背景与动机 (Motivation)

### 2.1 空间局部性问题 (Spatial Locality)

在检查一条连续的运动路径时，路径被离散化为一系列紧密的姿态点（Poses），记为 $p_0, p_1, p_2, \dots, p_N$。

- **物理邻近性**：$p_i$ 和 $p_{i+1}$ 在物理空间上非常接近。
    
- **结果相关性**：如果 $p_i$ 没有发生碰撞，那么 $p_{i+1}$ 发生碰撞的概率也很低；反之亦然。
    

### 2.2 朴素并行 (Naive Parallelism) 的缺陷

如果简单地将相邻的点（如 $p_0, p_1, p_2, p_3$）分配给并行的硬件单元同时检测：

1. **冗余计算**：由于空间重叠度高，这 4 个点可能都在检测同一个障碍物的同一部分，浪费了并行资源。
    
2. **无法快速退出**：如果障碍物位于路径的后半段（如 $p_{100}$），朴素并行依然会按顺序推进，无法尽早发现碰撞并终止整个路径的检查。
    

因此，需要一种策略能够**快速覆盖整个运动路径的物理空间**，而不是在局部死磕。

## 3. CSP 策略原理 (Core Mechanism)

CSP 采用一种**固定步长（Fixed Step Size）**的跳跃式调度方法。它不再按索引顺序连续检查，而是按照步长间隔选取点进行检查。

### 3.1 算法逻辑

假设一条路径上有 $N$ 个离散点，设定步长为 $K$（Step Size）。

调度器会按照以下轮次（Rounds）分发检测任务：

- **第一轮 (Offset = 0)**：检查索引为 $0, K, 2K, 3K, \dots$ 的点。
    
    - _目的_：以最稀疏的方式快速扫描整条路径，覆盖最大的物理范围。
        
- **第二轮 (Offset = 1)**：检查索引为 $1, 1+K, 1+2K, \dots$ 的点。
    
    - _目的_：填补第一轮扫描留下的空隙。
        
- **...**
    
- **第 K 轮 (Offset = K-1)**：检查索引为 $K-1, 2K-1, \dots$ 的点。
    

一旦在任何一轮中检测到**碰撞**，调度器立即停止该路径剩余所有点的检测任务（Early Exit）。

### 3.2 示例演示

假设路径有 $N$ 个点，步长 Step Size = 4。

检测顺序如下：

1. **Cycle 1**: $p_0, p_4, p_8, p_{12}, \dots$  
    
2. **Cycle 2**: $p_1, p_5, p_9, p_{13}, \dots$  
    
3. **Cycle 3**: $p_2, p_6, p_{10}, p_{14}, \dots$  
    
4. **Cycle 4**: $p_3, p_7, p_{11}, p_{15}, \dots$  
    

## 4. 数学描述 (Mathematical Representation)

对于给定的步长 $S$ 和路径总点数 $N$，第 $t$ 次调度选取的点 $P_{index}$ 可以描述为：

$$P_{index} = (offset) + k \cdot S$$

```
P_{index} = (offset) + k \cdot S
```

其中：

- $offset$ 从 $0$ 递增到 $S-1$（外层循环）。
    
- $k$ 为自然数 $0, 1, 2, \dots$，且满足 $P_{index} < N$（内层循环）。
    

## 5. CSP 的优势 (Advantages)

### 5.1 高效的空间探索 (Space Exploration)

相比于顺序检测，CSP 能更快地“撞上”障碍物。

- **图解理解**：如果路径穿过一个障碍物，顺序检测需要从头走到障碍物边缘才能发现。而 CSP 通过大步长跳跃，极大概率在第一轮或第二轮就落在障碍物内部或附近，从而触发“早退机制”。
    

### 5.2 硬件实现极其简单 (Hardware Simplicity)

论文将 CSP 与另一种策略 **BRP (Binary Recursive Policy，二分递归策略)** 进行了对比。

- **BRP**：需要维护一个复杂的队列（Queue）来存储待检测的区间中点（如先查 $N/2$，再查 $N/4, 3N/4$）。
    
- **CSP**：只需要简单的**计数器（Counter）和加法器（Adder）**。
    
    - 寄存器存储当前的 $offset$ 和 $index$。
        
    - 下一周期 $index = index + StepSize$。
        
    - 当 $index \ge N$ 时， $offset = offset + 1$。
        

这种简单的逻辑大大减少了硬件调度器（SAS）的面积和功耗。

### 5.3 能效比提升 (Energy Efficiency)

实验数据显示（参见论文 Figure 7）：

- 随着并行度（CDU 数量）增加，CSP 的检测次数（Work Efficiency）显著优于朴素并行（NP）。
    
- CSP 在保持高加速比的同时，并未显著增加为了并行而引入的额外冗余计算。
    

## 6. 进阶应用：MCSP

在论文提出的最终架构 **MPAccel** 中，CSP 并非独立使用，而是与“动作间并行”结合，形成了 **MCSP (Multi-motion Coarse-step Scheduling Policy)**。

- **Multi-motion**：同时选取一组动作（例如 16 条路径）。
    
- **Coarse-step**：对这 16 条路径中的每一条，都应用 CSP 策略（例如步长设为 8）。
    

**结论**：CSP 是 MPAccel 实现低功耗、实时运动规划的基石策略，它成功地解决了并行计算中常见的“冗余工作”难题。