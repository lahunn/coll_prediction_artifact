# 系统微架构式仿真与性能评估分析报告

> 本文档总结仓库中（主要为 `motion_planning_prediction/` 与 `prediction_approaches/` 目录）关于碰撞检测任务在抽象“微架构”层次上的仿真与性能评估方法。其核心思想是：将并行碰撞检测过程映射为一个离散周期 (cycle-based) 的多执行单元调度模型，对不同策略（CSP / 预测 / Oracle）的效率进行比较。

---
## 1. 总体概念框架
该项目并非精细 CPU/Cache/流水线模拟，而是一个“领域特化的高层微架构抽象 (domain-specific high-level micro-architecture approximation)”。

| 碰撞检测体系元素 | 在本模拟中的抽象 | 类比硬件架构元素 |
|------------------|------------------|------------------|
| 单个 link 的碰撞检测任务 | 最小可调度任务 (Task) | 微操作 (uop) / 指令 |
| 并行 OOCD 单元 (Out-of-Order Collision Detectors) | 固定数量的执行槽位 | 功能单元阵列 (ALUs) |
| 队列 (`qnoncoll`, `qcoll`) | 等待/优先级任务缓冲 | 发射队列 / Ready Queue |
| `cycle` 变量 | 离散时间推进 | 时钟周期 |
| 预测哈希表 `colldict` | 历史统计驱动决策 | 分支预测/局部历史表 |
| Oracle 策略 | 完全知识下的最优调度 | 理论上界（理想硬件） |
| CSP 重排 | 静态启发式顺序 | 静态编排 / 指令排序 |
| 统计输出 (cycles, checks) | 性能指标 | 吞吐/执行时间/指令数 |

目的：比较不同调度/预测策略在路径碰撞检测场景中的“时间”和“计算工作量”节省潜力。

---
## 2. 目录与职责

| 目录 | 角色 | 说明 |
|------|------|------|
| `prediction_approaches/` | 特征/哈希/编码 | 生成或评估位姿、坐标的哈希/编码，支撑预测策略或统计分析 |
| `motion_planning_prediction/` | 调度与仿真 | 搭建并行调度循环，比较 CSP / 预测 / Oracle 策略 |
| `trace_files/` | 输入数据 | 预生成的运动路径及逐 link 碰撞标注 (pickle) |
| `result_files/` | 输出结果 | 各脚本的统计 CSV / 图像 (PDF) |
| Shell 脚本 | 批处理与复现 | 参数扫描、运行多套实验、生成论文图 |

---
## 3. 数据流（端到端）
1. **轨迹来源**：`trace_files/motion_traces/.../*.pkl` 文件包含：
   - `edge_link_data`：路径（edge）序列，每条路径由多个姿态 (pose) 组成，姿态包含若干 link 的连续参数（如关节角 / AABB/OBB 几何编码）。
   - `edge_link_coll_data`：与上结构对应的标签矩阵（`1`=无碰撞，`0`=碰撞）。
2. **可选编码阶段**（`prediction_approaches/`）：对 pose 或 link 参数进行量化哈希或嵌入编码（`pose_hashing.py`，`enpose_hashing.py` 等）。
3. **仿真脚本加载**（`perf_*.py`, `prediction_simulation_*.py`, `CSP_simulation_*.py`）：按 benchid 遍历读取 pickle，生成待检测任务序列。
4. **顺序生成 & 重排**：调用 `csp_rearrange` / `csp_rearrange_oracle` 对 (pose, link) 的检查顺序重排。
5. **调度循环**：并行 OOCD 单元周期推进；按队列取任务、执行、完成、写回、统计。
6. **终止条件**：路径内发现碰撞（提早剪枝）或所有 link 检查完毕。
7. **统计聚合**：每路径和全局的周期数、执行次数、归一化比率、节省百分比。
8. **可视化/报告**：`plot_fig15.py` 等脚本将输出 CSV 分桶、绘制对比图（Fig15 / Fig16 等）。

---
## 4. 关键数据结构

### 4.1 任务表示
每个“任务”对应：某条路径中某一姿态下的一个 link 的碰撞检查。任务在加入调度前可以通过哈希编码生成 key。

### 4.2 Hash & 统计表 `colldict`
- Key：对连续值分箱 (binning) 后的整数序列拼接字符串。
- Value：`[collision_count, free_count]`。
- 用途：在预测策略中估计“该状态再触发碰撞的概率”，指导放入高优先级队列或普通队列（当前部分脚本中逻辑被简化，统计功能保留）。

### 4.3 OOCD 单元结构（逻辑形式）
```
[oocd_hash_key, label, active_flag, finish_cycle]
```
- `active_flag`：是否正在执行。
- `finish_cycle`：完成周期，用于在主循环中判断是否写回。

### 4.4 队列
- `qnoncoll`：默认队列。
- `qcoll`：潜在高碰撞概率（或计划用于优先调度）的专用队列（有的脚本中占位未充分启用）。
- 内部为 Python 列表 FIFO；并未使用优先级堆。

---
## 5. 执行循环（“微架构”仿真核心）
伪代码摘要：
```python
cycle = 0
while True:
    # 1. 回收完成单元
    for unit in units:
        if unit.active and unit.finish_cycle <= cycle:
            record_result(unit.label)
            update_stats(colldict, unit.key, unit.label)
            if unit.label == 0:   # 碰撞 => 路径剪枝
                path_done = True
            unit.active = False

    # 2. 补充任务到空闲单元
    for unit in units:
        if not unit.active and not path_done:
            task = pop(qcoll) or pop(qnoncoll)
            if task:
                unit.key, unit.label = task.key, task.label
                unit.finish_cycle = cycle + LATENCY  # 常量延迟模型
                unit.active = True

    # 3. 若终止条件满足则退出
    if path_done or (queues_empty and all_idle):
        break

    cycle += 1
```
特点：
- **时间模型**：固定延迟，无资源争用回退。
- **并行度**：由命令行提供（执行槽位数）。
- **路径级剪枝**：任何任务标记碰撞即可终止该路径剩余任务（节省模拟工作量）。

---
## 6. 策略差异
| 策略 | 使用信息 | 顺序生成方式 | 适用文件 | 目标 |
|------|----------|--------------|----------|------|
| CSP | 固定启发式重排（姿态末→前 + 内部索引模式） | `csp_rearrange` | `CSP_simulation_*.py`, `perf_csp_simulation_mpnet.py` | 减少冗余检查路径长度 |
| Prediction | 历史统计估计碰撞概率 | 哈希+统计分流（部分精简） | `prediction_simulation_*.py`, `perf_prediction_simulation_mpnet.py` | 提前暴露高概率碰撞 |
| Oracle | 完全真实标签（上界） | 优先抽取最早可判定碰撞的子集 | `perf_oracle_simulation_mpnet.py` | 估计理论最优上界 |

---
## 7. 指标与输出
| 指标 | 说明 |
|------|------|
| `total_cycles` | 单条路径的模拟周期数（时间代理） |
| `all_prediction` / `all_csp` / `all_oracle` | 执行的碰撞检测任务计数（工作量代理） |
| 归一化组均值 | 通过排序+分桶比较不同复杂度场景的策略退化/优势 |
| 节省率 | `(baseline - strategy)/baseline` 形式的相对改进 |

在绘图脚本（如 `plot_fig15.py`）中：
1. 对某列（如碰撞或任务量）排序。
2. 均匀分成若干“组”（groupsize）。
3. 组内求平均，转成相对比率做柱状图。

---
## 8. 预测与哈希（`prediction_approaches/`）
| 文件 | 可能作用 |
|------|----------|
| `pose_hashing.py`, `coord_hashing.py` | 对位姿/坐标进行均匀或自定义分箱，生成哈希 key |
| `enpose_hashing.py` / `_cpu` | 增强型编码（可能包含学习式嵌入 + 量化） |
| `models_new.py` | 定义轻量模型（MLP/编码器）用于碰撞预测或特征抽取 |
| 结果 CSV & pkl | 保存编码后的性能指标或嵌入向量 |
| `plot_fig9.py`, `plot_fig13.py`, `plot_fig14.py` | 评估不同编码在区分碰撞 vs 非碰撞上的效果 |

这些编码/哈希可作为：
- 统计表 key（提高预测泛化能力）
- 后续替换目前脚本中“简单 binning”的输入渠道
- 构建更低冲突/更稳定的预测分布

---
## 9. 与真实微架构模拟的对比
| 方面 | 本项目 | 传统微架构模拟 (gem5 / Sniper 等) |
|------|--------|-------------------------------|
| 指令/任务模型 | 仅一种任务类型，固定延迟 | 多种指令类型 + 复杂依赖 |
| 时间精度 | 整数周期，单阶段 | 多阶段流水线，乱序发射/提交 |
| 资源模型 | 均质执行槽位 | 异构功能单元 + Cache/内存层级 |
| 存储层/带宽 | 未建模 | Cache 命中/缺失、带宽争用 |
| 预测机制 | 统计频率 / 哈希 | 复杂分支预测 / 投机执行 |
| 目标 | 相对策略效果分析 | 绝对性能/能耗/延时预测 |

定位：本项目是一种 **任务调度敏感性分析器**，帮助回答：“若采用更智能的顺序/预测，碰撞检测在抽象并行阵列上的潜在加速是多少？” 而非：精确还原硬件执行细节。

---
## 10. 局限与改进空间
| 现状 | 改进方向 |
|------|----------|
| 延迟固定 | 建模多阶段：Broad-phase / Narrow-phase，不同概率路径不同延迟 |
| 队列为 FIFO | 引入优先队列 / 多级反馈 / aging 策略 |
| 预测准则粗糙 | 使用贝叶斯平滑、置信区间或在线轻量模型 |
| Oracle 仅作为对比 | 定义“信息部分可见”中间层（半 Oracle）估计可行上界 |
| 结果仅均值 | 增加方差/分位数(P50/P95/P99) 及箱线图支持 |
| 只支持离线 traces | 集成交互式 trace 生成器（随机障碍/密度扰动） |
| 缺乏模块化 | 拆分为 `sim_core/` + `strategies/` + `predictors/`，提升复用性 |
| 没有参数配置文件 | 引入 YAML/JSON 批量实验描述 + 统一 orchestrator |

---
## 11. 推荐重构结构（提案）
```
/sim_core
  scheduler.py        # OOCD 资源与循环
  task.py             # LinkTask / Edge 封装
  predictor.py        # 抽象基类 + 统计/学习实现
  metrics.py          # 统一指标聚合导出

/strategies
  csp.py              # 静态重排
  oracle.py           # 上界策略
  predictive.py       # 动态分流策略

/cli
  run_batch.py        # 读取配置矩阵
  run_single.py       # 单次调试
  analyze.py          # 结果后处理

/report
  plot_groups.py
  summarize.py
```
优点：清晰分层、易扩展、可测试。

---
## 12. 典型使用流程（示例）
1. 生成或准备 `trace_files/.../*.pkl`
2. 运行性能对比：
   - `python perf_csp_simulation_mpnet.py <并行度>`
   - `python perf_prediction_simulation_mpnet.py <阈值> <skip因子> <并行度>`
   - `python perf_oracle_simulation_mpnet.py <并行度> MPNET`
3. 聚合结果 (`result_files/perf_data.csv`)
4. 绘制论文图：`bash fig15.sh` 或单独运行 `plot_fig15.py`
5. 对比策略曲线 / 柱状图，分析不同复杂度区间性能差异。

---
## 13. 关键价值与总结
- 提供一种低成本快速实验框架，评估“任务顺序+预测”对碰撞检测并行效率的影响。
- 通过 Oracle 给出理论收益上界，为继续优化提供方向感。
- 预测机制的基础形态（哈希+频次）已验证可行性，后续可替换更精细的学习式模型。
- 该抽象可迁移到其它“可并行早停”类问题：如约束求解、启发式搜索剪枝、并行可满足性测试等。

---
## 14. 快速要点复盘
- 核心：并行调度仿真 + 策略对比，而非硬件精细建模。
- 三策略：CSP / Prediction / Oracle → 静态 vs 动态 vs 上界。
- 指标：周期数、执行次数、节省率、分桶趋势。
- 支撑：哈希/嵌入 → 任务分流潜力。
- 改进：变延迟、优先级队列、统计增强、模块化重构。

---
## 15. 后续可执行改进（建议优先级）
| 优先级 | 任务 | 目的 |
|--------|------|------|
| 高 | 抽象 Predictor 接口 + 迁移现有统计 | 解耦策略与框架 |
| 高 | 增加可变延迟模型 | 更贴近真实 broad/narrow-phase 分层 |
| 中 | 结果统计扩展（方差/分位数） | 体现尾部开销差异 |
| 中 | 队列策略实验（优先堆/aging） | 探索更优调度策略 |
| 低 | YAML 配置批量实验 | 自动化大规模扫描 |
| 低 | 可视化统一脚本化 | 复现稳定性与美观输出 |

---
如需我进一步：
- 添加重构骨架代码
- 引入 Predictor 抽象模块
- 增强统计输出
请告知优先级，我可以继续补充实施。
