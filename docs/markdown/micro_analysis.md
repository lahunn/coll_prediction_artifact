我已整理并扩展整个仓库（侧重 motion_planning_prediction 与 prediction_approaches 两个目录）中“类微架构”仿真与性能评估机制的工作方式。以下是完整分析报告。

## 需求清单
- [x] 中文说明
- [x] Markdown 格式
- [x] 解释项目如何进行“系统微架构”式仿真
- [x] 说明性能测试流程、指标与脚本关系
- [x] 结合两个核心子目录：预测/哈希 与 调度仿真
- [x] 给出关键机制、数据流、扩展建议

---

# 系统微架构式仿真与性能评估分析报告

## 1. 总体概念框架

本项目并未模拟真实 CPU（流水线 / cache / 发射队列），而是将“并行碰撞检测任务调度”抽象为一个极简的“微架构”：
- 碰撞检测请求 → 类似“微操作 (uop)”
- 并行 OOCD（Out-of-Order Collision Detectors）单元 → 类似“功能执行单元阵列”
- 队列（`qnoncoll`, `qcoll`）→ 类似“待发射队列 / 优先级队列”
- cycle 推进循环 → 离散事件/时钟节拍驱动
- 预测模块（基于哈希统计）→ 类似“分支预测 / 访问热点预测”
- Oracle / CSP / Prediction 多策略 → 类似不同调度/推测策略的体系结构配置
- 指标（周期数 / 计算次数 / 归一化查询数）→ 性能评估指标（吞吐、节省率）

核心目的：比较不同碰撞检测顺序与预测机制下的“计算开销”与“并行调度效率”，为运动规划系统的实时性优化提供依据。

---

## 2. 目录与职责映射

| 目录 | 角色 | 说明 |
|------|------|------|
| prediction_approaches | 低层特征哈希与嵌入 | 通过对位姿 / 占据 / 编码的哈希或嵌入，为快速预测/分类提供结构；可能作为训练或统计输入生成模块 |
| motion_planning_prediction | 调度仿真与策略评估 | 将已生成的路径 / 碰撞标注数据作为输入，模拟不同调度策略（CSP / 预测 / Oracle）并统计性能 |
| `trace_files/` | 原始轨迹/采样数据 | 预先离线生成的路径（关节/姿态序列）及其每 link 碰撞标注 |
| `result_files/` | 仿真输出 | 各策略运行后的统计数据（CSV / PDF 图） |
| Shell 脚本 | 批处理与论文复现实验 | 参数扫描、结果整合、图表生成（Fig 9/13/14/15/16 等） |

---

## 3. 数据流 (端到端)

1. 轨迹生成（外部已完成）：
   - 在 `trace_files/motion_traces/...` 下存放 `.pkl`：`(edge_link_data, edge_link_coll_data)`
   - `edge_link_data`: 路径列表，每条路径包含多个姿态，每个姿态包含多个 link 数值（可能为 AABB / OBB 参数或关节角）
   - `edge_link_coll_data`: 同结构二值标注（1=free, 0=collision）

2. 可选特征/哈希编码（prediction_approaches）：
   - `pose_hashing.py`, `coord_hashing.py`, `enpose_hashing*.py`：对 pose/link 连续值做量化与哈希，或嵌入（可能为学习到的编码）
   - 结果可用于：
     - 直接作为预测模型输入
     - 统计模式（在仿真中以 hash key 分类历史概率）

3. 仿真脚本加载数据（`perf_*` / `prediction_simulation_*` / `CSP_*`）：
   - 逐“基准编号 benchid”循环读取对应 `.pkl`
   - 针对每条路径（edge）枚举所有 link 的检查任务

4. 顺序/优先级构造：
   - 基础重排：`csp_rearrange()`（静态启发式序列）
   - Oracle：`csp_rearrange_oracle()`（利用真实标签尽快暴露碰撞，构造上界）
   - 预测：通过历史表（`colldict`）统计 key 的碰撞/非碰撞次数决定进入不同队列（在精简版本中逻辑部分残留/弱化）

5. 并行执行循环（核心微架构模拟）：
   - 初始化 N 个“执行槽位”（OOCD 单元） → idle
   - 循环推进 `cycle`：
     - 完成检测 → 写结果 → 更新统计表 → 若发现碰撞则路径提前结束
     - 空闲槽位拉取队列元素 → 计算 finish_cycle（简单加常量）→ 标记 busy
     - 按条件继续填充队列（保持流水）
   - 终止条件：
     - 发现碰撞（短路）
     - 所有 link 处理完毕
     - 两队列为空且无活跃槽位

6. 统计聚合：
   - 每路径：`total_cycles`, `all_prediction_this_edge`（或 `all_oracle` / CSP 计数）
   - 全局：均值、归一化、节省百分比

7. 可视化与报告：
   - `plot_fig15.py` 等将 CSV 分桶（按复杂度/碰撞分布排序），再归一化绘柱状图比较策略
   - 生成 PDF（论文图）

---

## 4. 微架构抽象细节

### 4.1 执行槽位模型
槽位结构（逻辑）：
```
[oocd_hash_key, label, active_flag, finish_cycle]
```
- `finish_cycle` = 当前 `cycle + latency（常量）`
- 不模拟乱序旁路 / 资源竞争，只模拟并行完成与循环填充

### 4.2 队列设计
- `qnoncoll`：默认任务缓冲
- `qcoll`：潜在“高风险”优先级（在部分版本中未完全使用）
- 使用长度阈值避免一次性灌满
- 无显式老化或重排（除初始 csp_rearrange 输出顺序）

### 4.3 预测统计表 `colldict`
- Key：对 link 数值范围进行 binning 后串联
- Value：计数对（碰撞数，非碰撞数）
- 预测策略（完整版设想）：
  - 若 P(碰撞) > 阈值 → 放入“高优先”队列
  - 否则放入普通队列
  - 当前脚本里多为记录/占位（简化）

### 4.4 Oracle 与 CSP 的对比
| 策略 | 信息来源 | 目标 | 近似意义 |
|------|----------|------|----------|
| CSP | 静态启发式顺序 | 减少冗余，尽量前置“代表性姿态” | 静态调度 |
| Prediction | 历史统计（哈希） | 动态挑出潜在碰撞热点 | 动态/自适应调度 |
| Oracle | 真实标签 | 最短路径揭露碰撞 | 理论上界（下界近似） |

---

## 5. 指标体系

| 指标 | 来源 | 含义 |
|------|------|------|
| `total_cycles` | 仿真循环迭代数 | 抽象“时间” |
| `all_prediction` / `all_csp` / `all_oracle` | 每路径累计已执行的 collision checks | 抽象“工作量” |
| 归一化柱状图 (plot_fig15) | 分桶均值 / 首桶均值 | 比较不同复杂度组的策略退化情况 |
| 计算节省率 | `(CSP_sum - Pred_sum)/CSP_sum` | 预测策略收益 |
| Group 分桶 | 按第二列排序后均匀切块 | 近似“环境难度/查询类型”分层 |

---

## 6. 与真实微架构仿真的类比与差异

| 方面 | 本项目 | 真实微架构仿真（Gem5、Sniper 等） |
|------|--------|------------------------------------|
| 时间建模 | 整数 cycle 自增，无 pipeline 阶段 | 细粒度（取指/译码/发射/执行/写回） |
| 资源 | 均质并行执行槽位 | 多类功能单元（ALU/FPU/LSU） |
| 等待/阻塞 | 无内存/带宽/依赖 | 有 cache miss / 结构 / 数据冒险 |
| 预测 | 基于统计哈希 | 分支预测 / 乱序窗口策略 |
| 终止 | 发现碰撞→短路 | 指令完毕 / 事件完成 |
| 精度目标 | 相对策略开销差异 | 绝对性能/能耗/时序精度 |

定位：这是一个“领域特化的高层架构敏感性模拟器”，用于探索“调度/预测策略”在给定任务结构下的潜在优化幅度。

---

## 7. 预测与哈希层（prediction_approaches）概述

虽然问题聚焦调度仿真，但该目录是上游支撑：

| 文件 | 推测用途 |
|------|----------|
| `pose_hashing.py` / `coord_hashing.py` | 将连续 pose / 坐标做区间量化 + 拼接 → 低冲突键 |
| `enpose_hashing.py` / `_cpu` | 可能含“嵌入 + 哈希”混合方案（en = encoded） |
| `models_new.py` | 可能定义简单神经网络（MLP/CNN）进行碰撞预测学习 |
| `high_obstacle_encodepose.pkl` / `low_obstacle_encodepose.pkl` | 训练后嵌入或编码映射 |
| `plot_fig9.py`, `plot_fig13.py`, `plot_fig14.py` | 比较不同哈希/编码策略区分度、碰撞可分性等 |
| 结果 CSV | 统计不同策略下错误率 / 分布特征 |

这些特征输出可作为：
- 离线：训练更精确预测模型 → 未来替换当前简单统计
- 在线：构造更判别性的 key，降低“预测冲突”与错误调度

---

## 8. 典型执行路径（示意）

以 `launch_perf.sh` 为例：

1. 清理旧结果：`rm result_files/perf_data.csv`
2. 运行：
   - `perf_csp_simulation_mpnet.py <并行度>`
   - `perf_prediction_simulation_mpnet.py <阈值> <跳过因子> <并行度>`
   - `perf_oracle_simulation_mpnet.py <并行度> MPNET`
3. 追加所有结果到 `perf_data.csv`
4. 后续使用 `plot_fig15.py` 聚合成图

---

## 9. 关键算法片段（语义级伪代码）

### 9.1 周期推进主循环（抽象）

```
while True:
    # 回收完成槽位
    for unit in units:
        if unit.active and unit.finish_cycle <= cycle:
            record(unit)
            if unit.collision: terminate_path
            refill_or_mark_idle(unit)

    # 空闲槽位填充
    for unit in units:
        if not unit.active:
            task = pop(qcoll) or pop(qnoncoll)
            if task: dispatch(unit, task, cycle)

    if termination_condition: break
    cycle += 1
```

### 9.2 预测分流（完整版设想）

```
key = hash(pose_link_values)
stats = colldict[key]
p = stats.collision_count / (stats.total + epsilon)
if p > threshold:
    push(qcoll, task)
else:
    push(qnoncoll, task)
```

---

## 10. 当前实现的局限与改进方向

| 类别 | 现状 | 改进建议 |
|------|------|----------|
| 预测判定 | 统计逻辑部分残缺（某些文件注释掉） | 封装 Predictor 接口，支持贝叶斯平滑 / 在线更新 |
| 任务粒度 | 单 link 一次性固定延迟 | 建模多阶段（broad-phase / narrow-phase）并引入可变 latency |
| 队列策略 | 两队列 + 顺序出队 | 引入优先级堆 / aging / 批量调度 |
| 终止条件 | 碰撞即终止 | 增加“置信停机”机制（概率界） |
| 资源模型 | 均质并行槽位 | 支持异构（快速粗检测 + 慢精检）流水 |
| 统计输出 | 平均值 + 分桶 | 增加方差/箱线/尾部 P95/P99 展开 |
| 可复现实验 | Shell 顺序运行 | 提供 JSON/CLI 参数矩阵与单一 orchestrator |
| 数据生成 | 依赖外部 pickle | 集成 trace 采样器（随机场景/密度自动生成） |
| 可视化 | 单图脚本 | 统一 `notebook` 或 `report generator` |

---

## 11. 推荐分层重构（若后续扩展）

```
/sim_core
  scheduler.py         # OOCD 资源与队列管理
  task_model.py        # LinkTask 定义
  predictor.py         # 抽象基类 + 统计/学习实现
  metrics.py           # 统一收集与导出

/strategies
  csp_strategy.py
  oracle_strategy.py
  predictive_strategy.py

/cli
  run_sim.py           # 参数解析 + 批量运行
  analyze_results.py

/report
  plot_groups.py
  aggregate_metrics.py
```

---

## 12. 结论总结

该项目采用“面向任务调度”的抽象微架构仿真框架，通过：
- 将碰撞检测离散化为可并行调度的最小计算单元
- 引入多策略（静态 / 预测 / 理想上界）比较
- 用 cycle 与计算次数模拟“时间 + 资源消耗”
- 通过分桶与归一化展示复杂度敏感性

有效评估了“预测指导的碰撞检测顺序调整”在不同场景下的潜在收益，为后续引入更真实的层次（分阶段检测、异构执行、概率停机）留下扩展空间。

---

## 13. 快速要点（便于复述）

- 本质：任务调度敏感性评估，而非精细硬件仿真  
- 资源：固定数量并行“检测单元”  
- 策略：CSP（静态）、Prediction（统计）、Oracle（上界）  
- 数据：预生成路径 + link 碰撞标签  
- 预测：哈希量化 + 历史统计（当前简化）  
- 指标：周期数 / 检测次数 / 节省率 / 分桶趋势  
- 可改进：预测模型、异构延迟、调度策略、统计精细度

---

## 14. 若需下一步协助

可继续提供：
- 抽象 Predictor 类代码草稿
- 统计增强（方差/分位数）补丁
- 重构脚手架
- 跑一组示例（需您确认可执行依赖）

如需我直接实现上述任一改进，请告诉我优先级。
