# 🔧 PyBullet GUI 错误修复说明

## 问题描述

错误信息：
```
pybullet.error: Only one local in-process GUI/GUI_SERVER connection allowed.
```

## 根本原因

PyBullet **只允许创建一个GUI连接**。原代码在循环中重复创建 `RobotEnv(GUI=True)`，导致第二次创建时报错。

## 解决方案

### ✅ 已修复的改动

1. **复用环境实例**：在循环外只创建一次环境
2. **动态更新障碍物**：每次迭代时清除旧障碍物，创建新障碍物
3. **添加障碍物ID跟踪**：记录每个障碍物的ID以便清除

### 修改前后对比

#### ❌ 修改前（会报错）
```python
while success_count < num_problems:
    # 每次都创建新环境 - 第二次会崩溃！
    env = RobotEnv(GUI=visualize, robot_file=robot_file)
    # ...
```

#### ✅ 修改后（正常工作）
```python
# 只创建一次环境
env = RobotEnv(GUI=visualize, robot_file=robot_file)
obstacle_ids = []

while success_count < num_problems:
    # 清除旧障碍物
    for obs_id in obstacle_ids:
        try:
            p.removeBody(obs_id)
        except Exception:
            pass
    obstacle_ids.clear()
    
    # 创建新障碍物
    for halfExtents, basePosition in obstacles:
        obs_id = env.create_voxel(halfExtents, basePosition)
        if obs_id is not None:
            obstacle_ids.append(obs_id)
```

## 使用方法

### 1. 可视化模式（调试用）

使用VS Code调试配置 **"调试 generate_problem_dataset.py (可视化)"**，或：

```bash
python generate_problem_dataset.py \
    --num-problems 3 \
    --num-obstacles 5 \
    --visualize \
    --max-time 30.0
```

**注意**：可视化模式会慢很多，仅用于调试！

### 2. 批量生成模式（推荐）

不使用 `--visualize` 标志，速度更快：

```bash
python generate_problem_dataset.py \
    --num-problems 100 \
    --num-obstacles 10 \
    --max-time 60.0
```

## 其他改进

### 安全区域保护

现在障碍物不会生成在机器人基座附近：

```bash
# 调整安全区域半径（默认0.4米）
python generate_problem_dataset.py \
    --safe-zone-radius 0.5 \
    --num-problems 10
```

### 工作空间范围

```bash
# 扩大工作空间
python generate_problem_dataset.py \
    --workspace-min -1.5 \
    --workspace-max 1.5 \
    --num-problems 10
```

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--visualize` | False | 是否开启GUI可视化（调试用） |
| `--safe-zone-radius` | 0.4 | 机器人基座安全区域半径(米) |
| `--workspace-min` | -1.5 | 工作空间最小坐标 |
| `--workspace-max` | 1.5 | 工作空间最大坐标 |
| `--num-obstacles` | 10 | 每个问题的障碍物数量 |
| `--max-time` | 60.0 | 每个问题的最大规划时间(秒) |

## 性能建议

1. **调试时**：使用 `--visualize` + 少量问题（3-5个）
2. **测试时**：不用 `--visualize` + 中等问题（50-100个）
3. **生产时**：不用 `--visualize` + 大量问题（1000-3000个）

## 验证修复

运行快速测试：

```bash
cd /home/lanh/project/robot_sim/coll_prediction_artifact/trace_generation/bit_planning

# 测试1: 无可视化（快速）
python generate_problem_dataset.py --num-problems 3 --num-obstacles 5

# 测试2: 带可视化（观察障碍物和机器人）
python generate_problem_dataset.py --num-problems 2 --num-obstacles 5 --visualize
```

如果看到问题顺利生成且没有错误，说明修复成功！✅

## 故障排除

### Q: 仍然看到GUI错误？
A: 确保没有其他PyBullet GUI程序在运行，关闭所有其他Python进程后重试。

### Q: 可视化模式看不到GUI窗口？
A: 检查是否在远程SSH会话中，GUI需要X11转发或本地运行。

### Q: 所有起点/终点都碰撞？
A: 增大 `--safe-zone-radius` 到 0.5-0.6米，或减少 `--num-obstacles`。

---

修复完成！现在可以安全地使用可视化模式调试了。🎉
