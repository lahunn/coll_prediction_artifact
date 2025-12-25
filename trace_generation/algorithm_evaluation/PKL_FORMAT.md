# PKL文件结构说明

## 数据结构

```python
problems = pickle.load(open('maze_files/kukas_7_3000.pkl', 'rb'))
problem = problems[i]  # 第i个问题
```

每个problem包含4个元素：

```python
obstacles, start, goal, path = problem
```

### 1. obstacles
类型：`list[(halfExtents, basePosition)]`

每个障碍物包含：
- `halfExtents`: `ndarray(3,)` - 半尺寸 [x, y, z]
- `basePosition`: `ndarray(3,)` - 中心位置 [x, y, z]

### 2. start
类型：`ndarray(n,)` - 起始关节角配置（n为自由度数）

### 3. goal
类型：`ndarray(n,)` - 目标关节角配置

### 4. path
类型：`list[ndarray]` - 参考路径点列表

## 使用示例

```python
import pickle
import numpy as np

# 加载数据
with open('maze_files/kukas_7_3000.pkl', 'rb') as f:
    problems = pickle.load(f)

# 读取第0个问题
obstacles, start, goal, path = problems[0]

print(f"障碍物数量: {len(obstacles)}")
print(f"起点: {start}")
print(f"终点: {goal}")
print(f"路径点数: {len(path)}")

# 访问第一个障碍物
half_extents, position = obstacles[0]
print(f"障碍物尺寸: {half_extents}")
print(f"障碍物位置: {position}")
```

## 环境加载

```python
from environment.kuka_env import KukaEnv

env = KukaEnv()
env.load_problems('maze_files/kukas_7_3000.pkl')

# 内部解包方式（kuka_env.py:85）
obstacles, start, goal, path = self.problems[index]
```
