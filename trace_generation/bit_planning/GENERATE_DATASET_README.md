# 问题集生成脚本使用指南

## 📝 概述

`generate_problem_dataset.py` 用于生成机器人路径规划问题数据集。每个问题包含障碍物、起点、终点和使用 BIT* 算法规划的路径。

## 🚀 快速开始

### 1. 超快速测试 (推荐首次使用)

生成3个简单问题，验证功能是否正常：

```bash
chmod +x quick_test.sh
./quick_test.sh
```

**预计时间**: 1-5分钟

---

### 2. 完整功能测试

运行完整测试套件，生成15个问题进行验证：

```bash
chmod +x test_generate_dataset.sh
./test_generate_dataset.sh
```

**预计时间**: 5-15分钟

**测试内容**:
- 测试1: 生成5个简单问题 (5个障碍物)
- 测试2: 生成10个中等难度问题 (10个障碍物)
- 数据集验证

---

### 3. 生成标准数据集

生成3000个问题的完整数据集：

```bash
chmod +x generate_standard_dataset.sh
./generate_standard_dataset.sh
```

**预计时间**: 数小时到数天（取决于硬件）

---

## 📋 手动运行示例

### 基本用法

```bash
# 生成默认配置 (3000个问题, 10个障碍物)
python generate_problem_dataset.py
```

### 自定义参数

```bash
# 生成100个问题，每个有15个障碍物
python generate_problem_dataset.py \
    --num-problems 100 \
    --num-obstacles 15 \
    --max-time 60.0
```

### 完整参数示例

```bash
python generate_problem_dataset.py \
    --robot-file kuka_iiwa/model_0.urdf \
    --num-problems 1000 \
    --num-obstacles 12 \
    --max-time 60.0 \
    --workspace-min -0.8 \
    --workspace-max 0.8 \
    --voxel-size-min 0.05 \
    --voxel-size-max 0.12 \
    --output-file maze_files/custom_dataset.pkl
```

---

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--robot-file` | `kuka_iiwa/model_0.urdf` | 机器人URDF文件路径 |
| `--num-problems` | `3000` | 要生成的问题数量 |
| `--num-obstacles` | `10` | 每个问题的障碍物数量 |
| `--max-time` | `60.0` | 每个问题的最大规划时间(秒) |
| `--workspace-min` | `-0.8` | 工作空间最小坐标(米) |
| `--workspace-max` | `0.8` | 工作空间最大坐标(米) |
| `--voxel-size-min` | `0.05` | 体素最小尺寸(米) |
| `--voxel-size-max` | `0.12` | 体素最大尺寸(米) |
| `--output-file` | 自动生成 | 输出文件路径 |

---

## 📊 输出格式

生成的 `.pkl` 文件包含一个问题列表：

```python
problems = [
    (obstacles_0, start_0, goal_0, path_0),
    (obstacles_1, start_1, goal_1, path_1),
    ...
]

# 每个问题的结构:
obstacles: List[Tuple[halfExtents, basePosition]]  # 障碍物体素列表
start: np.ndarray                                   # 起始配置 [q1, q2, ..., qn]
goal: np.ndarray                                    # 目标配置 [q1, q2, ..., qn]
path: List[np.ndarray]                             # 路径配置序列
```

### 读取数据集示例

```python
import pickle

with open('maze_files/kukas_7_3000.pkl', 'rb') as f:
    problems = pickle.load(f)

# 访问第一个问题
obstacles, start, goal, path = problems[0]

print(f"障碍物数量: {len(obstacles)}")
print(f"起点: {start}")
print(f"终点: {goal}")
print(f"路径长度: {len(path)}")
```

---

## 🔧 性能调优

### 快速生成（测试用）

```bash
python generate_problem_dataset.py \
    --num-problems 10 \
    --num-obstacles 5 \
    --max-time 30.0
```

### 标准配置（平衡）

```bash
python generate_problem_dataset.py \
    --num-problems 1000 \
    --num-obstacles 10 \
    --max-time 60.0
```

### 高难度配置（研究用）

```bash
python generate_problem_dataset.py \
    --num-problems 500 \
    --num-obstacles 20 \
    --max-time 120.0
```

---

## ⚠️ 注意事项

1. **时间估算**
   - 每个问题可能需要几秒到几分钟
   - 3000个问题可能需要数小时到数天
   - 建议先用小数据集测试

2. **存储空间**
   - 3000个问题约 10-50 MB
   - 确保有足够的磁盘空间

3. **规划失败**
   - 如果规划频繁失败，考虑：
     - 减少障碍物数量
     - 增加工作空间范围
     - 增加规划超时时间

4. **并行生成**
   - 可以同时运行多个脚本生成不同数据集
   - 注意指定不同的输出文件名

---

## 🐛 故障排除

### 问题1: ImportError

```
解决方案: 确保在 bit_planning 目录下运行
cd /path/to/trace_generation/bit_planning
python generate_problem_dataset.py
```

### 问题2: 规划失败率高

```
解决方案: 减少障碍物或增加超时
python generate_problem_dataset.py \
    --num-obstacles 8 \
    --max-time 90.0
```

### 问题3: 内存不足

```
解决方案: 分批生成
python generate_problem_dataset.py --num-problems 500 --output-file part1.pkl
python generate_problem_dataset.py --num-problems 500 --output-file part2.pkl
```

---

## 📈 示例输出

```
======================================================================
机器人路径规划问题数据集生成器
======================================================================
机器人文件: kuka_iiwa/model_0.urdf
目标问题数: 10
障碍物数量: 10
工作空间范围: (-0.8, 0.8)
体素尺寸范围: (0.05, 0.12)
最大规划时间: 60.0秒
======================================================================
机器人自由度: 7
输出文件: maze_files/model_0_7_10.pkl
======================================================================

生成问题: 100%|████████████████| 10/10 [05:23<00:00, 32.35s/问题]

正在保存 10 个问题到 maze_files/model_0_7_10.pkl...
======================================================================
✓ 数据集生成完成!
✓ 成功生成 10 个问题
✓ 已保存到: maze_files/model_0_7_10.pkl
======================================================================

路径长度统计:
  平均: 15.30
  最小: 8
  最大: 25
  中位数: 14.50
```

---

## 📚 相关文件

- `generate_problem_dataset.py` - 主生成脚本
- `quick_test.sh` - 快速测试 (3个问题)
- `test_generate_dataset.sh` - 完整测试套件 (15个问题)
- `generate_standard_dataset.sh` - 生成标准数据集 (3000个问题)
- `DATASET_GENERATION.md` - 详细文档

---

## 🎯 推荐工作流程

1. **首次使用**: 运行 `quick_test.sh` 验证功能
2. **测试调整**: 运行 `test_generate_dataset.sh` 测试不同参数
3. **小规模生成**: 生成100-500个问题进行初步实验
4. **大规模生成**: 生成完整的3000个问题数据集
