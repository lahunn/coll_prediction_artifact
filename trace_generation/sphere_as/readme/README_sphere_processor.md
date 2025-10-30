# Sphere Collision Processor

球体碰撞检测数据生成程序，用于处理机器人路径规划问题并生成球体级别的碰撞检测数据。

## 功能

该程序读取包含障碍物和配置列表的problem文件，使用球体模型进行碰撞检测，并输出与`collision_data_manager`兼容的数据格式。

### 主要特性

- **球体碰撞检测**：使用SphereEnv对每个机器人配置进行球体级别的碰撞检测
- **OBB比较功能**：可与OBB（link级别）碰撞检测结果进行比较，识别不一致的配置
- **不一致数据保存**：自动保存OBB和sphere结果不一致的配置及对应障碍物
- **兼容输出格式**：输出与`collision_data_manager`兼容的数据结构

## 输出格式

程序输出两个主要数据结构：
- `sphere_data`: 球体坐标数据列表，格式为 `[edge][config][sphere][x,y,z,radius]`
- `sphere_coll_data`: 球体碰撞标签列表，格式为 `[edge][config][sphere]`，其中0表示碰撞，1表示无碰撞

当启用OBB比较时，还会生成不一致数据文件：
- `*_inconsistent.pkl`: 包含不一致配置的障碍物和pose信息

## 使用方法

### 处理单个文件

```bash
python sphere_collision_processor.py --input-file path/to/problem.pkl --robot-name franka --output-file output.pkl
```

### 启用OBB比较

```bash
python sphere_collision_processor.py \
  --input-file path/to/problem.pkl \
  --robot-name franka \
  --output-file output.pkl \
  --obb-data-dir path/to/obb/data/
```

### 批量处理目录

```bash
python sphere_collision_processor.py --input-dir path/to/problems/ --robot-name franka --output-dir output/
```

### 批量处理并比较OBB

```bash
python sphere_collision_processor.py \
  --input-dir path/to/problems/ \
  --robot-name franka \
  --output-dir output/ \
  --obb-data-dir path/to/obb/data/
```

## 参数说明

- `--input-file`: 单个problem文件路径
- `--input-dir`: problem文件目录（批量处理）
- `--robot-name`: 机器人名称（默认: franka）
- `--output-file`: 输出文件路径（单个文件模式）
- `--output-dir`: 输出目录（批量模式）
- `--obb-data-dir`: OBB数据目录路径（用于比较不一致，可选）

## 示例

```bash
# 处理单个Franka问题文件
python sphere_collision_processor.py \
  --input-file ../../trace_files/bit_traces/franka_7_0200.pkl \
  --robot-name franka \
  --output-file franka_7_0200_sphere.pkl

# 处理单个文件并与OBB比较
python sphere_collision_processor.py \
  --input-file ../../trace_files/bit_traces/franka_7_0200.pkl \
  --robot-name franka \
  --output-file franka_7_0200_sphere.pkl \
  --obb-data-dir ../../trace_files/scene_benchmarks/bit_collision_data/

# 批量处理所有Franka问题并比较OBB
python sphere_collision_processor.py \
  --input-dir ../../trace_files/bit_traces/ \
  --robot-name franka \
  --output-dir ../../trace_files/sphere_collision_data/ \
  --obb-data-dir ../../trace_files/scene_benchmarks/bit_collision_data/
```

## 输出文件格式

### 球体数据文件 (pickle)
输出文件为pickle格式，包含两个列表：
1. `sphere_data`: 球体坐标数据
2. `sphere_coll_data`: 球体碰撞标签

### 不一致数据文件 (pickle)
当启用OBB比较时，如果发现不一致配置，会生成额外文件：
- 格式: `{basename}_inconsistent.pkl`
- 内容: `{'obstacles': [...], 'inconsistent_poses': [...]}`
- `inconsistent_poses`包含不一致配置的详细信息

数据结构与`collision_data_manager`中的`obb_link_data`和`obb_link_coll_data`兼容，但使用球体而不是OBB包围盒。