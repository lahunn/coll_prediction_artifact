# 球体近似碰撞检测数据生成

基于 `pred_trace_generation.py` 开发的球体近似版本，用于生成基于球体几何的碰撞检测数据。

## 主要差异

### 原始 OBB 方法 vs 球体方法

| 特性 | OBB 方法 | 球体方法 |
|------|----------|----------|
| 几何表示 | 有向包围盒 (OBB) | 球体近似 |
| 每个 link 元素数 | 固定 1 个 | 可变 (基于 YAML 配置) |
| 几何参数 | 中心坐标 + 方向编码 | 中心坐标 + 半径 |
| 配置来源 | 硬编码尺寸 | YAML 文件 (iiwa.yml) |
| 碰撞精度 | 较高 (矩形包围) | 中等 (球形包围) |
| 计算复杂度 | 中等 | 较低 |

## 文件说明

### 1. `sphere_trace_generation.py`
主要的球体数据生成脚本

**输入参数：**
```bash
python sphere_trace_generation.py <numqueries> <foldername> <filenumber>
```

**输出文件：**
- `obstacles_X_sphere.pkl`: 球体级数据
  - `qarr_sphere`: 球体中心坐标 (N×3)
  - `yarr_sphere`: 碰撞标签 (N×1, 0=碰撞, 1=自由)
  - `radius_arr`: 球体半径 (N×1)
  - `link_id_arr`: 所属 link ID (N×1)
  - `sphere_id_arr`: 球体 ID (N×1)
- `obstacles_X_pose.pkl`: 姿态级数据 (与原版相同)

### 2. `test_sphere_generation.sh`
测试脚本，验证球体数据生成是否正常工作

```bash
./test_sphere_generation.sh
```

### 3. `compare_data_formats.py`
数据格式对比脚本，分析 OBB 和球体方法的差异

```bash
python compare_data_formats.py <foldername> <filenumber>
```

## 球体配置

球体定义来自 `configs/robot/spheres/iiwa_allegro.yml`：

```yaml
collision_spheres:
  iiwa7_link_0:
    - center: [0.0, 0.0, 0.05]
      radius: 0.10
  iiwa7_link_1:
    - center: [0.0, 0.0, 0.0]
      radius: 0.08
    - center: [0.0, -0.05, 0.1]
      radius: 0.07
    # ...更多球体
```

每个 link 可以有多个球体，数量不固定。

## 使用示例

### 1. 生成球体数据
```bash
# 生成 1000 个姿态的球体碰撞数据
python sphere_trace_generation.py 1000 maze_data 1
```

### 2. 验证数据生成
```bash
# 运行测试脚本
./test_sphere_generation.sh
```

### 3. 对比 OBB 和球体数据
```bash
# 首先生成 OBB 数据 (如果没有)
python pred_trace_generation.py 1000 maze_data 1

# 然后对比两种方法
python compare_data_formats.py maze_data 1
```

## 数据结构对比

### OBB 数据 (coord.pkl)
```python
(qarr, dirarr, yarr) = pickle.load(file)
# qarr: (7×numqueries, 3) - OBB中心坐标
# dirarr: [string, ...] - 方向编码
# yarr: (7×numqueries, 1) - 碰撞标签
```

### 球体数据 (sphere.pkl)
```python
(qarr_sphere, yarr_sphere, radius_arr, link_id_arr, sphere_id_arr) = pickle.load(file)
# qarr_sphere: (total_spheres, 3) - 球体中心坐标
# yarr_sphere: (total_spheres, 1) - 碰撞标签
# radius_arr: (total_spheres,) - 球体半径
# link_id_arr: (total_spheres,) - 所属link ID
# sphere_id_arr: (total_spheres,) - 球体ID
```

## 微架构仿真集成

要在微架构仿真中使用球体数据，需要：

1. **修改数据加载部分**：
   ```python
   # 原始 OBB 加载
   (edge_link_data, edge_link_coll_data) = pickle.load(f)
   
   # 球体数据加载
   (qarr_sphere, yarr_sphere, radius_arr, link_id_arr, sphere_id_arr) = pickle.load(f)
   ```

2. **调整量化函数**：
   ```python
   # 原始: 使用 OBB 中心坐标
   code_quant = np.digitize(link, bins, right=True)
   
   # 球体: 可包含中心坐标和半径
   extended_features = np.append(sphere_center, sphere_radius)
   code_quant = np.digitize(extended_features, bins, right=True)
   ```

3. **修改重排函数**：
   适配变长的球体序列而不是固定的 7×pose 结构。

## 优势与局限

### 优势
- **更精细的几何表示**: 每个 link 用多个球体更准确地逼近复杂形状
- **配置化**: 球体定义外部化，易于调整和实验
- **碰撞检测效率**: 球体间碰撞检测计算简单
- **可扩展性**: 易于添加新的几何特征 (半径信息)

### 局限
- **数据量增加**: 每个 pose 产生更多数据点
- **不规则数据结构**: link 间球体数量不同，处理复杂
- **几何精度**: 球体包围可能过于宽松或保守
- **兼容性**: 需要修改现有仿真脚本以适配新数据格式

## 后续扩展

1. **自适应球体生成**: 根据 link 几何自动优化球体数量和位置
2. **混合几何表示**: 结合 OBB、球体、胶囊体等多种形状
3. **层次化碰撞检测**: 粗粒度+细粒度的多级检测策略
4. **性能优化**: 球体剔除、空间索引等加速技术

## 相关文件

- `pred_trace_generation.py`: 原始 OBB 方法
- `../configs/robot/iiwa.yml`: 机器人配置
- `../configs/robot/spheres/iiwa_allegro.yml`: 球体定义
