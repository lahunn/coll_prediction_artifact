# 通用机器人 OBB 计算系统

## 概述

原始的 `pred_trace_generation.py` 针对 KUKA iiwa 7-DOF 机器人进行了硬编码，现在已经扩展为支持任意机器人模型的通用版本。

## 主要改进

### 1. 移除硬编码限制
- **原版**: 固定支持 7-DOF KUKA iiwa 机器人
- **新版**: 自动适应任意 DOF 数量和 link 数量的机器人

### 2. 智能 OBB 计算
- **精确模式**: 使用 `obb_calculator` 模块的 CoACD + Open3D 方法
- **回退模式**: 基于 Klampt 几何包围盒的快速计算

### 3. 自动配置检测
- 自动获取机器人的 link 数量和 DOF 数量
- 动态调整数据结构大小
- 智能选择计算方法

## 使用方法

### 基本用法 (与原版相同)
```bash
python pred_trace_generation.py <numqueries> <foldername> <filenumber>
```

### 示例
```bash
# 为 Jaco 机器人生成 1000 个姿态的数据
python pred_trace_generation.py 1000 scene_benchmarks/dens3 0

# 为任意其他机器人模型生成数据
python pred_trace_generation.py 500 my_robot_scenes/obstacles 1
```

## 计算模式

### 模式 1: 精确 OBB 计算
**条件**: 
- `obb_calculator` 依赖库已安装 (yourdfpy, trimesh, open3d, coacd)
- 机器人有可用的 URDF 文件路径

**优势**:
- 基于 CoACD 凸分解的高精度 OBB
- 适合复杂几何形状
- 最小体积的最优 OBB

**安装依赖**:
```bash
pip install yourdfpy trimesh open3d coacd
```

### 模式 2: 几何包围盒 (回退模式)
**条件**: 
- 精确模式不可用时自动启用
- 只需要 Klampt 库

**特点**:
- 基于 Klampt 几何体的包围盒
- 快速计算，适合实时应用
- 精度略低但兼容性好

## 输出文件格式

### Link 级数据 (`obstacles_X_coord.pkl`)
```python
(qarr, dirarr, yarr) = pickle.load(file)
# qarr: (num_links × numqueries, 3) - OBB 中心坐标
# dirarr: 方向编码字符串列表
# yarr: (num_links × numqueries, 1) - 碰撞标签
```

### Pose 级数据 (`obstacles_X_pose.pkl`)  
```python
(qarr_pose, yarr_pose) = pickle.load(file)
# qarr_pose: (numqueries, num_dofs) - 关节配置
# yarr_pose: (numqueries, 1) - 整体碰撞标签
```

## 支持的机器人类型

### 已测试
- ✅ KUKA iiwa (7-DOF)
- ✅ Kinova Jaco (6/7-DOF)

### 理论支持
- 🔄 任意串联机械臂
- 🔄 移动机械臂
- 🔄 人形机器人
- 🔄 多臂系统

## 调试和故障排除

### 1. 检查依赖库
```bash
python test_universal_obb.py
```

### 2. 常见问题

**Q**: "Missing required libraries" 错误
**A**: 安装 OBB calculator 依赖: `pip install yourdfpy trimesh open3d coacd`

**Q**: "No valid OBB found" 警告
**A**: 正常现象，系统会自动回退到几何包围盒模式

**Q**: 输出数据维度不匹配
**A**: 检查机器人的 link 数量和 DOF 数量是否正确检测

### 3. 日志输出
```
Robot has 7 links and 7 DOFs
  Using precise OBB calculation for 7 links...
  Successfully computed precise OBBs for 7 links
  ✓ Link 0: OBB volume = 0.001234
  ✓ Link 1: OBB volume = 0.002456
  ...
```

## 扩展使用

### 1. 自定义机器人模型
1. 准备机器人的 XML/URDF 文件
2. 确保几何文件 (STL/OBJ/DAE) 路径正确
3. 运行数据生成脚本

### 2. 集成到现有工作流
```bash
# 修改 launch_pred.sh 脚本
for robot_type in jaco iiwa ur5 panda; do
    python pred_trace_generation.py 1000 ${robot_type}_scenes 0
done
```

### 3. 批量处理
```bash
# 使用修改后的 launch_pred_sphere.sh
./launch_pred_sphere.sh
```

## 性能对比

| 方法 | 精度 | 速度 | 内存 | 兼容性 |
|------|------|------|------|--------|
| 原版 (硬编码) | 中等 | 快 | 低 | 仅 KUKA iiwa |
| 精确 OBB | 高 | 中等 | 中等 | 通用 |
| 几何包围盒 | 中等 | 快 | 低 | 通用 |

## 未来改进

1. **自动 URDF 路径检测**: 改进 URDF 文件路径的自动识别
2. **并行计算**: 支持多线程 OBB 计算
3. **缓存机制**: 缓存已计算的 OBB 结果
4. **可视化工具**: 添加 OBB 可视化功能
5. **配置文件**: 支持外部配置文件定制计算参数

## 相关文件

- `pred_trace_generation.py` - 主要数据生成脚本 (已修改)
- `obb_calculator.py` - 精确 OBB 计算模块
- `test_universal_obb.py` - 功能测试脚本
- `sphere_trace_generation.py` - 球体近似数据生成
- `launch_pred_sphere.sh` - 批量执行脚本
