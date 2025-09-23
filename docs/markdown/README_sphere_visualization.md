# 机器人球体结构可视化系统

这是一个完整的机器人球体结构可视化系统，可以同时显示机器人模型和对应的球体碰撞表示。

## 功能特性

1. **球体配置解析**: 从YAML文件解析球体配置信息
2. **球体正向运动学**: 根据机器人关节配置实时计算球体位姿
3. **集成可视化**: 同时显示机器人和球体，支持实时交互
4. **多种模式**: GUI交互模式和静态演示模式

## 文件结构

```
trace_generation/
├── sphere_visualizer.py           # 球体YAML解析和基础可视化
├── sphere_forward_kinematics.py   # 球体正向运动学计算
├── robot_sphere_visualizer.py     # 集成机器人和球体可视化
└── README_sphere_visualization.md # 本文档
```

## 使用方法

### 1. 基础球体可视化

```bash
# 查看球体配置信息
python sphere_visualizer.py ../content/configs/robot/spheres/franka.yml --list-links

# 可视化球体结构
python sphere_visualizer.py ../content/configs/robot/spheres/franka.yml
```

### 2. 球体正向运动学

```bash
# 计算球体位姿
python sphere_forward_kinematics.py ../data/robots/panda/panda.urdf ../content/configs/robot/spheres/franka.yml
```

### 3. 集成机器人和球体可视化

```bash
# 交互式GUI模式（推荐）
python robot_sphere_visualizer.py ../data/robots/panda/panda.urdf ../content/configs/robot/spheres/franka.yml

# 静态演示模式
python robot_sphere_visualizer.py ../data/robots/panda/panda.urdf ../content/configs/robot/spheres/franka.yml --demo

# 无GUI模式
python robot_sphere_visualizer.py ../data/robots/panda/panda.urdf ../content/configs/robot/spheres/franka.yml --no-gui
```

## GUI控制说明

在交互式模式下，你可以使用以下控制：

- **关节控制**: 7个关节滑块控制机器人姿态
- **相机控制**: 距离、水平角度、俯仰角度滑块
- **透明度控制**: 机器人和球体的透明度调节
- **实时更新**: 球体位姿根据机器人关节配置实时更新

## 技术细节

### 球体配置格式

YAML文件包含每个连杆的球体定义：

```yaml
robot_name: "Franka Panda"
urdf_path: "urdf/franka_description/franka_panda_dyn.urdf"
links:
  panda_link0:
    spheres:
      - center: [0.0, 0.0, 0.05]
        radius: 0.06
      - center: [0.0, 0.0, 0.15]
        radius: 0.06
```

### 正向运动学计算

1. 获取机器人当前关节配置
2. 计算每个连杆的变换矩阵
3. 将球体中心从连杆坐标系变换到世界坐标系
4. 返回更新后的球体位姿

### 颜色编码

- 每个连杆使用不同颜色的球体
- 支持透明度调节
- 12种预定义颜色循环使用

## 依赖项

- PyBullet: 3D物理仿真和可视化
- NumPy: 数值计算
- PyYAML: YAML文件解析
- pathlib: 路径处理

## 测试结果

使用Franka Panda机器人测试：
- 成功解析61/65个有效球体
- 覆盖11个连杆
- 实时计算球体位姿
- 流畅的交互体验

## 扩展可能

1. **碰撞检测**: 球体间碰撞检测
2. **轨迹规划**: 基于球体表示的路径规划
3. **多机器人**: 支持多机器人可视化
4. **导出功能**: 导出球体位姿数据
5. **优化算法**: 球体参数优化

## 故障排除

### 常见问题

1. **找不到URDF文件**: 检查文件路径是否正确
2. **YAML解析错误**: 检查YAML文件格式
3. **球体数量不匹配**: 某些连杆在机器人中可能不存在
4. **GUI响应慢**: 减少球体数量或降低更新频率

### 性能优化

- 使用缓存避免重复计算
- 批量更新球体位姿
- 适当的帧率控制（60 FPS）

## 开发说明

代码遵循PEP 8规范，包含详细的类型注解和文档字符串。每个模块都有明确的职责分工：

- `sphere_visualizer.py`: 专注于YAML解析和基础可视化
- `sphere_forward_kinematics.py`: 专注于运动学计算
- `robot_sphere_visualizer.py`: 集成所有功能提供完整体验

这种模块化设计便于维护和扩展。