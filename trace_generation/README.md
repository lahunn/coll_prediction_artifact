# Trace Generation - 碰撞检测轨迹生成模块

> **重要更新**: 本模块已于 2025-11-06 进行了目录结构重构。详见 [REFACTORING_REPORT.md](REFACTORING_REPORT.md)

## 📁 目录结构

```
trace_generation/
├── core/              # 核心算法模块
├── scripts/           # 可执行脚本
├── config/            # 配置文件
├── visualization/     # 可视化工具
├── data/              # 数据文件
├── tests/             # 测试文件
└── bit_planning/      # 路径规划
```

## 🚀 实验步骤说明

### 1. 随机场景随机pose，测试预测策略的精确率和召回率

#### 步骤1: 生成预测轨迹数据
```bash
cd trace_generation/scripts
bash launch_pred.sh
```
- `launch_pred.sh` 脚本中可以设置不同的机器人、障碍物和场景参数
- 生成的数据会保存在 `../trace_files/scene_benchmarks/dens*` 文件夹下

#### 步骤2: 分析预测策略效果
```bash
# 运行球体代价分析
bash prediction_approaches/bash_script/run_sphere_cost_analysis.sh

# 运行坐标代价分析  
bash prediction_approaches/bash_script/run_coord_cost_analysis.sh
```
分析不同密度条件下碰撞预测策略的效果

#### 步骤3: 数据分析和可视化
```bash
python plot_comparison_results.py
```
进行数据分析，绘制各类图表

### 2. 实际碰撞检测算法测试

#### 步骤1: 生成标准数据集
```bash
cd trace_generation/bit_planning
bash generate_standard_dataset.sh
```

#### 步骤2: 生成球体碰撞检测数据
```bash
cd trace_generation/scripts
bash generate_sphere_data.sh
```

#### 步骤3: 硬件结构仿真测试
```bash
bash motion_planning_prediction/test_sphere_obb_simulation.sh
```

## 📖 使用示例

### 直接运行脚本
```bash
# 生成场景
cd scripts
python scene_generator.py franka 100

# 生成轨迹
python pred_trace_generation.py franka 100 ../trace_files/scene_benchmarks/dens3 1 --seed 0
```

### 在代码中使用
```python
# 新的导入方式（推荐）
from core.collision.geometric_collision_detection import sphere_aabb
from core.robot.environment import RobotEnv
from core.collision.sphere_detector import SphereEnvGeometric

# 旧的导入方式（向后兼容，仍然可用）
from geo_collision.geometric_collision_detection import sphere_aabb
from robot_as.robot_method import RobotEnv
from sphere_as.sphere_method_geometric import SphereEnvGeometric
```

## 🔧 配置说明

- **机器人配置**: `config/ana_parameters.py`
- **工作空间配置**: `data/workspace_bounds/`
- **URDF文件映射**: `core/robot/environment.py` 中的 `robot_urdf_mapping`

## 📊 性能优化

本模块包含C++加速的碰撞检测算法：
- **位置**: `core/collision/cpp_collision/`
- **编译**: 见 `core/collision/cpp_collision/build.sh`
- **性能提升**: 相比纯Python实现提速 10-50倍

## 📝 更多信息

- 完整重构报告: [REFACTORING_REPORT.md](REFACTORING_REPORT.md)
- C++扩展文档: [core/collision/cpp_collision/README.md](core/collision/cpp_collision/README.md)
- 原始README: 保留在下方供参考

---

## 原README内容（归档）