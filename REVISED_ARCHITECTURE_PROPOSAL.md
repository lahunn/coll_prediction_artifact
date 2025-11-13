# 碰撞检测架构重新设计方案（修订版）

## 核心理念：分层架构

按照**检测粒度**和**物理引擎依赖**进行分层：

```
┌─────────────────────────────────────────────────────────────┐
│                     应用层                                  │
│  (modular_env.py, modular_sphere_env.py)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Pose 级碰撞检测协调层                          │
│  (collision_env.py 在 collision 目录)                       │
│  - 负责 edge 的离散化                                       │
│  - 调用下层进行单个 pose 检测                              │
│  - 收集并汇总碰撞数据                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            Link/Sphere 级碰撞检测实现层                    │
│  (link_collision_detector.py, sphere_detector.py)           │
│  - Link级：PyBullet 碰撞检测 + OBB 数据收集               │
│  - Sphere级：几何球体碰撞检测 + Sphere 数据收集           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              数据管理与统计层                               │
│  (collision_data_manager.py 在 collision 目录)              │
│  - 统一的数据存储接口                                       │
│  - 支持多种碰撞模型（Link/Sphere）                         │
│  - 碰撞率统计与数据导出                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 目录结构改变

### 改前

```
trace_generation/core/
├── collision/
│   ├── geometric_collision_detection.py
│   ├── obb_detector.py
│   ├── sphere_detector.py
│   └── sphere_method.py
└── robot/
    ├── collision_check.py              ← 应该在 collision 目录
    ├── collision_data_manager.py       ← 应该在 collision 目录
    ├── environment.py
    ├── modular_env.py
    ├── modular_sphere_env.py
    └── ...
```

### 改后

```
trace_generation/core/
└── collision/
    ├── __init__.py
    ├── data_manager.py                 ← 从 robot/ 移来，重命名
    ├── collision_env.py                ← 从 robot/collision_check.py 移来，重命名
    ├── link_collision_detector.py      ← 新建，从 collision_check.py 提取
    ├── sphere_detector.py              ← 已有，保留
    ├── geometric_collision_detection.py
    ├── obb_detector.py
    ├── sphere_method.py
    ├── cpp_collision/
    │   └── ...
    └── tests/
        └── ...

robot/
├── environment.py
├── modular_env.py
├── modular_sphere_env.py
└── ...
```

---

## 关键文件的职责重新定义

### 1. collision_env.py（新位置：collision/collision_env.py）

**现有代码的问题**：
- `_get_link_collisions()` 混合了碰撞检测和数据收集
- 耦合了 PyBullet 的具体实现细节
- 无法支持其他类型的碰撞检测（如 Sphere）

**重新设计**：

```python
class CollisionEnv:
    """
    Pose 级碰撞检测协调层
    
    职责：
    - 处理边的离散化
    - 为每个 pose 调用适当的碰撞检测器
    - 汇总并管理碰撞数据
    
    不涉及具体的碰撞检测算法实现
    """
    
    def __init__(
        self,
        robot_env,
        collision_model_type: str = "link",  # "link" 或 "sphere"
        config_output_file=None,
    ):
        self.robot_env = robot_env
        self.collision_model_type = collision_model_type
        
        # 根据模型类型选择合适的检测器
        if collision_model_type == "link":
            self.detector = LinkCollisionDetector(robot_env)
        elif collision_model_type == "sphere":
            self.detector = SphereCollisionDetector(robot_env)
        else:
            raise ValueError(f"Unknown collision model: {collision_model_type}")
        
        self.data_manager = CollisionDataManager(model_type=collision_model_type)
        # ... 其他初始化 ...
    
    def _point_in_free_space(self, state):
        """
        Pose 级碰撞检测
        
        返回该 pose 是否无碰撞，以及对应的碰撞数据
        """
        is_free, data = self.detector.check_pose(state)
        
        # 由 detector 返回的数据格式已经标准化，直接存储
        self.data_manager._store_collision_data(data)
        
        return is_free, data
    
    def _discretize_edge(self, state, new_state, RRT_EPS=0.25):
        """边的离散化逻辑（保持不变）"""
        # ... 现有代码 ...
    
    def _edge_fp(self, state, new_state, RRT_EPS=0.25):
        """
        边级碰撞检测
        
        对边上的所有 pose 进行检测，汇总结果
        """
        edge_configs = self._discretize_edge(state, new_state, RRT_EPS)
        
        edge_free = True
        for config in edge_configs:
            is_free, data = self._point_in_free_space(config)
            if not is_free:
                edge_free = False
        
        return edge_free
```

**关键设计决定**：
- `CollisionEnv` 成为**协调层**，只知道如何组织检测流程
- 具体的碰撞检测逻辑下沉到 `LinkCollisionDetector` 和 `SphereCollisionDetector`
- 数据格式由 detector 提供，CollisionEnv 只负责存储和传递

---

### 2. link_collision_detector.py（新建文件：collision/link_collision_detector.py）

**来源**：从 collision_check.py 中提取

```python
from typing import Tuple, List, Any
import pybullet as p
import time

class LinkCollisionDetector:
    """
    Link 级碰撞检测实现
    
    使用 PyBullet 进行碰撞检测，返回标准化的数据格式
    """
    
    def __init__(self, robot_env):
        """
        初始化 Link 级碰撞检测器
        
        Args:
            robot_env: 机器人环境实例（提供 PyBullet 接口）
        """
        self.robot_env = robot_env
        self.collision_time = 0.0
        self.collision_check_count = 0
    
    def check_pose(self, state) -> Tuple[bool, dict]:
        """
        检查单个 pose 的碰撞状态
        
        Args:
            state: 机器人配置
        
        Returns:
            (is_free, collision_data)
            - is_free: 是否无碰撞
            - collision_data: 标准化的碰撞数据字典
              {
                  'link_coords': [...],      # 各 link 的位姿
                  'link_colls': [...],       # 各 link 的碰撞标签 (0/1)
                  'timestamp': ...            # 检测时间戳
              }
        """
        start_time = time.time()
        self.collision_check_count += 1
        
        # 设置机器人配置
        if not self.robot_env._valid_state(state):
            self.collision_time += time.time() - start_time
            return False, {'link_coords': [], 'link_colls': []}
        
        self.robot_env.set_config(state)
        
        # 获取 Link 级碰撞信息
        is_collision, link_colls = self._get_link_collisions()
        
        # 收集 Link 位姿数据
        link_coords = []
        for link_idx in self.robot_env.valid_collision_links:
            if link_idx == -1:
                continue
            pose = self.robot_env._get_link_pose(link_idx)
            link_coords.append(pose)
        
        self.collision_time += time.time() - start_time
        
        is_free = not is_collision
        collision_data = {
            'link_coords': link_coords,
            'link_colls': link_colls,
            'timestamp': time.time(),
        }
        
        return is_free, collision_data
    
    def _get_link_collisions(self) -> Tuple[bool, List[int]]:
        """
        获取各个 valid link 的碰撞结果
        
        Returns:
            (any_collision, link_collision_flags)
            - any_collision: 是否有任何 link 碰撞
            - link_collision_flags: 各 link 的碰撞标志列表 (0=碰撞, 1=自由)
        """
        any_coll = False
        link_colls = []
        
        p.performCollisionDetection(
            physicsClientId=self.robot_env.physics_client
        )
        
        for link_idx in self.robot_env.valid_collision_links:
            if link_idx == -1:
                continue
            
            contacts = p.getContactPoints(
                self.robot_env.robotId,
                linkIndexA=link_idx,
                physicsClientId=self.robot_env.physics_client,
            )
            
            is_colliding = len(contacts) > 0
            if is_colliding:
                any_coll = True
                link_colls.append(0)
            else:
                link_colls.append(1)
        
        return any_coll, link_colls
    
    def reset(self):
        """重置统计信息"""
        self.collision_time = 0.0
        self.collision_check_count = 0
    
    def get_stats(self) -> dict:
        """获取检测统计信息"""
        return {
            'collision_check_count': self.collision_check_count,
            'collision_time': self.collision_time,
        }
```

**关键设计决定**：
- `_get_link_collisions()` 现在是 `LinkCollisionDetector` 的**私有方法**
- `check_pose()` 返回**标准化的数据格式**（dict），而不是元组
- 每个 detector 维护自己的统计信息

---

### 3. sphere_detector.py（修改现有文件）

**现有位置**：`collision/sphere_detector.py`

**需要修改**：

```python
class SphereCollisionDetector:
    """
    Sphere 级碰撞检测实现（现有的 SphereEnvGeometric 的重命名和重构）
    """
    
    def __init__(self, robot_env, robot_name=None):
        self.robot_env = robot_env
        self.robot_name = robot_name
        self.collision_time = 0.0
        self.collision_check_count = 0
        
        # ... 现有的球体检测初始化 ...
    
    def check_pose(self, state) -> Tuple[bool, dict]:
        """
        检查单个 pose 的碰撞状态
        
        Args:
            state: 机器人配置
        
        Returns:
            (is_free, collision_data)
            - is_free: 是否无碰撞
            - collision_data: 标准化的碰撞数据字典
              {
                  'sphere_coords': [...],    # 各球体的位置
                  'sphere_colls': [...],     # 各球体的碰撞标签 (0/1)
                  'timestamp': ...
              }
        """
        start_time = time.time()
        self.collision_check_count += 1
        
        # 进行球体碰撞检测
        is_collision = self._check_sphere_collision(state)
        
        # 收集球体数据
        sphere_coords = self._get_sphere_coords(state)
        sphere_colls = self._get_sphere_collisions(state)
        
        self.collision_time += time.time() - start_time
        
        collision_data = {
            'sphere_coords': sphere_coords,
            'sphere_colls': sphere_colls,
            'timestamp': time.time(),
        }
        
        return not is_collision, collision_data
    
    def reset(self):
        """重置统计信息"""
        self.collision_time = 0.0
        self.collision_check_count = 0
    
    def get_stats(self) -> dict:
        """获取检测统计信息"""
        return {
            'collision_check_count': self.collision_check_count,
            'collision_time': self.collision_time,
        }
```

**关键修改**：
- 添加 `check_pose()` 方法（与 LinkCollisionDetector 兼容的接口）
- 返回**统一的数据格式**（dict，而不是元组）
- 保留现有的具体检测逻辑，只修改接口层

---

### 4. collision_data_manager.py（重命名为 data_manager.py）

**新位置**：`collision/data_manager.py`

**核心改变**：

```python
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class CollisionDataModel(ABC):
    """碰撞数据模型抽象基类"""
    
    @abstractmethod
    def store_collision_data(self, data: dict, is_edge: bool = True):
        """存储碰撞数据"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置数据"""
        pass
    
    @abstractmethod
    def get_collision_ratio(self) -> Tuple[float, float, float]:
        """获取碰撞率 (element_ratio, pose_ratio, edge_ratio)"""
        pass
    
    @abstractmethod
    def save_collision_data(self, output_file):
        """保存数据到文件"""
        pass


class LinkDataModel(CollisionDataModel):
    """Link 级数据模型"""
    
    def __init__(self):
        self.link_data = []           # Link 位姿数据
        self.link_coll_data = []      # Link 碰撞标签
    
    def store_collision_data(self, data: dict, is_edge: bool = True):
        """
        Args:
            data: {'link_coords': [...], 'link_colls': [...]}
        """
        if not data.get('link_coords'):
            return
        
        if is_edge:
            self.link_data.append(data['link_coords'])
            self.link_coll_data.append(data['link_colls'])
        else:
            self.link_data.append([data['link_coords']])
            self.link_coll_data.append([data['link_colls']])
    
    # ... 其他方法与现有相同 ...


class SphereDataModel(CollisionDataModel):
    """Sphere 级数据模型"""
    
    def __init__(self, return_cycles: bool = False):
        self.sphere_data = []        # Sphere 坐标数据
        self.sphere_coll_data = []   # Sphere 碰撞标签
        self.return_cycles = return_cycles
    
    def store_collision_data(self, data: dict, is_edge: bool = True):
        """
        Args:
            data: {'sphere_coords': [...], 'sphere_colls': [...]}
        """
        if not data.get('sphere_coords'):
            return
        
        if is_edge:
            self.sphere_data.append(data['sphere_coords'])
            self.sphere_coll_data.append(data['sphere_colls'])
        else:
            self.sphere_data.append([data['sphere_coords']])
            self.sphere_coll_data.append([data['sphere_colls']])
    
    # ... 其他方法 ...


class CollisionDataManager:
    """统一的碰撞数据管理器"""
    
    def __init__(self, model_type: str = "link"):
        self.model_type = model_type
        self.collision_data = self._create_model(model_type)
        self.edge_fp_call_count = 0
    
    def _create_model(self, model_type: str) -> CollisionDataModel:
        """工厂方法"""
        if model_type == "link":
            return LinkDataModel()
        elif model_type == "sphere":
            return SphereDataModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _store_collision_data(self, data: dict, is_edge: bool = True):
        """代理到具体模型"""
        self.collision_data.store_collision_data(data, is_edge=is_edge)
    
    # ... 其他代理方法 ...
```

---

## 修改清单（重新梳理）

### 阶段 1：基础设施（P1 - 必须）

| 序号 | 操作 | 文件 | 说明 |
|-----|------|------|------|
| 1 | 新建 | `collision/data_manager.py` | 从 robot/collision_data_manager.py 移来 + 重构 |
| 2 | 新建 | `collision/collision_env.py` | 从 robot/collision_check.py 移来 + 重构 |
| 3 | 新建 | `collision/link_collision_detector.py` | 从 collision_check.py 提取出来 |
| 4 | 修改 | `collision/sphere_detector.py` | 添加 check_pose() 接口 |
| 5 | 删除 | `robot/collision_data_manager.py` | 已移至 collision 目录 |
| 6 | 删除 | `robot/collision_check.py` | 已移至 collision 目录 |

### 阶段 2：应用层集成（P2 - 依赖 P1）

| 序号 | 操作 | 文件 | 说明 |
|-----|------|------|------|
| 7 | 修改 | `robot/modular_env.py` | 更新导入路径 + 调用新接口 |
| 8 | 修改 | `robot/modular_sphere_env.py` | 更新导入路径 + 调用新接口 |
| 9 | 修改 | `robot/environment.py` | 更新导入路径 |

### 阶段 3：测试验证（P3）

| 序号 | 操作 | 文件 | 说明 |
|-----|------|------|------|
| 10 | 新建 | `collision/tests/test_link_detector.py` | Link 检测器单元测试 |
| 11 | 新建 | `collision/tests/test_collision_env.py` | 集成测试 |

---

## 数据流图

### 改前

```
CollisionEnv._edge_fp()
  ├─ _discretize_edge()
  ├─ _collect_edge_collision_data()
  │  └─ _point_in_free_space()
  │     ├─ 设置机器人配置
  │     ├─ _get_link_collisions() ← 混合了 PyBullet + 数据收集
  │     └─ data_manager._store_collision_data()
  └─ data_manager.obb_link_data.append()
```

### 改后

```
CollisionEnv._edge_fp()
  ├─ _discretize_edge()
  └─ for each pose in edge:
     └─ _point_in_free_space(pose)
        ├─ detector.check_pose(state) ← 检测器处理所有逻辑
        │  └─ LinkCollisionDetector._get_link_collisions()
        └─ data_manager._store_collision_data(detector_result)
           └─ CollisionDataModel.store_collision_data()
```

**优势**：
- 关注点分离：detector 专注于碰撞检测，CollisionEnv 专注于流程控制
- 易于扩展：添加新的 detector 无需修改 CollisionEnv
- 易于测试：可独立测试 detector

---

## 向后兼容性

### 属性代理

在 `CollisionDataManager` 中添加向后兼容属性：

```python
class CollisionDataManager:
    @property
    def obb_link_data(self):
        """向后兼容：访问 link 数据"""
        if isinstance(self.collision_data, LinkDataModel):
            return self.collision_data.link_data
        raise AttributeError("Only available for 'link' model type")
    
    @property
    def obb_link_coll_data(self):
        """向后兼容：访问 link 碰撞数据"""
        if isinstance(self.collision_data, LinkDataModel):
            return self.collision_data.link_coll_data
        raise AttributeError("Only available for 'link' model type")
```

### 导入路径

在 `robot/` 目录的 `__init__.py` 中添加重导出：

```python
# robot/__init__.py
from collision.collision_env import CollisionEnv
from collision.data_manager import CollisionDataManager

__all__ = ['CollisionEnv', 'CollisionDataManager']
```

这样现有代码可以继续使用 `from robot import CollisionEnv`。

---

## 实现步骤

### 步骤 1：创建新的文件结构
1. 创建 `collision/link_collision_detector.py`
2. 复制 `collision_check.py` 内容到 `collision/collision_env.py`
3. 复制 `collision_data_manager.py` 到 `collision/data_manager.py`

### 步骤 2：重构代码
1. 在 `collision/collision_env.py` 中：
   - 删除 `_get_link_collisions()` 方法
   - 创建 `self.detector` 实例
   - 修改 `_point_in_free_space()` 使用 detector
   - 修改 `_edge_fp()` 逻辑

2. 在 `collision/link_collision_detector.py` 中：
   - 实现 `check_pose()` 方法
   - 实现 `_get_link_collisions()` 方法
   - 添加 `reset()` 和 `get_stats()` 方法

3. 在 `collision/data_manager.py` 中：
   - 添加 `CollisionDataModel` 抽象类
   - 添加 `LinkDataModel` 和 `SphereDataModel` 实现
   - 修改 `CollisionDataManager` 使用工厂方法

4. 在 `collision/sphere_detector.py` 中：
   - 添加 `check_pose()` 方法
   - 标准化返回数据格式

### 步骤 3：更新导入
1. 更新 `modular_env.py` 和 `modular_sphere_env.py` 的导入路径
2. 添加向后兼容的重导出

### 步骤 4：测试
1. 运行现有的测试套件确保功能不变
2. 添加新的单元测试

---

## 优势总结

| 方面 | 改前 | 改后 |
|------|------|------|
| **关注点分离** | 混合的 | 清晰的：detector 处理检测，CollisionEnv 处理流程 |
| **可扩展性** | 添加新模型需修改 CollisionEnv | 添加新 detector 无需修改 CollisionEnv |
| **可测试性** | CollisionEnv 内部逻辑难以单独测试 | 可独立测试 detector 和 CollisionEnv |
| **代码位置** | collision_check.py 混杂了多个职责 | 每个文件职责单一 |
| **目录组织** | 碰撞相关代码分散在 robot/ 和 collision/ | 所有碰撞代码在 collision/ 目录 |
| **数据格式** | 返回值是元组，不同类型差异大 | 统一的 dict 格式，易于扩展 |

---

## 风险评估与缓解

| 风险 | 影响 | 缓解方案 |
|-----|------|---------|
| **文件移动导致导入破坏** | 高 | 添加向后兼容的重导出；仔细检查所有调用点 |
| **API 变化影响现有脚本** | 中 | 使用属性代理保持原接口；提供迁移指南 |
| **性能下降** | 低 | 抽象层开销很小；可以性能测试验证 |
| **Sphere 检测器集成不当** | 中 | 充分测试 check_pose() 接口；编写集成测试 |

