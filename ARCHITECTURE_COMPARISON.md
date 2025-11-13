# 架构对比与设计总结

## 核心设计变化

### 旧架构的问题

```
collision_check.py
├── CollisionEnv._edge_fp()
├── CollisionEnv._point_in_free_space()
├── CollisionEnv._get_link_collisions() ❌ 问题1：职责混杂
│   ├── PyBullet 碰撞检测逻辑
│   └── Link 数据收集
└── data_manager._store_collision_data()
    ├── obb_link_data.append()
    └── obb_link_coll_data.append()

问题：
1. _get_link_collisions() 混合了两个职责：
   - 执行 PyBullet 碰撞检测
   - 返回原始碰撞结果
   
2. CollisionEnv 需要知道如何处理 Link 特定的数据结构
3. 难以支持其他碰撞模型（Sphere）
4. 碰撞检测和数据存储耦合紧密
```

### 新架构的改进

```
新分层结构：
┌─────────────────────────────────────┐
│     CollisionEnv（Pose级）          │
│  - 边离散化                         │
│  - 调用 detector 进行检测            │
│  - 管理 config_list                 │
└─────────────────────────────────────┘
           ↓ detector.check_pose()
┌─────────────────────────────────────┐
│   LinkCollisionDetector             │
│   (Link级 - 从CollisionEnv提取)     │
│  - _get_link_collisions()          │
│  - 返回标准化dict数据               │
└─────────────────────────────────────┘
           ↓ data_manager._store_collision_data()
┌─────────────────────────────────────┐
│   CollisionDataManager              │
│  - LinkDataModel                   │
│  - SphereDataModel                 │
└─────────────────────────────────────┘

改进：
✓ 职责清晰：每层只做一件事
✓ 易于扩展：添加 SphereCollisionDetector 无需修改 CollisionEnv
✓ 标准化接口：所有 detector 都返回相同格式 dict
✓ 易于测试：可独立测试每个 detector
```

---

## 文件位置变化

### 目录结构对比

#### 改前

```
trace_generation/core/
├── collision/
│   ├── geometric_collision_detection.py
│   ├── obb_detector.py
│   ├── sphere_detector.py
│   └── sphere_method.py
└── robot/
    ├── collision_check.py              ← Link级检测混在这里
    ├── collision_data_manager.py       ← 数据管理混在这里
    ├── environment.py
    ├── modular_env.py
    └── modular_sphere_env.py
```

#### 改后

```
trace_generation/core/
└── collision/
    ├── __init__.py                     ← 新增：统一导出
    ├── collision_env.py                ← 改名：从robot/collision_check.py
    ├── data_manager.py                 ← 改名：从robot/collision_data_manager.py
    ├── link_collision_detector.py      ← 新增：从collision_env提取
    ├── sphere_detector.py              ← 修改：添加check_pose()接口
    ├── geometric_collision_detection.py
    ├── obb_detector.py
    ├── sphere_method.py
    └── cpp_collision/
        └── ...
```

**好处**：
- 所有碰撞相关代码在一个目录
- 更清晰的代码组织
- 更容易找到相关代码

---

## API 变化对比

### LinkCollisionDetector（新）

```python
# 改前：在 CollisionEnv 内部
class CollisionEnv:
    def _get_link_collisions(self):
        # 内部实现，返回元组
        return any_coll, link_colls
    
    def _point_in_free_space(self, state):
        # ... 复杂的组织逻辑 ...
        is_collision, link_colls = self._get_link_collisions()
        # 直接操作 data_manager 内部字段
        self.data_manager.obb_link_data.append(link_coords)

# 改后：独立的检测器
class LinkCollisionDetector:
    def check_pose(self, state) -> Tuple[bool, Dict]:
        # 清晰的公开接口
        return is_free, {
            'link_coords': [...],
            'link_colls': [...],
            'timestamp': float,
        }
    
    def _get_link_collisions(self) -> Tuple[bool, List[int]]:
        # 私有方法，职责单一

class CollisionEnv:
    def _point_in_free_space(self, state):
        is_free, data = self.detector.check_pose(state)
        self.data_manager._store_collision_data(data)
```

**API 对比**：

| 方面 | 改前 | 改后 |
|------|------|------|
| **接口类型** | 私有方法 | 公开方法 |
| **返回值** | 元组 `(bool, List)` | 标准dict `{'link_coords': ..., 'link_colls': ...}` |
| **调用方** | CollisionEnv 内部 | 任何需要 Link 检测的地方 |
| **可测试性** | 难以单独测试 | 容易单独测试 |

---

### 数据流变化

#### 改前

```python
# _edge_fp() 方法
edge_configs = self._discretize_edge(state, new_state, RRT_EPS)

for config in edge_configs:
    is_free, link_coords, link_colls = self._point_in_free_space(config)
    
    if link_coords:
        # 直接操作 data_manager 的私有字段
        self.data_manager.obb_link_data.append(link_coords)
        self.data_manager.obb_link_coll_data.append(link_colls)
```

#### 改后

```python
# _edge_fp() 方法
edge_configs = self._discretize_edge(state, new_state, RRT_EPS)

for config in edge_configs:
    is_free, collision_data = self._point_in_free_space(config)
    # 通过公开方法存储数据
    self.data_manager._store_collision_data(collision_data, is_edge=False)
```

**改进**：
- 不再直接操作 `obb_link_data` 等内部字段
- 通过 `_store_collision_data()` 方法规范数据存储
- 更容易支持不同的模型类型

---

## 模型扩展示例

### 支持新的碰撞模型

假设要添加 `CapsuleCollisionDetector`：

#### 改前

```python
# 需要修改 CollisionEnv 和 CollisionDataManager
class CollisionEnv:
    def __init__(self):
        # 不知道如何处理 capsule 数据
        if model_type == "capsule":
            # ???

class CollisionDataManager:
    def __init__(self):
        if model_type == "capsule":
            self.capsule_data = []
            self.capsule_coll_data = []
```

#### 改后

```python
# 只需添加新的 detector 和数据模型，无需修改 CollisionEnv

# 1. 创建 capsule_collision_detector.py
class CapsuleCollisionDetector:
    def check_pose(self, state) -> Tuple[bool, Dict]:
        return is_free, {
            'capsule_coords': [...],
            'capsule_colls': [...],
            'timestamp': time.time(),
        }

# 2. 添加 CapsuleDataModel
class CapsuleDataModel(CollisionDataModel):
    def store_collision_data(self, data):
        # 处理 capsule 特定数据
        pass

# 3. CollisionDataManager 自动支持
manager = CollisionDataManager(model_type="capsule")  # 自动创建 CapsuleDataModel

# CollisionEnv 无需任何改动！
env = CollisionEnv(robot_env, collision_model_type="capsule")
```

---

## 向后兼容性保证

### 属性访问（直接兼容）

```python
# 改前：直接访问
data_manager.obb_link_data
data_manager.obb_link_coll_data

# 改后：通过属性代理（完全兼容）
@property
def obb_link_data(self):
    if isinstance(self.collision_data, LinkDataModel):
        return self.collision_data.link_data
    raise AttributeError(...)

# 现有代码无需改动！
data_manager.obb_link_data  # 仍然可用
```

### 导入路径（通过重导出）

```python
# 改前：从 robot 目录导入
from trace_generation.core.robot.collision_check import CollisionEnv
from trace_generation.core.robot.collision_data_manager import CollisionDataManager

# 改后：从 collision 目录导入（但通过 robot/__init__.py 重导出）
# robot/__init__.py
from trace_generation.core.collision.collision_env import CollisionEnv
from trace_generation.core.collision.data_manager import CollisionDataManager

# 现有代码可以继续使用旧导入路径
from trace_generation.core.robot import CollisionEnv  # 通过 __init__.py
# 或者使用新导入路径
from trace_generation.core.collision import CollisionEnv  # 直接从 collision
```

---

## 实现的核心要点

### 1. LinkCollisionDetector 提取

```python
# 这是原来在 CollisionEnv 中的方法
def _get_link_collisions(self):
    """获取各个valid link的碰撞结果"""
    any_coll = False
    link_colls = []
    p.performCollisionDetection(physicsClientId=self.robot_env.physics_client)
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

# 现在变成 LinkCollisionDetector 的私有方法
class LinkCollisionDetector:
    def _get_link_collisions(self):
        # 完全相同的实现
        ...
    
    def check_pose(self, state):
        # 公开接口：调用 _get_link_collisions() 并返回 dict
        is_collision, link_colls = self._get_link_collisions()
        return is_free, {
            'link_coords': [...],
            'link_colls': link_colls,
            'timestamp': time.time(),
        }
```

### 2. CollisionEnv 简化

```python
# 改前：知道如何处理 Link 特定的细节
class CollisionEnv:
    def _point_in_free_space(self, state):
        is_free, link_coords, link_colls = self._point_in_free_space_impl(state)
        self.data_manager.obb_link_data.append(link_coords)  # Link特定
        self.data_manager.obb_link_coll_data.append(link_colls)  # Link特定

# 改后：通用的处理逻辑
class CollisionEnv:
    def __init__(self, robot_env, collision_model_type="link"):
        if collision_model_type == "link":
            self.detector = LinkCollisionDetector(robot_env)
        elif collision_model_type == "sphere":
            self.detector = SphereCollisionDetector(robot_env)
        self.data_manager = CollisionDataManager(model_type=collision_model_type)
    
    def _point_in_free_space(self, state):
        is_free, data = self.detector.check_pose(state)  # 不关心具体类型
        self.data_manager._store_collision_data(data)  # 通用方法
        return is_free, data
```

### 3. 数据管理器抽象

```python
# 改前：Link 特定的硬编码
class CollisionDataManager:
    def __init__(self):
        self.obb_link_data = []
        self.obb_link_coll_data = []

# 改后：支持多种模型
class CollisionDataManager:
    def __init__(self, model_type="link"):
        self.collision_data = self._create_model(model_type)
    
    def _create_model(self, model_type):
        if model_type == "link":
            return LinkDataModel()
        elif model_type == "sphere":
            return SphereDataModel()
        # 易于添加新模型
```

---

## 关键决策的理由

| 决策 | 理由 |
|------|------|
| **Pose级协调** | CollisionEnv 只负责流程控制（边离散化、调用detector），使其与具体检测算法解耦 |
| **Link/Sphere检测下沉** | 每个检测器专注于自己的算法，与 Pose 级逻辑无关 |
| **统一的check_pose()接口** | 让 CollisionEnv 对所有类型的 detector 都适用 |
| **dict数据格式** | 比元组更清晰，更容易扩展（可以添加新字段而不影响兼容性） |
| **文件移到collision目录** | 所有碰撞相关代码在同一目录，更清晰的组织 |
| **属性代理向后兼容** | 现有代码不需要修改，减少迁移成本 |

---

## 测试策略

### 单元测试

```python
# test_link_collision_detector.py
def test_check_pose_returns_dict():
    detector = LinkCollisionDetector(robot_env)
    is_free, data = detector.check_pose(valid_state)
    assert isinstance(data, dict)
    assert 'link_coords' in data
    assert 'link_colls' in data
    assert 'timestamp' in data

def test_link_collision_detector_statistics():
    detector = LinkCollisionDetector(robot_env)
    detector.check_pose(state1)
    detector.check_pose(state2)
    assert detector.collision_check_count == 2
```

### 集成测试

```python
# test_collision_env_integration.py
def test_edge_fp_with_link_model():
    env = CollisionEnv(robot_env, collision_model_type="link")
    result = env._edge_fp(state1, state2)
    assert isinstance(result, bool)
    
def test_edge_fp_with_sphere_model():
    env = CollisionEnv(robot_env, collision_model_type="sphere")
    result = env._edge_fp(state1, state2)
    assert isinstance(result, bool)
```

### 向后兼容性测试

```python
# test_backward_compatibility.py
def test_obb_link_data_property():
    manager = CollisionDataManager(model_type="link")
    # 应该能够访问旧属性
    assert hasattr(manager, 'obb_link_data')
    assert isinstance(manager.obb_link_data, list)
```

---

## 性能影响评估

| 操作 | 开销 | 说明 |
|------|------|------|
| **多一层函数调用** | ~0-1% | detector.check_pose() 的开销可忽略 |
| **dict 创建** | ~1-2% | 每次 check_pose() 创建一个 dict |
| **数据存储委托** | ~0% | _store_collision_data() 只是转发，无额外计算 |
| **总体** | **<5%** | 可接受的开销，换来的收益远大于成本 |

---

## 文件清单汇总

### 新建文件（3个）

| 文件 | 代码行数 | 说明 |
|------|---------|------|
| collision/link_collision_detector.py | ~200 | 从 CollisionEnv 提取 |
| collision/collision_env.py | ~250 | 从 robot/collision_check.py 移来 |
| collision/data_manager.py | ~400 | 从 robot/collision_data_manager.py 移来，重构 |

### 修改文件（4个）

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| collision/sphere_detector.py | 添加 check_pose() 接口 | +50 |
| robot/modular_env.py | 更新导入路径 | 2 |
| robot/modular_sphere_env.py | 更新导入路径 | 2 |
| collision/__init__.py | 新建，统一导出 | 10 |

### 删除文件（2个）

| 文件 | 原因 |
|------|------|
| robot/collision_check.py | 已移至 collision/collision_env.py |
| robot/collision_data_manager.py | 已移至 collision/data_manager.py |

**总代码量**：~915 行（新增 + 修改）

---

## 风险和缓解方案

| 风险 | 等级 | 缓解方案 |
|------|------|---------|
| 文件移动导致导入破裂 | 高 | 通过 __init__.py 重导出，充分的导入测试 |
| SphereDetector 集成不当 | 中 | 详细的单元测试，验证 check_pose() 返回格式 |
| 现有脚本无法运行 | 中 | 属性代理保持兼容性，渐进式迁移 |
| 性能下降 | 低 | 开销 <5%，可接受；需要性能测试验证 |
| 意外的 API 变化 | 低 | 充分的文档和测试，代码评审把关 |

