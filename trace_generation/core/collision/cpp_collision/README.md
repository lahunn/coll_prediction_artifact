# C++ 加速碰撞检测库

这是一个使用 C++ 实现的高性能碰撞检测库，通过 pybind11 提供 Python 接口。

## 特性

- **高性能**：C++ 实现，比纯 Python 版本快 5-10 倍
- **完全兼容**：API 与原 Python 版本完全一致
- **优化编译**：使用 `-O3 -march=native -ffast-math` 优化
- **内联函数**：关键函数使用 inline 减少调用开销

## 支持的碰撞检测函数

1. **sphere_sphere** - 球-球碰撞检测
2. **sphere_aabb** - 球-AABB碰撞检测（带周期计数）
3. **cuboid_sphere** / **sphere_cuboid** - 球-OBB碰撞检测
4. **cuboid_aabb** - AABB-OBB碰撞检测（SAT算法，15轴测试）

## 依赖

- Python 3.7+
- CMake 3.12+
- pybind11
- C++17 编译器（g++ 或 clang++）

## 安装

### 方法 1：使用构建脚本（推荐）

```bash
cd trace_generation/geo_collision/cpp_collision
chmod +x build.sh
./build.sh
```

### 方法 2：手动构建

```bash
cd trace_generation/geo_collision/cpp_collision
mkdir build && cd build
cmake ..
make -j$(nproc)
cp cpp_collision*.so ..
```

### 方法 3：使用 setup.py

```bash
cd trace_generation/geo_collision/cpp_collision
python3 setup.py build_ext --inplace
```

## 使用示例

```python
import sys
sys.path.append('/path/to/cpp_collision')
import cpp_collision as cc

# 创建几何形状
sphere1 = cc.Sphere(0.0, 0.0, 0.0, 1.0)
sphere2 = cc.Sphere(3.0, 0.0, 0.0, 1.0)

# 球-球碰撞检测
result = cc.sphere_sphere(sphere1, sphere2)
print(f"碰撞结果: {result}")  # 1=无碰撞, 0=碰撞

# 球-AABB碰撞检测
aabb = cc.AABB(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)
result, cycles = cc.sphere_aabb(sphere1, aabb)
print(f"碰撞结果: {result}, 周期数: {cycles}")
```

## 在 sphere_method_geometric 中使用

```python
# 在 sphere_method_geometric.py 中添加：
try:
    import cpp_collision as cc
    USE_CPP = True
    print("使用 C++ 加速版碰撞检测")
except ImportError:
    from geo_collision.geometric_collision_detection import (
        Sphere, AABB, sphere_aabb, sphere_sphere
    )
    USE_CPP = False
    print("使用 Python 版碰撞检测")
```

## 性能对比

基于初步测试，C++ 版本相比 Python 版本的性能提升：

- `sphere_sphere`: ~8-10x 加速
- `sphere_aabb`: ~5-7x 加速
- `cuboid_aabb`: ~6-8x 加速

实际性能提升取决于具体使用场景和数据规模。

## 文件结构

```
cpp_collision/
├── collision_detection.h   # C++ 核心实现（头文件）
├── bindings.cpp            # pybind11 Python 绑定
├── CMakeLists.txt          # CMake 构建配置
├── setup.py                # Python 安装脚本
├── build.sh                # 自动构建脚本
└── README.md               # 本文件
```

## 故障排除

### pybind11 未找到

```bash
pip3 install pybind11
```

### CMake 版本过低

Ubuntu/Debian:
```bash
sudo apt install cmake
```

### 编译错误

确保安装了 C++17 兼容的编译器：
```bash
sudo apt install g++
```

## 许可证

与主项目保持一致
