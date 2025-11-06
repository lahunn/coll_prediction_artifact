#!/bin/bash
# C++ 碰撞检测库构建和安装脚本

set -e  # 遇到错误立即退出

echo "========================================"
echo "C++ 碰撞检测库构建脚本"
echo "========================================"

# 进入cpp_collision目录
cd "$(dirname "$0")"

# 清理旧的构建
echo ""
echo "[1/4] 清理旧构建..."
rm -rf build
rm -f *.so cpp_collision*.so

# 创建构建目录
echo ""
echo "[2/4] 创建构建目录..."
mkdir -p build
cd build

# 运行CMake配置
echo ""
echo "[3/4] 配置CMake..."
cmake ..

# 编译
echo ""
echo "[4/4] 编译C++扩展..."
make -j$(nproc)

# .so文件已经直接输出到正确位置了
cd ..

echo ""
echo "========================================"
echo "✓ 构建完成！"
echo "========================================"
echo ""
echo "生成的文件："
ls -lh cpp_collision*.so
echo ""
echo "使用方法："
echo "  import sys"
echo "  sys.path.append('/path/to/cpp_collision')"
echo "  import cpp_collision"
echo ""
