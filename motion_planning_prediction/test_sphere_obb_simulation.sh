#!/bin/bash
# 球体和OBB碰撞检测仿真测试脚本

echo "========================================"
echo "球体和OBB碰撞检测仿真测试"
echo "========================================"

# 检查数据文件夹
SPHERE_DATA="../trace_files/sphere_data"
OBB_DATA="../trace_files/obb_data"

echo -e "\n[1/4] 检查数据文件夹..."
if [ ! -d "$SPHERE_DATA" ]; then
    echo "警告: 球体数据文件夹不存在: $SPHERE_DATA"
    echo "创建文件夹..."
    mkdir -p "$SPHERE_DATA"
fi

if [ ! -d "$OBB_DATA" ]; then
    echo "警告: OBB数据文件夹不存在: $OBB_DATA"
    echo "创建文件夹..."
    mkdir -p "$OBB_DATA"
fi

# 检查是否有数据文件
echo -e "\n[2/4] 检查数据文件..."
SPHERE_FILES=$(ls $SPHERE_DATA/obstacles_*_sphere.pkl 2>/dev/null | wc -l)
OBB_FILES=$(ls $OBB_DATA/obstacles_*_obb.pkl 2>/dev/null | wc -l)

echo "找到 $SPHERE_FILES 个球体数据文件"
echo "找到 $OBB_FILES 个OBB数据文件"

if [ $SPHERE_FILES -eq 0 ] || [ $OBB_FILES -eq 0 ]; then
    echo -e "\n警告: 数据文件不足，需要先生成数据"
    echo "请运行以下命令生成数据:"
    echo "  cd ../trace_generation"
    echo "  python generate_collision_data.py <obstacle_config_file> <robot_urdf> <robot_name> <obb_output> <sphere_output>"
    exit 1
fi

# 运行球体仿真测试
echo -e "\n[3/4] 运行球体碰撞检测仿真..."
NUM_TESTS=$(($SPHERE_FILES < 10 ? $SPHERE_FILES : 10))
echo "测试前 $NUM_TESTS 个基准..."

python prediction_simulation_nDOF_sphere.py 0.5 0.1 8 $SPHERE_DATA $NUM_TESTS

if [ $? -eq 0 ]; then
    echo "✓ 球体仿真测试通过"
else
    echo "✗ 球体仿真测试失败"
    exit 1
fi

# 运行OBB仿真测试
echo -e "\n[4/4] 运行OBB碰撞检测仿真..."
NUM_TESTS=$(($OBB_FILES < 10 ? $OBB_FILES : 10))
echo "测试前 $NUM_TESTS 个基准..."

python prediction_simulation_nDOF_obb.py 0.5 0.1 8 $OBB_DATA $NUM_TESTS

if [ $? -eq 0 ]; then
    echo "✓ OBB仿真测试通过"
else
    echo "✗ OBB仿真测试失败"
    exit 1
fi

echo -e "\n========================================"
echo "所有测试通过! ✓"
echo "========================================"
echo -e "\n使用方法:"
echo "  球体仿真: python prediction_simulation_nDOF_sphere.py 0.5 0.1 8 $SPHERE_DATA 100"
echo "  OBB仿真:  python prediction_simulation_nDOF_obb.py 0.5 0.1 8 $OBB_DATA 100"
echo -e "\n查看详细使用指南: cat SPHERE_OBB_SIMULATION_GUIDE.md"
