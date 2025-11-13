#!/bin/bash
# 球体和OBB碰撞检测仿真测试脚本

# 默认机器人名称
ROBOT_NAME="iiwa"
# 检查数据文件夹
SPHERE_DATA="../trace_files/scene_benchmarks/bit_collision_data"
OBB_DATA="../trace_files/scene_benchmarks/bit_collision_data"
BASE_NAME="iiwa_7"
NUM_TESTS=50


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
SPHERE_FILES=$(ls $SPHERE_DATA/$BASE_NAME_*_sphere.pkl 2>/dev/null | wc -l)
LINK_FILES=$(ls $OBB_DATA/$BASE_NAME_*_link.pkl 2>/dev/null | wc -l)

echo "找到 $SPHERE_FILES 个球体数据文件"
echo "找到 $LINK_FILES 个Link级数据文件"

# 运行球体仿真测试
echo -e "\n[3/4] 运行球体碰撞检测仿真..."
echo "测试前 $NUM_TESTS 个基准..."
# python prediction_simulation_nDOF_sphere.py 碰撞阈值 更新率 非碰撞队列的长度比 SPHERE_DATA BASE_NAME NUM_TESTS [robot_name]
python prediction_simulation_nDOF_sphere.py 0.5 0.1 8 $SPHERE_DATA $BASE_NAME $NUM_TESTS $ROBOT_NAME

if [ $? -eq 0 ]; then
    echo "✓ 球体仿真测试通过"
else
    echo "✗ 球体仿真测试失败"
    exit 1
fi

# 运行OBB仿真测试
echo -e "\n[4/4] 运行Link级碰撞检测仿真..."
echo "测试前 $NUM_TESTS 个基准..."

# python prediction_simulation_nDOF.py 碰撞阈值 更新率 非碰撞队列的长度比 OBB_DATA BASE_NAME NUM_TESTS [robot_name]
python prediction_simulation_nDOF.py 0.5 0.1 8 $OBB_DATA $BASE_NAME $NUM_TESTS $ROBOT_NAME

if [ $? -eq 0 ]; then
    echo "✓ Link级仿真测试通过"
else
    echo "✗ Link级仿真测试失败"
    exit 1 
fi

echo -e "\n========================================"
echo "所有测试通过! ✓"
echo "========================================"
echo -e "\n详细命令:"
echo "  球体仿真: python prediction_simulation_nDOF_sphere.py 0.5 0.1 8 $SPHERE_DATA $BASE_NAME 100 $ROBOT_NAME"
echo "  Link仿真: python prediction_simulation_nDOF.py 0.5 0.1 8 $OBB_DATA $BASE_NAME 100 $ROBOT_NAME"
echo -e "\n查看详细使用指南: cat SPHERE_OBB_SIMULATION_GUIDE.md"
