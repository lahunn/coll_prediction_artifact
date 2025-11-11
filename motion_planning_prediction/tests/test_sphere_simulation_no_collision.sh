#!/bin/bash
# 球体碰撞检测仿真测试脚本（无碰撞边筛选版本）

# 默认参数
ROBOT_NAME="iiwa"
BASE_NAME="iiwa_7"
NUM_TESTS=10
THRESHOLD=0.5
SAMPLE_RATE=0.1
QNONCOLL_MULTIPLIER=8
NUM_OOCDS_LIST="1,2,4,7,14,28"

# 数据文件夹
DATA_FOLDER="../trace_files/scene_benchmarks/bit_collision_data"

echo "=== 球体碰撞检测仿真测试（无碰撞边筛选 + OOCD分析）==="
echo "机器人: $ROBOT_NAME"
echo "基准名称: $BASE_NAME"
echo "测试数量: $NUM_TESTS"
echo "阈值: $THRESHOLD"
echo "采样率: $SAMPLE_RATE"
echo "队列长度倍数: $QNONCOLL_MULTIPLIER"
echo "OOCD数量列表: $NUM_OOCDS_LIST"
echo "数据文件夹: $DATA_FOLDER"
echo "========================================"

# 运行仿真
python3 prediction_simulation_nDOF_sphere_no_collision.py \
    $THRESHOLD \
    $SAMPLE_RATE \
    $QNONCOLL_MULTIPLIER \
    $DATA_FOLDER \
    $BASE_NAME \
    $NUM_TESTS \
    $ROBOT_NAME \
    $NUM_OOCDS_LIST

echo "测试完成！"