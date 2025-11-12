#!/bin/bash

# 性能瓶颈分析脚本
# 分析并行碰撞检测仿真中的主要性能限制

echo "=== 性能瓶颈分析 ==="
echo "分析OOCD数量与周期数非线性关系的根本原因"
echo

# 参数设置
THRESHOLD=0.5
SAMPLE_RATE=0.1
QNONCOLL_MULTIPLIER=6
DATA_FOLDER="/home/lanh/project/robot_sim/coll_prediction_artifact/trace_files/scene_benchmarks/bit_collision_data"
BASENAME="iiwa_7"
NUM_BENCHMARKS=10  # 减少基准测试数量以加快分析
ROBOT_NAME="iiwa"
NUM_OOCDS=22

echo "分析参数:"
echo "  阈值: $THRESHOLD"
echo "  采样率: $SAMPLE_RATE"
echo "  队列倍数: $QNONCOLL_MULTIPLIER"
echo "  数据文件夹: $DATA_FOLDER"
echo "  基准测试: $BASENAME (前$NUM_BENCHMARKS个)"
echo "  机器人: $ROBOT_NAME"
echo "  OOCD数量: $NUM_OOCDS"
echo

# 创建结果目录
mkdir -p result_files

# 运行性能瓶颈分析
echo "开始性能瓶颈分析..."
python3 performance_bottleneck_analysis.py \
    $THRESHOLD \
    $SAMPLE_RATE \
    $QNONCOLL_MULTIPLIER \
    $DATA_FOLDER \
    $BASENAME \
    $NUM_BENCHMARKS \
    $ROBOT_NAME \
    $NUM_OOCDS

echo
echo "性能瓶颈分析完成!"
echo "结果已保存到 result_files/performance_bottleneck_analysis.csv"