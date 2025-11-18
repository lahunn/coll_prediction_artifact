#!/bin/bash

# =================================================================
# 运行nDOF球体碰撞预测仿真（抢占式）的脚本
# =================================================================

# --- 仿真参数 ---

# 预测阈值 (0.0-1.0)
THRESHOLD=0.5

# 自由空间配置的采样率 (0.0-1.0)
SAMPLE_RATE=0.1

# NONCOLL队列长度相对于机器人球体数量的倍数
QN_MULTIPLIER=6

# 包含碰撞数据的文件夹路径
DATA_FOLDER="../trace_files/scene_benchmarks/bit_collision_data"

# 基准测试文件的基础名称 (例如 franka_14, iiwa_7)
BASENAME="iiwa_7"

# 要运行的基准测试数量
NUM_BENCHMARKS=10

# 机器人名称 (例如 franka, iiwa)
ROBOT_NAME="iiwa"

# OOCD（硬件碰撞检测器）的数量
NUM_OOCDS=22

# --- 执行仿真 ---

echo "开始运行抢占式球体碰撞仿真..."
echo "机器人: $ROBOT_NAME, OOCD数量: $NUM_OOCDS"

cd ..
python prediction_simulation_nDOF_sphere_preemptive.py \
    "$THRESHOLD" \
    "$SAMPLE_RATE" \
    "$QN_MULTIPLIER" \
    "$DATA_FOLDER" \
    "$BASENAME" \
    "$NUM_BENCHMARKS" \
    "$ROBOT_NAME" \
    "$NUM_OOCDS"

echo "仿真完成."
