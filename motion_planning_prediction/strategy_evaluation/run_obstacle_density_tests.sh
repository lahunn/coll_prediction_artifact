#!/bin/bash

# 批量运行不同障碍物密度的预测仿真
# 遍历碰撞模型: link, sphere
# 遍历障碍物数量: 3, 6, 9, 12

THRESHOLD=0.5
SAMPLE_RATE=0.1
QNONCOLL_MULTIPLIER=8
DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
BASENAME="iiwa_7"
NUM_BENCHMARKS=100
ROBOT_NAME="iiwa"

echo "=== 批量运行不同障碍物密度的预测仿真 ==="
echo "参数配置:"
echo "  阈值: $THRESHOLD"
echo "  采样率: $SAMPLE_RATE"
echo "  队列长度倍数: $QNONCOLL_MULTIPLIER"
echo "  数据文件夹: $DATA_FOLDER"
echo "  基准测试名称: $BASENAME"
echo "  基准测试数量: $NUM_BENCHMARKS"
echo "  机器人: $ROBOT_NAME"
echo "=========================================="

# 遍历碰撞模型
for COLLISION_MODEL in link sphere; do
    echo ""
    echo ">>> 运行碰撞模型: $COLLISION_MODEL <<<"

    # 根据碰撞模型确定结果文件并删除旧结果
    [ "$COLLISION_MODEL" = "sphere" ] && RESULT_FILE="./result_files/sphere_results.csv" || RESULT_FILE="./result_files/obb_results.csv"
    rm -rf "$RESULT_FILE"

    # 遍历障碍物数量
    for num_obstacles in 3 6 9 12; do
        echo ""
        echo "  >>> 运行障碍物数量: $num_obstacles <<<"

        python prediction_simulation_nDOF.py \
            $THRESHOLD $SAMPLE_RATE $QNONCOLL_MULTIPLIER \
            $DATA_FOLDER $BASENAME $NUM_BENCHMARKS \
            $ROBOT_NAME $COLLISION_MODEL $num_obstacles

        echo "  >>> 障碍物数量 $num_obstacles 运行完成 <<<"
    done

    echo ">>> 碰撞模型 $COLLISION_MODEL 运行完成 <<<"
done