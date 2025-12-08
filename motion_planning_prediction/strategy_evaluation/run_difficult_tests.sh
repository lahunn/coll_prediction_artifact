#!/bin/bash

# 批量运行不同难度等级的预测仿真
# 遍历碰撞模型: link, sphere
# 遍历难度等级: G1, G2, G3, G4, G5

THRESHOLD=0.5
SAMPLE_RATE=0.1
QNONCOLL_MULTIPLIER=8
BASE_DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
BASENAME="iiwa_7"
NUM_BENCHMARKS=10
ROBOT_NAME="iiwa"

echo "=== 批量运行不同难度等级的预测仿真 ==="
echo "参数配置:"
echo "  阈值: $THRESHOLD"
echo "  采样率: $SAMPLE_RATE"
echo "  队列长度倍数: $QNONCOLL_MULTIPLIER"
echo "  基准数据文件夹: $BASE_DATA_FOLDER"
echo "  基准测试名称: $BASENAME"
echo "  基准测试数量: $NUM_BENCHMARKS"
echo "  机器人: $ROBOT_NAME"
echo "=========================================="

# 遍历碰撞模型
for COLLISION_MODEL in link sphere; do
    echo ""
    echo ">>> 运行碰撞模型: $COLLISION_MODEL <<<"

    # 根据碰撞模型确定结果文件并删除旧结果
    RESULT_FILE="../result_files/${COLLISION_MODEL}_results.csv"
    rm -rf "$RESULT_FILE"

    # 遍历难度等级
    for difficulty_level in G1 G2 G3 G4 G5; do
        echo ""
        echo "  >>> 运行难度等级: $difficulty_level <<<"

        # 设置当前难度等级的数据文件夹
        DATA_FOLDER="$BASE_DATA_FOLDER/$difficulty_level"

        # 初始化CSV文件头 (如果文件不存在)
        if [ ! -f "$RESULT_FILE" ]; then
            echo "Scene,Threshold,Sample_Rate,QNonColl_Mult,Total_Checks,Total_Pred_Queries,Total_Oracle_Queries,Total_Cycles,Total_Oracle_Cycles,Reduction_Rate,Cycle_Efficiency,CDU_Utilization" > "$RESULT_FILE"
        fi

        # 运行仿真并捕获输出
        OUTPUT=$(python prediction_simulation_nDOF.py \
            $THRESHOLD $SAMPLE_RATE $QNONCOLL_MULTIPLIER \
            $DATA_FOLDER $BASENAME $NUM_BENCHMARKS \
            $ROBOT_NAME $COLLISION_MODEL)
        
        # 显示输出
        echo "$OUTPUT"

        # 提取数据
        TOTAL_CHECKS=$(echo "$OUTPUT" | grep "Total Actual Checks:" | awk -F': ' '{print $2}')
        FALL_PREDICTION=$(echo "$OUTPUT" | grep "Total Prediction Queries:" | awk -F': ' '{print $2}')
        FALL_ORACLE=$(echo "$OUTPUT" | grep "Total Oracle Queries:" | awk -F': ' '{print $2}')
        REDUCTION_RATE=$(echo "$OUTPUT" | grep "Query Reduction Rate:" | awk -F': ' '{print $2}' | sed 's/%//')
        FALL_CYCLE=$(echo "$OUTPUT" | grep "Total Cycles (Prediction):" | awk -F': ' '{print $2}')
        THEORETICAL_MIN_CYCLES=$(echo "$OUTPUT" | grep "Total Cycles (Oracle):" | awk -F': ' '{print $2}')
        CYCLE_EFFICIENCY=$(echo "$OUTPUT" | grep "Cycle Efficiency:" | awk -F': ' '{print $2}' | sed 's/%//')

        # 写入CSV (CDU_Utilization 设为 0)
        echo "$difficulty_level,$THRESHOLD,$SAMPLE_RATE,$QNONCOLL_MULTIPLIER,$TOTAL_CHECKS,$FALL_PREDICTION,$FALL_ORACLE,$FALL_CYCLE,$THEORETICAL_MIN_CYCLES,$REDUCTION_RATE,$CYCLE_EFFICIENCY,0" >> "$RESULT_FILE"

        echo "  >>> 难度等级 $difficulty_level 运行完成 <<<"
    done

    echo ">>> 碰撞模型 $COLLISION_MODEL 运行完成 <<<"
done