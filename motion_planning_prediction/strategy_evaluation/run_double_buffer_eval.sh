#!/bin/bash

# 脚本功能：遍历G1-G5场景，对double_buffer策略进行评估
# 使用方法：./run_double_buffer_eval.sh

set -e  # 遇到错误立即退出

# === 配置参数 ===
THRESHOLD=1
SAMPLE_RATE=0.125
QNONCOLL_MULTIPLIER=8
BASENAME="iiwa_7"
BENCHID="1-10"
# 基础数据路径
BASE_DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
ROBOT_NAME="iiwa"
NUM_PREDICTIONS=2
# 总CDU数量（传递给仿真脚本的 num_oocds）
TOTAL_OOCDS=8
# 专用CDU数量（传递给仿真脚本的 num_dedicated_oocds）
DEDICATED_OOCDS=8
# 创建结果目录（如果不存在）
mkdir -p "../result_files"

# 遍历碰撞模型
for COLLISION_MODEL in sphere link; do
    # 结果文件路径
    RESULT_CSV="../result_files/double_buffer_${COLLISION_MODEL}_results.csv"

    # 初始化CSV文件头部
    echo "Scene,Threshold,Sample_Rate,QNonColl_Mult,Total_Checks,Total_Pred_Queries,Total_Oracle_Queries,Total_Cycles,Total_Oracle_Cycles,Reduction_Rate,Cycle_Efficiency,CDU_Utilization" > "$RESULT_CSV"

    echo "=========================================="
    echo "开始执行Double Buffer策略评估遍历..."
    echo "数据集: $BASENAME, Benchmark范围: $BENCHID"
    echo "阈值: $THRESHOLD, 采样率: $SAMPLE_RATE"
    echo "非碰撞队列乘数: $QNONCOLL_MULTIPLIER"
    echo "机器人: $ROBOT_NAME, 碰撞模型: $COLLISION_MODEL"
    echo "总CDU数量: $TOTAL_OOCDS"
    echo "专用CDU数量: $DEDICATED_OOCDS"
    echo "=========================================="

    # 遍历场景 G1-G5
    for SCENE in G1 G2 G3 G4 G5; do
        DATA_FOLDER="$BASE_DATA_FOLDER/$SCENE"
        echo "正在处理场景: $SCENE"

        # 运行仿真并将输出捕获到变量
        # 使用 2>&1 将 stderr 也捕获
        OUTPUT=$(python3 prediction_simulation_nDOF_double_buffer.py \
            "$THRESHOLD" \
            "$SAMPLE_RATE" \
            "$QNONCOLL_MULTIPLIER" \
            "$DATA_FOLDER" \
            "$BASENAME" \
            "$BENCHID" \
            "$ROBOT_NAME" \
            "$NUM_PREDICTIONS" \
            "$COLLISION_MODEL" \
            "$DEDICATED_OOCDS" \
            "$TOTAL_OOCDS")
        
        # 检查 python 脚本是否执行成功
        if [ $? -ne 0 ]; then
            echo "Error running simulation for $SCENE"
            echo "$OUTPUT"
            exit 1
        fi

        # 解析输出
        TOTAL_CHECKS=$(echo "$OUTPUT" | grep "Total Actual Checks:" | tail -n 1 | awk -F': ' '{print $2}')
        TOTAL_PRED_QUERIES=$(echo "$OUTPUT" | grep "Total Prediction Queries:" | tail -n 1 | awk -F': ' '{print $2}')
        TOTAL_ORACLE_QUERIES=$(echo "$OUTPUT" | grep "Total Oracle Queries:" | tail -n 1 | awk -F': ' '{print $2}')
        TOTAL_CYCLES=$(echo "$OUTPUT" | grep "Total Cycles (Prediction):" | tail -n 1 | awk -F': ' '{print $2}')
        TOTAL_ORACLE_CYCLES=$(echo "$OUTPUT" | grep "Total Cycles (Oracle):" | tail -n 1 | awk -F': ' '{print $2}')
        REDUCTION_RATE=$(echo "$OUTPUT" | grep "Query Reduction Rate:" | tail -n 1 | awk -F': ' '{print $2}' | sed 's/%//')
        CYCLE_EFFICIENCY=$(echo "$OUTPUT" | grep "Cycle Efficiency:" | tail -n 1 | awk -F': ' '{print $2}' | sed 's/%//')
        CDU_UTILIZATION=$(echo "$OUTPUT" | grep "Average CDU Utilization:" | tail -n 1 | awk -F': ' '{print $2}' | sed 's/%//')
        # 写入CSV
        echo "$SCENE,$THRESHOLD,$SAMPLE_RATE,$QNONCOLL_MULTIPLIER,$TOTAL_CHECKS,$TOTAL_PRED_QUERIES,$TOTAL_ORACLE_QUERIES,$TOTAL_CYCLES,$TOTAL_ORACLE_CYCLES,$REDUCTION_RATE,$CYCLE_EFFICIENCY,$CDU_UTILIZATION" >> "$RESULT_CSV"
    done

    echo "=========================================="
    echo "评估完成！结果已保存至: $RESULT_CSV"
    echo "=========================================="
done
