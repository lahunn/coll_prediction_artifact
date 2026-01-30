#!/bin/bash

# 脚本功能：遍历G1-G5场景，对double_buffer策略进行评估
# 使用方法：./run_double_buffer_eval.sh

set -e  # 遇到错误立即退出

# === 配置参数 ===
BASENAME="iiwa_7"
BENCHID="1-10"
# 遍历的 QNONCOLL 值
# 基础数据路径
BASE_DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
THRESHOLD=0.5
SAMPLE_RATE=0.1
MULTIPLIERS="4 8 16 32 64"
ROBOT_NAME="iiwa"
NUM_OOCDS=8
NUM_PREDICTIONS=2
# 专用CDU数量（传递给仿真脚本的 num_dedicated_oocds）
DEDICATED_OOCDS=8
# 创建结果目录（如果不存在）
mkdir -p "../result_files"

# 遍历碰撞模型
for COLLISION_MODEL in sphere link; do
    # 结果文件路径
    RESULT_CSV="../result_files/double_buffer_${COLLISION_MODEL}_results.csv"

    # 初始化CSV文件头部 (统一格式)
    echo "Scene,Collision_Model,QNonColl_Mult,Threshold,Sample_Rate,Total_Checks,Total_Pred_Queries,Total_Oracle_Queries,Reduction_Rate,Query_Diff,Total_Pred_Cycles,Total_Oracle_Cycles,Cycle_Efficiency,OOCD_Utilization" > "$RESULT_CSV"

    echo "=========================================="
    echo "开始执行Double Buffer策略评估遍历..."
    echo "碰撞模型: $COLLISION_MODEL"
    echo "数据集: $BASENAME, Benchmark范围: $BENCHID"
    echo "阈值: $THRESHOLD, 采样率: $SAMPLE_RATE"
    echo "机器人: $ROBOT_NAME, OOCDs: $NUM_OOCDS"
    echo "专用CDU数量: $DEDICATED_OOCDS"
    echo "=========================================="

    # 遍历 QNONCOLL 值
    for QNONCOLL_MULTIPLIER in $MULTIPLIERS; do
        echo "--- QNonColl Multiplier = $QNONCOLL_MULTIPLIER ---"
            # 遍历场景 G1-G5
            for SCENE in G1 G2 G3 G4 G5; do
                    DATA_FOLDER="$BASE_DATA_FOLDER/$SCENE"
                    echo "正在处理场景: $SCENE, 队列倍率: $QNONCOLL_MULTIPLIER"
        
                    # 运行仿真并将输出捕获到变量
                    # 修正参数顺序：... robot_name collision_model num_oocds num_pred dedicated
                    OUTPUT=$(python3 prediction_simulation_nDOF_double_buffer.py \
                        "$THRESHOLD" \
                        "$SAMPLE_RATE" \
                        "$QNONCOLL_MULTIPLIER" \
                        "$DATA_FOLDER" \
                        "$BASENAME" \
                        "$BENCHID" \
                        "$ROBOT_NAME" \
                        "$COLLISION_MODEL" \
                        "$NUM_OOCDS" \
                        "$NUM_PREDICTIONS" \
                        "$DEDICATED_OOCDS")
                    # 检查 python 脚本是否执行成功
              if [ $? -ne 0 ]; then
                  echo "Error running simulation for $SCENE (QNonColl=$QNONCOLL_MULTIPLIER)"
                  echo "$OUTPUT"
                  exit 1
              fi

              # 解析输出 (基于 print_final_statistics 的标准输出)
              TOTAL_CHECKS=$(echo "$OUTPUT" | grep "Total Actual Checks:" | tail -n 1 | awk -F': ' '{print $2}')
              TOTAL_PRED_QUERIES=$(echo "$OUTPUT" | grep "Total Prediction Queries:" | tail -n 1 | awk -F': ' '{print $2}')
              TOTAL_ORACLE_QUERIES=$(echo "$OUTPUT" | grep "Total Oracle Queries:" | tail -n 1 | awk -F': ' '{print $2}')
              TOTAL_CYCLES=$(echo "$OUTPUT" | grep "Total Cycles (Prediction):" | tail -n 1 | awk -F': ' '{print $2}')
              TOTAL_ORACLE_CYCLES=$(echo "$OUTPUT" | grep "Total Cycles (Oracle):" | tail -n 1 | awk -F': ' '{print $2}')
              REDUCTION_RATE=$(echo "$OUTPUT" | grep "Query Reduction Rate:" | tail -n 1 | awk -F': ' '{print $2}' | sed 's/%//')
              QUERY_DIFF=$(echo "$OUTPUT" | grep "Query Difference (Prediction - Oracle):" | tail -n 1 | awk -F': ' '{print $2}' | sed 's/%//')
              CYCLE_EFFICIENCY=$(echo "$OUTPUT" | grep "Cycle Efficiency:" | tail -n 1 | awk -F': ' '{print $2}' | sed 's/%//')
              CDU_UTILIZATION=$(echo "$OUTPUT" | grep "Average OOCD Utilization:" | tail -n 1 | awk -F': ' '{print $2}' | sed 's/%//')

              # 写入CSV (统一顺序)
              echo "$SCENE,$COLLISION_MODEL,$QNONCOLL_MULTIPLIER,$THRESHOLD,$SAMPLE_RATE,$TOTAL_CHECKS,$TOTAL_PRED_QUERIES,$TOTAL_ORACLE_QUERIES,$REDUCTION_RATE,$QUERY_DIFF,$TOTAL_CYCLES,$TOTAL_ORACLE_CYCLES,$CYCLE_EFFICIENCY,$CDU_UTILIZATION" >> "$RESULT_CSV"
            done
    done

    echo "=========================================="
    echo "评估完成！结果已保存至: $RESULT_CSV"
    echo "=========================================="
done
