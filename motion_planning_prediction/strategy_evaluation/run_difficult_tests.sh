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

        # 初始化CSV文件头 (如果文件不存在) - 统一格式
        if [ ! -f "$RESULT_FILE" ]; then
            echo "Scene,Collision_Model,QNonColl_Mult,Threshold,Sample_Rate,Total_Checks,Total_Pred_Queries,Total_Oracle_Queries,Reduction_Rate,Query_Diff,Total_Pred_Cycles,Total_Oracle_Cycles,Cycle_Efficiency,OOCD_Utilization" > "$RESULT_FILE"
        fi

        # 运行仿真并捕获输出
        # 注意: prediction_simulation_nDOF.py 参数顺序: threshold, sample_rate, qnoncoll_multiplier, data_folder, basename, benchmarks, robot_name, collision_model_type, num_oocds
        # 原脚本调用似乎缺少 num_oocds，依赖默认值? 或者参数解析器处理了? 
        # 最好显式传递以防万一，但为了保持兼容性，我们先按原样传递，如果报错再修。
        # 修正：原脚本参数传递方式是位置参数。prediction_simulation_nDOF.py 使用 create_common_parser，它接受可选参数。
        # 我们按照标准位置传递参数。
        OUTPUT=$(python prediction_simulation_nDOF.py \
            "$THRESHOLD" "$SAMPLE_RATE" "$QNONCOLL_MULTIPLIER" \
            "$DATA_FOLDER" "$BASENAME" "$NUM_BENCHMARKS" \
            "$ROBOT_NAME" "$COLLISION_MODEL")
        
        # 显示输出
        echo "$OUTPUT"

        # 提取数据 (基于 print_final_statistics 的标准输出)
        TOTAL_CHECKS=$(echo "$OUTPUT" | grep "Total Actual Checks:" | awk -F': ' '{print $2}')
        TOTAL_PRED_QUERIES=$(echo "$OUTPUT" | grep "Total Prediction Queries:" | awk -F': ' '{print $2}')
        TOTAL_ORACLE_QUERIES=$(echo "$OUTPUT" | grep "Total Oracle Queries:" | awk -F': ' '{print $2}')
        REDUCTION_RATE=$(echo "$OUTPUT" | grep "Query Reduction Rate:" | awk -F': ' '{print $2}' | sed 's/%//')
        QUERY_DIFF=$(echo "$OUTPUT" | grep "Query Difference (Prediction - Oracle):" | awk -F': ' '{print $2}' | sed 's/%//')
        TOTAL_PRED_CYCLES=$(echo "$OUTPUT" | grep "Total Cycles (Prediction):" | awk -F': ' '{print $2}')
        TOTAL_ORACLE_CYCLES=$(echo "$OUTPUT" | grep "Total Cycles (Oracle):" | awk -F': ' '{print $2}')
        CYCLE_EFFICIENCY=$(echo "$OUTPUT" | grep "Cycle Efficiency:" | awk -F': ' '{print $2}' | sed 's/%//')
        OOCD_UTILIZATION=$(echo "$OUTPUT" | grep "Average OOCD Utilization:" | awk -F': ' '{print $2}' | sed 's/%//')

        # 写入CSV (统一顺序)
        echo "$difficulty_level,$COLLISION_MODEL,$QNONCOLL_MULTIPLIER,$THRESHOLD,$SAMPLE_RATE,$TOTAL_CHECKS,$TOTAL_PRED_QUERIES,$TOTAL_ORACLE_QUERIES,$REDUCTION_RATE,$QUERY_DIFF,$TOTAL_PRED_CYCLES,$TOTAL_ORACLE_CYCLES,$CYCLE_EFFICIENCY,$OOCD_UTILIZATION" >> "$RESULT_FILE"

        echo "  >>> 难度等级 $difficulty_level 运行完成 <<<"
    done

    echo ">>> 碰撞模型 $COLLISION_MODEL 运行完成 <<<"
done