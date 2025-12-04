#!/bin/bash

# 脚本用于批量运行 sphere_coord 和 link_coord 的对比仿真
# 并将结果保存到 CSV 文件中

# 参数设置
THRESHOLD=0.5
SAMPLE_RATE=0.1
QNONCOLL_MULTIPLIER=8
BASENAME="iiwa_7"
ROBOT_NAME="iiwa"
START_BENCH=1
END_BENCH=100

# 结果文件目录
RESULT_DIR="../result_files"
mkdir -p "$RESULT_DIR"
RESULT_FILE="${RESULT_DIR}/sphere_link_comparison_results.csv"

# 基础数据路径 (相对于脚本执行位置)
BASE_DATA_DIR="../../trace_files/scene_benchmarks/bit_collision_data"

# 初始化 CSV 文件
echo "Difficulty,Strategy,Threshold,Sample_Rate,Total_Checks,Total_Pred_Queries,Total_Oracle_Queries,Query_Reduction_Rate,Query_Difference,Total_Pred_Cycles,Total_Oracle_Cycles,Cycle_Efficiency" > "$RESULT_FILE"

echo "=== 开始运行对比仿真 ==="

# 修改遍历顺序：先遍历策略，再遍历难度
for STRATEGY in sphere_coord link_coord; do
    for DIFFICULTY in G1 G2 G3 G4 G5; do
        DATA_FOLDER="${BASE_DATA_DIR}/${DIFFICULTY}"
        
        # 检查数据目录是否存在
        if [ ! -d "$DATA_FOLDER" ]; then
            echo "警告: 数据目录不存在 $DATA_FOLDER，跳过..."
            continue
        fi

        echo "--------------------------------------------------"
        echo "正在处理: 策略=$STRATEGY, 难度=$DIFFICULTY"
        
        # 运行 Python 脚本并捕获输出
        OUTPUT=$(python prediction_simulation_sphere_link.py \
            $THRESHOLD \
            $SAMPLE_RATE \
            $QNONCOLL_MULTIPLIER \
            "$DATA_FOLDER" \
            "$BASENAME" \
            $START_BENCH \
            $END_BENCH \
            "$ROBOT_NAME" \
            "$STRATEGY")
            
        # 解析结果
        TOTAL_CHECKS=$(echo "$OUTPUT" | grep "Total Actual Checks:" | awk -F': ' '{print $2}')
        PRED_QUERIES=$(echo "$OUTPUT" | grep "Total Prediction Queries:" | awk -F': ' '{print $2}')
        ORACLE_QUERIES=$(echo "$OUTPUT" | grep "Total Oracle Queries:" | awk -F': ' '{print $2}')
        REDUCTION_RATE=$(echo "$OUTPUT" | grep "Query Reduction Rate:" | awk -F': ' '{print $2}' | sed 's/%//')
        QUERY_DIFF=$(echo "$OUTPUT" | grep "Query Difference (Prediction - Oracle):" | awk -F': ' '{print $2}' | sed 's/%//')
        PRED_CYCLES=$(echo "$OUTPUT" | grep "Total Cycles (Prediction):" | awk -F': ' '{print $2}')
        ORACLE_CYCLES=$(echo "$OUTPUT" | grep "Total Cycles (Oracle):" | awk -F': ' '{print $2}')
        CYCLE_EFFICIENCY=$(echo "$OUTPUT" | grep "Cycle Efficiency:" | awk -F': ' '{print $2}' | sed 's/%//')
        
        # 检查是否成功提取到数据
        if [ -z "$TOTAL_CHECKS" ]; then
            echo "错误: 无法从输出中提取数据。请检查 Python 脚本是否运行成功。"
            echo "Python 脚本输出片段:"
            echo "$OUTPUT" | tail -n 10
        else
            # 写入 CSV
            echo "$DIFFICULTY,$STRATEGY,$THRESHOLD,$SAMPLE_RATE,$TOTAL_CHECKS,$PRED_QUERIES,$ORACLE_QUERIES,$REDUCTION_RATE,$QUERY_DIFF,$PRED_CYCLES,$ORACLE_CYCLES,$CYCLE_EFFICIENCY" >> "$RESULT_FILE"
            echo "结果已写入 CSV: Checks=$TOTAL_CHECKS, Efficiency=$CYCLE_EFFICIENCY%"
        fi
    done
done

echo "=== 所有仿真完成 ==="
echo "结果保存在: $(readlink -f $RESULT_FILE)"
