#!/bin/bash

# 脚本功能：在 G1-G5 场景上评估 Global OOCD 调度策略，输出结果到 CSV
# 使用方法：./run_global_oocd_sim.sh

set -e  # 遇到错误立即退出

# === 配置参数 ===
BASENAME="iiwa_7"
BENCHID="1-10"
BASE_DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
NUM_OOCDS=56
THRESHOLD=0.5
SAMPLE_RATE=0.1
NUM_PRED_LIST=(8 16)
NUM_BANKS=8
RESULT_DIR="../result_files"

# 创建结果目录（如果不存在）
mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "开始执行 Global OOCD 仿真遍历..."
echo "数据集: $BASENAME, Benchmark范围: $BENCHID"
echo "OOCD数: $NUM_OOCDS, 采样率: $SAMPLE_RATE"
echo "=========================================="

# 遍历 CHT 类型
auth_types=(dual_port multi_bank)
for CHT_TYPE in "${auth_types[@]}"; do
    if [ "$CHT_TYPE" == "multi_bank" ]; then
        echo ""
        echo "=========================================="
        echo "CHT类型: $CHT_TYPE (Bank数: $NUM_BANKS)"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "CHT类型: $CHT_TYPE"
        echo "=========================================="
    fi

    # 遍历 Prediction 缓冲配置
    for NUM_PRED in "${NUM_PRED_LIST[@]}"; do
        echo ""
        echo "  Prediction缓冲数: $NUM_PRED"
        echo "  ------------------------------------------"

        # 准备 CSV 文件及表头
        if [ "$CHT_TYPE" == "dual_port" ]; then
            CSV_FILE="$RESULT_DIR/global_oocd_dual_port_pred${NUM_PRED}_results.csv"
            echo "Scene,Num_Pred,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts" > "$CSV_FILE"
        else
            CSV_FILE="$RESULT_DIR/global_oocd_multi_bank_pred${NUM_PRED}_results.csv"
            echo "Scene,Num_Banks,Num_Pred,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts" > "$CSV_FILE"
        fi

        # 遍历场景 G1-G5
        for SCENE in G1 G2 G3 G4 G5; do
            DATA_FOLDER="$BASE_DATA_FOLDER/$SCENE"
            echo -n "    场景 $SCENE: "

            # 运行仿真并捕获输出
            OUTPUT=$(python3 global_oocd_simulation.py \
                "$BASENAME" \
                "$BENCHID" \
                "$DATA_FOLDER" \
                "$NUM_OOCDS" \
                "$THRESHOLD" \
                "$SAMPLE_RATE" \
                "$NUM_PRED" \
                --max-oocd-per-pred 10 \
                --cht-type "$CHT_TYPE" \
                --num-banks "$NUM_BANKS" 2>&1)

            # 检查执行是否成功
            if [ $? -ne 0 ]; then
                echo "✗ 失败"
                echo "Error details: $OUTPUT"
                exit 1
            fi

            # 解析输出
            TOTAL_CYCLES=$(echo "$OUTPUT" | grep "总周期:" | tail -n 1 | awk -F': ' '{print $2}')
            TOTAL_QUERIES=$(echo "$OUTPUT" | grep "总查询数:" | tail -n 1 | awk -F': ' '{print $2}')
            THROUGHPUT=$(echo "$OUTPUT" | grep "系统吞吐量:" | tail -n 1 | awk -F': ' '{print $2}' | awk '{print $1}')
            UTILIZATION=$(echo "$OUTPUT" | grep "平均占用率:" | tail -n 1 | awk -F': ' '{print $2}')
            CONFLICTS=$(echo "$OUTPUT" | grep "CHT冲突数:" | tail -n 1 | awk -F': ' '{print $2}')

            echo "✓ (cycles=$TOTAL_CYCLES, throughput=$THROUGHPUT, utilization=$UTILIZATION)"

            # 写入 CSV
            if [ "$CHT_TYPE" == "dual_port" ]; then
                echo "$SCENE,$NUM_PRED,$TOTAL_CYCLES,$TOTAL_QUERIES,$THROUGHPUT,$UTILIZATION,$CONFLICTS" >> "$CSV_FILE"
            else
                echo "$SCENE,$NUM_BANKS,$NUM_PRED,$TOTAL_CYCLES,$TOTAL_QUERIES,$THROUGHPUT,$UTILIZATION,$CONFLICTS" >> "$CSV_FILE"
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "Global OOCD 仿真完成！"
echo "结果已保存至:"
echo "  - $RESULT_DIR/global_oocd_dual_port_pred1_results.csv"
echo "  - $RESULT_DIR/global_oocd_dual_port_pred2_results.csv"
echo "  - $RESULT_DIR/global_oocd_multi_bank_pred1_results.csv"
echo "  - $RESULT_DIR/global_oocd_multi_bank_pred2_results.csv"
echo "=========================================="
