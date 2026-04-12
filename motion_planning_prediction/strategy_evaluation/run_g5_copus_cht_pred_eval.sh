#!/bin/bash

# 脚本功能：固定 G5 场景，遍历 COPU 数量、CHT 类型和预测通道数
# 用于分析在不同硬件配置下，dual_port / multi_bank 与 单通道 / 双通道预测的相对优势
# 使用方法：./run_g5_copus_cht_pred_eval.sh

set -e

# === 配置参数 ===
THRESHOLD=1
SAMPLE_RATE=0.125
QNONCOLL_MULTIPLIER=8
BASE_DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
WARMSTART_BASE_FOLDER="../../trace_files/cht_pre_load"
BASENAME="iiwa_7"
BENCHID="1-10"
COLLISION_TYPE="sphere"
FIXED_QNONCOLL_LEN=128

# === 固定场景 ===
SCENE="G5"
DATA_FOLDER="$BASE_DATA_FOLDER/$SCENE"
WARMSTART_DIR="$WARMSTART_BASE_FOLDER/$SCENE"

# === 遍历维度 ===
SWEEP_NUM_COPUS=(1 2 4 8 16 32)
SWEEP_CHT_TYPES=("dual_port" "distri_multi_bank")
SWEEP_NUM_PRED=(1 2)

# === 固定基准参数 ===
BASE_NUM_OOCDS=8
BASE_NUM_BANKS=8
BASE_COPUS_PER_EDGE=4

RESULT_DIR="../result_files"
mkdir -p "$RESULT_DIR"
RESULT_FILE="$RESULT_DIR/g5_copus_cht_pred_eval_${COLLISION_TYPE}.csv"

cat > "$RESULT_FILE" << EOF
Num_COPUS,CHT_Type,Pred,Scene,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts,Avg_Wait_Cycles,DEAD_AVG_RATIO
EOF

echo "=========================================="
echo "开始执行 G5 固定场景评测..."
echo "数据集: $BASENAME, Benchmark范围: $BENCHID, Scene: $SCENE"
echo "遍历维度: COPUS={1,2,4,8,16,32}, CHT_TYPE={dual_port,distri_multi_bank}, PRED={1,2}"
echo "固定参数: OOCDS=$BASE_NUM_OOCDS, BANKS=$BASE_NUM_BANKS, THRESHOLD=$THRESHOLD, SAMPLE_RATE=$SAMPLE_RATE"
echo "=========================================="

for num_copus in "${SWEEP_NUM_COPUS[@]}"; do
    for cht_type in "${SWEEP_CHT_TYPES[@]}"; do
        for num_pred in "${SWEEP_NUM_PRED[@]}"; do
            echo ""
            echo "=========================================="
            echo "组合: COPUS=$num_copus, CHT_TYPE=$cht_type, PRED=$num_pred"
            echo "输出文件: $RESULT_FILE"
            echo "=========================================="

            copus_per_edge="$BASE_COPUS_PER_EDGE"
            if [ "$num_copus" = "1" ] || [ "$num_copus" = "2" ]; then
                copus_per_edge="$num_copus"
            fi

            qnoncoll_multiplier=$((FIXED_QNONCOLL_LEN / BASE_NUM_OOCDS))

            cht_args=(--cht-type "$cht_type")
            if [ "$cht_type" = "multi_bank" ] || [ "$cht_type" = "distri_multi_bank" ]; then
                cht_args+=(--num-banks "$BASE_NUM_BANKS")
            fi

            OUTPUT=$(python3 multi_copu_real_data_simulation.py \
                "$BASENAME" \
                "$BENCHID" \
                "$DATA_FOLDER" \
                "$num_copus" \
                "$THRESHOLD" \
                "$BASE_NUM_OOCDS" \
                "$SAMPLE_RATE" \
                "$num_pred" \
                --copus-per-edge "$copus_per_edge" \
                "${cht_args[@]}" \
                --collision-type "$COLLISION_TYPE" \
                --cht-warmstart-dir "$WARMSTART_DIR" \
                --qnoncoll-multiplier "$qnoncoll_multiplier" 2>&1)

            if [ $? -ne 0 ]; then
                echo "✗ 失败"
                echo "Error details: $OUTPUT"
                exit 1
            fi

            TOTAL_CYCLES=$(echo "$OUTPUT" | grep "总周期:" | tail -n 1 | awk -F': ' '{print $2}')
            TOTAL_QUERIES=$(echo "$OUTPUT" | grep "总查询数:" | tail -n 1 | awk -F': ' '{print $2}')
            THROUGHPUT=$(echo "$OUTPUT" | grep "系统吞吐量:" | tail -n 1 | awk -F': ' '{print $2}' | awk '{print $1}')
            UTILIZATION=$(echo "$OUTPUT" | grep "平均COPU占用率:" | tail -n 1 | awk -F': ' '{print $2}')
            CONFLICTS=$(echo "$OUTPUT" | grep "CHT冲突数:" | tail -n 1 | awk -F': ' '{print $2}')
            AVG_WAIT_CYCLES=$(echo "$OUTPUT" | grep "平均等待周期:" | tail -n 1 | awk -F': ' '{print $2}')
            DEAD_AVG_RATIO=$(echo "$OUTPUT" | grep "Dead Time Avg Ratio Per Edge:" | tail -n 1 | awk -F': ' '{print $2}' | sed 's/%//')

            if [ -z "$AVG_WAIT_CYCLES" ]; then
                AVG_WAIT_CYCLES=0
            fi
            if [ -z "$DEAD_AVG_RATIO" ]; then
                DEAD_AVG_RATIO=0
            fi

            echo "✓ (cycles=$TOTAL_CYCLES, throughput=$THROUGHPUT, utilization=$UTILIZATION, conflicts=$CONFLICTS, avg_wait=$AVG_WAIT_CYCLES, dead_avg_ratio=${DEAD_AVG_RATIO}%)"
            echo "$num_copus,$cht_type,$num_pred,$SCENE,$TOTAL_CYCLES,$TOTAL_QUERIES,$THROUGHPUT,$UTILIZATION,$CONFLICTS,$AVG_WAIT_CYCLES,$DEAD_AVG_RATIO" >> "$RESULT_FILE"
        done
    done
done

echo ""
echo "=========================================="
echo "G5 固定场景评测完成！"
echo "结果已保存至: $RESULT_FILE"
echo "=========================================="
