#!/bin/bash

# 脚本功能：在经典配置下做消融实验
# 遍历 pred = 1/2，以及 CHT 类型 = dual_port / multi_bank / distri_dual_port / distri_multi_bank
# 使用方法：./run_ablation_pred_sram.sh

set -e  # 遇到错误立即退出

# === 配置参数 ===
THRESHOLD=1
SAMPLE_RATE=0.125
# 基础数据路径
BASE_DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
# warm-start 数据路径
WARMSTART_BASE_FOLDER="../../trace_files/cht_pre_load"
BASENAME="iiwa_7"
BENCHID="1-10"

# === 经典配置（固定） ===
BASE_NUM_COPUS=16
BASE_NUM_OOCDS=8
BASE_NUM_PRED=2
BASE_NUM_BANKS=8
COLLISION_TYPE="sphere"
COPUS_PER_EDGE=4 # 最好不要让COPUS_PER_EDGE过小，否则在 pred=2 时可能出现大量 edge 被预分配到 COPU 的情况，导致部分copu执行完任务后没有下一个任务,进入空转，从而影响吞吐量和利用率的评估(会导致pred = 2 劣于 pred = 1)
FIXED_QNONCOLL_LEN=128

# === 消融维度 ===
SWEEP_NUM_PRED=(1 2)
SWEEP_CHT_TYPES=("distri_multi_bank" "dual_port" "multi_bank" "distri_dual_port")

# 结果文件路径
RESULT_DIR="../result_files"
mkdir -p "$RESULT_DIR"
RESULT_FILE="$RESULT_DIR/ablation_pred_sram_${COLLISION_TYPE}_results.csv"

qnoncoll_multiplier=$((FIXED_QNONCOLL_LEN / BASE_NUM_OOCDS))

cat > "$RESULT_FILE" << EOF
Pred,CHT_Type,Num_COPUS,Num_OOCDS,Num_BANKS,Scene,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts,Avg_Wait_Cycles,DEAD_AVG_RATIO
EOF

echo "=========================================="
echo "开始执行消融实验（经典配置）..."
echo "数据集: $BASENAME, Benchmark范围: $BENCHID"
echo "固定场景: COPUS=$BASE_NUM_COPUS, OOCDS=$BASE_NUM_OOCDS, BANKS=$BASE_NUM_BANKS"
echo "遍历维度: PRED={1,2}, CHT_TYPE={dual_port,multi_bank,distri_dual_port,distri_multi_bank}"
echo "CHT_TYPE基准: $COLLISION_TYPE"
echo "固定QNONCOLL_LEN: $FIXED_QNONCOLL_LEN"
echo "=========================================="


for cht_type in "${SWEEP_CHT_TYPES[@]}"; do
    for num_pred in "${SWEEP_NUM_PRED[@]}"; do
        echo ""
        echo "=========================================="
        echo "开始组合: PRED=$num_pred, CHT_TYPE=$cht_type"
        echo "输出文件: $RESULT_FILE"
        echo "=========================================="

        for SCENE in G1 G2 G3 G4 G5; do
            DATA_FOLDER="$BASE_DATA_FOLDER/$SCENE"
            WARMSTART_DIR="$WARMSTART_BASE_FOLDER/$SCENE"
            echo -n "  场景 $SCENE: "

            cht_args=(--cht-type "$cht_type")
            if [ "$cht_type" = "multi_bank" ] || [ "$cht_type" = "distri_multi_bank" ]; then
                cht_args+=(--num-banks "$BASE_NUM_BANKS")
            fi

            OUTPUT=$(python3 multi_copu_real_data_simulation.py \
                "$BASENAME" \
                "$BENCHID" \
                "$DATA_FOLDER" \
                "$BASE_NUM_COPUS" \
                "$THRESHOLD" \
                "$BASE_NUM_OOCDS" \
                "$SAMPLE_RATE" \
                "$num_pred" \
                --copus-per-edge "$COPUS_PER_EDGE" \
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
            echo "$num_pred,$cht_type,$BASE_NUM_COPUS,$BASE_NUM_OOCDS,$BASE_NUM_BANKS,$SCENE,$TOTAL_CYCLES,$TOTAL_QUERIES,$THROUGHPUT,$UTILIZATION,$CONFLICTS,$AVG_WAIT_CYCLES,$DEAD_AVG_RATIO" >> "$RESULT_FILE"
        done
    done
done

echo ""
echo "=========================================="
echo "消融实验完成！"
echo "结果已保存至: $RESULT_FILE"
echo "=========================================="
