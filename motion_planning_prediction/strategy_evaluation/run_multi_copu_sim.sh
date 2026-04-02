#!/bin/bash

# 脚本功能：运行多COPU实际数据仿真（仅sphere），在标准场景下做单变量遍历
# 使用方法：./run_multi_copu_sim.sh

set -e  # 遇到错误立即退出

# === 配置参数 ===
THRESHOLD=1
SAMPLE_RATE=0.125
QNONCOLL_MULTIPLIER=8
# 基础数据路径
BASE_DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
BASENAME="iiwa_7"
BENCHID="1-10"

# === 标准场景（baseline） ===
BASE_NUM_COPUS=16
BASE_NUM_OOCDS=8
BASE_NUM_PRED=2
BASE_NUM_BANKS=8
BASE_CHT_TYPE="multi_bank"
COLLISION_TYPE="sphere"
COPUS_PER_EDGE=1
FIXED_QNONCOLL_LEN=128  # 固定非碰撞队列长度

# === 遍历列表（其余参数固定为标准场景） ===
SWEEP_NUM_OOCDS=(2 4 8 12 16)
SWEEP_NUM_COPUS=(2 4 8 12 16)
SWEEP_NUM_PRED=(1 2 4 6 8)
SWEEP_NUM_BANKS=(1 2 4 6 8)

# 结果文件路径
RESULT_DIR="../result_files"

# 创建结果目录（如果不存在）
mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "开始执行多COPU仿真遍历（仅sphere）..."
echo "数据集: $BASENAME, Benchmark范围: $BENCHID"
echo "标准场景: COPUS=$BASE_NUM_COPUS, PRED=$BASE_NUM_PRED, OOCDS=$BASE_NUM_OOCDS, BANKS=$BASE_NUM_BANKS"
echo "CHT_TYPE: $BASE_CHT_TYPE, COLLISION_TYPE: $COLLISION_TYPE"
echo "固定QNONCOLL_LEN: $FIXED_QNONCOLL_LEN"
echo "=========================================="

run_sweep() {
    local sweep_name="$1"
    local csv_file="$2"
    local values_name="$3"

    echo ""
    echo "=========================================="
    echo "开始遍历: $sweep_name"
    echo "输出文件: $csv_file"
    echo "=========================================="

    echo "Sweep,Value,Scene,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts,Avg_Wait_Cycles,DEAD_AVG_RATIO" > "$csv_file"

    local -n values_ref="$values_name"
    for value in "${values_ref[@]}"; do
        # 默认使用标准场景参数
        local num_copus="$BASE_NUM_COPUS"
        local num_oocds="$BASE_NUM_OOCDS"
        local num_pred="$BASE_NUM_PRED"
        local num_banks="$BASE_NUM_BANKS"

        # 仅替换当前遍历变量
        if [ "$sweep_name" == "NUM_OOCDS" ]; then
            num_oocds="$value"
        elif [ "$sweep_name" == "NUM_COPUS" ]; then
            num_copus="$value"
        elif [ "$sweep_name" == "NUM_PRED" ]; then
            num_pred="$value"
        elif [ "$sweep_name" == "NUM_BANKS" ]; then
            num_banks="$value"
        fi

        echo ""
        echo "[$sweep_name=$value] COPUS=$num_copus, PRED=$num_pred, OOCDS=$num_oocds, BANKS=$num_banks"

        # 计算multiplier以保持QNONCOLL_LEN固定为64
        qnoncoll_multiplier=$((FIXED_QNONCOLL_LEN / num_oocds))

        for SCENE in G1 G2 G3 G4 G5; do
            DATA_FOLDER="$BASE_DATA_FOLDER/$SCENE"
            echo -n "  场景 $SCENE: "

            OUTPUT=$(python3 multi_copu_real_data_simulation.py \
                "$BASENAME" \
                "$BENCHID" \
                "$DATA_FOLDER" \
                "$num_copus" \
                "$THRESHOLD" \
                "$num_oocds" \
                "$SAMPLE_RATE" \
                "$num_pred" \
                --copus-per-edge "$COPUS_PER_EDGE" \
                --cht-type "$BASE_CHT_TYPE" \
                --num-banks "$num_banks" \
                --collision-type "$COLLISION_TYPE" \
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

            echo "✓ (cycles=$TOTAL_CYCLES, throughput=$THROUGHPUT, utilization=$UTILIZATION, avg_wait=$AVG_WAIT_CYCLES, dead_avg_ratio=${DEAD_AVG_RATIO}%)"
            echo "$sweep_name,$value,$SCENE,$TOTAL_CYCLES,$TOTAL_QUERIES,$THROUGHPUT,$UTILIZATION,$CONFLICTS,$AVG_WAIT_CYCLES,$DEAD_AVG_RATIO" >> "$csv_file"
        done
    done
}

run_sweep "NUM_OOCDS" "$RESULT_DIR/sweep_num_oocds_sphere_results.csv" "SWEEP_NUM_OOCDS"
run_sweep "NUM_COPUS" "$RESULT_DIR/sweep_num_copus_sphere_results.csv" "SWEEP_NUM_COPUS"
run_sweep "NUM_PRED" "$RESULT_DIR/sweep_num_pred_sphere_results.csv" "SWEEP_NUM_PRED"
run_sweep "NUM_BANKS" "$RESULT_DIR/sweep_num_banks_sphere_results.csv" "SWEEP_NUM_BANKS"

echo ""
echo "=========================================="
echo "仿真完成！"
echo "结果已保存至:"
echo "  - $RESULT_DIR/sweep_num_oocds_sphere_results.csv"
echo "  - $RESULT_DIR/sweep_num_copus_sphere_results.csv"
echo "  - $RESULT_DIR/sweep_num_pred_sphere_results.csv"
echo "  - $RESULT_DIR/sweep_num_banks_sphere_results.csv"
echo "=========================================="

