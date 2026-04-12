#!/bin/bash

# 脚本功能：运行多COPU实际数据仿真，遍历G1-G5场景，分别测试dual_port和multi_bank
# 使用方法：./run_multi_copu_sim.sh

set -e  # 遇到错误立即退出

# === 配置参数 ===
THRESHOLD=1
SAMPLE_RATE=0.125
LINK_QNONCOLL_MULTIPLIER=8
NUM_COPUS=8
NUM_OOCDS=8
# 基础数据路径
BASE_DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
BASENAME="iiwa_7"
BENCHID="1-10"
COPUS_PER_EDGE=1  # 每个Edge分配的COPU数量
NUM_BANKS=8

# 结果文件路径
RESULT_DIR="../result_files"

# 创建结果目录（如果不存在）
mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "开始执行多COPU仿真遍历..."
echo "数据集: $BASENAME, Benchmark范围: $BENCHID"
echo "总COPU数: $NUM_COPUS, 每Edge分配: $COPUS_PER_EDGE"
echo "OOCD数: $NUM_OOCDS, 采样率: $SAMPLE_RATE"
echo "Collision类型: link + sphere"
echo "QNONCOLL multiplier (link): $LINK_QNONCOLL_MULTIPLIER"
echo "QNONCOLL multiplier (sphere): $((LINK_QNONCOLL_MULTIPLIER * 4))"
echo "=========================================="

# 遍历 Collision 类型、CHT 类型和 Prediction 配置
for COLLISION_TYPE in link sphere; do
    if [ "$COLLISION_TYPE" == "sphere" ]; then
        QNONCOLL_MULTIPLIER=$((LINK_QNONCOLL_MULTIPLIER * 4))
    else
        QNONCOLL_MULTIPLIER=$LINK_QNONCOLL_MULTIPLIER
    fi

    echo ""
    echo "=========================================="
    echo "Collision类型: $COLLISION_TYPE (QNONCOLL multiplier=$QNONCOLL_MULTIPLIER)"
    echo "=========================================="

for CHT_TYPE in dual_port multi_bank; do
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
    for NUM_PRED in 1 2; do
        echo ""
        echo "  Prediction缓冲数: $NUM_PRED"
        echo "  ------------------------------------------"

        # 在进入场景循环前，预先确定 CSV 文件并写入表头（每个 CHT_TYPE + NUM_PRED 只写一次）
        if [ "$CHT_TYPE" == "dual_port" ]; then
            CSV_FILE="$RESULT_DIR/dual_port_pred${NUM_PRED}_${COLLISION_TYPE}_results.csv"
            echo "Scene,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts" > "$CSV_FILE"
        else
            CSV_FILE="$RESULT_DIR/multi_bank_pred${NUM_PRED}_${COLLISION_TYPE}_results.csv"
            echo "Scene,Num_Banks,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts" > "$CSV_FILE"
        fi

        # 遍历场景 G1-G5
        for SCENE in G1 G2 G3 G4 G5; do
            DATA_FOLDER="$BASE_DATA_FOLDER/$SCENE"
            echo -n "    场景 $SCENE: "

            # 输出执行的命令
            # echo "python3 multi_copu_real_data_simulation.py $BASENAME $BENCHID $DATA_FOLDER $NUM_COPUS $THRESHOLD $NUM_OOCDS $SAMPLE_RATE $NUM_PRED --copus-per-edge $COPUS_PER_EDGE --cht-type $CHT_TYPE --num-banks $NUM_BANKS"

            # 运行仿真并将输出捕获到变量
            # 使用 2>&1 将 stderr 也捕获，防止 python 报错漏掉
            OUTPUT=$(python3 multi_copu_real_data_simulation.py \
                "$BASENAME" \
                "$BENCHID" \
                "$DATA_FOLDER" \
                "$NUM_COPUS" \
                "$THRESHOLD" \
                "$NUM_OOCDS" \
                "$SAMPLE_RATE" \
                "$NUM_PRED" \
                --copus-per-edge "$COPUS_PER_EDGE" \
                --cht-type "$CHT_TYPE" \
                --num-banks "$NUM_BANKS" \
                --collision-type "$COLLISION_TYPE" \
                --qnoncoll-multiplier "$QNONCOLL_MULTIPLIER" 2>&1)

            # 检查 python 脚本是否执行成功
            if [ $? -ne 0 ]; then
                echo "✗ 失败"
                echo "Error details: $OUTPUT"
                exit 1
            fi

            # 解析输出（取最后一行匹配以避免干扰的调试输出）
            TOTAL_CYCLES=$(echo "$OUTPUT" | grep "总周期:" | tail -n 1 | awk -F': ' '{print $2}')
            TOTAL_QUERIES=$(echo "$OUTPUT" | grep "总查询数:" | tail -n 1 | awk -F': ' '{print $2}')
            THROUGHPUT=$(echo "$OUTPUT" | grep "系统吞吐量:" | tail -n 1 | awk -F': ' '{print $2}' | awk '{print $1}')
            UTILIZATION=$(echo "$OUTPUT" | grep "平均COPU占用率:" | tail -n 1 | awk -F': ' '{print $2}')
            CONFLICTS=$(echo "$OUTPUT" | grep "CHT冲突数:" | tail -n 1 | awk -F': ' '{print $2}')

            echo "✓ (cycles=$TOTAL_CYCLES, throughput=$THROUGHPUT, utilization=$UTILIZATION)"

            # 追加写入 CSV（头部已在循环外写入）
            if [ "$CHT_TYPE" == "dual_port" ]; then
                echo "$SCENE,$TOTAL_CYCLES,$TOTAL_QUERIES,$THROUGHPUT,$UTILIZATION,$CONFLICTS" >> "$CSV_FILE"
            else
                echo "$SCENE,$NUM_BANKS,$TOTAL_CYCLES,$TOTAL_QUERIES,$THROUGHPUT,$UTILIZATION,$CONFLICTS" >> "$CSV_FILE"
            fi
        done
    done
done
done

echo ""
echo "=========================================="
echo "仿真完成！"
echo "结果已保存至:"
echo "  - $RESULT_DIR/dual_port_pred1_link_results.csv     (Dual Port, Prediction=1, link)"
echo "  - $RESULT_DIR/dual_port_pred2_link_results.csv     (Dual Port, Prediction=2, link)"
echo "  - $RESULT_DIR/multi_bank_pred1_link_results.csv    (Multi Bank, Prediction=1, link)"
echo "  - $RESULT_DIR/multi_bank_pred2_link_results.csv    (Multi Bank, Prediction=2, link)"
echo "  - $RESULT_DIR/dual_port_pred1_sphere_results.csv   (Dual Port, Prediction=1, sphere)"
echo "  - $RESULT_DIR/dual_port_pred2_sphere_results.csv   (Dual Port, Prediction=2, sphere)"
echo "  - $RESULT_DIR/multi_bank_pred1_sphere_results.csv  (Multi Bank, Prediction=1, sphere)"
echo "  - $RESULT_DIR/multi_bank_pred2_sphere_results.csv  (Multi Bank, Prediction=2, sphere)"
echo "=========================================="

