#!/bin/bash

# 脚本功能：生成 Figure 2 所需的数据
# 对比 Shared Dual-Port SRAM (有冲突) 与 Conflict-Free SRAM (无冲突) 的性能
# 遍历 G1-G5 场景
# 配置 1: Shared Dual-Port (Default)
# 配置 2: Conflict-Free (--no-cht-conflict)

set -e  # 遇到错误立即退出

# === 配置参数 ===
BASENAME="iiwa_7"
BENCHID="1-10"
BASE_DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
NUM_COPUS=16
THRESHOLD=0.5
COPUS_PER_EDGE=1
NUM_OOCDS=8
SAMPLE_RATE=0.1
NUM_PRED=2  # 显式指定 pred=2
CHT_TYPE="dual_port"
NUM_BANKS=8

RESULT_DIR="../result_files"
mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "开始执行冲突分析仿真 (Figure 2 Data Generation)..."
echo "对比: Shared Dual-Port vs Conflict-Free (link + sphere)"
echo "COPU: $NUM_COPUS, Pred: $NUM_PRED"
echo "CHT_TYPE: $CHT_TYPE"
echo "NUM_BANKS: $NUM_BANKS"
echo "=========================================="

for COLLISION_TYPE in link sphere; do
    echo "------------------------------------------"
    echo "处理碰撞模型: $COLLISION_TYPE"
    echo "------------------------------------------"

    FILE_SHARED="$RESULT_DIR/shared_dual_port_${COLLISION_TYPE}_results.csv"
    FILE_NO_CONFLICT="$RESULT_DIR/no_conflict_${COLLISION_TYPE}_results.csv"

    # 初始化 CSV
    echo "Scene,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts" > "$FILE_SHARED"
    echo "Scene,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts" > "$FILE_NO_CONFLICT"

    for SCENE in G1 G2 G3 G4 G5; do
        DATA_FOLDER="$BASE_DATA_FOLDER/$SCENE"
        echo "正在处理场景: $SCENE ($COLLISION_TYPE)"

        # --- 1. Shared Dual-Port (Default) ---
        echo -n "  [Shared] "
        # 调用 python 脚本，注意参数顺序需与 multi_copu_real_data_simulation.py 一致
        OUTPUT_SHARED=$(python3 multi_copu_real_data_simulation.py \
            "$BASENAME" "$BENCHID" "$DATA_FOLDER" "$NUM_COPUS" \
            "$THRESHOLD" "$NUM_OOCDS" "$SAMPLE_RATE" "$NUM_PRED" \
            --copus-per-edge "$COPUS_PER_EDGE" \
            --cht-type "$CHT_TYPE" \
            --num-banks "$NUM_BANKS" \
            --collision-type "$COLLISION_TYPE" 2>&1)
    
        if [ $? -ne 0 ]; then 
            echo "Failed"
            echo "$OUTPUT_SHARED"
            exit 1
        fi

        # 提取数据
        CYCLES=$(echo "$OUTPUT_SHARED" | grep "总周期:" | tail -n 1 | awk '{print $2}')
        QUERIES=$(echo "$OUTPUT_SHARED" | grep "总查询数:" | tail -n 1 | awk '{print $2}')
        THROUGHPUT=$(echo "$OUTPUT_SHARED" | grep "系统吞吐量:" | tail -n 1 | awk '{print $2}')
        UTIL=$(echo "$OUTPUT_SHARED" | grep "平均COPU占用率:" | tail -n 1 | awk '{print $2}')
        CONFLICTS=$(echo "$OUTPUT_SHARED" | grep "CHT冲突数:" | tail -n 1 | awk '{print $2}')
    
        echo "Cycles: $CYCLES, Conflicts: $CONFLICTS"
        echo "$SCENE,$CYCLES,$QUERIES,$THROUGHPUT,$UTIL,$CONFLICTS" >> "$FILE_SHARED"

        # --- 2. Conflict-Free (--no-cht-conflict) ---
        echo -n "  [No-Conflict] "
        OUTPUT_NC=$(python3 multi_copu_real_data_simulation.py \
            "$BASENAME" "$BENCHID" "$DATA_FOLDER" "$NUM_COPUS" \
            "$THRESHOLD" "$NUM_OOCDS" "$SAMPLE_RATE" "$NUM_PRED" \
            --copus-per-edge "$COPUS_PER_EDGE" \
            --cht-type "$CHT_TYPE" \
            --num-banks "$NUM_BANKS" \
            --collision-type "$COLLISION_TYPE" \
            --no-cht-conflict 2>&1)

        if [ $? -ne 0 ]; then 
            echo "Failed"
            echo "$OUTPUT_NC"
            exit 1
        fi

        CYCLES=$(echo "$OUTPUT_NC" | grep "总周期:" | tail -n 1 | awk '{print $2}')
        QUERIES=$(echo "$OUTPUT_NC" | grep "总查询数:" | tail -n 1 | awk '{print $2}')
        THROUGHPUT=$(echo "$OUTPUT_NC" | grep "系统吞吐量:" | tail -n 1 | awk '{print $2}')
        UTIL=$(echo "$OUTPUT_NC" | grep "平均COPU占用率:" | tail -n 1 | awk '{print $2}')
        CONFLICTS=$(echo "$OUTPUT_NC" | grep "CHT冲突数:" | tail -n 1 | awk '{print $2}')

        echo "Cycles: $CYCLES"
        echo "$SCENE,$CYCLES,$QUERIES,$THROUGHPUT,$UTIL,$CONFLICTS" >> "$FILE_NO_CONFLICT"
    done

    echo "输出文件: $FILE_SHARED"
    echo "输出文件: $FILE_NO_CONFLICT"
done

echo "=========================================="
echo "仿真完成！"
echo "Shared Results: $RESULT_DIR/shared_dual_port_link_results.csv"
echo "Shared Results: $RESULT_DIR/shared_dual_port_sphere_results.csv"
echo "No-Conflict Results: $RESULT_DIR/no_conflict_link_results.csv"
echo "No-Conflict Results: $RESULT_DIR/no_conflict_sphere_results.csv"
echo "=========================================="
