#!/bin/bash

# 脚本功能：运行多COPU实际数据仿真，遍历G1-G5场景，分别测试dual_port和multi_bank
# 使用方法：./run_multi_copu_sim.sh

set -e  # 遇到错误立即退出

# === 配置参数 ===
BASENAME="iiwa_7"
BENCHID="1-50"
# 基础数据路径
BASE_DATA_FOLDER="../trace_files/scene_benchmarks/bit_collision_data"
NUM_COPUS=8
THRESHOLD=1.0
COPUS_PER_EDGE=1  # 每个Edge分配的COPU数量
NUM_OOCDS=7
SAMPLE_RATE=0.1

# 结果文件路径
RESULT_DIR="result_files"
DUAL_PORT_CSV="$RESULT_DIR/dual_port_results.csv"
MULTI_BANK_CSV="$RESULT_DIR/multi_bank_results.csv"

# 创建结果目录（如果不存在）
mkdir -p "$RESULT_DIR"

# 初始化CSV文件头部
echo "Scene,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts" > "$DUAL_PORT_CSV"
echo "Scene,Num_Banks,Total_Cycles,Total_Queries,Throughput,Utilization,Conflicts" > "$MULTI_BANK_CSV"

echo "=========================================="
echo "开始执行多COPU仿真遍历..."
echo "数据集: $BASENAME, Benchmark范围: $BENCHID"
echo "总COPU数: $NUM_COPUS, 每Edge分配: $COPUS_PER_EDGE"
echo "OOCD数: $NUM_OOCDS, 采样率: $SAMPLE_RATE"
echo "=========================================="

# 遍历场景 G1-G5
for SCENE in G1 G2 G3 G4 G5; do
    DATA_FOLDER="$BASE_DATA_FOLDER/$SCENE"
    echo "正在处理场景: $SCENE"

    # 遍历 CHT 类型
    for CHT_TYPE in dual_port multi_bank; do
        if [ "$CHT_TYPE" == "multi_bank" ]; then
            NUM_BANKS=8
            echo "  - CHT类型: $CHT_TYPE, Bank数: $NUM_BANKS"
        else
            NUM_BANKS=8 # 对于dual_port，此参数无效，但为了命令一致性保留
            echo "  - CHT类型: $CHT_TYPE"
        fi

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
            --copus-per-edge "$COPUS_PER_EDGE" \
            --cht-type "$CHT_TYPE" \
            --num-banks "$NUM_BANKS" 2>&1)
        
        # 检查 python 脚本是否执行成功
        if [ $? -ne 0 ]; then
            echo "Error running simulation for $SCENE with $CHT_TYPE"
            echo "$OUTPUT"
            exit 1
        fi

        # 解析输出
        # 注意：grep 可能会匹配到多行（如果有调试输出），这里取最后一行匹配的
        TOTAL_CYCLES=$(echo "$OUTPUT" | grep "总周期:" | tail -n 1 | awk -F': ' '{print $2}')
        TOTAL_QUERIES=$(echo "$OUTPUT" | grep "总查询数:" | tail -n 1 | awk -F': ' '{print $2}')
        THROUGHPUT=$(echo "$OUTPUT" | grep "系统吞吐量:" | tail -n 1 | awk -F': ' '{print $2}' | awk '{print $1}')
        UTILIZATION=$(echo "$OUTPUT" | grep "平均COPU占用率:" | tail -n 1 | awk -F': ' '{print $2}')
        CONFLICTS=$(echo "$OUTPUT" | grep "CHT冲突数:" | tail -n 1 | awk -F': ' '{print $2}')

        # 写入CSV
        if [ "$CHT_TYPE" == "dual_port" ]; then
            echo "$SCENE,$TOTAL_CYCLES,$TOTAL_QUERIES,$THROUGHPUT,$UTILIZATION,$CONFLICTS" >> "$DUAL_PORT_CSV"
        else
            echo "$SCENE,$NUM_BANKS,$TOTAL_CYCLES,$TOTAL_QUERIES,$THROUGHPUT,$UTILIZATION,$CONFLICTS" >> "$MULTI_BANK_CSV"
        fi
    done
done

echo "=========================================="
echo "仿真完成！"
echo "结果已保存至:"
echo "  - $DUAL_PORT_CSV"
echo "  - $MULTI_BANK_CSV"
echo "=========================================="

