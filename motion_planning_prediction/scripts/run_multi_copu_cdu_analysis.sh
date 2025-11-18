#!/bin/bash
#
# 多COPU-CDU组合性能分析脚本
# 统计不同COPU数量(1,2,4,8)和CDU数量(1,2,4,6,8)下的仿真结果
# 将结果输出到CSV文件供后续分析
#

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置参数
BASENAME="iiwa_7"
BENCHID="1-10"
DATA_FOLDER="$SCRIPT_DIR/../../trace_files/scene_benchmarks/bit_collision_data"
THRESHOLD="1.0"
SAMPLE_RATE="0.1"
MAX_CYCLES="10000"
OUTPUT_DIR="result_files"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# CSV输出文件
CSV_FILE="$OUTPUT_DIR/multi_copu_cdu_analysis.csv"

# 初始化CSV文件（写入表头）
echo "COPU_Num,CDU_Num,Total_Edges,Total_Cycles,Total_Queries,System_Throughput,Avg_COPU_Utilization,CHT_Conflicts" > "$CSV_FILE"

echo "=========================================="
echo "多COPU-CDU组合性能分析"
echo "=========================================="
echo "数据集: $BASENAME"
echo "Benchmark: $BENCHID"
echo "数据文件夹: $DATA_FOLDER"
echo "采样率: $SAMPLE_RATE"
echo "最大周期: $MAX_CYCLES"
echo "结果输出: $CSV_FILE"
echo "工作目录: $(pwd)"
echo "=========================================="

# COPU数量
COPU_NUMS=(1 2 4 8)

# CDU数量
CDU_NUMS=(1 2 4 6 8)

# 循环遍历所有组合
for num_copus in "${COPU_NUMS[@]}"; do
    for num_cdus in "${CDU_NUMS[@]}"; do
        echo ""
        echo "【测试】COPU=$num_copus, CDU=$num_cdus"
        
        # 运行仿真并捕获输出，传递CDU参数
        if output=$(python ../multi_copu_real_data_simulation.py \
            "$BASENAME" "$BENCHID" "$DATA_FOLDER" "$num_copus" \
            "$THRESHOLD" "$num_cdus" "$SAMPLE_RATE" "$MAX_CYCLES" 2>&1); then
            
            # 从输出中提取关键指标
            total_edges=$(echo "$output" | grep "总Edge数:" | awk '{print $NF}')
            total_cycles=$(echo "$output" | grep "总周期:" | awk '{print $NF}')
            total_queries=$(echo "$output" | grep "总查询数:" | awk '{print $NF}')
            throughput=$(echo "$output" | grep "系统吞吐量:" | awk '{print $2}')
            copu_util=$(echo "$output" | grep "平均COPU占用率:" | awk '{print $NF}' | sed 's/%//')
            cht_conflicts=$(echo "$output" | grep "CHT冲突数:" | awk '{print $NF}')
            
            # 安全检查：如果提取失败，设置默认值
            total_edges=${total_edges:-"N/A"}
            total_cycles=${total_cycles:-"N/A"}
            total_queries=${total_queries:-"N/A"}
            throughput=${throughput:-"N/A"}
            copu_util=${copu_util:-"N/A"}
            cht_conflicts=${cht_conflicts:-"N/A"}
            
            # 输出当前结果
            echo "  总Edge数: $total_edges"
            echo "  总周期: $total_cycles"
            echo "  总查询数: $total_queries"
            echo "  系统吞吐量: $throughput queries/cycle"
            echo "  平均COPU占用率: $copu_util%"
            echo "  CHT冲突数: $cht_conflicts"
            
            # 追加到CSV文件
            echo "$num_copus,$num_cdus,$total_edges,$total_cycles,$total_queries,$throughput,$copu_util,$cht_conflicts" >> "$CSV_FILE"
        else
            echo "  ✗ 仿真失败"
            echo "$num_copus,$num_cdus,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$CSV_FILE"
        fi
    done
done

echo ""
echo "=========================================="
echo "分析完成！结果已保存到: $CSV_FILE"
echo "=========================================="
echo ""
echo "结果预览："
column -t -s ',' "$CSV_FILE" 2>/dev/null || cat "$CSV_FILE"
echo ""
echo "可使用以下命令查看完整CSV文件："
echo "  cat $CSV_FILE"
