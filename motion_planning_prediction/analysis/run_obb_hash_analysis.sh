#!/bin/bash
#
# OBB哈希编码分析 - 快速运行脚本
#
# 用途: 快速执行OBB哈希编码变化规律分析，并查看结果
# 用法: bash run_obb_hash_analysis.sh [benchmark_id]
#      如果不提供benchmark_id，默认使用46

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASENAME="iiwa_7"
BENCHID="${1:-46}"
DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"

cd "$SCRIPT_DIR"

echo "=================================================================================="
echo "OBB哈希编码变化规律分析"
echo "=================================================================================="
echo ""
echo "参数配置："
echo "  数据集: $BASENAME"
echo "  Benchmark: $BENCHID"
echo "  脚本位置: $SCRIPT_DIR"
echo ""

# 运行分析脚本
python analyze_obb_hash_patterns.py "$BASENAME" "$BENCHID" "$DATA_FOLDER"

# 检查结果
RESULT_DIR="result_files/obb_hash_analysis"
if [ -f "$RESULT_DIR/hash_analysis_report.txt" ]; then
    echo ""
    echo "=================================================================================="
    echo "分析结果摘要"
    echo "=================================================================================="
    echo ""
    head -30 "$RESULT_DIR/hash_analysis_report.txt"
    echo ""
    echo "... (更多内容请查看报告文件)"
    echo ""
    echo "=================================================================================="
    echo "输出文件列表"
    echo "=================================================================================="
    ls -lh "$RESULT_DIR"
    echo ""
    echo "关键文件："
    echo "  [1] 文本报告      : $RESULT_DIR/hash_analysis_report.txt"
    echo "  [2] 统计数据      : $RESULT_DIR/hash_statistics.csv"
    echo "  [3] 维度差异图    : $RESULT_DIR/dimension_diff_frequency.pdf"
    echo "  [4] Bit位热力图   : $RESULT_DIR/bit_diff_heatmap.pdf"
    echo "  [5] 差异类型图    : $RESULT_DIR/difference_types_distribution.pdf"
else
    echo "✗ 分析失败，未生成结果文件"
    exit 1
fi

echo ""
echo "=================================================================================="
echo "分析完成！"
echo "=================================================================================="
