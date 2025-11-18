#!/bin/bash

# qnoncoll_multiplier参数分析脚本
# 分析队列长度倍数与预测周期总数、查询减少率的关系

echo "=== qnoncoll_multiplier参数分析 ==="
echo "分析队列长度倍数对预测性能的影响"
echo

# 参数设置
THRESHOLD=0.5
SAMPLE_RATE=0.1
DATA_FOLDER="../../trace_files/scene_benchmarks/bit_collision_data"
BASENAME="iiwa_7"
NUM_BENCHMARKS=10  # 使用较少的基准测试以加快分析
ROBOT_NAME="iiwa"

# 要测试的qnoncoll_multiplier值
QNONCOLL_VALUES=(1 2 4 6 8 10)

echo "分析参数:"
echo "  阈值: $THRESHOLD"
echo "  采样率: $SAMPLE_RATE"
echo "  数据文件夹: $DATA_FOLDER"
echo "  基准测试: $BASENAME (前$NUM_BENCHMARKS个)"
echo "  机器人: $ROBOT_NAME"
echo "  测试的队列倍数: ${QNONCOLL_VALUES[*]}"
echo

# 创建结果目录
mkdir -p result_files

# 清理旧结果文件
RESULT_FILE="result_files/qnoncoll_analysis.csv"
rm -f $RESULT_FILE

# 添加CSV头部
echo "qnoncoll_multiplier,total_sphere_checks,prediction_queries,oracle_queries,prediction_cycles,reduction_rate" > $RESULT_FILE

echo "开始参数分析..."
echo "进度:"

# 对每个qnoncoll_multiplier值运行仿真
for multiplier in "${QNONCOLL_VALUES[@]}"; do
    echo "  测试 qnoncoll_multiplier = $multiplier..."

    # 运行仿真并捕获输出
    output=$(python3 ../prediction_simulation_nDOF_sphere.py \
        $THRESHOLD \
        $SAMPLE_RATE \
        $multiplier \
        $DATA_FOLDER \
        $BASENAME \
        $NUM_BENCHMARKS \
        $ROBOT_NAME 2>/dev/null)

    # 从输出中提取关键指标
    total_checks=$(echo "$output" | grep "实际查询总数" | sed 's/.*: \([0-9]*\)/\1/')
    pred_queries=$(echo "$output" | grep "预测查询总数" | sed 's/.*: \([0-9.]*\)/\1/')
    oracle_queries=$(echo "$output" | grep "Oracle查询总数" | sed 's/.*: \([0-9]*\)/\1/')
    pred_cycles=$(echo "$output" | grep "预测周期总数" | sed 's/.*: \([0-9]*\)/\1/')
    reduction_rate=$(echo "$output" | grep "查询减少率" | sed 's/.*: \([0-9.-]*\)%/\1/')

    # 保存到CSV文件
    echo "$multiplier,$total_checks,$pred_queries,$oracle_queries,$pred_cycles,$reduction_rate" >> $RESULT_FILE

    echo "    完成 - 减少率: ${reduction_rate}%"
done

echo
echo "参数分析完成!"
echo "结果已保存到: $RESULT_FILE"
echo

# 显示结果摘要
echo "=== 结果摘要 ==="
echo "qnoncoll_multiplier | 预测周期 | 查询减少率"
echo "-------------------|-----------|------------"
tail -n +2 $RESULT_FILE | while IFS=',' read -r mult total pred oracle cycles reduction; do
    printf "%-18s | %-9s | %-10s%%\n" "$mult" "$cycles" "$reduction"
done

echo
echo "=== 分析建议 ==="
echo "1. 查看 $RESULT_FILE 获取详细数据"
echo "2. 运行以下命令生成分析图表："
echo "   python3 -c \""
echo "   import pandas as pd"
echo "   import matplotlib.pyplot as plt"
echo "   df = pd.read_csv('$RESULT_FILE')"
echo "   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))"
echo "   ax1.plot(df['qnoncoll_multiplier'], df['prediction_cycles'], 'o-', label='Prediction Cycles')"
echo "   ax1.set_xlabel('qnoncoll_multiplier'); ax1.set_ylabel('Prediction Cycles'); ax1.set_title('Cycles vs qnoncoll_multiplier'); ax1.grid(True)"
echo "   ax2.plot(df['qnoncoll_multiplier'], df['reduction_rate'], 's-', color='red', label='Reduction Rate')"
echo "   ax2.set_xlabel('qnoncoll_multiplier'); ax2.set_ylabel('Reduction Rate (%)'); ax2.set_title('Reduction Rate vs qnoncoll_multiplier'); ax2.grid(True)"
echo "   plt.tight_layout(); plt.savefig('qnoncoll_analysis.png', dpi=150); plt.show()"
echo "   \""

echo
echo "分析完成！"