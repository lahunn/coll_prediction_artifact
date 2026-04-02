#!/bin/bash
# 准确率分析实验运行脚本
# 此脚本应在项目根目录下运行，或者直接执行：bash motion_planning_prediction/scripts/run_accuracy_analysis.sh

# 设置基本路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"
MPP_DIR="$PROJECT_ROOT/motion_planning_prediction"

# 进入运动规划预测目录
cd "$MPP_DIR" || exit

# 确保 Python 路径包含项目根目录，以便导入 trace_generation 等模块
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT:$MPP_DIR

# 参数配置
THRESHOLDS=(0.1)
SAMPLE_RATES=(1)
QNONCOLL_MULTIPLIERS=(4)
# 数据文件夹路径（相对于 MPP_DIR）
DATA_FOLDER="../trace_files/scene_benchmarks/bit_collision_data/G3"
BASENAME="iiwa_7"
NUM_BENCHMARKS=50
START_BENCHID=1
ROBOT_NAME="iiwa"
COLLISION_TYPE="sphere"

# 创建结果和图表目录
mkdir -p result_files
mkdir -p plots

echo "=== 启动批量准确率分析实验 ==="
echo "配置数量: $(( ${#THRESHOLDS[@]} * ${#SAMPLE_RATES[@]} * ${#QNONCOLL_MULTIPLIERS[@]} ))"
echo "基准测试数量: $NUM_BENCHMARKS (起始 ID: $START_BENCHID)"
echo "数据目录: $DATA_FOLDER"
echo "================================="

# 清理旧的结果文件（可选）
rm -f result_files/sphere_results.csv
rm -f result_files/sphere_accuracy_curve.csv

# 计数器
total_configs=$(( ${#THRESHOLDS[@]} * ${#SAMPLE_RATES[@]} * ${#QNONCOLL_MULTIPLIERS[@]} ))
current_config=0

# 运行所有配置组合
for threshold in "${THRESHOLDS[@]}"; do
    for sample_rate in "${SAMPLE_RATES[@]}"; do
        for qnoncoll_multiplier in "${QNONCOLL_MULTIPLIERS[@]}"; do
            current_config=$((current_config + 1))
            
            echo ""
            echo "[$current_config/$total_configs] 正在运行配置: 阈值=$threshold, 采样率=$sample_rate, 队列倍数=$qnoncoll_multiplier"
            
            # 运行仿真程序
            python3 strategy_evaluation/prediction_simulation_nDOF_accuracy_tracking.py \
                $threshold \
                $sample_rate \
                $qnoncoll_multiplier \
                $DATA_FOLDER \
                $BASENAME \
                $NUM_BENCHMARKS \
                $ROBOT_NAME \
                $COLLISION_TYPE \
                $START_BENCHID
            
            if [ $? -eq 0 ]; then
                echo "  ✓ 配置运行成功"
            else
                echo "  ✗ 配置运行失败"
            fi
        done
    done
done

echo ""
echo "=== 仿真阶段完成 ==="
echo "正在生成可视化图表..."

# 生成详细的可视化
# 修正了绘图脚本的路径
ACCURACY_DATA="result_files/sphere_accuracy_curve.csv"

if [ -f "$ACCURACY_DATA" ]; then
    python3 plots/plot_accuracy_learning_curve.py \
        "$ACCURACY_DATA" \
        --output plots/detailed_learning_curves.pdf \
        --mode aggregated
    
    if [ $? -eq 0 ]; then
        echo "✓ 可视化图表生成成功，已保存至: plots/detailed_learning_curves.pdf"
    else
        echo "✗ 可视化脚本执行失败"
    fi
else
    echo "✗ 错误: 未找到准确率数据文件 $ACCURACY_DATA"
fi

echo ""
echo "=== 实验全部完成 ==="
echo "结果文件:"
echo "  - 仿真统计: result_files/sphere_results.csv"
echo "  - 准确率曲线: result_files/sphere_accuracy_curve.csv"
echo "  - 分析图表: plots/detailed_learning_curves.pdf"
echo ""
