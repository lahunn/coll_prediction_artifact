#!/bin/bash
# 球体碰撞检测仿真测试脚本（支持几何和PyBullet检测器）

# 默认参数
ROBOT_NAME="iiwa"
BASE_NAME="iiwa_7"
NUM_TESTS=50
THRESHOLD=0.5
SAMPLE_RATE=0.1
QNONCOLL_MULTIPLIER=8

# 数据文件夹
DATA_FOLDER="../trace_files/scene_benchmarks/bit_collision_data"

# 检测器类型 (pybullet 或 geometric)
DETECTOR_TYPE="geometric"

# 使用说明
usage() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -r, --robot ROBOT_NAME        机器人名称 (默认: iiwa)"
    echo "  -b, --basename BASE_NAME       基准名称 (默认: iiwa_7)"
    echo "  -n, --num-tests NUM_TESTS      测试数量 (默认: 50)"
    echo "  -d, --data-folder DATA_FOLDER  数据文件夹 (默认: ../trace_files/scene_benchmarks/bit_collision_data)"
    echo "  -t, --threshold THRESHOLD      碰撞阈值 (默认: 0.5)"
    echo "  -s, --sample-rate SAMPLE_RATE  更新率 (默认: 0.1)"
    echo "  -q, --qnoncoll MULTIPLIER      非碰撞队列长度比 (默认: 8)"
    echo "  -D, --detector TYPE            检测器类型: pybullet 或 geometric (默认: geometric)"
    echo "  -h, --help                     显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -r iiwa -b iiwa_7 -n 100"
    echo "  $0 -D geometric -n 50"
    echo "  $0 -D pybullet -n 50"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--robot)
            ROBOT_NAME="$2"
            shift 2
            ;;
        -b|--basename)
            BASE_NAME="$2"
            shift 2
            ;;
        -n|--num-tests)
            NUM_TESTS="$2"
            shift 2
            ;;
        -d|--data-folder)
            DATA_FOLDER="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -s|--sample-rate)
            SAMPLE_RATE="$2"
            shift 2
            ;;
        -q|--qnoncoll)
            QNONCOLL_MULTIPLIER="$2"
            shift 2
            ;;
        -D|--detector)
            DETECTOR_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            usage
            exit 1
            ;;
    esac
done

echo "========================================"
echo "球体碰撞检测仿真测试"
echo "========================================"
echo "检测器类型: $DETECTOR_TYPE"
echo "机器人: $ROBOT_NAME"
echo "基准名称: $BASE_NAME"
echo "测试数量: $NUM_TESTS"
echo "碰撞阈值: $THRESHOLD"
echo "更新率: $SAMPLE_RATE"
echo "非碰撞队列比: $QNONCOLL_MULTIPLIER"
echo "数据文件夹: $DATA_FOLDER"
echo "========================================"

# 检查数据文件夹
echo -e "\n[1/3] 检查数据文件夹..."
if [ ! -d "$DATA_FOLDER" ]; then
    echo "警告: 数据文件夹不存在: $DATA_FOLDER"
    echo "创建文件夹..."
    mkdir -p "$DATA_FOLDER"
fi

# 检查是否有数据文件
echo -e "\n[2/3] 检查数据文件..."
if [ "$DETECTOR_TYPE" = "geometric" ]; then
    # 优先查找带周期信息的文件
    CYCLE_FILES=$(ls $DATA_FOLDER/${BASE_NAME}_*_sphere_geometric_cycles.pkl 2>/dev/null | wc -l)
    GEOM_FILES=$(ls $DATA_FOLDER/${BASE_NAME}_*_sphere_geometric.pkl 2>/dev/null | wc -l)
    SPHERE_FILES=$(ls $DATA_FOLDER/${BASE_NAME}_*_sphere.pkl 2>/dev/null | wc -l)
    
    echo "找到 $CYCLE_FILES 个几何检测器数据文件（带周期信息）"
    echo "找到 $GEOM_FILES 个几何检测器数据文件"
    echo "找到 $SPHERE_FILES 个通用球体数据文件"
    
    TOTAL_FILES=$((CYCLE_FILES + GEOM_FILES + SPHERE_FILES))
    
    if [ $TOTAL_FILES -eq 0 ]; then
        echo ""
        echo "错误: 未找到几何检测器的数据文件！"
        echo "请先运行以下命令生成数据:"
        echo "  cd ../trace_generation/scripts"
        echo "  bash generate_sphere_data.sh --detector-type geometric --return-cycles"
        exit 1
    fi
    
    if [ $CYCLE_FILES -gt 0 ]; then
        echo "✓ 将使用带周期统计信息的数据文件"
    fi
elif [ "$DETECTOR_TYPE" = "pybullet" ]; then
    SPHERE_FILES=$(ls $DATA_FOLDER/${BASE_NAME}_*_sphere.pkl 2>/dev/null | wc -l)
    
    echo "找到 $SPHERE_FILES 个PyBullet球体数据文件"
    
    if [ $SPHERE_FILES -eq 0 ]; then
        echo ""
        echo "错误: 未找到PyBullet的数据文件！"
        echo "请先运行以下命令生成数据:"
        echo "  cd ../trace_generation/scripts"
        echo "  bash generate_sphere_data.sh --detector-type pybullet"
        exit 1
    fi
else
    echo "错误: 未知的检测器类型: $DETECTOR_TYPE"
    echo "支持的类型: pybullet, geometric"
    exit 1
fi

# 运行球体仿真测试
echo -e "\n[3/3] 运行球体碰撞检测仿真..."
echo "测试前 $NUM_TESTS 个基准..."
echo ""

# 保存当前目录
CURRENT_DIR=$(pwd)

# 切换到motion_planning_prediction目录（如果不在的话）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "执行命令:"
echo "python3 prediction_simulation_nDOF_sphere.py $THRESHOLD $SAMPLE_RATE $QNONCOLL_MULTIPLIER $DATA_FOLDER $BASE_NAME $NUM_TESTS $ROBOT_NAME"
echo ""

python3 prediction_simulation_nDOF_sphere.py \
    $THRESHOLD \
    $SAMPLE_RATE \
    $QNONCOLL_MULTIPLIER \
    $DATA_FOLDER \
    $BASE_NAME \
    $NUM_TESTS \
    $ROBOT_NAME

# 返回原目录
cd "$CURRENT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 球体仿真测试通过!"
    echo "========================================"
    echo ""
    echo "结果已保存到: result_files/sphere_results.csv"
    echo ""
    echo "提示:"
    echo "  - 查看CSV结果: cat result_files/sphere_results.csv"
    echo "  - 使用不同参数测试: $0 -t 0.3 -s 0.2 -q 10 -n 100"
    echo "  - 切换到PyBullet检测器: $0 -D pybullet"
    echo "  - 切换到几何检测器: $0 -D geometric"
    
    if [ "$DETECTOR_TYPE" = "geometric" ] && [ $CYCLE_FILES -gt 0 ]; then
        echo ""
        echo "周期统计信息已包含在结果中："
        echo "  - 碰撞边总周期数 / 平均周期数"
        echo "  - 无碰撞边总周期数 / 平均周期数"
    fi
else
    echo ""
    echo "========================================"
    echo "✗ 球体仿真测试失败"
    echo "========================================"
    exit 1
fi
