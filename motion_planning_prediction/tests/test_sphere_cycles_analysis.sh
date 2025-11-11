#!/bin/bash
# 球体碰撞检测周期数分析测试脚本

# 默认参数
BASE_NAME="iiwa_7"
NUM_TESTS=50
DATA_FOLDER="../trace_files/scene_benchmarks/bit_collision_data"

# 使用说明
usage() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -b, --basename BASE_NAME       基准名称 (默认: iiwa_7)"
    echo "  -n, --num-tests NUM_TESTS      测试数量 (默认: 50)"
    echo "  -d, --data-folder DATA_FOLDER  数据文件夹 (默认: ../trace_files/scene_benchmarks/bit_collision_data)"
    echo "  -h, --help                     显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -b iiwa_7 -n 100"
    echo "  $0 -d /path/to/data -n 50"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
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
echo "球体碰撞检测周期数统计分析"
echo "========================================"
echo "基准名称: $BASE_NAME"
echo "测试数量: $NUM_TESTS"
echo "数据文件夹: $DATA_FOLDER"
echo "========================================"

# 检查数据文件夹
echo -e "\n[1/2] 检查数据文件..."
if [ ! -d "$DATA_FOLDER" ]; then
    echo "错误: 数据文件夹不存在: $DATA_FOLDER"
    exit 1
fi

# 检查是否有周期数据文件
CYCLE_FILES=$(ls $DATA_FOLDER/${BASE_NAME}_*_sphere_geometric_cycles.pkl 2>/dev/null | wc -l)

if [ $CYCLE_FILES -eq 0 ]; then
    echo "错误: 未找到周期数据文件！"
    echo "请先运行以下命令生成周期数据:"
    echo "  cd ../trace_generation/scripts"
    echo "  bash generate_sphere_data.sh --detector-type geometric --return-cycles"
    exit 1
fi

echo "找到 $CYCLE_FILES 个周期数据文件"

# 运行周期分析
echo -e "\n[2/2] 运行周期数统计分析..."
echo ""

# 切换到脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

python3 analyze_sphere_cycles.py $DATA_FOLDER $BASE_NAME $NUM_TESTS

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 周期统计分析完成!"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "✗ 周期统计分析失败"
    echo "========================================"
    exit 1
fi
