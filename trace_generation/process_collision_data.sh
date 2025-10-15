#!/bin/bash
################################################################################
# 处理碰撞检测数据脚本
# 调用generate_collision_data.py处理obstacle_config_pairs目录下的配对文件
################################################################################
# set -e  # 遇到错误时退出

# 切换到脚本所在目录
cd "$(dirname "${BASH_SOURCE[0]}")"

# ==================== 配置参数 ====================
PAIR_DIR="bit_planning/obstacle_config_pairs"
OUTPUT_DIR="collision_results"

# 机器人配置
ROBOT_URDF="../data/robots/franka_description/franka_panda.urdf"
ROBOT_MODEL_NAME="franka"

# 文件名模式和序号范围
PAIR_FILE_PREFIX="franka_14_30"  # 文件名前缀
START_INDEX=1                     # 起始序号
END_INDEX=30                      # 结束序号
# ==================================================

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "处理碰撞检测数据"
echo "========================================================================"
echo "障碍物-配置配对目录: ${PAIR_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "机器人URDF: ${ROBOT_URDF}"
echo "机器人模型名称: ${ROBOT_MODEL_NAME}"
echo ""

# 检查URDF文件是否存在
if [ ! -f "$ROBOT_URDF" ]; then
    echo "错误: URDF文件不存在: $ROBOT_URDF"
    exit 1
fi

# 检查配对目录是否存在
if [ ! -d "$PAIR_DIR" ]; then
    echo "错误: 配对目录不存在: $PAIR_DIR"
    exit 1
fi

# 处理计数
total_processed=0
total_failed=0
total_files=$((END_INDEX - START_INDEX + 1))

echo "文件名前缀: ${PAIR_FILE_PREFIX}"
echo "处理范围: ${START_INDEX} 到 ${END_INDEX} (共 ${total_files} 个文件)"
echo ""

# 遍历指定序号范围的文件
for idx in $(seq -f "%04g" $START_INDEX $END_INDEX); do
    # 构造文件名
    pair_basename="${PAIR_FILE_PREFIX}_${idx}"
    pair_file="${PAIR_DIR}/${pair_basename}.pkl"
    
    # 检查文件是否存在
    if [ ! -f "$pair_file" ]; then
        echo "警告: 文件不存在,跳过: $pair_file"
        ((total_failed++))
        continue
    fi
    
    # 生成输出文件名
    obb_output="${OUTPUT_DIR}/${pair_basename}_obb.pkl"
    sphere_output="${OUTPUT_DIR}/${pair_basename}_sphere.pkl"
    
    # 移除前导零避免八进制解析错误
    current=$((10#$idx - START_INDEX + 1))
    echo "========================================================================"
    echo "[$current/$total_files] 处理: $pair_basename"
    echo "------------------------------------------------------------------------"
    echo "  配对文件: $pair_file"
    echo "  输出: ${pair_basename}_*.pkl"
    echo ""
    
    # 调用generate_collision_data.py (新参数格式: 5个参数)
    if python generate_collision_data.py \
        "$pair_file" \
        "$ROBOT_URDF" \
        "$ROBOT_MODEL_NAME" \
        "$obb_output" \
        "$sphere_output"; then
        
        echo "✓ 成功处理: $pair_basename"
        ((total_processed++))
    else
        echo "✗ 处理失败: $pair_basename"
        ((total_failed++))
    fi
    echo ""
done

echo "========================================================================"
echo "处理完成"
echo "========================================================================"
echo "总文件数: $total_files"
echo "成功处理: $total_processed"
echo "失败: $total_failed"
echo "成功率: $(awk "BEGIN {printf \"%.1f\", $total_processed/$total_files*100}")%"
echo ""
if [ $total_processed -gt 0 ]; then
    echo "输出文件示例:"
    ls -lh "${OUTPUT_DIR}"/*.pkl 2>/dev/null | head -10
    echo ""
    echo "总文件数: $(ls -1 "${OUTPUT_DIR}"/*.pkl 2>/dev/null | wc -l)"
else
    echo "没有生成输出文件"
fi
