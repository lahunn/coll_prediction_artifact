#!/bin/bash
################################################################################
# 处理碰撞检测数据脚本
# 调用generate_collision_data.py处理obstacle_config_pairs目录下的配对文件
################################################################################

set -e  # 遇到错误时退出

# 切换到脚本所在目录
cd "$(dirname "${BASH_SOURCE[0]}")"

# ==================== 配置参数 ====================
PAIR_DIR="bit_planning/obstacle_config_pairs"
OUTPUT_DIR="collision_results"

# 机器人配置
ROBOT_URDF="../data/robots/franka_description/franka_panda.urdf"
ROBOT_MODEL_NAME="franka"
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

# 遍历所有配对文件
for pair_file in "${PAIR_DIR}"/*.pkl; do
    if [ ! -f "$pair_file" ]; then
        echo "没有找到配对文件"
        continue
    fi
    
    # 提取文件名（不含路径和扩展名）
    pair_basename=$(basename "$pair_file" .pkl)
    
    # 生成输出文件名
    obb_output="${OUTPUT_DIR}/${pair_basename}_obb.pkl"
    sphere_output="${OUTPUT_DIR}/${pair_basename}_sphere.pkl"
    
    echo "------------------------------------------------------------------------"
    echo "处理: $pair_basename"
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
echo "成功处理: $total_processed 个配对文件"
echo "失败: $total_failed"
echo ""
if [ $total_processed -gt 0 ]; then
    echo "输出文件示例:"
    ls -lh "${OUTPUT_DIR}"/*.pkl 2>/dev/null | head -10
    echo ""
    echo "总文件数: $(ls -1 "${OUTPUT_DIR}"/*.pkl 2>/dev/null | wc -l)"
else
    echo "没有生成输出文件"
fi
