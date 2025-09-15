#!/bin/bash
# 两步式XML路径替换脚本

INPUT_DIR="../scene_benchmarks"
OUTPUT_DIR="../scene_benchmarks_urdf"

echo "🔧 开始两步式路径替换..."
echo "输入: $INPUT_DIR"
echo "输出: $OUTPUT_DIR"

# 删除可能存在的输出目录并创建新的
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 复制整个目录结构
cp -r "$INPUT_DIR"/* "$OUTPUT_DIR"/

# 获取data目录的绝对路径
DATA_ABS_PATH=$(realpath ../../data)
ROBOT_FILE="jaco_7/jaco_7s.urdf"

echo "📝 执行两步替换..."
echo "Data绝对路径: $DATA_ABS_PATH"
echo "目标机器人文件: $ROBOT_FILE"

# 步骤1: 替换所有 ../data 为绝对路径
echo "步骤1: 将 ../data 替换为绝对路径"
find "$OUTPUT_DIR" -name "*.xml" -type f -exec sed -i "s|../data|$DATA_ABS_PATH|g" {} \;

# 步骤2: 替换机器人文件名
echo "步骤2: 替换机器人文件名"
find "$OUTPUT_DIR" -name "*.xml" -type f -exec sed -i \
    -e "s|/robots/jaco_mod\.rob|/robots/$ROBOT_FILE|g" \
    -e "s|/robots/jaco\.rob|/robots/$ROBOT_FILE|g" \
    -e "s|/robots/iiwa_arm\.urdf|/robots/$ROBOT_FILE|g" \
    {} \;

# 统计结果
total=$(find "$OUTPUT_DIR" -name "*.xml" | wc -l)
converted=$(find "$OUTPUT_DIR" -name "*.xml" -exec grep -l "jaco_7s\.urdf" {} \; | wc -l)

echo "✅ 替换完成!"
echo "   总文件: $total"
echo "   已转换: $converted"

# 显示示例
echo ""
echo "🔍 转换示例:"
find "$OUTPUT_DIR" -name "*.xml" | head -1 | xargs grep "file=.*\.urdf" | head -1
