#!/bin/bash

# filepath: /home/lanh/project/robot_sim/coll_prediction_artifact/trace_generation/batch_replace_paths.sh

echo "批量替换场景文件中的机器人模型和路径..."

# 配置参数
SOURCE_DIR="../scene_benchmarks"
TARGET_DIR="../scene_benchmarks_abs"
OLD_ROBOT='file="../data/robots/jaco_mod.rob"'
NEW_ROBOT='file="../data/robots/jaco_7/jaco_7s.urdf"'  # 修改为你需要的机器人模型

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
ABS_DATA_DIR="${SCRIPT_DIR}/data"

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源目录 $SOURCE_DIR 不存在"
    exit 1
fi

# 创建目标目录
echo "创建目标目录: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# 计数器
total_files=0
processed_files=0

# 统计总文件数
echo "统计需要处理的文件数量..."
total_files=$(find "$SOURCE_DIR" -name "*.xml" | wc -l)
echo "找到 $total_files 个 XML 文件需要处理"

# 遍历所有密度目录
for density_dir in "$SOURCE_DIR"/dens*; do
    if [ -d "$density_dir" ]; then
        # 提取密度目录名
        density_name=$(basename "$density_dir")
        target_density_dir="$TARGET_DIR/$density_name"
        
        echo "处理密度目录: $density_name"
        
        # 创建对应的目标密度目录
        mkdir -p "$target_density_dir"
        
        # 遍历该密度目录下的所有 XML 文件
        for xml_file in "$density_dir"/*.xml; do
            if [ -f "$xml_file" ]; then
                # 提取文件名
                filename=$(basename "$xml_file")
                target_file="$target_density_dir/$filename"
                
                # 处理文件内容
                sed -e "s|$OLD_ROBOT|$NEW_ROBOT|g" \
                    -e "s|file=\"../data/terrains/|file=\"$ABS_DATA_DIR/terrains/|g" \
                    -e "s|file=\"../data/robots/|file=\"$ABS_DATA_DIR/robots/|g" \
                    "$xml_file" > "$target_file"
                
                processed_files=$((processed_files + 1))
                
                # 显示进度
                if [ $((processed_files % 10)) -eq 0 ]; then
                    echo "  已处理: $processed_files/$total_files 文件"
                fi
            fi
        done
        
        # 复制非 XML 文件（如 .pkl 文件）
        echo "  复制 $density_name 中的数据文件..."
        for data_file in "$density_dir"/*.pkl; do
            if [ -f "$data_file" ]; then
                cp "$data_file" "$target_density_dir/"
            fi
        done
    fi
done

echo ""
echo "批量替换完成!"
echo "处理统计:"
echo "  总文件数: $total_files"
echo "  已处理: $processed_files"
echo "  源目录: $SOURCE_DIR"
echo "  目标目录: $TARGET_DIR"
echo "  机器人模型: $OLD_ROBOT -> $NEW_ROBOT"
echo "  路径转换: 相对路径 -> 绝对路径 ($ABS_DATA_DIR)"

# 验证处理结果
echo ""
echo "验证处理结果..."
sample_file=$(find "$TARGET_DIR" -name "*.xml" | head -1)
if [ -f "$sample_file" ]; then
    echo "示例文件内容预览:"
    echo "文件: $sample_file"
    echo "机器人行:"
    grep "robot.*file=" "$sample_file" || echo "  未找到机器人定义"
    echo "地形行 (前3行):"
    grep "terrain.*file=" "$sample_file" | head -3 || echo "  未找到地形定义"
else
    echo "警告: 未找到处理后的文件进行验证"
fi

echo ""
echo "脚本执行完成!"