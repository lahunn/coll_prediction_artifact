#!/bin/bash
################################################################################
# 生成标准数据集脚本
# 生成不同配置的标准数据集 (3000个问题)
################################################################################

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "========================================================================"
echo "生成标准数据集"
echo "========================================================================"

# 确保输出目录存在
mkdir -p maze_files

# 机器人配置
ROBOT_NAME="iiwa"
WORKSPACE_FILE="../data/workspace_bounds/${ROBOT_NAME}_workspace.json"
# 注意: --robot-file 参数已弃用，现在使用 robot_urdf_mapping 从 robot_name 查找 URDF

# 分析工作空间
echo ""
echo "========================================================================"
echo "分析机器人工作空间"
echo "========================================================================"
if [ -f "$WORKSPACE_FILE" ]; then
    echo "✓ 工作空间文件 '$WORKSPACE_FILE' 已存在, 跳过分析."
else
    echo "i 工作空间文件 '$WORKSPACE_FILE' 不存在, 开始分析..."
    python ../data/workspace_bounds/workspace_analyzer.py "$ROBOT_NAME" "$WORKSPACE_FILE"

    if [ ! -f "$WORKSPACE_FILE" ]; then
        echo "✗ 工作空间分析失败, 未能创建 '$WORKSPACE_FILE'."
        exit 1
    else
        echo "✓ 工作空间分析成功, 文件已创建: '$WORKSPACE_FILE'."
    fi
fi

# 从JSON文件读取工作空间范围
X_START=$(python -c "import json; print(json.load(open('$WORKSPACE_FILE'))['x_start'])")
X_END=$(python -c "import json; print(json.load(open('$WORKSPACE_FILE'))['x_end'])")
Z_START=$(python -c "import json; print(json.load(open('$WORKSPACE_FILE'))['z_start'])")
Z_END=$(python -c "import json; print(json.load(open('$WORKSPACE_FILE'))['z_end'])")

echo "使用工作空间范围: X=[$X_START, $X_END], Z=[$Z_START, $Z_END]"

# 生成数据集
echo ""
echo "========================================================================"
echo "生成 $ROBOT_NAME 数据集 (双模型: Link + Sphere)"
echo "========================================================================"

# 遍历不同的障碍物数量
# for NUM_OBSTACLES in 10; do
#     echo ""
#     echo "----------------------------------------------------------------"
#     echo "生成障碍物数量: $NUM_OBSTACLES"
#     echo "----------------------------------------------------------------"

    python generate_problem_dataset.py \
        --robot-name "$ROBOT_NAME" \
        --num-problems 200 \
        --num-obstacles 8 \
        --max-time 100.0 \
        --workspace-min "$X_START" \
        --workspace-max "$X_END" \
        --safe-zone-radius 0.15 \
        --voxel-size-min 0.12 \
        --voxel-size-max 0.20

    if [ $? -eq 0 ]; then
        echo "✓ $ROBOT_NAME 数据集 (障碍物: $NUM_OBSTACLES) 生成成功"
    else
        echo "✗ $ROBOT_NAME 数据集 (障碍物: $NUM_OBSTACLES) 生成失败"
    fi
# done

echo ""
echo "========================================================================"
echo "数据集生成完成"
echo "========================================================================"