#!/bin/bash
################################################################################
# 生成标准数据集脚本
# 生成不同配置的标准数据集 (3000个问题)
################################################################################

cd "$(dirname "${BASH_SOURCE[0]}")"

# ========================================================================
# 配置区域
# ========================================================================
# 工作模式: 
#   "dual"   - 使用 generate_problem_dataset.py (Link 搜索 + Sphere 验证)
#   "sphere" - 使用 generate_problem_dataset_sphere.py (全程仅使用 Sphere)
MODE="sphere" 

# 机器人配置
ROBOT_NAME="iiwa"

# 规划参数
MAX_TIME=300.0
PATH_LIMIT=1.2
NUM_PROBLEMS=200
MIN_OBSTACLES=5
MAX_OBSTACLES=9
# ========================================================================

echo "========================================================================"
echo "生成标准数据集 (模式: $MODE)"
echo "========================================================================"

# 确保输出目录存在
mkdir -p maze_files

WORKSPACE_FILE="../data/workspace_bounds/${ROBOT_NAME}_workspace.json"

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

# 选择脚本
if [ "$MODE" == "dual" ]; then
    TARGET_SCRIPT="generate_problem_dataset.py"
    DESC="双模型: Link + Sphere"
    OBS_ARGS=("--num-obstacles" "$MAX_OBSTACLES") # dual 脚本暂不支持 range
else
    TARGET_SCRIPT="generate_problem_dataset_sphere.py"
    DESC="单模型: Sphere"
    OBS_ARGS=("--min-obstacles" "$MIN_OBSTACLES" "--max-obstacles" "$MAX_OBSTACLES")
fi

# 生成数据集
echo ""
echo "========================================================================"
echo "生成 $ROBOT_NAME 数据集 ($DESC)"
echo "========================================================================"

    python "$TARGET_SCRIPT" \
        --robot-name "$ROBOT_NAME" \
        --num-problems "$NUM_PROBLEMS" \
        "${OBS_ARGS[@]}" \
        --max-time "$MAX_TIME" \
        --path-length-limit "$PATH_LIMIT" \
        --workspace-min "$X_START" \
        --workspace-max "$X_END" \
        --safe-zone-radius 0.15 \
        --voxel-size-min 0.12 \
        --voxel-size-max 0.20

    if [ $? -eq 0 ]; then
        echo "✓ $ROBOT_NAME 数据集 (模式: $MODE) 生成成功"
    else
        echo "✗ $ROBOT_NAME 数据集 (模式: $MODE) 生成失败"
    fi

echo ""
echo "========================================================================"
echo "数据集生成完成"
echo "========================================================================"