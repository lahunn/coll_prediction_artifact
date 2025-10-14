#!/bin/bash
################################################################################
# 生成标准数据集脚本
# 生成不同配置的标准数据集 (3000个问题)
################################################################################

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "========================================================================"
echo "生成标准数据集 (3000个问题)"
echo "========================================================================"
echo "警告: 这可能需要数小时到数天时间，取决于硬件性能"
echo ""
read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 确保输出目录存在
mkdir -p maze_files

# 生成 Kuka 7DOF 数据集
echo ""
echo "========================================================================"
echo "生成 franka 数据集"
echo "========================================================================"
# cd /home/lanh/project/robot_sim/coll_prediction_artifact/trace_generation/bit_planning
python generate_problem_dataset.py \
    --robot-file /home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf \
    --robot-name franka \
    --num-problems 30 \
    --num-obstacles 10 \
    --max-time 6.0 \
    --workspace-min -2.0 \
    --workspace-max 2.0 \
    --voxel-size-min 0.05 \
    --voxel-size-max 0.12

if [ $? -eq 0 ]; then
    echo "✓ Kuka 7DOF 数据集生成成功"
else
    echo "✗ Kuka 7DOF 数据集生成失败"
fi

echo ""
echo "========================================================================"
echo "数据集生成完成"
echo "========================================================================"
echo "生成的文件:"
ls -lh maze_files/*.pkl 2>/dev/null || echo "没有找到生成的文件"
