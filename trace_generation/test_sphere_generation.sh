#!/bin/bash

# 球体数据生成测试脚本
# 用于验证 sphere_trace_generation.py 的功能

echo "Testing sphere-based collision detection data generation..."

# 检查是否存在环境文件
ENV_FOLDER="/home/lanh/project/robot_sim/coll_prediction_artifact/trace_files/scene_benchmarks/dens3"
ENV_NUMBER="1"

if [ ! -d "$ENV_FOLDER" ]; then
    echo "Creating test environment folder: $ENV_FOLDER"
    mkdir -p "$ENV_FOLDER"
fi

# 检查是否存在环境文件
ENV_FILE="$ENV_FOLDER/obstacles_$ENV_NUMBER.xml"
if [ ! -f "$ENV_FILE" ]; then
    echo "Warning: Environment file $ENV_FILE not found."
    echo "Please ensure you have a proper obstacle environment file."
    echo "You can generate one using the scene generator or copy from existing data."
fi

# 生成少量样本进行测试
NUM_QUERIES=50
echo "Generating $NUM_QUERIES poses with sphere-based collision detection..."

python sphere_trace_generation.py $NUM_QUERIES $ENV_FOLDER $ENV_NUMBER

# 检查输出文件
SPHERE_FILE="$ENV_FOLDER/obstacles_${ENV_NUMBER}_sphere.pkl"
POSE_FILE="$ENV_FOLDER/obstacles_${ENV_NUMBER}_pose.pkl"

if [ -f "$SPHERE_FILE" ]; then
    echo "✓ Sphere data file generated successfully: $SPHERE_FILE"
    python -c "
import pickle
import numpy as np
with open('$SPHERE_FILE', 'rb') as f:
    qarr_sphere, yarr_sphere, radius_arr, link_id_arr, sphere_id_arr = pickle.load(f)
print(f'Sphere data shape: {qarr_sphere.shape}')
print(f'Collision labels shape: {yarr_sphere.shape}')
print(f'Total spheres: {len(qarr_sphere)}')
print(f'Unique links: {len(np.unique(link_id_arr))}')
print(f'Collision rate: {np.sum(yarr_sphere == 0) / len(yarr_sphere) * 100:.2f}%')
"
else
    echo "✗ Sphere data file not generated"
fi

if [ -f "$POSE_FILE" ]; then
    echo "✓ Pose data file generated successfully: $POSE_FILE"
    python -c "
import pickle
import numpy as np
with open('$POSE_FILE', 'rb') as f:
    qarr_pose, yarr_pose = pickle.load(f)
print(f'Pose data shape: {qarr_pose.shape}')
print(f'Pose collision labels shape: {yarr_pose.shape}')
print(f'Pose collision rate: {np.sum(yarr_pose == 0) / len(yarr_pose) * 100:.2f}%')
"
else
    echo "✗ Pose data file not generated"
fi

echo "Test completed!"
