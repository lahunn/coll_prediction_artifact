#!/bin/bash

echo "Starting data generation pipeline..."

# 生成场景文件
# echo "Step 1: Generating scene files..."
ROBOT_URDF="/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf" 

python scene_generator.py  $ROBOT_URDF

# 生成 OBB 和球体数据
echo "Step 2: Generating collision detection data..."
for i in {0..99}
do
    echo "Processing environment ${i}/99..."
    for j in dens3 dens6 dens9 dens12
    do  
        echo "  Processing density: ${j}"
        
        # 生成 OBB 数据 (原有功能)
        echo "    Generating OBB data..."
        python pred_trace_generation.py 1000 scene_benchmarks/${j} ${i}
        
        echo "    Completed ${j} environment ${i}"
    done
done

echo "Data generation pipeline completed!"
echo "Generated files for each environment:"
echo "  - obstacles_X_coord.pkl (OBB data)"
echo "  - obstacles_X_sphere.pkl (Sphere data)" 
echo "  - obstacles_X_pose.pkl (Pose data)"