#!/bin/bash

echo "Starting data generation pipeline..."

# 生成场景文件
echo "Step 1: Generating scene files..."
python scene_generator.py

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
        
        # 生成球体数据 (新增功能)
        echo "    Generating sphere data..."
        python sphere_trace_generation.py 1000 scene_benchmarks/${j} ${i}
        
        echo "    Completed ${j} environment ${i}"
    done
done

echo "Data generation pipeline completed!"
echo "Generated files for each environment:"
echo "  - obstacles_X_coord.pkl (OBB data)"
echo "  - obstacles_X_sphere.pkl (Sphere data)" 
echo "  - obstacles_X_pose.pkl (Pose data)"