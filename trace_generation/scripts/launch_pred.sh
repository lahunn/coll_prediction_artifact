#!/bin/bash

echo "Starting data generation pipeline..."

# 生成场景文件
echo "Step 1: Generating scene files..."
ROBOT_NAME="iiwa"
NUM_PROBLEMS=100
NUM_SAMPLES=1000
BASE_SEED=0
DENSITIES=("dens3" "dens6" "dens9" "dens12")

python scene_generator.py "$ROBOT_NAME" "$NUM_PROBLEMS"

# 生成 OBB 和球体数据
echo "Step 2: Generating collision detection data..."
for ((i = 0; i < NUM_PROBLEMS; ++i))
do
    echo "Processing environment ${i}/99..."
    for density in "${DENSITIES[@]}"
    do  
        echo "  Processing density: ${density}"
        obstacle_file="../../trace_files/scene_benchmarks/${density}/obstacles_${i}.pkl"
        if [[ ! -f "${obstacle_file}" ]]; then
            echo "    Warning: obstacle file not found at ${obstacle_file}, skipping."
            continue
        fi
      
        # 生成碰撞数据 (默认包含球体)
        echo "    Generating collision trace..."
        python pred_trace_generation.py \
            "$ROBOT_NAME" \
            "$NUM_SAMPLES" \
            "../trace_files/scene_benchmarks/${density}" \
            "${i}" \
            --seed "$((BASE_SEED + i))" \
            --obstacle-file "${obstacle_file}"
        
        echo "    Completed ${density} environment ${i}"
    done
done

echo "Data generation pipeline completed!"
echo "Generated files for each environment:"
echo "  - obstacles_X_coord.pkl (OBB data)"
echo "  - obstacles_X_sphere.pkl (Sphere data)" 
echo "  - obstacles_X_pose.pkl (Pose data)"