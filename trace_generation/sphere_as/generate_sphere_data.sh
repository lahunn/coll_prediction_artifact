#!/bin/bash

# 批量对比OBB和球体碰撞检测结果

OBSTACLE_DIR="../../trace_files/bit_traces"
COLLISION_DIR="../../trace_files/scene_benchmarks/bit_collision_data"
ROBOT_NAME="franka"

# 默认范围
START=1
END=200

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --start)
      START="$2"
      shift 2
      ;;
    --end)
      END="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--start START_ID] [--end END_ID]"
      exit 1
      ;;
  esac
done

echo "处理范围: $START 到 $END"
echo "开始批量处理文件..."

for obstacle_file in "$OBSTACLE_DIR"/${ROBOT_NAME}_7_*.pkl; do
    if [ -f "$obstacle_file" ]; then
        # 提取文件名（不含路径和扩展名）
        base_name=$(basename "$obstacle_file" .pkl)

        # 构造对应的OBB文件路径
        collision_file="$COLLISION_DIR/${base_name}_obb.pkl"

        # 提取benchmark_id（文件名中的最后一部分）
        benchmark_id=${base_name##*_}
        benchmark_id_int=$((10#$benchmark_id))  # 转换为整数

        if (( benchmark_id_int >= START && benchmark_id_int <= END )); then
            if [ -f "$collision_file" ]; then
                echo "处理文件: $base_name"
                # 生成球体数据输出文件路径
                sphere_file="$COLLISION_DIR/${base_name}_sphere.pkl"
                python generate_sphere_data.py \
                    --obstacle-config-file "$obstacle_file" \
                    --collision-data-file "$collision_file" \
                    --robot-name "$ROBOT_NAME" \
                    --benchmark-id "$benchmark_id" \
                    --output-file "$sphere_file"
                echo "------------------------"
            else
                echo "警告: 未找到OBB文件 $collision_file"
            fi
        fi
    fi
done

echo "批量处理完成。"