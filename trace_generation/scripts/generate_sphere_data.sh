#!/bin/bash

# 批量对比OBB和球体碰撞检测结果

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OBSTACLE_DIR="$SCRIPT_DIR/../../trace_files/bit_traces"
COLLISION_DIR="$SCRIPT_DIR/../../trace_files/scene_benchmarks/bit_collision_data"
ROBOT_NAME="iiwa"

# 默认范围
START=1
END=50

# 新增：检测器类型和周期计数选项
DETECTOR_TYPE="geometric"  # 默认使用geometric
RETURN_CYCLES="--return-cycles"  # 默认返回周期数

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
    --detector-type)
      DETECTOR_TYPE="$2"
      shift 2
      ;;
    --return-cycles)
      RETURN_CYCLES="--return-cycles"
      shift
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--start START_ID] [--end END_ID] [--detector-type pybullet|geometric] [--return-cycles]"
      exit 1
      ;;
  esac
done

echo "处理范围: $START 到 $END"
echo "检测器类型: $DETECTOR_TYPE"
if [ -n "$RETURN_CYCLES" ]; then
    echo "周期计数: 启用"
fi
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
                # 根据检测器类型和是否返回周期数调整输出文件名
                if [ "$DETECTOR_TYPE" = "geometric" ] && [ -n "$RETURN_CYCLES" ]; then
                    sphere_file="$COLLISION_DIR/${base_name}_sphere_geometric_cycles.pkl"
                elif [ "$DETECTOR_TYPE" = "geometric" ]; then
                    sphere_file="$COLLISION_DIR/${base_name}_sphere_geometric.pkl"
                else
                    sphere_file="$COLLISION_DIR/${base_name}_sphere.pkl"
                fi
                
                # 使用 python -m 方式运行，确保能正确导入模块
                cd "$PROJECT_ROOT" || exit 1  # 切换到项目根目录
                python3 -m trace_generation.scripts.generate_sphere_data \
                    --obstacle-config-file "$obstacle_file" \
                    --collision-data-file "$collision_file" \
                    --robot-name "$ROBOT_NAME" \
                    --benchmark-id "$benchmark_id" \
                    --detector-type "$DETECTOR_TYPE" \
                    $RETURN_CYCLES \
                    --output-file "$sphere_file"
                cd "$SCRIPT_DIR" > /dev/null || exit 1  # 切换回脚本目录
                echo "------------------------"
            else
                echo "警告: 未找到OBB文件 $collision_file"
            fi
        fi
    fi
done

echo "批量处理完成。"