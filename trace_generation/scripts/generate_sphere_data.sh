#!/bin/bash

# 批量对比OBB和球体碰撞检测结果

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ROBOT_NAME="iiwa"

# 默认范围
START=1
END=40

# 支持选择算法类型（bit_star 或 gnnmp）。当选择 gnnmp 时，会使用 gnn_traces / gnn_collision_data 目录
ALGO="bit"

# 难度等级列表
DIFFICULTY_LEVELS=("G1" "G2" "G3" "G4" "G5")

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
    --algo)
      ALGO="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--start START_ID] [--end END_ID] [--algo bit|gnnmp]"
      exit 1
      ;;
  esac
done

echo "算法: $ALGO"

echo "处理范围: $START 到 $END"
echo "难度等级: ${DIFFICULTY_LEVELS[*]}"
echo "开始批量处理文件..."

# 遍历难度等级
for DIFFICULTY in "${DIFFICULTY_LEVELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "处理难度等级: $DIFFICULTY"
    echo "=========================================="
    
    # 根据算法类型选择目录
    if [ "$ALGO" = "gnnmp" ]; then
        OBSTACLE_DIR="$SCRIPT_DIR/../../trace_files/gnn_traces/$DIFFICULTY"
        COLLISION_DIR="$SCRIPT_DIR/../../trace_files/scene_benchmarks/gnn_collision_data/$DIFFICULTY"
    else
        OBSTACLE_DIR="$SCRIPT_DIR/../../trace_files/bit_traces/$DIFFICULTY"
        COLLISION_DIR="$SCRIPT_DIR/../../trace_files/scene_benchmarks/bit_collision_data/$DIFFICULTY"
    fi

    # 检查目录是否存在
    if [ ! -d "$OBSTACLE_DIR" ]; then
        echo "警告: 障碍物目录不存在，跳过 $DIFFICULTY: $OBSTACLE_DIR"
        continue
    fi

    if [ ! -d "$COLLISION_DIR" ]; then
        echo "碰撞数据目录不存在，创建$COLLISION_DIR"
        mkdir -p "$COLLISION_DIR"
    fi

    for obstacle_file in "$OBSTACLE_DIR"/${ROBOT_NAME}_7_*.pkl; do
        if [ -f "$obstacle_file" ]; then
            # 提取文件名（不含路径和扩展名）
            base_name=$(basename "$obstacle_file" .pkl)

            # 构造对应的OBB文件路径
            collision_file="$COLLISION_DIR/${base_name}_link.pkl"

            # 提取benchmark_id（文件名中的最后一部分）
            benchmark_id=${base_name##*_}
            benchmark_id_int=$((10#$benchmark_id))  # 转换为整数

            if (( benchmark_id_int >= START && benchmark_id_int <= END )); then
                if [ -f "$collision_file" ]; then
                    echo "处理文件: $base_name ($DIFFICULTY)"
                    echo "处理障碍物文件: $obstacle_file"
                    # 生成球体数据输出文件路径
                    # 根据检测器类型和是否返回周期数调整输出文件名
                    sphere_file="$COLLISION_DIR/${base_name}_sphere_link.pkl"
                    # 使用 python -m 方式运行，确保能正确导入模块
                    cd "$PROJECT_ROOT" || exit 1  # 切换到项目根目录
                    python3 -m trace_generation.scripts.generate_sphere_data \
                        --obstacle-config-file "$obstacle_file" \
                        --collision-data-file "$collision_file" \
                        --robot-name "$ROBOT_NAME" \
                        --benchmark-id "$benchmark_id" \
                        --output-file "$sphere_file"
                    cd "$SCRIPT_DIR" > /dev/null || exit 1  # 切换回脚本目录
                    echo "------------------------"
                else
                    echo "警告: 未找到OBB文件 $collision_file"
                fi
            fi
        fi
    done
    
    echo "难度等级 $DIFFICULTY 处理完成"
done

echo "批量处理完成。"