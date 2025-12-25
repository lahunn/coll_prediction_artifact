#!/bin/bash
# 选择规划算法: gnnmp 或 bit_star
PLANNER="gnnmp"
MODEL_KEY="kuka7"   # 仅gnnmp需要
ROBOT_NAME="iiwa"
PROBLEMS_ROOT="../../trace_files/problems"
PAIR_DIR="../../trace_files/bit_traces"
COLLISION_DIR="../../trace_files/scene_benchmarks/bit_collision_data"
START_INDEX=1
END_INDEX=50   # 可根据需要调整

for LEVEL in G1 G2 G3 G4 G5; do
    PROBLEMS_FILE="${PROBLEMS_ROOT}/${LEVEL}/problems.pkl"
    echo "处理 ${LEVEL} ..."

    if [ "$PLANNER" = "gnnmp" ]; then
        python3 generate_collision_data.py \
            --problems-file "$PROBLEMS_FILE" \
            --robot-name "$ROBOT_NAME" \
            --planner gnnmp \
            --model-key "$MODEL_KEY" \
            --level "$LEVEL" \
            --pair-dir "$PAIR_DIR" \
            --collision-dir "$COLLISION_DIR" \
            --start-index $START_INDEX \
            --end-index $END_INDEX
    else
        python3 generate_collision_data.py \
            --problems-file "$PROBLEMS_FILE" \
            --robot-name "$ROBOT_NAME" \
            --planner bit_star \
            --level "$LEVEL" \
            --pair-dir "$PAIR_DIR" \
            --collision-dir "$COLLISION_DIR" \
            --start-index $START_INDEX \
            --end-index $END_INDEX
    fi
done

echo "全部完成"