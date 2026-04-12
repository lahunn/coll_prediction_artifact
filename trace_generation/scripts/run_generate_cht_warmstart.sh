#!/bin/bash

# 批量生成CHT warm-start包，按G1-G5分别保存到 trace_files/cht_pre_load 下

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ROBOT_NAME="iiwa_7"
SAMPLES_PER_PROBLEM=10000
QUANT_BITS=4
SEED=0
OVERWRITE=""

SCENES=("G1" "G2" "G3" "G4" "G5")

while [[ $# -gt 0 ]]; do
  case $1 in
    --robot-name)
      ROBOT_NAME="$2"
      shift 2
      ;;
    --samples-per-problem)
      SAMPLES_PER_PROBLEM="$2"
      shift 2
      ;;
    --quant-bits)
      QUANT_BITS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="--overwrite"
      shift
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--robot-name iiwa_7] [--samples-per-problem 1000] [--quant-bits 4] [--seed 0] [--overwrite]"
      exit 1
      ;;
  esac
done

echo "开始批量生成CHT warm-start包"
echo "机器人: $ROBOT_NAME"
echo "每个problem采样数: $SAMPLES_PER_PROBLEM"
echo "量化位数: $QUANT_BITS"
echo "随机种子: $SEED"

cd "$PROJECT_ROOT"

for SCENE in "${SCENES[@]}"; do
  PROBLEMS_PKL="$PROJECT_ROOT/trace_files/problems/$SCENE/problems.pkl"
  OUTPUT_DIR="$PROJECT_ROOT/trace_files/cht_pre_load/$SCENE"

  if [[ ! -f "$PROBLEMS_PKL" ]]; then
    echo "警告: 未找到 $PROBLEMS_PKL，跳过 $SCENE"
    continue
  fi

  mkdir -p "$OUTPUT_DIR"

  echo ""
  echo "=========================================="
  echo "处理场景: $SCENE"
  echo "输入: $PROBLEMS_PKL"
  echo "输出: $OUTPUT_DIR"
  echo "=========================================="

  python3 -m trace_generation.scripts.generate_cht_warmstart \
    --problems-pkl "$PROBLEMS_PKL" \
    --output-dir "$OUTPUT_DIR" \
    --basename "$ROBOT_NAME" \
    --samples-per-problem "$SAMPLES_PER_PROBLEM" \
    --quant-bits "$QUANT_BITS" \
    --seed "$SEED" \
    $OVERWRITE
done

echo ""
echo "批量生成完成，结果保存在 trace_files/cht_pre_load/G1-G5 下"