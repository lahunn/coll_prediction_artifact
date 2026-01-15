#!/bin/bash
# 一键运行 generate_collision_data.py 的脚本
# 用法示例:
#   ./generate_collision_data.sh --start 0 --end 4000 --batch 50 --t_max 1000 --collision-model link --seed 1234

set -euo pipefail
cd "$(dirname "$0")"

# 默认参数
START=2000
END=2550
BATCH=50
T_MAX=1000
SEED=1234
COLLISION_MODEL="link"
PLANNER="bit_star"
# PLANNER="gnnmp"
DRY_RUN=0
# 清理选项
CLEAN=0
FORCE=0

print_usage() {
  echo "Usage: $0 [--planner gnnmp|bit_star]"
}

# 只保留 --planner 参数以简化脚本
while [[ $# -gt 0 ]]; do
  case $1 in
    --planner)
      PLANNER="$2"; shift 2;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown option: $1"; print_usage; exit 1;;
  esac
done

# 根据所选算法调整默认目录
if [[ "$PLANNER" == "bit_star" ]]; then
  OUT="../../trace_files/bit_kuka7_difficulty"
  TRACE_DIR="../../trace_files/bit_traces"
  COLLISION_DIR="../../trace_files/scene_benchmarks/bit_collision_data"
else
  OUT="../../trace_files/gnn_kuka7_difficulty"
  TRACE_DIR="../../trace_files/gnn_traces"
  COLLISION_DIR="../../trace_files/scene_benchmarks/gnn_collision_data"
fi

# 可选的删除操作（仅在 --clean 指定时执行）
if [[ $CLEAN -eq 1 ]]; then
  echo "Requested clean of:\n  traces: $TRACE_DIR\n  collision data: $COLLISION_DIR"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY-RUN: would remove contents of $TRACE_DIR and $COLLISION_DIR (no files will be deleted)"
  else
    if [[ $FORCE -ne 1 ]]; then
      read -p "Confirm deletion of contents under $TRACE_DIR and $COLLISION_DIR? [y/N]: " CONFIRM
      case "$CONFIRM" in
        [yY]|[yY][eE][sS])
          ;; # proceed
        *)
          echo "Aborted by user. No files were deleted."; exit 0;;
      esac
    fi

    # 删除目录内所有内容（保留目录本身），如果目录不存在则忽略
    if [[ -d "$TRACE_DIR" ]]; then
      echo "Removing contents of $TRACE_DIR ..."
      find "$TRACE_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} + || true
      echo "Removed contents of $TRACE_DIR"
    else
      echo "Warning: $TRACE_DIR not found, skipping."
    fi

    if [[ -d "$COLLISION_DIR" ]]; then
      echo "Removing contents of $COLLISION_DIR ..."
      find "$COLLISION_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} + || true
      echo "Removed contents of $COLLISION_DIR"
    else
      echo "Warning: $COLLISION_DIR not found, skipping."
    fi
  fi
fi

# 构建命令
CMD=(python3 generate_collision_data.py 
  --start "$START" --end "$END" 
  --batch "$BATCH" --t_max "$T_MAX" 
  --seed "$SEED" --collision-model "$COLLISION_MODEL" --planner "$PLANNER"
  --out "$OUT"
)
if [[ $DRY_RUN -eq 1 ]]; then
  CMD+=(--dry-run)
  echo "Running in dry-run mode"
fi

echo "Running collision data generation with parameters:"
echo "  start=$START end=$END batch=$BATCH t_max=$T_MAX seed=$SEED collision_model=$COLLISION_MODEL planner=$PLANNER out=$OUT trace_dir=$TRACE_DIR collision_dir=$COLLISION_DIR dry_run=$DRY_RUN"

# 执行
"${CMD[@]}"
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "generate_collision_data.py failed with exit code $RC"
  exit $RC
fi

echo "Done. Output saved to: $OUT (if not dry-run)"
exit 0
