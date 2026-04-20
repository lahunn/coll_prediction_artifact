#!/bin/bash
# 一键运行 generate_collision_data.py 的脚本
# 用法示例:
#   ./generate_collision_data.sh --start 0 --end 4000 --batch 50 --t_max 1000 --collision-model link --seed 1234

set -euo pipefail
cd "$(dirname "$0")"

# 默认参数（逐项说明见下）
# START: 要处理的问题索引起始值（包含），用于选择问题子集进行评估
START=2000
# END: 要处理的问题索引结束值（不包含或包含，取决于脚本内部实现），通常与 START 一起限定范围
END=2550
# BATCH: 每个批次处理的问题数量（控制一次运行中加载/评估的问题数）
BATCH=50
# T_MAX: 单个轨迹/规划器的最大搜索/时间限制（单位取决于上游代码，通常为迭代或时间步数）
T_MAX=100
# TIME_BUDGET: 每个问题的总时间预算（单位同上），用于限制整个评估过程的时间消耗
TIME_BUDGET=100
# SEED: 随机数种子，用于可重复的随机抽样/排序（确保试验可复现）
SEED=1234
# COLLISION_MODEL: 使用的碰撞检测模式，可选值示例："link"（逐连杆检测）、"sphere"（球元近似）
COLLISION_MODEL="sphere"
# PLANNER: 要评估的规划器/追踪器，脚本在不同 planner 值下选择不同的 trace/collision 目录
PLANNER="bit_star"
# 可选: PLANNER="gnnmp"
# DRY_RUN: 是否为演习模式（不实际写入/移动文件），1=仅打印要做的事，0=执行真实操作
DRY_RUN=0
# 清理选项：在运行脚本前是否删除目标目录内容（默认不删除）
CLEAN=0
# FORCE: 在执行删除操作时跳过交互确认（谨慎使用）
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

# 根据所选算法调整默认目录（说明：
# - OUT: 脚本输出/结果保存路径（例如问题难度分层结果）
# 选择不同的 PLANNER 会映射到不同的一组目录，使得同一脚本可处理多种算法产生的数据
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
# 构建要执行的 Python 命令及其参数说明：
# --start/--end: 指定要评估的问题索引范围
# --batch: 每次处理的问题数量
# --t_max: 单次评估的时间/迭代上限
# --seed: 随机种子以保证可复现性
# --collision-model: 碰撞检测类型（link/sphere 等）
# --planner: 使用的规划器名（决定 trace/collision 目录的选择）
# --out: 保存评估结果的目标目录
CMD=(python3 generate_collision_data.py
  --start "$START" --end "$END"
  --batch "$BATCH" --t_max "$T_MAX" --time-budget "$TIME_BUDGET"
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
