#!/bin/bash

# 该脚本用于在动态变化的环境中(从低密度到高密度)评估和比较
# 固定S策略和自适应S策略的性能。

RESULT_FILE="result_files/adaptive_evaluation.csv"

# 清理之前的结果
rm -f $RESULT_FILE

echo "Starting evaluation for changing environments..."

# --- 通用参数 ---
BIN_BITS=4      # 哈希箱的比特数 (2^4 = 16 bins)
UPDATE_PROB=1   # 哈希表更新概率

# --- 1. 评估固定S策略 (使用 coord_hashing_changing_env.py) ---

echo "--- Running Fixed S Strategies ---"


# 策略一: 随机基线 (S=4)
python coord_hashing_changing_env.py $BIN_BITS 4 $UPDATE_PROB >> $RESULT_FILE

# 策略二: 高S值 (适合低密度环境)
python coord_hashing_changing_env.py $BIN_BITS 2 $UPDATE_PROB >> $RESULT_FILE
python coord_hashing_changing_env.py $BIN_BITS 1 $UPDATE_PROB >> $RESULT_FILE

# 策略三: 中等S值
python coord_hashing_changing_env.py $BIN_BITS 0.5 $UPDATE_PROB >> $RESULT_FILE

# 策略四: 低S值 (适合高密度环境)
python coord_hashing_changing_env.py $BIN_BITS 0.125 $UPDATE_PROB >> $RESULT_FILE
python coord_hashing_changing_env.py $BIN_BITS 0.03125 $UPDATE_PROB >> $RESULT_FILE


# --- 2. 评估自适应S策略 (使用 coord_hashing_adaptive.py) ---

echo "--- Running Adaptive S Strategies ---"

# 策略五: 自适应S, 范围较窄
python coord_hashing_adaptive.py $BIN_BITS 0.1 1.0 $UPDATE_PROB >> $RESULT_FILE

# 策略六: 自适应S, 范围适中
python coord_hashing_adaptive.py $BIN_BITS 0.1 1.5 $UPDATE_PROB >> $RESULT_FILE

# 策略七: 自适应S, 范围较宽
python coord_hashing_adaptive.py $BIN_BITS 0.05 2.0 $UPDATE_PROB >> $RESULT_FILE

echo "Evaluation finished. Results are in $RESULT_FILE"
