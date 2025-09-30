#!/bin/bash

# ==============================================================================
# 脚本: run_coord_analysis.sh
# 功能: 遍历 coord_hashing.py 的不同参数设置,评估其对预测性能的影响。
#
# 该脚本会自动运行一系列实验,并将结果保存到CSV文件中,
# 以便对坐标量化、碰撞阈值、自由样本采样率和链接数等参数
# 如何影响精确率(Precision)和召回率(Recall)进行分析。
# ==============================================================================

# --- 配置 ---

# 定义结果输出文件
OUTPUT_FILE="./result_files/coord_hashing_results.csv"

# 定义要测试的参数范围
DENSITY_LEVELS=("low" "mid" "high")       # 目标场景密度
QUANTIZE_BITS_LIST=(3 4 5 6)           # 坐标量化位数
THRESHOLDS=(0.01 0.05 0.1 0.5 1.0 2.0)       # 碰撞阈值 (S)
SAMPLE_RATES=(0.01 0.05 0.1 0.5 1.0)     # 自由样本采样率 (U)
# NUM_LINKS_LIST=(7)                        # 机器人链接数 (通常是固定值)

# --- 执行 ---

# 检查Python脚本是否存在
if [ ! -f "coord_hashing.py" ]; then
    echo "错误: 脚本 'coord_hashing.py' 未找到"
    exit 1
fi

# 写入CSV文件的表头
echo "Density,QuantBits,Threshold,SampleRate,NumLinks,Precision,Recall" > "$OUTPUT_FILE"

echo "🚀 开始参数扫描, 结果将保存到 $OUTPUT_FILE"

# 使用嵌套循环遍历所有参数组合
for density in "${DENSITY_LEVELS[@]}"; do
  for quant_bits in "${QUANTIZE_BITS_LIST[@]}"; do
    for threshold in "${THRESHOLDS[@]}"; do
      for sample_rate in "${SAMPLE_RATES[@]}"; do

          echo "  - 正在运行: 密度=$density, 量化位数=$quant_bits, 阈值=$threshold, 采样率=$sample_rate"

          # 执行Python脚本并捕获输出
          result=$(python coord_hashing.py "$density" "$quant_bits" "$threshold" "$sample_rate")

          # 清理输出,移除空格和百分号,然后追加到CSV文件
          cleaned_result=$(echo "$result" | sed 's/%, /,/g' | sed 's/%//' | sed 's/,  ,/,/g')
          echo "$cleaned_result" >> "$OUTPUT_FILE"

      done
    done
  done
done

echo "✅ 参数扫描完成!"

