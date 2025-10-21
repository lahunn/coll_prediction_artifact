#!/bin/bash

# ==============================================================================
# 脚本: run_sphere_cost_analysis.sh
# 功能: 遍历 coord_hashing_sphere.py 的不同参数设置，评估其对预测性能和计算成本的影响。
#
# 该脚本会自动运行一系列实验，并将结果保存到CSV文件中，
# 包括精确率、召回率、碰撞率、预期计算成本、baseline成本和加速比。
# ==============================================================================

# --- 配置 ---

# 定义结果输出文件
OUTPUT_FILE="../result_files/sphere_hashing_cost_results.csv"

# 定义要测试的参数范围
DENSITY_LEVELS=("dens3" "dens6" "dens9" "dens12")  # 目标场景密度
COORD_BITS_LIST=(3 4 5 6)                      # 坐标量化位数
RADIUS_BITS_LIST=(1 2 3 4)                    # 半径量化位数
THRESHOLDS=(0.0 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0)  # 碰撞阈值 (S)
SAMPLE_RATES=(1.0)    # 自由样本采样率 (U)
NUM_PROBLEMS=100                                # 评估的问题数量

# --- 执行 ---

# 检查Python脚本是否存在
if [ ! -f "../coord_hashing_sphere.py" ]; then
    echo "错误: 脚本 'coord_hashing_sphere.py' 未找到。"
    exit 1
fi

# 创建result_files目录（如果不存在）
mkdir -p ../result_files

# 写入CSV文件的表头
echo "Density,CoordBits,RadiusBits,Threshold,SampleRate,PosePrecision,PoseRecall,PoseCollisionRatio,ElemPrecision,ElemRecall,ElemCollisionRatio,PredCost,BaselineCost,Speedup" > "$OUTPUT_FILE"

echo "🚀 开始球体碰撞预测参数扫描 (包含成本分析)"
echo "   结果将保存到 $OUTPUT_FILE"
echo ""

# 计数器
total_combinations=$((${#DENSITY_LEVELS[@]} * ${#COORD_BITS_LIST[@]} * ${#RADIUS_BITS_LIST[@]} * ${#THRESHOLDS[@]} * ${#SAMPLE_RATES[@]}))
current=0

# 使用嵌套循环遍历所有参数组合
for density in "${DENSITY_LEVELS[@]}"; do
  echo "📊 处理密度级别: $density"
  
  for coord_bits in "${COORD_BITS_LIST[@]}"; do
    for radius_bits in "${RADIUS_BITS_LIST[@]}"; do
      for threshold in "${THRESHOLDS[@]}"; do
        for sample_rate in "${SAMPLE_RATES[@]}"; do
          
          current=$((current + 1))
          
          # 显示进度（每10个输出一次）
          if [ $((current % 10)) -eq 0 ] || [ $current -eq 1 ]; then
            echo "  [$current/$total_combinations] 坐标位数=$coord_bits, 半径位数=$radius_bits, 阈值=$threshold, 采样率=$sample_rate"
          fi

          # 执行Python脚本并捕获输出（需要在上级目录执行）
          result=$(cd .. && python coord_hashing_sphere.py "$density" "$coord_bits" "$radius_bits" "$threshold" "$sample_rate" "$NUM_PROBLEMS" 2>&1)

          # 检查是否执行成功
          if [ $? -eq 0 ]; then
            # 清理输出，移除百分号、标签和多余空格
            cleaned_result=$(echo "$result" | sed 's/Pose://g' | sed 's/Elem://g' | sed 's/Cost://g' | sed 's/%, /,/g' | sed 's/%//g' | sed 's/ //g')
            echo "$cleaned_result" >> "$OUTPUT_FILE"
          else
            echo "  ⚠️  警告: 参数组合执行失败 ($density, $coord_bits, $radius_bits, $threshold, $sample_rate)"
            echo "  错误信息: $result"
          fi

        done
      done
    done
  done
  
  echo "  ✓ 完成密度级别 $density 的所有测试"
  echo ""
done

echo ""
echo "✅ 参数扫描完成!"
echo "📄 结果已保存到: $OUTPUT_FILE"
echo "📊 总共测试了 $total_combinations 个参数组合"

# 显示文件大小
if [ -f "$OUTPUT_FILE" ]; then
  file_size=$(du -h "$OUTPUT_FILE" | cut -f1)
  line_count=$(wc -l < "$OUTPUT_FILE")
  echo "📈 结果文件大小: $file_size, 包含 $line_count 行数据"
fi
