"""
测试CHT继承效果 - 基于coord_hashing的轨迹数据
在同一个benchmark内比较:
1. 前半部分训练CHT
2. 后半部分:
   - 策略1: 清零CHT从零开始
   - 策略2: 继承前半部分的CHT
"""

import sys
import numpy as np
import pickle
from collision_prediction_strategies import (
    FixedThresholdStrategy,
    evaluate_strategy_on_trajectory,
)

# 解析命令行参数
if len(sys.argv) != 5:
    print(
        "用法: python test_cht_inheritance_same_benchmark.py <密度等级> <量化位数> <碰撞阈值> <自由样本采样率>"
    )
    print("示例: python test_cht_inheritance_same_benchmark.py mid 8 0.1 0.3")
    sys.exit(1)

# 解析命令行参数到变量
density_level = sys.argv[1]  # 密度等级: "low", "mid", "high"
quantize_bits = int(sys.argv[2])  # 量化位数 (如: 8)
collision_threshold = float(sys.argv[3])  # 碰撞阈值 (如: 0.1)
free_sample_rate = float(sys.argv[4])  # 自由样本采样率 (如: 0.3)
num_links = 11

print("=" * 80)
print("CHT继承效果测试 - 同一Benchmark前后半数据对比")
print("=" * 80)
print("\n参数设置:")
print(f"  密度级别: {density_level}")
print(f"  量化位数: {quantize_bits}")
print(f"  碰撞阈值: {collision_threshold}")
print(f"  自由样本采样率: {free_sample_rate}")
print(f"  关节数量: {num_links}")

# 设置量化参数
binnumber = 2**quantize_bits
intervalsize = 2.24 / binnumber
bins = np.zeros(binnumber)
start = -1.12
for i in range(0, binnumber):
    bins[i] = start
    start += intervalsize

# 全局统计变量 (策略1: 清零重启)
total_zerozero_reset = 0
total_onezero_reset = 0
total_colliding_reset = 0

# 全局统计变量 (策略2: 继承CHT)
total_zerozero_inherit = 0
total_onezero_inherit = 0
total_colliding_inherit = 0

# 全局统计变量 (训练阶段)
total_zerozero_train = 0
total_onezero_train = 0
total_colliding_train = 0

# 统计CHT大小
total_cht_size_after_train = 0
total_cht_size_after_reset = 0
total_cht_size_after_inherit = 0

print("\n开始处理100个benchmark...")
print("-" * 80)

# 主循环：遍历100个基准场景
for benchid in range(0, 100):
    benchidstr = str(benchid)

    # 加载数据
    if density_level == "low":
        f = open(
            f"../trace_generation/scene_benchmarks/dens6_rs/obstacles_{benchidstr}_coord.pkl",
            "rb",
        )
    elif density_level == "mid":
        f = open(
            f"../trace_generation/scene_benchmarks/dens9_rs/obstacles_{benchidstr}_coord.pkl",
            "rb",
        )
    else:
        f = open(
            f"../trace_generation/scene_benchmarks/dens12_rs/obstacles_{benchidstr}_coord.pkl",
            "rb",
        )

    xtest_pred, dirr_pred, label_pred = pickle.load(f)
    f.close()

    # 对坐标进行量化
    code_pred_quant = np.digitize(xtest_pred, bins, right=True)

    # 数据总量
    total_samples = len(code_pred_quant)

    # 按group_size分组后的总组数
    num_groups = (total_samples + num_links - 1) // num_links
    half_groups = num_groups // 2

    # 前半部分的索引范围
    first_half_end_idx = half_groups * num_links

    # 分割数据
    code_first_half = code_pred_quant[:first_half_end_idx]
    label_first_half = label_pred[:first_half_end_idx]

    code_second_half = code_pred_quant[first_half_end_idx:]
    label_second_half = label_pred[first_half_end_idx:]

    if benchid % 20 == 0:
        print(
            f"Benchmark {benchid}: 总样本={total_samples}, 前半={len(code_first_half)}, 后半={len(code_second_half)}"
        )

    # ========== 阶段1: 在前半部分训练CHT ==========
    strategy_train = FixedThresholdStrategy(
        threshold=collision_threshold,
        update_prob=free_sample_rate,
        max_count=255,
    )

    # 在前半部分训练
    evaluate_strategy_on_trajectory(
        strategy_train, code_first_half, label_first_half, group_size=num_links
    )

    # 累积训练阶段的统计
    total_zerozero_train += strategy_train.all_zerozero
    total_onezero_train += strategy_train.all_onezero
    total_colliding_train += strategy_train.all_total_colliding
    total_cht_size_after_train += len(strategy_train.colldict)

    # ========== 阶段2: 策略1 - 清零CHT,在后半部分从零开始 ==========
    strategy_reset = FixedThresholdStrategy(
        threshold=collision_threshold,
        update_prob=free_sample_rate,
        max_count=255,
    )
    # 不继承任何历史,从零开始

    evaluate_strategy_on_trajectory(
        strategy_reset, code_second_half, label_second_half, group_size=num_links
    )

    # 累积策略1的统计
    total_zerozero_reset += strategy_reset.all_zerozero
    total_onezero_reset += strategy_reset.all_onezero
    total_colliding_reset += strategy_reset.all_total_colliding
    total_cht_size_after_reset += len(strategy_reset.colldict)

    # ========== 阶段3: 策略2 - 继承CHT,在后半部分继续学习 ==========
    strategy_inherit = FixedThresholdStrategy(
        threshold=collision_threshold,
        update_prob=free_sample_rate,
        max_count=255,
    )
    # 继承前半部分训练的CHT
    strategy_inherit.inherit_collision_history(strategy_train, rate=1.0)

    evaluate_strategy_on_trajectory(
        strategy_inherit, code_second_half, label_second_half, group_size=num_links
    )

    # 累积策略2的统计
    total_zerozero_inherit += strategy_inherit.all_zerozero
    total_onezero_inherit += strategy_inherit.all_onezero
    total_colliding_inherit += strategy_inherit.all_total_colliding
    total_cht_size_after_inherit += len(strategy_inherit.colldict)

print("\n处理完成!")
print("=" * 80)

# 计算最终指标
# 训练阶段
precision_train = (
    (total_zerozero_train * 100.0 / (total_zerozero_train + total_onezero_train))
    if (total_zerozero_train + total_onezero_train) > 0
    else 0.0
)
recall_train = (
    (total_zerozero_train * 100.0 / total_colliding_train)
    if total_colliding_train > 0
    else 0.0
)

# 策略1: 清零重启
precision_reset = (
    (total_zerozero_reset * 100.0 / (total_zerozero_reset + total_onezero_reset))
    if (total_zerozero_reset + total_onezero_reset) > 0
    else 0.0
)
recall_reset = (
    (total_zerozero_reset * 100.0 / total_colliding_reset)
    if total_colliding_reset > 0
    else 0.0
)

# 策略2: 继承CHT
precision_inherit = (
    (total_zerozero_inherit * 100.0 / (total_zerozero_inherit + total_onezero_inherit))
    if (total_zerozero_inherit + total_onezero_inherit) > 0
    else 0.0
)
recall_inherit = (
    (total_zerozero_inherit * 100.0 / total_colliding_inherit)
    if total_colliding_inherit > 0
    else 0.0
)

# 输出结果
print("\n【结果总结】")
print("=" * 80)

print("\n1. 训练阶段 (前半部分数据):")
print(f"   精确率: {precision_train:.2f}%")
print(f"   召回率: {recall_train:.2f}%")
print(
    f"   TP={total_zerozero_train}, FP={total_onezero_train}, Total_Collisions={total_colliding_train}"
)
print(f"   平均CHT大小: {total_cht_size_after_train / 100:.1f}")

print("\n2. 策略1: 清零CHT从零开始 (后半部分数据):")
print(f"   精确率: {precision_reset:.2f}%")
print(f"   召回率: {recall_reset:.2f}%")
print(
    f"   TP={total_zerozero_reset}, FP={total_onezero_reset}, Total_Collisions={total_colliding_reset}"
)
print(f"   平均CHT大小: {total_cht_size_after_reset / 100:.1f}")

print("\n3. 策略2: 继承CHT (后半部分数据):")
print(f"   精确率: {precision_inherit:.2f}%")
print(f"   召回率: {recall_inherit:.2f}%")
print(
    f"   TP={total_zerozero_inherit}, FP={total_onezero_inherit}, Total_Collisions={total_colliding_inherit}"
)
print(f"   平均CHT大小: {total_cht_size_after_inherit / 100:.1f}")

print("\n【性能对比】")
print("=" * 80)

# 召回率提升
recall_improvement = recall_inherit - recall_reset
recall_improvement_pct = (
    (recall_improvement / recall_reset * 100) if recall_reset > 0 else 0
)

print("\n召回率 (Recall):")
print(f"  清零重启: {recall_reset:.2f}%")
print(f"  继承CHT:   {recall_inherit:.2f}%")
print(
    f"  提升:      {recall_improvement:+.2f}% (相对提升: {recall_improvement_pct:+.1f}%)"
)

# 精确率变化
precision_change = precision_inherit - precision_reset
precision_change_pct = (
    (precision_change / precision_reset * 100) if precision_reset > 0 else 0
)

print("\n精确率 (Precision):")
print(f"  清零重启: {precision_reset:.2f}%")
print(f"  继承CHT:   {precision_inherit:.2f}%")
print(f"  变化:      {precision_change:+.2f}% (相对变化: {precision_change_pct:+.1f}%)")

# F1分数
f1_reset = (
    (2 * precision_reset * recall_reset / (precision_reset + recall_reset))
    if (precision_reset + recall_reset) > 0
    else 0.0
)
f1_inherit = (
    (2 * precision_inherit * recall_inherit / (precision_inherit + recall_inherit))
    if (precision_inherit + recall_inherit) > 0
    else 0.0
)
f1_improvement = f1_inherit - f1_reset

print("\nF1分数:")
print(f"  清零重启: {f1_reset:.2f}")
print(f"  继承CHT:   {f1_inherit:.2f}")
print(f"  提升:      {f1_improvement:+.2f}")

# CHT大小对比
cht_size_change = (total_cht_size_after_inherit - total_cht_size_after_reset) / 100

print("\n平均CHT大小:")
print(f"  训练后:    {total_cht_size_after_train / 100:.1f}")
print(f"  清零重启:  {total_cht_size_after_reset / 100:.1f}")
print(f"  继承CHT:   {total_cht_size_after_inherit / 100:.1f}")
print(f"  差异:      {cht_size_change:+.1f}")

print("\n【结论】")
print("=" * 80)

if recall_improvement > 5:
    print("✓ 继承CHT显著提升召回率 (>5%)")
elif recall_improvement > 2:
    print("✓ 继承CHT明显提升召回率 (2-5%)")
elif recall_improvement > 0:
    print("• 继承CHT略微提升召回率 (<2%)")
else:
    print("✗ 继承CHT未能提升召回率")

if precision_change > -2:
    print("✓ 继承CHT基本维持精确率")
elif precision_change > -5:
    print("• 继承CHT轻微降低精确率 (2-5%)")
else:
    print("⚠ 继承CHT明显降低精确率 (>5%)")

if f1_improvement > 3:
    print("✓ 整体性能(F1)显著提升,强烈推荐使用继承策略")
elif f1_improvement > 1:
    print("✓ 整体性能(F1)有所提升,建议使用继承策略")
elif f1_improvement > 0:
    print("• 整体性能(F1)略有提升")
else:
    print("• 整体性能(F1)未见提升")

print("\n推荐:")
if recall_improvement > 3 and precision_change > -3:
    print("  在相同场景的连续任务中,强烈建议使用CHT继承策略(rate=1.0)")
    print("  可以显著提升性能,同时避免冷启动问题")
elif recall_improvement > 1:
    print("  在相同场景的连续任务中,建议使用CHT继承策略")
    print("  可以提升召回率,改善整体性能")
else:
    print("  在当前参数设置下,CHT继承效果有限")
    print("  可以尝试调整阈值和采样率参数")

print("\n" + "=" * 80)
print(
    f"最终输出: {density_level}, {quantize_bits}, {collision_threshold}, {free_sample_rate}, "
    f"Reset: {precision_reset:.2f}%/{recall_reset:.2f}%, "
    f"Inherit: {precision_inherit:.2f}%/{recall_inherit:.2f}%, "
    f"Improvement: {recall_improvement:+.2f}%"
)
print("=" * 80)
