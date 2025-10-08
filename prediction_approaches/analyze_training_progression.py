"""
分析训练过程中准确率和召回率的变化趋势
统计在不同数据量下的性能表现:
- 前1000个数据
- 1000~2000个数据
- 2000~3000个数据
- ...

支持两种数据类型:
1. 机器人轨迹数据 (coord_hashing)
2. 球体数据 (coord_hashing_sphere)
"""

import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from collision_prediction_strategies import (
    FixedThresholdStrategy,
    generate_hash_key,
    generate_sphere_hash_key,
)

# 配置matplotlib以支持中文显示
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 解析命令行参数
if len(sys.argv) < 6:
    print(
        "用法: python analyze_training_progression.py <数据类型> <密度等级> <量化参数> <碰撞阈值> <自由样本采样率> [benchmark_id] [步长]"
    )
    print("\n数据类型:")
    print("  trajectory - 机器人轨迹数据")
    print("  sphere - 球体数据")
    print("\n示例:")
    print("  python analyze_training_progression.py trajectory mid 8 0.1 0.3")
    print("  python analyze_training_progression.py trajectory mid 8 0.1 0.3 0")
    print("  python analyze_training_progression.py trajectory mid 8 0.1 0.3 0 500")
    print("  python analyze_training_progression.py sphere mid 8,6 0.1 0.3")
    print("  python analyze_training_progression.py sphere mid 8,6 0.1 0.3 0 2000")
    sys.exit(1)

data_type = sys.argv[1]  # 数据类型: "trajectory" 或 "sphere"
density_level = sys.argv[2]  # 密度等级: "low", "mid", "high"
quantize_param = sys.argv[3]  # 量化参数 (trajectory: "8", sphere: "8,6")
collision_threshold = float(sys.argv[4])  # 碰撞阈值
free_sample_rate = float(sys.argv[5])  # 自由样本采样率

# 解析可选参数: benchmark_id 和 step_size
benchmark_id = 0  # 默认使用第0个benchmark
step_size = 1000  # 默认步长1000

if len(sys.argv) > 6:
    try:
        benchmark_id = int(sys.argv[6])
        if len(sys.argv) > 7:
            step_size = int(sys.argv[7])
    except ValueError:
        # 如果第6个参数不是数字，可能是旧版本调用方式（直接是步长）
        step_size = int(sys.argv[6])

print("=" * 80)
print("训练过程分析 - 准确率和召回率变化趋势")
print("=" * 80)
print("\n参数设置:")
print(f"  数据类型: {data_type}")
print(f"  密度级别: {density_level}")
print(f"  量化参数: {quantize_param}")
print(f"  碰撞阈值: {collision_threshold}")
print(f"  自由样本采样率: {free_sample_rate}")
print(f"  Benchmark ID: {benchmark_id}")
print(f"  数据步长: {step_size}")

# 根据数据类型解析量化参数
if data_type == "trajectory":
    quantize_bits = int(quantize_param)
    num_links = 11
    print(f"  关节数量: {num_links}")
elif data_type == "sphere":
    coord_bits, radius_bits = map(int, quantize_param.split(","))
    print(f"  坐标量化位数: {coord_bits}")
    print(f"  半径量化位数: {radius_bits}")
else:
    print(f"错误: 不支持的数据类型 '{data_type}'")
    sys.exit(1)

# 选择数据路径
if density_level == "low":
    data_prefix = "../trace_generation/scene_benchmarks/dens6_rs/obstacles_"
elif density_level == "mid":
    data_prefix = "../trace_generation/scene_benchmarks/dens9_rs/obstacles_"
else:
    data_prefix = "../trace_generation/scene_benchmarks/dens12_rs/obstacles_"

# 数据后缀
if data_type == "trajectory":
    data_suffix = "_coord.pkl"
else:
    data_suffix = "_sphere.pkl"


def create_bins_for_range(min_val, max_val, num_bins):
    """创建等间距的分桶边界"""
    margin = (max_val - min_val) * 0.01
    return np.linspace(min_val - margin, max_val + margin, num_bins + 1)[:-1]


# ========== 数据预处理 ==========
if data_type == "trajectory":
    # 轨迹数据的量化设置
    binnumber = 2**quantize_bits
    intervalsize = 2.24 / binnumber
    bins = np.zeros(binnumber)
    start = -1.12
    for i in range(binnumber):
        bins[i] = start
        start += intervalsize

else:  # sphere
    # 球体数据：仅加载指定的benchmark来确定数据范围
    print(f"\n正在加载benchmark {benchmark_id}的球体数据...")
    f = open(f"{data_prefix}{benchmark_id}{data_suffix}", "rb")
    all_positions, all_radii_temp, _ = pickle.load(f)
    f.close()
    all_radii = all_radii_temp.flatten()

    x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
    y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
    z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])
    r_min, r_max = np.min(all_radii), np.max(all_radii)

    binnumber_coord = 2**coord_bits
    binnumber_radius = 2**radius_bits

    x_bins = create_bins_for_range(x_min, x_max, binnumber_coord)
    y_bins = create_bins_for_range(y_min, y_max, binnumber_coord)
    z_bins = create_bins_for_range(z_min, z_max, binnumber_coord)
    r_bins = create_bins_for_range(r_min, r_max, binnumber_radius)

    consider_radius = False  # 默认仅使用位置
    print(
        f"数据范围: X[{x_min:.3f},{x_max:.3f}] Y[{y_min:.3f},{y_max:.3f}] Z[{z_min:.3f},{z_max:.3f}] R[{r_min:.3f},{r_max:.3f}]"
    )


# ========== 加载单个benchmark数据 ==========
print(f"\n正在加载benchmark {benchmark_id}的数据...")

f = open(f"{data_prefix}{benchmark_id}{data_suffix}", "rb")

if data_type == "trajectory":
    xtest, dirr, label = pickle.load(f)
    f.close()

    # 量化
    all_codes = np.digitize(xtest, bins, right=True)
    all_labels = label
    all_radii_quant = None

else:  # sphere
    qarr_sphere, rarr_sphere, yarr_sphere = pickle.load(f)
    f.close()

    # 量化位置
    all_codes = np.zeros_like(qarr_sphere, dtype=int)
    all_codes[:, 0] = np.digitize(qarr_sphere[:, 0], x_bins, right=True)
    all_codes[:, 1] = np.digitize(qarr_sphere[:, 1], y_bins, right=True)
    all_codes[:, 2] = np.digitize(qarr_sphere[:, 2], z_bins, right=True)

    # 量化半径
    all_radii_quant = np.digitize(rarr_sphere.flatten(), r_bins, right=True)
    all_labels = yarr_sphere.flatten()

total_samples = len(all_codes)
print(f"总样本数: {total_samples}")

# 计算需要评估的区间数
num_intervals = (total_samples + step_size - 1) // step_size
print(f"将划分为 {num_intervals} 个区间 (每个区间 {step_size} 个样本)")


# ========== 增量训练和评估 ==========
print("\n" + "=" * 80)
print("开始增量训练和评估...")
print("=" * 80)

results = []

for interval_idx in range(num_intervals):
    start_idx = interval_idx * step_size
    end_idx = min((interval_idx + 1) * step_size, total_samples)

    # 创建新策略（从头开始累积训练）
    strategy = FixedThresholdStrategy(
        threshold=collision_threshold,
        update_prob=free_sample_rate,
        max_count=255,
    )

    # 统计变量
    tp = 0  # True Positives
    fp = 0  # False Positives
    total_collisions = 0

    # 使用前0~end_idx的数据进行训练和评估
    if data_type == "trajectory":
        # 轨迹数据：按group_size处理
        for i in range(0, end_idx, num_links):
            predicted = 1  # 默认预测为非碰撞
            true_ans = 1  # 默认真实为非碰撞

            # 检查一组样本
            for j in range(i, min(i + num_links, end_idx)):
                keyy = generate_hash_key(all_codes[j], len(all_codes[j]))

                # 预测
                if strategy.predict_collision(keyy):
                    predicted = 0

                # 更新历史
                strategy.update_history(keyy, all_labels[j])

                # 检查真实标签
                if all_labels[j] < 0.5:
                    true_ans = 0
                    if predicted == 0:
                        break

            # 统计
            if true_ans < 0.5:  # 真实为碰撞
                total_collisions += 1
                if predicted == 0:  # 预测也为碰撞
                    tp += 1
            elif predicted == 0:  # 真实为自由但预测为碰撞
                fp += 1

    else:  # sphere
        # 球体数据：逐样本处理
        for i in range(end_idx):
            keyy = generate_sphere_hash_key(
                all_codes[i],
                all_radii_quant[i] if consider_radius else None,
                consider_radius=consider_radius,
            )

            # 预测
            predicted = 1 if not strategy.predict_collision(keyy) else 0

            # 更新历史
            strategy.update_history(keyy, all_labels[i])

            # 统计
            true_ans = 0 if all_labels[i] < 0.5 else 1
            if true_ans == 0:  # 真实为碰撞
                total_collisions += 1
                if predicted == 0:  # 预测也为碰撞
                    tp += 1
            elif predicted == 0:  # 真实为自由但预测为碰撞
                fp += 1

    # 计算指标
    precision = (tp * 100.0 / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp * 100.0 / total_collisions) if total_collisions > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    cht_size = len(strategy.colldict)

    results.append(
        {
            "interval": interval_idx + 1,
            "start": start_idx,
            "end": end_idx,
            "samples": end_idx,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "total_collisions": total_collisions,
            "cht_size": cht_size,
        }
    )

    if interval_idx % 5 == 0 or interval_idx == num_intervals - 1:
        print(
            f"区间 {interval_idx + 1}/{num_intervals}: 样本[0-{end_idx}] "
            f"精确率={precision:.2f}% 召回率={recall:.2f}% F1={f1:.2f} CHT大小={cht_size}"
        )


# ========== 输出结果 ==========
print("\n" + "=" * 80)
print("详细结果")
print("=" * 80)
print(
    f"{'区间':<6} {'样本范围':<20} {'累积样本':<10} {'精确率%':<10} {'召回率%':<10} {'F1':<8} {'CHT大小':<10}"
)
print("-" * 80)

for r in results:
    print(
        f"{r['interval']:<6} "
        f"[{r['start']:>6} - {r['end']:>6}]    "
        f"{r['samples']:<10} "
        f"{r['precision']:<10.2f} "
        f"{r['recall']:<10.2f} "
        f"{r['f1']:<8.2f} "
        f"{r['cht_size']:<10}"
    )

# ========== 绘制趋势图 ==========
print("\n" + "=" * 80)
print("生成趋势图...")
print("=" * 80)

samples = [r["samples"] for r in results]
precisions = [r["precision"] for r in results]
recalls = [r["recall"] for r in results]
f1_scores = [r["f1"] for r in results]
cht_sizes = [r["cht_size"] for r in results]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Training Progression Analysis - {data_type} ({density_level}, {quantize_param}, t={collision_threshold}, r={free_sample_rate})",
    fontsize=14,
)

# Subplot 1: Precision
axes[0, 0].plot(samples, precisions, "b-o", linewidth=2, markersize=4)
axes[0, 0].set_xlabel("Cumulative Samples", fontsize=11)
axes[0, 0].set_ylabel("Precision (%)", fontsize=11)
axes[0, 0].set_title("Precision vs Data Size", fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 100])

# Subplot 2: Recall
axes[0, 1].plot(samples, recalls, "r-s", linewidth=2, markersize=4)
axes[0, 1].set_xlabel("Cumulative Samples", fontsize=11)
axes[0, 1].set_ylabel("Recall (%)", fontsize=11)
axes[0, 1].set_title("Recall vs Data Size", fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 100])

# Subplot 3: F1 Score
axes[1, 0].plot(samples, f1_scores, "g-^", linewidth=2, markersize=4)
axes[1, 0].set_xlabel("Cumulative Samples", fontsize=11)
axes[1, 0].set_ylabel("F1 Score", fontsize=11)
axes[1, 0].set_title("F1 Score vs Data Size", fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: CHT Size
axes[1, 1].plot(samples, cht_sizes, "m-d", linewidth=2, markersize=4)
axes[1, 1].set_xlabel("Cumulative Samples", fontsize=11)
axes[1, 1].set_ylabel("CHT Size (entries)", fontsize=11)
axes[1, 1].set_title("CHT Size vs Data Size", fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
output_filename = f"training_progression_{data_type}_{density_level}_{quantize_param.replace(',', '_')}_t{collision_threshold}_r{free_sample_rate}_step{step_size}.png"
plt.savefig(output_filename, dpi=150, bbox_inches="tight")
print(f"\n趋势图已保存到: {output_filename}")

# ========== 统计摘要 ==========
print("\n" + "=" * 80)
print("统计摘要")
print("=" * 80)

initial_precision = results[0]["precision"]
final_precision = results[-1]["precision"]
precision_change = final_precision - initial_precision

initial_recall = results[0]["recall"]
final_recall = results[-1]["recall"]
recall_change = final_recall - initial_recall

initial_f1 = results[0]["f1"]
final_f1 = results[-1]["f1"]
f1_change = final_f1 - initial_f1

print("\n精确率:")
print(f"  初始 (前{step_size}样本): {initial_precision:.2f}%")
print(f"  最终 (全部{total_samples}样本): {final_precision:.2f}%")
print(f"  变化: {precision_change:+.2f}%")

print("\n召回率:")
print(f"  初始 (前{step_size}样本): {initial_recall:.2f}%")
print(f"  最终 (全部{total_samples}样本): {final_recall:.2f}%")
print(f"  变化: {recall_change:+.2f}%")

print("\nF1分数:")
print(f"  初始 (前{step_size}样本): {initial_f1:.2f}")
print(f"  最终 (全部{total_samples}样本): {final_f1:.2f}")
print(f"  变化: {f1_change:+.2f}")

print("\nCHT大小:")
print(f"  初始: {results[0]['cht_size']}")
print(f"  最终: {results[-1]['cht_size']}")
print(
    f"  增长: {results[-1]['cht_size'] - results[0]['cht_size']} (+{(results[-1]['cht_size'] / results[0]['cht_size'] - 1) * 100:.1f}%)"
)

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)
