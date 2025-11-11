#!/usr/bin/env python3
"""
Analyze prediction accuracy changes with training data size

Read accuracy curve data from multiple benchmarks and analyze learning curve characteristics
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def load_accuracy_data(csv_file):
    """Load accuracy curve data"""
    if not os.path.exists(csv_file):
        print(f"File {csv_file} does not exist")
        return None

    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error reading file {csv_file}: {e}")
        return None


def aggregate_accuracy_curves(
    df, groupby_cols=["threshold", "sample_rate", "qnoncoll_multiplier"]
):
    """
    Aggregate accuracy curves for different configurations

    Args:
        df: Accuracy data DataFrame
        groupby_cols: Grouping columns

    Returns:
        dict: Configuration -> (training size list, accuracy list, standard deviation list)
    """
    results = {}

    for group_key, group_df in df.groupby(groupby_cols):
        # 按训练大小排序
        sorted_df = group_df.sort_values("training_size")

        # 计算每个训练大小的平均准确率和标准差
        size_stats = []
        for size in sorted_df["training_size"].unique():
            size_data = sorted_df[sorted_df["training_size"] == size]["accuracy"]
            mean_acc = size_data.mean()
            std_acc = size_data.std()
            size_stats.append((size, mean_acc, std_acc))

        training_sizes, mean_accuracies, std_accuracies = zip(*size_stats)
        results[group_key] = (training_sizes, mean_accuracies, std_accuracies)

    return results


def plot_learning_curves(results, save_path=None):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 8))

    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    color_idx = 0

    for config, (sizes, accuracies, stds) in results.items():
        threshold, sample_rate, qnoncoll_multiplier = config

        label = f"Threshold={threshold}, Sample Rate={sample_rate}, Queue Multiplier={qnoncoll_multiplier}"
        color = colors[color_idx % len(colors)]

        plt.plot(sizes, accuracies, label=label, color=color, linewidth=2)

        # 添加误差条
        if stds and any(stds):
            plt.fill_between(
                sizes,
                np.array(accuracies) - np.array(stds),
                np.array(accuracies) + np.array(stds),
                alpha=0.3,
                color=color,
            )

        color_idx += 1

    plt.xlabel("Training Data Size (History Dictionary Size)")
    plt.ylabel("Prediction Accuracy")
    plt.title("Learning Curve of Collision Prediction Accuracy vs Training Data Size")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning curve plot saved to: {save_path}")

    plt.show()


def analyze_convergence(results):
    """Analyze convergence characteristics"""
    print("\n=== Convergence Analysis ===")

    for config, (sizes, accuracies, stds) in results.items():
        threshold, sample_rate, qnoncoll_multiplier = config

        if len(accuracies) < 2:
            continue

        # Calculate convergence rate (accuracy change rate of last 10% data)
        final_accuracies = accuracies[-max(1, len(accuracies) // 10) :]
        convergence_rate = (
            np.mean(np.diff(final_accuracies)) if len(final_accuracies) > 1 else 0
        )

        # Calculate final accuracy
        final_accuracy = accuracies[-1]

        # Calculate training data size needed to reach 90% of final accuracy
        target_accuracy = 0.9 * final_accuracy
        reached_target_idx = None
        for i, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                reached_target_idx = i
                break

        training_size_for_target = (
            sizes[reached_target_idx]
            if reached_target_idx is not None
            else float("inf")
        )

        print(
            f"\nConfiguration (Threshold={threshold}, Sample Rate={sample_rate}, Queue Multiplier={qnoncoll_multiplier}):"
        )
        print(f"  Final Accuracy: {final_accuracy:.4f}")
        print(f"  Convergence Rate: {convergence_rate:.6f}")
        print(
            f"  Training Data Size for 90% Final Accuracy: {training_size_for_target}"
        )


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python analyze_accuracy_vs_training_size.py <accuracy_csv_file> [output_plot_file]"
        )
        print(
            "Example: python analyze_accuracy_vs_training_size.py result_files/sphere_accuracy_curve.csv plots/learning_curve.png"
        )
        sys.exit(1)

    csv_file = sys.argv[1]
    output_plot = sys.argv[2] if len(sys.argv) > 2 else None

    # 加载数据
    print(f"Loading accuracy data: {csv_file}")
    df = load_accuracy_data(csv_file)

    if df is None or df.empty:
        print("No valid data found")
        return

    print(f"Loaded {len(df)} accuracy records")
    print(
        f"Number of configurations: {len(df.groupby(['threshold', 'sample_rate', 'qnoncoll_multiplier']))}"
    )

    # 聚合数据
    results = aggregate_accuracy_curves(df)

    # 绘制学习曲线
    plot_learning_curves(results, output_plot)

    # 分析收敛特性
    analyze_convergence(results)

    print(f"\nAnalysis completed, processed {len(results)} different configurations")


if __name__ == "__main__":
    main()
