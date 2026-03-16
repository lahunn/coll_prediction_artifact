import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib

# 1. 设置绘图风格（白底、带刻度）
sns.set_style("ticks") 

# 2. 设置调色板（推荐色盲友好型）
sns.set_palette("colorblind")

# 3. 设置上下文（自动调整线条粗细和字体大小，'paper' 适合论文）
sns.set_context("paper", font_scale=1.5)

# 确保在PDF和PS文件中正确嵌入字体
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def plot_comparison(model_type):
    # 文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "../result_files")

    # 根据模型类型构建文件名
    # Single Buffer: link_results.csv 或 sphere_results.csv
    # Double Buffer: double_buffer_link_results.csv 或 double_buffer_sphere_results.csv
    single_buffer_file = os.path.join(results_dir, f"{model_type}_results.csv")
    double_buffer_file = os.path.join(
        results_dir, f"double_buffer_{model_type}_results.csv"
    )

    print(f"Processing {model_type} model...")

    # 检查文件是否存在
    if not os.path.exists(single_buffer_file):
        print(f"Error: File not found: {single_buffer_file}")
        return
    if not os.path.exists(double_buffer_file):
        print(f"Error: File not found: {double_buffer_file}")
        return

    # 读取数据
    try:
        df_sb = pd.read_csv(single_buffer_file)
        df_db = pd.read_csv(double_buffer_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # 确保Scene列存在且为字符串类型，方便排序
    if "Scene" not in df_sb.columns or "Scene" not in df_db.columns:
        print("Error: 'Scene' column missing in one of the CSV files.")
        print(f"Single Buffer columns: {df_sb.columns}")
        print(f"Double Buffer columns: {df_db.columns}")
        return

    df_sb["Scene"] = df_sb["Scene"].astype(str)
    df_db["Scene"] = df_db["Scene"].astype(str)

    # 过滤出 G1-G5
    scenes = ["G1", "G2", "G3", "G4", "G5"]

    # 筛选并去重（如果同一场景有多行，取最后一行或平均，这里假设取最后一行作为最新结果）
    df_sb = (
        df_sb[df_sb["Scene"].isin(scenes)]
        .drop_duplicates(subset=["Scene"], keep="last")
        .set_index("Scene")
    )
    df_db = (
        df_db[df_db["Scene"].isin(scenes)]
        .drop_duplicates(subset=["Scene"], keep="last")
        .set_index("Scene")
    )

    # 重新索引以确保顺序和完整性
    df_sb = df_sb.reindex(scenes)
    df_db = df_db.reindex(scenes)

    # 检查是否有缺失数据
    if df_sb.isnull().all().all() or df_db.isnull().all().all():
        print(
            f"Warning: One of the datasets for {model_type} is empty or missing G1-G5 scenes."
        )

    # 准备绘图数据
    x = np.arange(len(scenes))
    width = 0.35

    # 获取 Seaborn 颜色
    colors = sns.color_palette()
    sb_color = colors[0]
    db_color = colors[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    model_label = "OBB" if model_type == "link" else "Sphere"

    # Plot 1: Total Prediction Queries
    # 使用 fillna(0) 处理可能的缺失值
    vals_sb_queries = df_sb["Total_Pred_Queries"].fillna(0)
    vals_db_queries = df_db["Total_Pred_Queries"].fillna(0)

    rects1 = ax1.bar(
        x - width / 2,
        vals_sb_queries,
        width,
        label=f"Single Buffer",
        color=sb_color,
        edgecolor='white'
    )
    rects2 = ax1.bar(
        x + width / 2, vals_db_queries, width, label="Double Buffer", color=db_color,
        edgecolor='white'
    )

    ax1.set_ylabel("Total Prediction Queries")
    ax1.set_title(f"Total Prediction Queries Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenes)
    ax1.legend()
    # grid removed per project style

    # 添加数值标签
    ax1.bar_label(rects1, padding=3, fmt="%.0f", fontsize=8)
    ax1.bar_label(rects2, padding=3, fmt="%.0f", fontsize=8)

    # Plot 2: Total Cycles
    vals_sb_cycles = df_sb["Total_Cycles"].fillna(0)
    vals_db_cycles = df_db["Total_Cycles"].fillna(0)

    rects3 = ax2.bar(
        x - width / 2,
        vals_sb_cycles,
        width,
        label=f"Single Buffer",
        color=sb_color,
        edgecolor='white'
    )
    rects4 = ax2.bar(
        x + width / 2, vals_db_cycles, width, label="Double Buffer", color=db_color,
        edgecolor='white'
    )

    ax2.set_ylabel("Total Cycles")
    ax2.set_title(f"Total Cycles Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenes)
    ax2.legend()
    # grid removed per project style

    # 添加数值标签
    ax2.bar_label(rects3, padding=3, fmt="%.0f", fontsize=8)

    # 计算差异比例并作为标签
    labels = []
    for sb, db in zip(vals_sb_cycles, vals_db_cycles):
        if sb != 0:
            diff = (db - sb) / sb * 100
            labels.append(f"{db:.0f}\n({diff:+.1f}%)")
        else:
            labels.append(f"{db:.0f}")

    ax2.bar_label(rects4, labels=labels, padding=3, fontsize=8)

    # 移除顶部和右侧边框
    sns.despine()

    plt.tight_layout()

    output_path = os.path.join(base_dir, f"figs/buffer_comparison_{model_type}.png")
    # 确保figs目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # 遍历两种模型
    for model in ["link", "sphere"]:
        plot_comparison(model)
