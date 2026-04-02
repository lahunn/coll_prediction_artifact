import math
import sys

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.ticker import FuncFormatter
import os
# --- 统一绘图风格配置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")


# 1. 设置绘图风格（白底、带刻度）
sns.set_style("ticks") 

# 2. 设置调色板（推荐色盲友好型）
sns.set_palette("colorblind")

# 3. 设置上下文（自动调整线条粗细和字体大小，'paper' 适合论文）
sns.set_context("paper", font_scale=1.5)

# 确保在PDF和PS文件中正确嵌入字体
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = 'SimSun'


def build_result_files(collision_type, include_multi_bank=True):
    files = {
        "共享双端口SRAM（预测通道数=1）": f"dual_port_pred1_{collision_type}_results.csv",
        "共享双端口SRAM（预测通道数=2）": f"dual_port_pred2_{collision_type}_results.csv",
    }
    if include_multi_bank:
        files.update(
            {
                "分布式多Bank SRAM（预测通道数=1）": f"multi_bank_pred1_{collision_type}_results.csv",
                "分布式多Bank SRAM（预测通道数=2）": f"multi_bank_pred2_{collision_type}_results.csv",
            }
        )
    return files


def plot_cycle_comparison(collision_type="link"):
    # 数据文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result_files")
    
    files = build_result_files(collision_type)
    
    data = {}
    scenes = None

    print("正在读取数据文件...")
    # 读取数据
    for label, filename in files.items():
        filepath = os.path.join(result_dir, filename)
        if not os.path.exists(filepath):
            print(f"警告：未找到文件: {filepath}")
            continue

        try:
            df = pd.read_csv(filepath)
            df["Scene_Num"] = df["Scene"].str.extract(r"(\d+)").astype(int)
            df = df.sort_values("Scene_Num")

            current_scenes = df["Scene"].tolist()
            if scenes is None:
                scenes = current_scenes
            
            data[label] = df["Total_Cycles"].tolist()
            print(f"已加载 {label}: {len(df)} 条记录")

        except Exception as e:
            print(f"读取 {filename} 时出错: {e}")

    if not data or scenes is None:
        print("未加载到数据或场景，退出。")
        return

    x = np.arange(len(scenes))
    num_strategies = len(data)
    width = 0.18  # 柱状图宽度

    # 从 Seaborn 调色板获取颜色
    palette = sns.color_palette()
    # 映射策略到颜色：共享双端口SRAM使用红色系，分布式多Bank SRAM使用绿色系
    colors_map = {
        "共享双端口SRAM（预测通道数=1）": palette[3],
        "共享双端口SRAM（预测通道数=2）": sns.light_palette(palette[3], n_colors=3)[1],
        "分布式多Bank SRAM（预测通道数=1）": palette[2],
        "分布式多Bank SRAM（预测通道数=2）": sns.light_palette(palette[2], n_colors=3)[1],
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    # 绘制柱状图
    for i, (strategy_name, cycles) in enumerate(data.items()):
        offset = width * (i - num_strategies / 2 + 0.5)
        ax.bar(x + offset, cycles, width, label=strategy_name,
               color=colors_map.get(strategy_name, palette[7]), 
               edgecolor='black', linewidth=1, alpha=0.85)

    # 标注共享双端口SRAM（预测通道数=1）与分布式多Bank SRAM（预测通道数=2）之间的差距
    if "共享双端口SRAM（预测通道数=1）" in data and "分布式多Bank SRAM（预测通道数=2）" in data:
        dp1 = data["共享双端口SRAM（预测通道数=1）"]
        mb2 = data["分布式多Bank SRAM（预测通道数=2）"]

        for idx in range(len(scenes)):
            reduction_pct = (dp1[idx] - mb2[idx]) / dp1[idx] * 100 if dp1[idx] > 0 else 0
            if reduction_pct > 0:
                ax.text(idx + width * 0.5, max(dp1[idx], mb2[idx]) * 1.05, 
                        f"-{reduction_pct:.1f}%", ha='center', va='bottom', 
                        fontsize=10, color='darkgreen', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkgreen", alpha=0.8))

    # 设置标签和标题
    ax.set_ylabel("总周期数")
    ax.set_xlabel("场景")
    ax.set_title(f"多COPU策略对比（{collision_type}）：各场景周期数")

    # 设置X轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    
    # grid removed per project style
    sns.despine()

    # 图例
    ax.legend(loc="upper left", ncol=2, frameon=True)

    plt.tight_layout()
    output_path = os.path.join(
        current_dir, f"figs/cycle_comparison_strategies_{collision_type}.pdf"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"图表已保存至 {output_path}")


def plot_prediction_impact_comparison(collision_type="link"):
    """对比共享双端口SRAM（预测通道数=1）和共享双端口SRAM（预测通道数=2）的性能差异"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result_files")

    files = build_result_files(collision_type, include_multi_bank=False)

    cycles_data = {}
    queries_data = {}
    scenes = None

    for label, filename in files.items():
        filepath = os.path.join(result_dir, filename)
        if not os.path.exists(filepath):
            continue
        try:
            df = pd.read_csv(filepath)
            df["Scene_Num"] = df["Scene"].str.extract(r"(\d+)").astype(int)
            df = df.sort_values("Scene_Num")
            if scenes is None:
                scenes = df["Scene"].tolist()
            cycles_data[label] = df["Total_Cycles"].tolist()
            queries_data[label] = df["Total_Queries"].tolist()
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not cycles_data or scenes is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(scenes))
    width = 0.35
    palette = sns.color_palette()
    colors = [palette[0], palette[1]]

    for ax, data_dict, title, ylabel in zip([ax1, ax2], [cycles_data, queries_data], 
                                            ["总周期数", "总查询数"], ["周期数", "查询数"]):
        for i, (label, vals) in enumerate(data_dict.items()):
            ax.bar(x + (i-0.5)*width, vals, width, label=label, color=colors[i], edgecolor='black', alpha=0.8)
        
        ax.set_title(f"{title}（{collision_type}）")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
        ax.legend()
        sns.despine(ax=ax)

    plt.tight_layout()
    output_path = os.path.join(
        current_dir, f"figs/prediction_impact_comparison_{collision_type}.pdf"
    )
    plt.savefig(output_path, dpi=300)
    print(f"图表已保存至 {output_path}")


def plot_cht_cycles_conflicts_comparison(collision_type="link"):
    """比较共享双端口SRAM（预测通道数=1）和分布式多Bank SRAM（预测通道数=1）的总周期与冲突指标"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result_files")

    files = {
        "共享双端口SRAM（预测通道数=1）": f"dual_port_pred1_{collision_type}_results.csv",
        "分布式多Bank SRAM（预测通道数=1）": f"multi_bank_pred1_{collision_type}_results.csv",
    }

    cycles = {}
    conflicts = {}
    scenes = None

    for label, fname in files.items():
        fp = os.path.join(result_dir, fname)
        if not os.path.exists(fp):
            return
        df = pd.read_csv(fp)
        df["Scene_Num"] = df["Scene"].str.extract(r"(\d+)").astype(int)
        df = df.sort_values("Scene_Num")
        if scenes is None:
            scenes = df["Scene"].tolist()
        cycles[label] = df["Total_Cycles"].tolist()
        conflicts[label] = (
            df["Conflicts"].tolist() if "Conflicts" in df.columns else [0] * len(df)
        )

    if scenes is None:
        return
    x = np.arange(len(scenes))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.35
    palette = sns.color_palette()
    
    for ax, data_dict, title, ylabel in zip([ax1, ax2], [cycles, conflicts], 
                                            ["总周期数", "冲突数"], ["数量", "数量"]):
        for i, (label, vals) in enumerate(data_dict.items()):
            ax.bar(x + (i-0.5)*width, vals, width, label=label, color=palette[i], edgecolor='black', alpha=0.8)
        
        ax.set_title(f"{title}（{collision_type}）")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
        ax.legend()
        sns.despine(ax=ax)

    plt.tight_layout()
    output_path = os.path.join(
        current_dir, f"figs/cht_cycles_conflicts_comparison_pred1_{collision_type}.pdf"
    )
    plt.savefig(output_path, dpi=300)
    print(f"图表已保存至 {output_path}")


if __name__ == "__main__":
    for collision_type in ["link", "sphere"]:
        plot_cycle_comparison(collision_type)
        plot_cht_cycles_conflicts_comparison(collision_type)
        plot_prediction_impact_comparison(collision_type)
