import math
import sys
import os
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm

# --- 统一绘图风格配置 ---
sns.set_theme(style="whitegrid")
sns.set_style("white")
sns.set_palette("colorblind")

# 字体加载与配置
font_path = os.path.expanduser("~/.local/share/fonts/simsun.ttc")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

plt.rcParams.update({
    'font.sans-serif': ['SimSun', 'NSimSun', 'Arial Unicode MS', 'sans-serif'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def plot_aggregated_metrics(df, parameter_name, density, output_dir="plots"):
    """
    对指定参数进行分组，计算精确率和召回率的平均值，并绘制柱状图。
    """
    # 确保输出目录存在
    density_dir = os.path.join(output_dir, density)
    if not os.path.exists(density_dir):
        os.makedirs(density_dir)

    # 映射密度名称
    density_map = {"low": "低", "mid": "中", "high": "高"}
    cn_density = density_map.get(density, density)

    # 按指定参数分组，并计算精确率和召回率的平均值
    aggregated_data = (
        df.groupby(parameter_name)[['精确率', '召回率']].mean().reset_index()
    )

    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"{parameter_name} 对精确率和召回率的影响 ({cn_density} 密度)",
        fontsize=16,
    )

    # --- 绘制精确率柱状图 ---
    palette = sns.color_palette("colorblind")
    ax1.bar(
        aggregated_data[parameter_name].astype(str),
        aggregated_data['精确率'],
        color=palette[0],
    )
    ax1.set_title("平均精确率")
    ax1.set_xlabel(parameter_name)
    ax1.set_ylabel("精确率 %")
    ax1.tick_params(axis="x", rotation=45)

    # --- 绘制召回率柱状图 ---
    ax2.bar(
        aggregated_data[parameter_name].astype(str),
        aggregated_data['召回率'],
        color=palette[2],
    )
    ax2.set_title("平均召回率")
    ax2.set_xlabel(parameter_name)
    ax2.set_ylabel("召回率 %")
    ax2.tick_params(axis="x", rotation=45)

    # 调整布局并保存图表
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    output_path = os.path.join(density_dir, f"{parameter_name}_performance.pdf")
    plt.savefig(output_path)
    plt.close()
    print(f"图表已保存至: {output_path}")

def main():
    """
    主函数，加载数据并为每个密度和参数组合生成图表。
    """
    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    output_dir = os.path.join(script_dir, "plots")

    # 构建结果文件的完整路径
    results_file = os.path.join(
        script_dir, "result_files", "sphere_hashing_results.csv"
    )

    # 检查结果文件是否存在
    if not os.path.exists(results_file):
        print(f"错误: 结果文件 '{results_file}' 未找到。")
        return

    # 加载CSV文件
    try:
        df = pd.read_csv(results_file)
    except pd.errors.EmptyDataError:
        print(f"错误: 结果文件 '{results_file}' 为空。")
        return

    # 定义要分析的参数列表
    parameters_to_analyze = ["CoordBits", "RadiusBits", "Threshold", "SampleRate"]

    # 定义密度级别
    densities = ["low", "mid", "high"]

    # 为每个密度和参数组合生成图表
    for density in densities:
        density_df = df[df["Density"] == density]
        if density_df.empty:
            print(f"警告: 密度 '{density}' 没有数据。")
            continue

        print(f"\n处理密度: {density}")
        for param in parameters_to_analyze:
            if param in density_df.columns:
                plot_aggregated_metrics(
                    density_df, param, density, output_dir=output_dir
                )
            else:
                print(f"警告: 在密度 '{density}' 的数据中未找到列 '{param}'。")

if __name__ == "__main__":
    main()
