import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_aggregated_metrics(df, parameter_name, density, output_dir="plots"):
    """
    对指定参数进行分组，计算Precision和Recall的平均值，并绘制柱状图。

    Args:
        df (pd.DataFrame): 包含结果数据的DataFrame。
        parameter_name (str): 要分析的参数列名 (例如 'CoordBits')。
        density (str): 密度级别 ('low', 'medium', 'high')。
        output_dir (str): 保存图表的目录。
    """
    # 确保输出目录存在
    density_dir = os.path.join(output_dir, density)
    if not os.path.exists(density_dir):
        os.makedirs(density_dir)

    # 按指定参数分组，并计算Precision和Recall的平均值
    aggregated_data = (
        df.groupby(parameter_name)[["Precision", "Recall"]].mean().reset_index()
    )

    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Impact of {parameter_name} on Precision and Recall ({density} density)",
        fontsize=16,
    )

    # --- 绘制Precision柱状图 ---
    ax1.bar(
        aggregated_data[parameter_name].astype(str),
        aggregated_data["Precision"],
        color="skyblue",
    )
    ax1.set_title("Average Precision")
    ax1.set_xlabel(parameter_name)
    ax1.set_ylabel("Precision (%)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # --- 绘制Recall柱状图 ---
    ax2.bar(
        aggregated_data[parameter_name].astype(str),
        aggregated_data["Recall"],
        color="lightgreen",
    )
    ax2.set_title("Average Recall")
    ax2.set_xlabel(parameter_name)
    ax2.set_ylabel("Recall (%)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # 调整布局并保存图表
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    output_path = os.path.join(density_dir, f"{parameter_name}_performance.png")
    plt.savefig(output_path)
    plt.close()
    print(f"图表已保存至: {output_path}")


def main():
    """
    主函数，加载数据并为每个密度和参数组合生成图表。
    """
    # 获取当前脚本的目录  # 获取当前脚本的绝对路径
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
