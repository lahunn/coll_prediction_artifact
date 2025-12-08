import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter


def plot_cycle_comparison():
    # 设置风格
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")  # Fallback for older matplotlib versions

    # 数据文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result_files")

    files = {
        "Dual Port (Pred=1)": "dual_port_pred1_results.csv",
        "Dual Port (Pred=2)": "dual_port_pred2_results.csv",
        "Multi-Bank (Pred=1)": "multi_bank_pred1_results.csv",
        "Multi-Bank (Pred=2)": "multi_bank_pred2_results.csv",
    }

    data = {}
    scenes = None

    print("Reading data files...")
    # 读取数据
    for label, filename in files.items():
        filepath = os.path.join(result_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        try:
            df = pd.read_csv(filepath)
            # 确保按 Scene 排序
            # 假设 Scene 列是 G1, G2... 这样的字符串，直接排序可能按字母序 G1, G10, G2...
            # 这里简单处理，假设数据量小且格式规范
            df["Scene_Num"] = df["Scene"].str.extract(r"(\d+)").astype(int)
            df = df.sort_values("Scene_Num")

            current_scenes = df["Scene"].tolist()
            if scenes is None:
                scenes = current_scenes
            else:
                # 简单检查场景是否对齐
                if scenes != current_scenes:
                    print(
                        f"Warning: Scenes mismatch in {filename}. Expected {scenes}, got {current_scenes}"
                    )

            data[label] = df["Total_Cycles"].tolist()
            print(f"Loaded {label}: {len(df)} records")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not data or scenes is None:
        print("No data or scenes loaded. Exiting.")
        return

    # 绘图配置
    if scenes is None:
        print("No scenes loaded. Exiting.")
        return
    x = np.arange(len(scenes))
    num_strategies = len(data)
    width = 0.18  # 柱状图宽度

    # 颜色配置 - 区分Dual Port和Multi-Bank
    colors = {
        "Dual Port (Pred=1)": "#2E86AB",  # 深蓝
        "Dual Port (Pred=2)": "#A23B72",  # 深紫红
        "Multi-Bank (Pred=1)": "#F18F01",  # 橙色
        "Multi-Bank (Pred=2)": "#C73E1D",  # 深红
    }

    # 边框样式

    fig, ax = plt.subplots(figsize=(14, 8))

    # 绘制柱状图
    bar_positions = {}
    for i, (strategy_name, cycles) in enumerate(data.items()):
        offset = width * (i - num_strategies / 2 + 0.5)
        ax.bar(
            x + offset,
            cycles,
            width,
            label=strategy_name,
            color=colors.get(strategy_name, "#999999"),
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )
        bar_positions[strategy_name] = (x + offset, cycles)

    # 标注 Dual Port (Pred=1) 与 Multi-Bank (Pred=2) 之间的差距
    if "Dual Port (Pred=1)" in data and "Multi-Bank (Pred=2)" in data:
        dual_port_pred1 = data["Dual Port (Pred=1)"]
        multi_bank_pred2 = data["Multi-Bank (Pred=2)"]

        for idx, scene in enumerate(scenes):
            dp_cycles = dual_port_pred1[idx]
            mb_cycles = multi_bank_pred2[idx]
            reduction = dp_cycles - mb_cycles
            reduction_pct = (reduction / dp_cycles) * 100 if dp_cycles > 0 else 0

            # 计算标注位置 (在两个柱子之间的上方)
            dp_offset = width * (0 - num_strategies / 2 + 0.5)
            mb_offset = width * (3 - num_strategies / 2 + 0.5)
            mid_x = idx + (dp_offset + mb_offset) / 2
            max_height = max(dp_cycles, mb_cycles)

            # 绘制连线和标注
            if reduction > 0:
                # 绘制箭头线
                ax.plot(
                    [idx + dp_offset, idx + mb_offset],
                    [dp_cycles, mb_cycles],
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=10,
                )

                # 添加文本标注
                ax.text(
                    mid_x,
                    max_height * 1.05,
                    f"-{reduction_pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="darkgreen",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="darkgreen",
                        alpha=0.8,
                    ),
                )

    # 设置标签和标题
    ax.set_ylabel("Total Cycles", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scene", fontsize=14, fontweight="bold")
    ax.set_title(
        "Multi-COPU Strategy Comparison: Cycle Count Across Scenes",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # 设置X轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=12)
    # 设置Y轴格式
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # 设置网格
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # 图例 - 分组显示
    ax.legend(
        loc="upper left",
        fontsize=11,
        frameon=True,
        shadow=True,
        ncol=2,
        columnspacing=1.5,
    )

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_dir = os.path.join(current_dir, "figs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cycle_comparison_strategies.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    print(f"Strategies compared: {list(data.keys())}")
    print(f"Scenes: {scenes}")
    # plt.show()


def plot_prediction_impact_comparison():
    """对比 Dual Port (Pred=1) 和 Dual Port (Pred=2) 的性能差异（Cycles和Queries）"""
    # 设置风格
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    # 数据文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result_files")

    files = {
        "Dual Port (Pred=1)": "dual_port_pred1_results.csv",
        "Dual Port (Pred=2)": "dual_port_pred2_results.csv",
    }

    cycles_data = {}
    queries_data = {}
    scenes = None

    print("Reading data files for prediction impact comparison...")
    # 读取数据
    for label, filename in files.items():
        filepath = os.path.join(result_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        try:
            df = pd.read_csv(filepath)
            df["Scene_Num"] = df["Scene"].str.extract(r"(\d+)").astype(int)
            df = df.sort_values("Scene_Num")

            current_scenes = df["Scene"].tolist()
            if scenes is None:
                scenes = current_scenes

            cycles_data[label] = df["Total_Cycles"].tolist()
            queries_data[label] = df["Total_Queries"].tolist()
            print(f"Loaded {label}: {len(df)} records")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not cycles_data or not queries_data or scenes is None:
        print("No data or scenes loaded. Exiting.")
        return

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(scenes))
    width = 0.35

    colors = {
        "Dual Port (Pred=1)": "#2E86AB",  # 深蓝
        "Dual Port (Pred=2)": "#A23B72",  # 深紫红
    }

    # ============ 第一个子图：Total Cycles ============
    for i, (strategy_name, cycles) in enumerate(cycles_data.items()):
        offset = width * (i - 0.5)
        ax1.bar(
            x + offset,
            cycles,
            width,
            label=strategy_name,
            color=colors.get(strategy_name, "#999999"),
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )

    # 标注Cycles的差距
    if "Dual Port (Pred=1)" in cycles_data and "Dual Port (Pred=2)" in cycles_data:
        pred1_cycles = cycles_data["Dual Port (Pred=1)"]
        pred2_cycles = cycles_data["Dual Port (Pred=2)"]

        for idx, scene in enumerate(scenes):
            p1_cycles = pred1_cycles[idx]
            p2_cycles = pred2_cycles[idx]
            reduction = p1_cycles - p2_cycles
            reduction_pct = (reduction / p1_cycles) * 100 if p1_cycles > 0 else 0

            dp_offset = width * (-0.5)
            dp2_offset = width * (0.5)
            mid_x = idx + (dp_offset + dp2_offset) / 2
            max_height = max(p1_cycles, p2_cycles)

            if reduction > 0:
                ax1.plot(
                    [idx + dp_offset, idx + dp2_offset],
                    [p1_cycles, p2_cycles],
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=10,
                )

                ax1.text(
                    mid_x,
                    max_height * 1.08,
                    f"-{reduction_pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="darkgreen",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="darkgreen",
                        linewidth=1.2,
                        alpha=0.9,
                    ),
                )

    ax1.set_ylabel("Total Cycles", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Scene", fontsize=12, fontweight="bold")
    ax1.set_title("Total Cycles Comparison", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenes, fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", fontsize=10, frameon=True, shadow=True)

    # ============ 第二个子图：Total Queries ============
    for i, (strategy_name, queries) in enumerate(queries_data.items()):
        offset = width * (i - 0.5)
        ax2.bar(
            x + offset,
            queries,
            width,
            label=strategy_name,
            color=colors.get(strategy_name, "#999999"),
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )

    # 标注Queries的差距
    if "Dual Port (Pred=1)" in queries_data and "Dual Port (Pred=2)" in queries_data:
        pred1_queries = queries_data["Dual Port (Pred=1)"]
        pred2_queries = queries_data["Dual Port (Pred=2)"]

        for idx, scene in enumerate(scenes):
            p1_queries = pred1_queries[idx]
            p2_queries = pred2_queries[idx]
            diff = p1_queries - p2_queries
            diff_pct = (diff / p1_queries) * 100 if p1_queries > 0 else 0

            dp_offset = width * (-0.5)
            dp2_offset = width * (0.5)
            mid_x = idx + (dp_offset + dp2_offset) / 2
            max_height = max(p1_queries, p2_queries)

            # 显示差异（无论增加还是减少）
            color = "darkgreen" if diff > 0 else "darkred"
            symbol = "-" if diff > 0 else "+"
            ax2.plot(
                [idx + dp_offset, idx + dp2_offset],
                [p1_queries, p2_queries],
                color="gray",
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                zorder=10,
            )

            ax2.text(
                mid_x,
                max_height * 1.08,
                f"{symbol}{abs(diff_pct):.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.2,
                    alpha=0.9,
                ),
            )

    ax2.set_ylabel("Total Queries", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Scene", fontsize=12, fontweight="bold")
    ax2.set_title("Total Queries Comparison", fontsize=13, fontweight="bold", pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenes, fontsize=11)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", fontsize=10, frameon=True, shadow=True)

    # 总标题
    fig.suptitle(
        "Prediction Impact: Dual Port (Pred=1) vs Dual Port (Pred=2)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_dir = os.path.join(current_dir, "figs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prediction_impact_comparison.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    print(f"Strategies compared: {list(cycles_data.keys())}")
    print(f"Scenes: {scenes}")


def plot_cht_cycles_conflicts_comparison():
    """比较 Dual Port (Pred=1) 和 Multi-Bank (Pred=1) 的 Total_Cycles 与 Conflicts 指标。

    生成一个两行一列的子图：上图显示 Total_Cycles 的对比，下图显示 Conflicts 的对比，
    并在每个场景上标出百分比差异（DualPort 相对于 MultiBank 的减少百分比，正为 DualPort 更高）。
    """
    # 风格与路径
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result_files")

    files = {
        "Dual Port (Pred=1)": "dual_port_pred1_results.csv",
        "Multi-Bank (Pred=1)": "multi_bank_pred1_results.csv",
    }

    cycles = {}
    conflicts = {}
    scenes = None

    for label, fname in files.items():
        fp = os.path.join(result_dir, fname)
        if not os.path.exists(fp):
            print(f"Warning: File not found: {fp}")
            return
        df = pd.read_csv(fp)
        df["Scene_Num"] = df["Scene"].str.extract(r"(\d+)").astype(int)
        df = df.sort_values("Scene_Num")
        if scenes is None:
            scenes = df["Scene"].tolist()
        cycles[label] = df["Total_Cycles"].tolist()
        # 有些CSV里'Conflicts'列可能存在或缺失，兼容处理
        if "Conflicts" in df.columns:
            conflicts[label] = df["Conflicts"].tolist()
        else:
            # 若不存在则填0占位，避免索引错误
            conflicts[label] = [0] * len(df)

    if scenes is None:
        print("No scenes loaded. Exiting.")
        return
    x = np.arange(len(scenes))

    # 并排子图：一行两列
    fig, (ax_cycles, ax_conf) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    width = 0.35

    # 绘制 Cycles（左图）
    ax_cycles.bar(
        x - width / 2,
        cycles["Dual Port (Pred=1)"],
        width,
        label="Dual Port (Pred=1)",
        color="#2E86AB",
        edgecolor="black",
    )
    ax_cycles.bar(
        x + width / 2,
        cycles["Multi-Bank (Pred=1)"],
        width,
        label="Multi-Bank (Pred=1)",
        color="#F18F01",
        edgecolor="black",
    )

    # 在Cycles上标注百分比差异（DualPort 相对 MultiBank）
    for idx in range(len(scenes)):
        dp = cycles["Dual Port (Pred=1)"][idx]
        mb = cycles["Multi-Bank (Pred=1)"][idx]
        pct = (dp - mb) / dp * 100 if dp > 0 else 0
        ymax = max(dp, mb)
        ax_cycles.text(
            idx,
            ymax * 1.02,
            f"{pct:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax_cycles.set_ylabel("Total Cycles", fontsize=12, fontweight="bold")
    ax_cycles.set_title("Total Cycles (Dual Port vs Multi-Bank, Pred=1)")
    ax_cycles.legend()
    ax_cycles.grid(True, axis="y", linestyle="--", alpha=0.4)

    # 绘制 Conflicts（右图）
    ax_conf.bar(
        x - width / 2,
        conflicts["Dual Port (Pred=1)"],
        width,
        label="Dual Port (Pred=1)",
        color="#2E86AB",
        edgecolor="black",
    )
    ax_conf.bar(
        x + width / 2,
        conflicts["Multi-Bank (Pred=1)"],
        width,
        label="Multi-Bank (Pred=1)",
        color="#F18F01",
        edgecolor="black",
    )

    # 在Conflicts上标注百分比差异
    for idx in range(len(scenes)):
        dp = conflicts["Dual Port (Pred=1)"][idx]
        mb = conflicts["Multi-Bank (Pred=1)"][idx]
        pct = (dp - mb) / dp * 100 if dp > 0 else 0
        ymax = max(dp, mb)
        ax_conf.text(
            idx,
            (ymax * 1.02 if ymax > 0 else 0.5),
            f"{pct:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax_conf.set_ylabel("Conflicts", fontsize=12, fontweight="bold")
    ax_conf.set_title("Conflicts (Dual Port vs Multi-Bank, Pred=1)")
    ax_conf.legend()
    ax_conf.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax_conf.set_xticks(x)
    ax_conf.set_xticklabels(scenes, fontsize=11)

    plt.tight_layout()
    out_dir = os.path.join(current_dir, "figs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cht_cycles_conflicts_comparison_pred1.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    plot_cycle_comparison()
    plot_cht_cycles_conflicts_comparison()
    plot_prediction_impact_comparison()
