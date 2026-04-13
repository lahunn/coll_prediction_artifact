import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


# Match color/style settings with plot_cycle_comparison_sphere_link.py
sns.set_theme(style="whitegrid")
sns.set_style("white")
sns.set_palette("colorblind")
PALETTE = sns.color_palette("colorblind")
PRED1_COLOR = PALETTE[0]
PRED2_COLOR = PALETTE[1]

font_path = os.path.expanduser("~/.local/share/fonts/simsun.ttc")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

plt.rcParams.update(
    {
        "font.sans-serif": [
            "SimSun",
            "NSimSun",
            "STSong",
            "Songti SC",
            "Noto Sans CJK SC",
            "WenQuanYi Micro Hei",
            "Droid Sans Fallback",
            "Arial Unicode MS",
            "sans-serif",
        ],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


CSV_PATH = "../result_files/ablation_pred_sram_sphere_results.csv"
SCENE_ORDER = ["G1", "G2", "G3", "G4", "G5"]

TARGET_STRATEGIES = ["dual_port", "distri_multi_bank"]
PINGPONG_OUTPUT = "figs/ablation_pingpong_effect_comparison.pdf"
SINGLE_DUAL_OUTPUT = "figs/ablation_single_dual_comparison.pdf"
COMBINED_CYCLE_OUTPUT = "figs/ablation_cycle_combined_comparison.pdf"
THROUGHPUT_UTIL_OUTPUT = "figs/ablation_throughput_utilization_comparison.pdf"
WAIT_DEAD_OUTPUT = "figs/ablation_wait_dead_mechanism_comparison.pdf"
CONFLICT_OUTPUT = "figs/ablation_conflict_pred_comparison.pdf"
PERFORMANCE_OUTPUT = "figs/ablation_total_cycles_pred_comparison.pdf"
QUERY_OUTPUT = "figs/ablation_query_strategy_comparison.pdf"

PERFORMANCE_STRATEGIES = [
    "dual_port",
    "distri_multi_bank",
]

PRED_ORDER = [1, 2]
PRED_LABEL_MAP = {
    1: "单通道预测（Pred=1）",
    2: "双通道乒乓预测（Pred=2）",
}
CHT_TITLE_MAP = {
    "dual_port": "共享双端口",
    "multi_bank": "多Bank",
    "distri_dual_port": "分布式双端口",
    "distri_multi_bank": "分布式多Bank",
}
STRATEGY_SHORT_LABEL = {
    "dual_port": "共享双端口",
    "multi_bank": "多Bank",
    "distri_dual_port": "分布式双端口",
    "distri_multi_bank": "分布式多Bank",
}

PRED_LABEL_CHINESE = {
    1: "单通道预测",
    2: "双通道预测",
}

ARCH_MARKER = {
    "dual_port": "o",
    "distri_multi_bank": "s",
}

ARCH_BASE_COLOR = {
    "dual_port": PALETTE[0],
    "distri_multi_bank": PALETTE[1],
}

CONFLICT_STRATEGIES = [
    "dual_port",
    "distri_multi_bank",
]

QUERY_STRATEGIES = [
    "dual_port",
    "distri_multi_bank",
]


# 统一字体大小控制
FONT_SIZE_LABEL = 16  # 其它字体大小（如标签、标题等）
TICK_FONT_SIZE = 12   # 坐标轴刻度字体大小
LEGEND_FONT_SIZE = 12 # legend字体大小
FONT_SIZE_TITLE = 16
FONT_SIZE_SUPTITLE = 16
MARKER_SIZE = 16
LINE_WIDTH = 2.0


def _build_pred_label(pred: int) -> str:
    return PRED_LABEL_MAP.get(pred, f"Pred={pred}")


def _mix_with_white(color: tuple, white_ratio: float) -> tuple:
    return tuple((1.0 - white_ratio) * c + white_ratio for c in color)


def _color_by_arch_pred(cht_type: str, pred: int) -> tuple:
    base = ARCH_BASE_COLOR.get(cht_type, PALETTE[0])
    if pred == 1:
        return _mix_with_white(base, 0.35)
    return base


def _plot_single_strategy_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    cht_type: str,
    show_ylabel: bool,
) -> bool:
    subset = df[df["CHT_Type"] == cht_type].copy()
    if subset.empty:
        print(f"No data found for CHT_Type={cht_type}")
        return False

    subset["Pred"] = pd.to_numeric(subset["Pred"], errors="coerce").astype("Int64")
    subset = subset[subset["Pred"].isin(PRED_ORDER)]
    if subset.empty:
        print(f"No Pred=1/2 data for CHT_Type={cht_type}")
        return False

    subset["Pred_Label"] = subset["Pred"].map(_build_pred_label)
    pred_labels = [_build_pred_label(p) for p in PRED_ORDER]
    pivot = (
        subset.pivot_table(
            index="Scene",
            columns="Pred_Label",
            values="Total_Cycles",
            aggfunc="mean",
        )
        .reindex(index=SCENE_ORDER, columns=pred_labels)
    )

    valid_labels = [c for c in pivot.columns if not pivot[c].isna().all()]
    if len(valid_labels) < 2:
        print(f"Insufficient Pred data to compare for CHT_Type={cht_type}")
        return False

    x = np.arange(len(SCENE_ORDER))
    n = len(valid_labels)
    width = 0.78 / n
    color_map = {
        _build_pred_label(1): PRED1_COLOR,
        _build_pred_label(2): PRED2_COLOR,
    }

    for i, label in enumerate(valid_labels):
        values = pivot[label].to_numpy(dtype=float)
        offset = (i - (n - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=color_map.get(label, PALETTE[i % len(PALETTE)]),
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCENE_ORDER, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("总周期数" if show_ylabel else "", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.legend(title="流水线配置", ncol=2, frameon=True, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

    return True


def plot_pingpong_effect_comparison(df: pd.DataFrame, output_file: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.5), sharex=True, sharey=True)

    left_ok = _plot_single_strategy_on_ax(axes[0], df, "dual_port", show_ylabel=True)
    right_ok = _plot_single_strategy_on_ax(axes[1], df, "distri_multi_bank", show_ylabel=False)

    if not (left_ok or right_ok):
        plt.close(fig)
        print("No valid data found for pingpong effect comparison.")
        return


    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to: {output_file}")


def plot_throughput_utilization_comparison(df: pd.DataFrame, output_file: str) -> None:
    required_columns = {"Scene", "CHT_Type", "Pred", "Throughput", "Utilization"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing required columns for throughput/utilization plot: {sorted(missing_columns)}")
        return

    subset = df[df["CHT_Type"].isin(TARGET_STRATEGIES)].copy()
    if subset.empty:
        print("No strategy data available for throughput/utilization plot.")
        return

    subset["Pred"] = pd.to_numeric(subset["Pred"], errors="coerce").astype("Int64")
    subset = subset[subset["Pred"].isin(PRED_ORDER)]
    subset["Throughput"] = pd.to_numeric(subset["Throughput"], errors="coerce")
    subset["Utilization"] = (
        subset["Utilization"].astype(str).str.rstrip("%").pipe(pd.to_numeric, errors="coerce")
    )

    if subset.empty:
        print("No valid Pred=1/2 throughput/utilization data.")
        return

    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    metric_col, y_label = (
        "Utilization",
        "平均CDU占用率（%）",
    )

    pred_style_map = {
        1: {"color": PRED1_COLOR, "linestyle": "-"},
        2: {"color": PRED2_COLOR, "linestyle": "-"},
    }

    for cht_type in TARGET_STRATEGIES:
        for pred in PRED_ORDER:
            line_df = subset[(subset["CHT_Type"] == cht_type) & (subset["Pred"] == pred)]
            if line_df.empty:
                continue

            values = (
                line_df.pivot_table(
                    index="Scene",
                    values=metric_col,
                    aggfunc="mean",
                )
                .reindex(index=SCENE_ORDER)[metric_col]
                .to_numpy(dtype=float)
            )

            style = pred_style_map[pred].copy()
            style["marker"] = ARCH_MARKER.get(cht_type, "o")
            style["markersize"] = MARKER_SIZE
            style["linewidth"] = LINE_WIDTH
            style["markeredgecolor"] = "black"
            style["markeredgewidth"] = 0.7
            label = f"{STRATEGY_SHORT_LABEL.get(cht_type, cht_type)} / {PRED_LABEL_CHINESE.get(pred, f'Pred={pred}') }"
            ax.plot(
                SCENE_ORDER,
                values,
                label=label,
                **style,
            )

    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        bbox_to_anchor=(0.5, 1.10),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")


def plot_wait_dead_mechanism_comparison(df: pd.DataFrame, output_file: str) -> None:
    required_columns = {"Scene", "CHT_Type", "Pred", "Avg_Wait_Cycles", "DEAD_AVG_RATIO"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing required columns for wait/dead plot: {sorted(missing_columns)}")
        return

    subset = df[df["CHT_Type"].isin(TARGET_STRATEGIES)].copy()
    if subset.empty:
        print("No strategy data available for wait/dead plot.")
        return

    subset["Pred"] = pd.to_numeric(subset["Pred"], errors="coerce").astype("Int64")
    subset = subset[subset["Pred"].isin(PRED_ORDER)]
    subset["Avg_Wait_Cycles"] = pd.to_numeric(subset["Avg_Wait_Cycles"], errors="coerce")
    subset["DEAD_AVG_RATIO"] = pd.to_numeric(subset["DEAD_AVG_RATIO"], errors="coerce")

    if subset.empty:
        print("No valid Pred=1/2 wait/dead data.")
        return

    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    metric_col, y_label = (
        "DEAD_AVG_RATIO",
        "死区时间占比（%）",
    )

    pred_style_map = {
        1: {"color": PRED1_COLOR, "linestyle": "-"},
        2: {"color": PRED2_COLOR, "linestyle": "-"},
    }

    for cht_type in TARGET_STRATEGIES:
        for pred in PRED_ORDER:
            line_df = subset[(subset["CHT_Type"] == cht_type) & (subset["Pred"] == pred)]
            if line_df.empty:
                continue

            values = (
                line_df.pivot_table(
                    index="Scene",
                    values=metric_col,
                    aggfunc="mean",
                )
                .reindex(index=SCENE_ORDER)[metric_col]
                .to_numpy(dtype=float)
            )

            style = pred_style_map[pred].copy()
            style["marker"] = ARCH_MARKER.get(cht_type, "o")
            style["markersize"] = MARKER_SIZE
            style["linewidth"] = LINE_WIDTH
            style["markeredgecolor"] = "black"
            style["markeredgewidth"] = 0.7

            label = f"{STRATEGY_SHORT_LABEL.get(cht_type, cht_type)} / {PRED_LABEL_CHINESE.get(pred, f'Pred={pred}') }"
            ax.plot(
                SCENE_ORDER,
                values,
                label=label,
                **style,
            )

    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        bbox_to_anchor=(0.5, 1.10),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")


def _plot_conflict_subplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    pred_value: int,
    title: str,
    palette: list,
    show_ylabel: bool,
) -> None:
    required_columns = {"Scene", "CHT_Type", "Pred", "Conflicts"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing required columns for conflict plot: {sorted(missing_columns)}")
        return

    subset = df[df["Pred"] == pred_value].copy()
    subset = subset[subset["CHT_Type"].isin(CONFLICT_STRATEGIES)]
    if subset.empty:
        print(f"No valid Pred={pred_value} conflict data.")
        return

    subset["Conflicts"] = pd.to_numeric(subset["Conflicts"], errors="coerce")
    pivot = (
        subset.pivot_table(
            index="Scene",
            columns="CHT_Type",
            values="Conflicts",
            aggfunc="mean",
        )
        .reindex(index=SCENE_ORDER, columns=CONFLICT_STRATEGIES)
    )

    valid_strategies = [col for col in pivot.columns if not pivot[col].isna().all()]
    if not valid_strategies:
        print("No conflict data available for plotting.")
        return

    x = np.arange(len(SCENE_ORDER))
    n = len(valid_strategies)
    width = 0.8 / n

    for i, cht_type in enumerate(valid_strategies):
        values = pivot[cht_type].to_numpy(dtype=float)
        safe_values = np.where(values > 0, values, np.nan)
        offset = (i - (n - 1) / 2) * width
        ax.bar(
            x + offset,
            safe_values,
            width=width,
            label=STRATEGY_SHORT_LABEL.get(cht_type, cht_type),
            color=palette[i],
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(SCENE_ORDER, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("冲突数" if show_ylabel else "", fontsize=FONT_SIZE_LABEL)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.legend(title="CHT 架构", ncol=2, frameon=True, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)


def plot_conflict_comparison(df: pd.DataFrame, output_file: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    palette = sns.color_palette("colorblind", len(CONFLICT_STRATEGIES))

    _plot_conflict_subplot(
        axes[0],
        df,
        pred_value=1,
        title="单通道下两类 CHT 架构的访存冲突数",
        palette=palette,
        show_ylabel=True,
    )
    _plot_conflict_subplot(
        axes[1],
        df,
        pred_value=2,
        title="双通道下两类 CHT 架构的访存冲突数",
        palette=palette,
        show_ylabel=False,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")


def plot_total_cycles_pred_comparison(df: pd.DataFrame, output_file: str) -> None:
    required_columns = {"Scene", "CHT_Type", "Pred", "Total_Cycles"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing required columns for performance plot: {sorted(missing_columns)}")
        return

    subset = df[df["CHT_Type"].isin(PERFORMANCE_STRATEGIES)].copy()
    if subset.empty:
        print("No valid Total_Cycles data for performance plot.")
        return

    subset["Pred"] = pd.to_numeric(subset["Pred"], errors="coerce").astype("Int64")
    subset["Total_Cycles"] = pd.to_numeric(subset["Total_Cycles"], errors="coerce")

    if subset.empty:
        print("No valid Pred=1/2 Total_Cycles data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.5), sharex=True, sharey=True)
    pred_panels = [
        (1, "单通道架构总周期"),
        (2, "双通道架构总周期"),
    ]
    palette = sns.color_palette("colorblind", len(PERFORMANCE_STRATEGIES))

    for ax, (pred_value, title) in zip(axes, pred_panels):
        pred_df = subset[subset["Pred"] == pred_value]
        if pred_df.empty:
            continue

        pivot = (
            pred_df.pivot_table(
                index="Scene",
                columns="CHT_Type",
                values="Total_Cycles",
                aggfunc="mean",
            )
            .reindex(index=SCENE_ORDER, columns=PERFORMANCE_STRATEGIES)
        )

        valid_strategies = [col for col in pivot.columns if not pivot[col].isna().all()]
        if not valid_strategies:
            continue

        x = np.arange(len(SCENE_ORDER))
        n = len(valid_strategies)
        width = 0.8 / n

        for i, cht_type in enumerate(valid_strategies):
            values = pivot[cht_type].to_numpy(dtype=float)
            offset = (i - (n - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=STRATEGY_SHORT_LABEL.get(cht_type, cht_type),
                color=palette[i],
                edgecolor="black",
                linewidth=0.7,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(SCENE_ORDER, fontsize=TICK_FONT_SIZE)
        ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("总周期数" if pred_value == 1 else "", fontsize=FONT_SIZE_LABEL)
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
        ax.legend(title="CHT 架构", ncol=2, frameon=True, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)


    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")


def plot_cycle_combined_comparison(df: pd.DataFrame, output_file: str) -> None:
    required_columns = {"Scene", "CHT_Type", "Pred", "Total_Cycles"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing required columns for combined cycle plot: {sorted(missing_columns)}")
        return

    subset = df[df["CHT_Type"].isin(TARGET_STRATEGIES)].copy()
    if subset.empty:
        print("No strategy data available for combined cycle plot.")
        return

    subset["Pred"] = pd.to_numeric(subset["Pred"], errors="coerce").astype("Int64")
    subset = subset[subset["Pred"].isin(PRED_ORDER)]
    subset["Total_Cycles"] = pd.to_numeric(subset["Total_Cycles"], errors="coerce")

    strategy_keys = [
        ("dual_port", 1),
        ("dual_port", 2),
        ("distri_multi_bank", 1),
        ("distri_multi_bank", 2),
    ]

    pivot = (
        subset.pivot_table(
            index="Scene",
            columns=["CHT_Type", "Pred"],
            values="Total_Cycles",
            aggfunc="mean",
        )
        .reindex(
            index=SCENE_ORDER,
            columns=pd.MultiIndex.from_tuples(strategy_keys, names=["CHT_Type", "Pred"]),
        )
    )

    valid_keys = [key for key in strategy_keys if key in pivot.columns and not pivot[key].isna().all()]
    if not valid_keys:
        print("No valid Total_Cycles data available for combined cycle plot.")
        return

    fig, ax = plt.subplots(figsize=(9.8, 4.5))

    x = np.arange(len(SCENE_ORDER))
    n = len(valid_keys)
    width = 0.8 / n

    for i, (cht_type, pred) in enumerate(valid_keys):
        values = pivot[(cht_type, pred)].to_numpy(dtype=float)
        offset = (i - (n - 1) / 2) * width
        label = f"{STRATEGY_SHORT_LABEL.get(cht_type, cht_type)} / {PRED_LABEL_CHINESE.get(pred, f'Pred={pred}') }"
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=_color_by_arch_pred(cht_type, pred),
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCENE_ORDER, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("总周期数", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.legend(title="策略", ncol=2, frameon=True, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")

# 新增：只比较单通道+双口SRAM 与 双通道+分布式多bank SRAM
def plot_single_dual_comparison(df: pd.DataFrame, output_file: str) -> None:
    """
    只比较：
      - 单通道 + 共享双端口（dual_port, Pred=1）
      - 双通道 + 分布式多Bank（distri_multi_bank, Pred=2）
    """
    # 选择数据
    subset = df[((df["CHT_Type"] == "dual_port") & (df["Pred"] == 1)) |
                ((df["CHT_Type"] == "distri_multi_bank") & (df["Pred"] == 2))].copy()
    subset = subset[subset["Scene"].isin(SCENE_ORDER)]
    if subset.empty:
        print("No valid data for single-dual comparison plot.")
        return

    # 构建pivot
    subset["Pred"] = pd.to_numeric(subset["Pred"], errors="coerce").astype("Int64")
    subset["Total_Cycles"] = pd.to_numeric(subset["Total_Cycles"], errors="coerce")
    label_map = {
        ("dual_port", 1): "现有方案",
        ("distri_multi_bank", 2): "本研究"
    }
    subset["GroupLabel"] = subset.apply(lambda r: label_map.get((r["CHT_Type"], r["Pred"]), ""), axis=1)
    pivot = subset.pivot_table(index="Scene", columns="GroupLabel", values="Total_Cycles", aggfunc="mean")
    # 保证列顺序：现有方案在左，本研究在右
    pivot = pivot.reindex(index=SCENE_ORDER, columns=["现有方案", "本研究"])

    valid_labels = [col for col in pivot.columns if not pivot[col].isna().all()]
    if len(valid_labels) < 2:
        print("Not enough valid data for both groups.")
        return

    x = np.arange(len(SCENE_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.8, 4.5))

    colors = [PRED1_COLOR, PRED2_COLOR]
    for i, label in enumerate(valid_labels):
        values = pivot[label].to_numpy(dtype=float)
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCENE_ORDER, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("总周期数", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.legend(frameon=True, fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")

def plot_query_strategy_comparison(df: pd.DataFrame, output_file: str) -> None:
    required_columns = {"Scene", "CHT_Type", "Pred", "Total_Queries"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing required columns for query plot: {sorted(missing_columns)}")
        return

    subset = df[df["CHT_Type"].isin(QUERY_STRATEGIES)].copy()
    if subset.empty:
        print("No valid strategy data for query plot.")
        return

    subset["Pred"] = pd.to_numeric(subset["Pred"], errors="coerce").astype("Int64")
    subset = subset[subset["Pred"].isin(PRED_ORDER)]
    subset["Total_Queries"] = pd.to_numeric(subset["Total_Queries"], errors="coerce")

    pivot = (
        subset.pivot_table(
            index="Scene",
            columns=["CHT_Type", "Pred"],
            values="Total_Queries",
            aggfunc="mean",
        )
        .reindex(
            index=SCENE_ORDER,
            columns=pd.MultiIndex.from_product([QUERY_STRATEGIES, PRED_ORDER]),
        )
    )

    valid_columns = [col for col in pivot.columns if not pivot[col].isna().all()]
    if not valid_columns:
        print("No query data available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    x = np.arange(len(SCENE_ORDER))
    n = len(valid_columns)
    width = 0.8 / n
    for i, (cht_type, pred) in enumerate(valid_columns):
        values = pivot[(cht_type, pred)].to_numpy(dtype=float)
        offset = (i - (n - 1) / 2) * width
        pred_text = PRED_LABEL_CHINESE.get(pred, f"Pred={pred}")
        label = f"{STRATEGY_SHORT_LABEL.get(cht_type, cht_type)} / {pred_text}"
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=_color_by_arch_pred(cht_type, pred),
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCENE_ORDER, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("总查询数", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.legend(title="CHT 架构", ncol=2, frameon=True, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")


def plot_ablation_cycle_comparison() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, CSV_PATH)

    if not os.path.exists(csv_file):
        print(f"Data file not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    required_columns = {"Pred", "CHT_Type", "Scene", "Total_Cycles"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing required columns: {sorted(missing_columns)}")
        return

    df = df[df["Scene"].isin(SCENE_ORDER)].copy()

    if df.empty:
        print("No valid data found for requested scenes.")
        return

    combined_cycle_output = os.path.join(current_dir, COMBINED_CYCLE_OUTPUT)
    plot_cycle_combined_comparison(df, combined_cycle_output)

    single_dual_output = os.path.join(current_dir, SINGLE_DUAL_OUTPUT)
    plot_single_dual_comparison(df, single_dual_output)

    throughput_util_output = os.path.join(current_dir, THROUGHPUT_UTIL_OUTPUT)
    plot_throughput_utilization_comparison(df, throughput_util_output)

    wait_dead_output = os.path.join(current_dir, WAIT_DEAD_OUTPUT)
    plot_wait_dead_mechanism_comparison(df, wait_dead_output)

    conflict_output = os.path.join(current_dir, CONFLICT_OUTPUT)
    plot_conflict_comparison(df, conflict_output)

    query_output = os.path.join(current_dir, QUERY_OUTPUT)
    plot_query_strategy_comparison(df, query_output)
    


if __name__ == "__main__":
    plot_ablation_cycle_comparison()
