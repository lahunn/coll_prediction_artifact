import argparse
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "../result_files")
OUTPUT_DIR = os.path.join(BASE_DIR, "figs")
DEFAULT_INPUT = "g5_copus_cht_pred_eval_sphere.csv"
DEFAULT_OUTPUT = "g5_copus_scalability.pdf"

SCENE_NAME = "G5"
COPU_ORDER = [1, 2, 4, 8, 16, 32]
TARGET_STRATEGIES = ["dual_port", "distri_multi_bank"]
PRED_ORDER = [1, 2]

sns.set_theme(style="whitegrid")
sns.set_style("white")
sns.set_palette("colorblind")
PALETTE = sns.color_palette("colorblind")

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


def _mix_with_white(color: tuple, white_ratio: float) -> tuple:
    return tuple((1.0 - white_ratio) * component + white_ratio for component in color)


def _color_by_arch_pred(cht_type: str, pred: int) -> tuple:
    base_color = PALETTE[0] if cht_type == "dual_port" else PALETTE[1]
    if pred == 1:
        return _mix_with_white(base_color, 0.35)
    return base_color


def _load_data(input_file: str) -> pd.DataFrame:
    file_path = os.path.join(RESULT_DIR, input_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)
    required_columns = {"Num_COPUS", "CHT_Type", "Pred", "Scene", "Total_Cycles"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{input_file} missing columns: {sorted(missing_columns)}")

    df = df[df["Scene"] == SCENE_NAME].copy()
    df["Num_COPUS"] = pd.to_numeric(df["Num_COPUS"], errors="coerce")
    df["Pred"] = pd.to_numeric(df["Pred"], errors="coerce").astype("Int64")
    df["Total_Cycles"] = pd.to_numeric(df["Total_Cycles"], errors="coerce")
    df = df[df["CHT_Type"].isin(TARGET_STRATEGIES)]
    df = df[df["Pred"].isin(PRED_ORDER)]
    return df.dropna(subset=["Num_COPUS", "Pred", "Total_Cycles"]).copy()


def plot_copu_scalability(input_file: str, output_file: str) -> None:
    output_path = os.path.join(OUTPUT_DIR, output_file)
    fig, ax = plt.subplots(figsize=(10.8, 5.6))

    try:
        df = _load_data(input_file)
    except (FileNotFoundError, ValueError) as exc:
        ax.text(0.5, 0.5, str(exc), ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {output_path}")
        return

    if df.empty:
        ax.text(0.5, 0.5, "无有效数据", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {output_path}")
        return

    copu_values = [value for value in COPU_ORDER if value in set(df["Num_COPUS"].dropna().astype(int))]
    if not copu_values:
        copu_values = sorted(df["Num_COPUS"].dropna().unique().tolist())

    strategy_keys = [
        ("dual_port", 1),
        ("dual_port", 2),
        ("distri_multi_bank", 1),
        ("distri_multi_bank", 2),
    ]

    pivot = (
        df.pivot_table(
            index="Num_COPUS",
            columns=["CHT_Type", "Pred"],
            values="Total_Cycles",
            aggfunc="mean",
        )
        .reindex(index=copu_values, columns=pd.MultiIndex.from_tuples(strategy_keys, names=["CHT_Type", "Pred"]))
    )

    valid_keys = [key for key in strategy_keys if key in pivot.columns and not pivot[key].isna().all()]
    if not valid_keys:
        ax.text(0.5, 0.5, "无有效策略数据", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {output_path}")
        return

    x = np.arange(len(copu_values))
    width = 0.8 / len(valid_keys)

    for i, (cht_type, pred) in enumerate(valid_keys):
        values = pivot[(cht_type, pred)].to_numpy(dtype=float)
        offset = (i - (len(valid_keys) - 1) / 2) * width
        label = f"{('共享双端口' if cht_type == 'dual_port' else '分布式多Bank')} / {'单通道预测' if pred == 1 else '双通道预测'}"
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=_color_by_arch_pred(cht_type, pred),
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_title("G5 场景下不同 COPU 数的总周期数对比")
    ax.set_xlabel("COPU 数")
    ax.set_ylabel("总周期数")
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in copu_values])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(title="策略", ncol=2, frameon=True, fontsize=11, title_fontsize=11)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot COPU scalability bars for G5")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="input CSV file under result_files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output PDF file under plots/figs")
    args = parser.parse_args()

    plot_copu_scalability(args.input, args.output)


if __name__ == "__main__":
    main()