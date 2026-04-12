import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "../result_files")
OUTPUT_DIR = os.path.join(BASE_DIR, "figs")

SWEEP_FILES = {
    "COPU 数量": "sweep_num_copus_distri_multi_bank_sphere_results.csv",
    "OOCD 数量": "sweep_num_oocds_distri_multi_bank_sphere_results.csv",
    "Bank 数量": "sweep_num_banks_distri_multi_bank_sphere_results.csv",
    "预测通道数": "sweep_num_pred_distri_multi_bank_sphere_results.csv",
}

SWEEP_OUTPUTS = {
    "COPU 数量": "total_cycles_scalability_num_copus.pdf",
    "OOCD 数量": "total_cycles_scalability_num_oocds.pdf",
    "Bank 数量": "total_cycles_scalability_num_banks.pdf",
    "预测通道数": "total_cycles_scalability_num_pred.pdf",
}

CONFLICT_BANK_FILE = "sweep_num_banks_distri_multi_bank_sphere_results.csv"
CONFLICT_BANK_OUTPUT = "memory_conflicts_vs_banks.pdf"
PRED_UTILIZATION_FILE = "sweep_num_pred_distri_multi_bank_sphere_results.csv"
PRED_UTILIZATION_OUTPUT = "cdu_utilization_vs_pred.pdf"
PRED_DEADTIME_FILE = "sweep_num_pred_distri_multi_bank_sphere_results.csv"
PRED_DEADTIME_OUTPUT = "dead_time_ratio_vs_pred.pdf"
PRED_THROUGHPUT_FILE = "sweep_num_pred_distri_multi_bank_sphere_results.csv"
PRED_THROUGHPUT_OUTPUT = "throughput_vs_pred.pdf"
G5_BANK_CYCLES_FILE = "sweep_num_banks_distri_multi_bank_sphere_results.csv"
G5_BANK_CYCLES_OUTPUT = "g5_total_cycles_vs_banks.pdf"

# Keep visual style aligned with existing plotting scripts.
sns.set_theme(style="whitegrid")
sns.set_style("white")
sns.set_palette("colorblind")

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


def _load_sweep(file_name: str) -> pd.DataFrame:
    file_path = os.path.join(RESULT_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)
    required_columns = {"Value", "Scene", "Total_Cycles"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{file_name} missing columns: {sorted(missing_columns)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Total_Cycles"] = pd.to_numeric(df["Total_Cycles"], errors="coerce")
    return df.dropna(subset=["Value", "Total_Cycles"]).copy()


def _load_conflict_sweep(file_name: str) -> pd.DataFrame:
    file_path = os.path.join(RESULT_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)
    required_columns = {"Value", "Scene", "Conflicts"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{file_name} missing columns: {sorted(missing_columns)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Conflicts"] = pd.to_numeric(df["Conflicts"], errors="coerce")
    return df.dropna(subset=["Value", "Conflicts"]).copy()


def _load_utilization_sweep(file_name: str) -> pd.DataFrame:
    file_path = os.path.join(RESULT_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)
    required_columns = {"Value", "Scene", "Utilization"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{file_name} missing columns: {sorted(missing_columns)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Utilization"] = (
        df["Utilization"].astype(str).str.rstrip("%").pipe(pd.to_numeric, errors="coerce")
    )
    return df.dropna(subset=["Value", "Utilization"]).copy()


def _load_deadtime_sweep(file_name: str) -> pd.DataFrame:
    file_path = os.path.join(RESULT_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)
    required_columns = {"Value", "Scene", "DEAD_AVG_RATIO"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{file_name} missing columns: {sorted(missing_columns)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["DEAD_AVG_RATIO"] = pd.to_numeric(df["DEAD_AVG_RATIO"], errors="coerce")
    return df.dropna(subset=["Value", "DEAD_AVG_RATIO"]).copy()


def _load_throughput_sweep(file_name: str) -> pd.DataFrame:
    file_path = os.path.join(RESULT_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)
    required_columns = {"Value", "Scene", "Throughput"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{file_name} missing columns: {sorted(missing_columns)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Throughput"] = pd.to_numeric(df["Throughput"], errors="coerce")
    return df.dropna(subset=["Value", "Throughput"]).copy()


def _plot_single_sweep(sweep_label: str, file_name: str) -> None:
    output_name = SWEEP_OUTPUTS[sweep_label]
    output_path = os.path.join(OUTPUT_DIR, output_name)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    try:
        df = _load_sweep(file_name)
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

    scene_order = ["G1", "G2", "G3", "G4", "G5"]
    palette = sns.color_palette("colorblind", len(scene_order))
    x_ticks = sorted(df["Value"].dropna().unique())
    x_pos_map = {v: i for i, v in enumerate(x_ticks)}

    for i, scene in enumerate(scene_order):
        scene_df = df[df["Scene"] == scene].copy()
        if scene_df.empty:
            continue
        scene_df = scene_df.sort_values("Value")
        x_pos = [x_pos_map[v] for v in scene_df["Value"].to_numpy()]
        ax.plot(
            x_pos,
            scene_df["Total_Cycles"].to_numpy(),
            marker="o",
            linewidth=2.0,
            color=palette[i],
            label=scene,
        )

    ax.set_title(f"总周期数 vs {sweep_label}")
    ax.set_xlabel(sweep_label)
    ax.set_ylabel("总周期数")
    # Keep y-axis auto-scaled; do not force starting from 0.
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in x_ticks])
    ax.legend(title="场景", frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def plot_conflicts_vs_banks() -> None:
    output_path = os.path.join(OUTPUT_DIR, CONFLICT_BANK_OUTPUT)
    fig, ax = plt.subplots(figsize=(7, 3.5))

    try:
        df = _load_conflict_sweep(CONFLICT_BANK_FILE)
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

    scene_order = ["G1", "G2", "G3", "G4", "G5"]
    palette = sns.color_palette("colorblind", len(scene_order))
    x_ticks = sorted(df["Value"].dropna().unique())
    x_pos_map = {v: i for i, v in enumerate(x_ticks)}

    for i, scene in enumerate(scene_order):
        scene_df = df[df["Scene"] == scene].copy()
        if scene_df.empty:
            continue
        scene_df = scene_df.sort_values("Value")
        x_pos = [x_pos_map[v] for v in scene_df["Value"].to_numpy()]
        ax.plot(
            x_pos,
            scene_df["Conflicts"].to_numpy(),
            marker="o",
            linewidth=2.0,
            color=palette[i],
            label=scene,
        )

    ax.set_title("访存冲突数 vs Bank 数量")
    ax.set_xlabel("Bank 数量")
    ax.set_ylabel("访存冲突数")
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in x_ticks])
    ax.legend(title="场景", frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def plot_cdu_utilization_vs_pred() -> None:
    output_path = os.path.join(OUTPUT_DIR, PRED_UTILIZATION_OUTPUT)
    fig, ax = plt.subplots(figsize=(7, 3.5))

    try:
        df = _load_utilization_sweep(PRED_UTILIZATION_FILE)
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

    scene_order = ["G1", "G2", "G3", "G4", "G5"]
    palette = sns.color_palette("colorblind", len(scene_order))
    x_ticks = sorted(df["Value"].dropna().unique())
    x_pos_map = {v: i for i, v in enumerate(x_ticks)}

    for i, scene in enumerate(scene_order):
        scene_df = df[df["Scene"] == scene].copy()
        if scene_df.empty:
            continue
        scene_df = scene_df.sort_values("Value")
        x_pos = [x_pos_map[v] for v in scene_df["Value"].to_numpy()]
        ax.plot(
            x_pos,
            scene_df["Utilization"].to_numpy(),
            marker="o",
            linewidth=2.0,
            color=palette[i],
            label=scene,
        )

    ax.set_title("CDU 利用率 vs 预测通道数")
    ax.set_xlabel("预测通道数")
    ax.set_ylabel("CDU 利用率 (%)")
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in x_ticks])
    ax.legend(title="场景", frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def plot_deadtime_ratio_vs_pred() -> None:
    output_path = os.path.join(OUTPUT_DIR, PRED_DEADTIME_OUTPUT)
    fig, ax = plt.subplots(figsize=(7, 3.5))

    try:
        df = _load_deadtime_sweep(PRED_DEADTIME_FILE)
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
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {output_path}")
        return

    scene_order = ["G1", "G2", "G3", "G4", "G5"]
    palette = sns.color_palette("colorblind", len(scene_order))
    x_ticks = sorted(df["Value"].dropna().unique())
    x_pos_map = {v: i for i, v in enumerate(x_ticks)}

    for i, scene in enumerate(scene_order):
        scene_df = df[df["Scene"] == scene].copy()
        if scene_df.empty:
            continue
        scene_df = scene_df.sort_values("Value")
        x_pos = [x_pos_map[v] for v in scene_df["Value"].to_numpy()]
        ax.plot(
            x_pos,
            scene_df["DEAD_AVG_RATIO"].to_numpy(),
            marker="o",
            linewidth=2.0,
            color=palette[i],
            label=scene,
        )

    ax.set_title("Dead-Time Ratio vs Pred Count")
    ax.set_xlabel("Pred Count")
    ax.set_ylabel("Dead-Time Ratio (%)")
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in x_ticks])
    ax.legend(title="Scene", frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def plot_throughput_vs_pred() -> None:
    output_path = os.path.join(OUTPUT_DIR, PRED_THROUGHPUT_OUTPUT)
    fig, ax = plt.subplots(figsize=(7, 3.5))

    try:
        df = _load_throughput_sweep(PRED_THROUGHPUT_FILE)
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
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {output_path}")
        return

    scene_order = ["G1", "G2", "G3", "G4", "G5"]
    palette = sns.color_palette("colorblind", len(scene_order))
    x_ticks = sorted(df["Value"].dropna().unique())
    x_pos_map = {v: i for i, v in enumerate(x_ticks)}

    for i, scene in enumerate(scene_order):
        scene_df = df[df["Scene"] == scene].copy()
        if scene_df.empty:
            continue
        scene_df = scene_df.sort_values("Value")
        x_pos = [x_pos_map[v] for v in scene_df["Value"].to_numpy()]
        ax.plot(
            x_pos,
            scene_df["Throughput"].to_numpy(),
            marker="o",
            linewidth=2.0,
            color=palette[i],
            label=scene,
        )

    ax.set_title("Throughput vs Pred Count")
    ax.set_xlabel("Pred Count")
    ax.set_ylabel("Throughput")
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in x_ticks])
    ax.legend(title="Scene", frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def plot_g5_total_cycles_vs_banks() -> None:
    output_path = os.path.join(OUTPUT_DIR, G5_BANK_CYCLES_OUTPUT)
    fig, ax = plt.subplots(figsize=(7, 3.5))

    try:
        df = _load_sweep(G5_BANK_CYCLES_FILE)
    except (FileNotFoundError, ValueError) as exc:
        ax.text(0.5, 0.5, str(exc), ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {output_path}")
        return

    g5_df = df[df["Scene"] == "G5"].copy()
    if g5_df.empty:
        ax.text(0.5, 0.5, "No valid G5 data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {output_path}")
        return

    g5_df = g5_df.sort_values("Value")
    x_ticks = sorted(g5_df["Value"].dropna().unique())
    x_pos_map = {v: i for i, v in enumerate(x_ticks)}
    x_pos = [x_pos_map[v] for v in g5_df["Value"].to_numpy()]

    ax.plot(
        x_pos,
        g5_df["Total_Cycles"].to_numpy(),
        marker="o",
        linewidth=2.2,
        color=sns.color_palette("colorblind", 1)[0],
        label="G5",
    )

    ax.set_title("G5 Total Cycles vs Bank Count")
    ax.set_xlabel("Bank Count")
    ax.set_ylabel("Total Cycles")
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in x_ticks])
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def plot_total_cycles_scalability() -> None:
    for sweep_label, file_name in SWEEP_FILES.items():
        _plot_single_sweep(sweep_label, file_name)
    plot_conflicts_vs_banks()
    plot_cdu_utilization_vs_pred()
    plot_deadtime_ratio_vs_pred()
    plot_throughput_vs_pred()
    plot_g5_total_cycles_vs_banks()


if __name__ == "__main__":
    plot_total_cycles_scalability()
