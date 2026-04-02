import math
import sys

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib

# --- 统一绘图风格配置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
球体哈希成本分析可视化工具

功能：
1. 读取 sphere_hashing_cost_results.csv 文件
2. 指定某个变量作为自变量，固定其他变量的值
3. 绘制 Precision、Recall 和 Cost 随该变量变化的曲线

支持的变量：
- Density: 障碍物密度 (dens3, dens6, dens9, dens12)
- CoordBits: 坐标量化位数 (3, 4, 5, 6)
- RadiusBits: 半径量化位数 (1, 2, 3, 4)
- Threshold: 碰撞阈值 (0.0, 0.03125, 0.0625, ..., 4.0)
- SampleRate: 采样率 (0.01, 0.05, 0.1, ..., 1.0)

使用示例：
    from plot_sphere_hashing_analysis import SphereHashingPlotter

    plotter = SphereHashingPlotter("../result_files/sphere_hashing_cost_results.csv")

    # 示例1: 分析 Threshold 变化的影响
    plotter.plot_variable_analysis(
        variable='Threshold',
        fixed_params={'Density': 'dens6', 'CoordBits': 4, 'RadiusBits': 2, 'SampleRate': 1.0},
        output_file='threshold_analysis.pdf'
    )

    # 示例2: 对比不同密度下的表现
    plotter.plot_multi_curves(
        x_variable='Threshold',
        group_variable='Density',
        fixed_params={'CoordBits': 4, 'RadiusBits': 2, 'SampleRate': 1.0},
        output_file='density_comparison.pdf'
    )
"""

import os
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
colors = sns.color_palette("deep")


# Unified plotting style
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
sns.set_style("white")
sns.set_palette("colorblind")




class SphereHashingPlotter:
    """球体哈希结果绘图类"""

    def __init__(self, csv_path):
        """
        初始化绘图器

        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.df = None
        self.palette = sns.color_palette("colorblind")
        self.load_data()

    def load_data(self):
        """加载CSV数据"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if self.df is None:
            raise ValueError("Failed to load CSV data")
        print(f"✓ 成功加载数据: {len(self.df)} 行")
        print(f"✓ 列名: {list(self.df.columns)}")

    def filter_data(self, **fixed_params):
        """
        根据固定参数过滤数据

        Args:
            **fixed_params: 固定的参数，例如 Density='dens6', CoordBits=4

        Returns:
            过滤后的DataFrame
        """
        if self.df is None:
            raise ValueError("数据未加载，请先调用 load_data()")
        filtered_df = self.df.copy()

        for param, value in fixed_params.items():
            if param not in filtered_df.columns:
                raise ValueError(f"参数 '{param}' 不存在于数据中")
            filtered_df = filtered_df[filtered_df[param] == value]

        if len(filtered_df) == 0:
            raise ValueError(f"没有数据满足条件: {fixed_params}")

        print(f"✓ 过滤后数据: {len(filtered_df)} 行")
        return filtered_df

    def plot_variable_analysis(
        self, variable, fixed_params, output_file=None, show_plot=True
    ):
        """
        绘制指定变量的分析图

        Args:
            variable: 作为自变量的参数名 (例如 'Threshold', 'CoordBits')
            fixed_params: 固定的其他参数字典
            output_file: 输出文件路径 (可选)
            show_plot: 是否显示图形
        """
        # 检查数据是否加载
        if self.df is None:
            raise ValueError("数据未加载，请先调用 load_data()")

        # 检查变量是否存在
        if variable not in self.df.columns:
            raise ValueError(f"变量 '{variable}' 不存在于数据中")

        # 过滤数据
        filtered_df = self.filter_data(**fixed_params)

        # 按变量排序
        filtered_df = filtered_df.sort_values(by=variable)

        # 提取数据
        x_values = filtered_df[variable].values
        pose_precision = filtered_df["PosePrecision"].values
        pose_recall = filtered_df["PoseRecall"].values
        elem_precision = filtered_df["ElemPrecision"].values
        elem_recall = filtered_df["ElemRecall"].values
        pred_cost = filtered_df["PredCost"].values
        baseline_cost = filtered_df["BaselineCost"].values

        # 创建图形：2个子图（精确率/召回率、Cost）
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1: Precision and Recall
        ax1 = axes[0]
        ax1.plot(
            x_values,
            pose_precision,
            "o-",
            label="Pose Precision",
            linewidth=2,
            markersize=6,
            color=self.palette[0],
        )
        ax1.plot(
            x_values,
            pose_recall,
            "s-",
            label="Pose Recall",
            linewidth=2,
            markersize=6,
            color=self.palette[1],
        )
        ax1.plot(
            x_values,
            elem_precision,
            "o--",
            label="Elem Precision",
            linewidth=2,
            markersize=6,
            color=self.palette[0],
            alpha=0.6,
        )
        ax1.plot(
            x_values,
            elem_recall,
            "s--",
            label="Elem Recall",
            linewidth=2,
            markersize=6,
            color=self.palette[1],
            alpha=0.6,
        )
        ax1.set_xlabel(self._get_label(variable), fontsize=12)
        ax1.set_ylabel("Percentage (%)", fontsize=12)
        ax1.set_title("Precision & Recall", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        # grid removed per style requirement

        # Subplot 2: Cost (Prediction vs Baseline)
        ax2 = axes[1]
        ax2.plot(
            x_values,
            pred_cost,
            "o-",
            label="Prediction Cost",
            linewidth=2,
            markersize=6,
            color=self.palette[2],
        )
        ax2.plot(
            x_values,
            baseline_cost,
            "s--",
            label="Baseline Cost",
            linewidth=2,
            markersize=6,
            color=self.palette[3],
            alpha=0.7,
        )
        ax2.set_xlabel(self._get_label(variable), fontsize=12)
        ax2.set_ylabel("Computation Cost", fontsize=12)
        ax2.set_title("Cost Comparison", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        # grid removed per style requirement

        # Set overall title
        title_parts = [f"{k}={v}" for k, v in fixed_params.items()]
        fig.suptitle(
            f"Sphere Hashing Analysis - {variable} Variation ({', '.join(title_parts)})",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"✓ 图形已保存到: {output_file}")

        # 显示图形
        if show_plot:
            plt.show()
        else:
            plt.close()

    def _get_label(self, variable):
        """Get English label for variable"""
        labels = {
            "Density": "Obstacle Density",
            "CoordBits": "Coordinate Quantization Bits",
            "RadiusBits": "Radius Quantization Bits",
            "Threshold": "Collision Threshold (S)",
            "SampleRate": "Sampling Rate (U)",
            '精确率': "Precision (%)",
            '召回率': "Recall (%)",
            "PredCost": "Prediction Cost",
            "BaselineCost": "Baseline Cost",
            "Speedup": "Speedup",
        }
        return labels.get(variable, variable)

    def plot_multi_curves(
        self, x_variable, group_variable, fixed_params, output_file=None, show_plot=True
    ):
        """
        绘制多条曲线，按某个变量分组

        Args:
            x_variable: X轴变量
            group_variable: 分组变量
            fixed_params: 固定的其他参数
            output_file: 输出文件路径
            show_plot: 是否显示图形
        """
        # 过滤数据
        filtered_df = self.filter_data(**fixed_params)

        # 获取分组的唯一值
        group_values = sorted(filtered_df[group_variable].unique())

        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        colors = sns.color_palette("colorblind", n_colors=max(len(group_values), 1))

        for idx, group_val in enumerate(group_values):
            group_df = filtered_df[filtered_df[group_variable] == group_val]
            group_df = group_df.sort_values(by=x_variable)

            x_values = group_df[x_variable].values
            pose_precision = group_df["PosePrecision"].values
            pose_recall = group_df["PoseRecall"].values
            elem_precision = group_df["ElemPrecision"].values
            elem_recall = group_df["ElemRecall"].values

            label = f"{group_variable}={group_val}"

            # Precision & Recall
            axes[0].plot(
                x_values,
                pose_precision,
                "o-",
                label=f"{label} (Pose P)",
                linewidth=2,
                markersize=5,
                color=colors[idx],
            )
            axes[0].plot(
                x_values,
                pose_recall,
                "s-",
                label=f"{label} (Pose R)",
                linewidth=2,
                markersize=5,
                color=colors[idx],
                alpha=0.6,
            )
            axes[0].plot(
                x_values,
                elem_precision,
                "o--",
                label=f"{label} (Elem P)",
                linewidth=1.5,
                markersize=4,
                color=colors[idx],
                alpha=0.4,
            )
            axes[0].plot(
                x_values,
                elem_recall,
                "s--",
                label=f"{label} (Elem R)",
                linewidth=1.5,
                markersize=4,
                color=colors[idx],
                alpha=0.3,
            )

            # Precision vs Recall (Pose level)
            axes[1].plot(
                pose_recall,
                pose_precision,
                "o-",
                label=f"{label} (Pose)",
                linewidth=2,
                markersize=5,
                color=colors[idx],
            )
            axes[1].plot(
                elem_recall,
                elem_precision,
                "s--",
                label=f"{label} (Elem)",
                linewidth=1.5,
                markersize=4,
                color=colors[idx],
                alpha=0.5,
            )

        # Setup Subplot 1: Precision & Recall
        axes[0].set_xlabel(self._get_label(x_variable), fontsize=12)
        axes[0].set_ylabel("Percentage (%)", fontsize=12)
        axes[0].set_title("Precision & Recall", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=9, ncol=2)
        # grid removed per style requirement

        # Setup Subplot 2: P-R Curve
        axes[1].set_xlabel("Recall (%)", fontsize=12)
        axes[1].set_ylabel("Precision (%)", fontsize=12)
        axes[1].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=9)
        # grid removed per style requirement

        # Overall title
        title_parts = [f"{k}={v}" for k, v in fixed_params.items()]
        fig.suptitle(
            f"Sphere Hashing Analysis - {x_variable} vs {group_variable} ({', '.join(title_parts)})",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"✓ 图形已保存到: {output_file}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_elem_metrics(
        self, x_variable, group_variable, fixed_params, output_file=None, show_plot=True
    ):
        """
        绘制元素级指标：ElemPrecision、ElemRecall 和 PredCost

        Args:
            x_variable: X轴变量
            group_variable: 分组变量
            fixed_params: 固定的其他参数
            output_file: 输出文件路径
            show_plot: 是否显示图形
        """
        filtered_df = self.filter_data(**fixed_params)
        group_values = sorted(filtered_df[group_variable].unique())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        colors = sns.color_palette("colorblind", n_colors=max(len(group_values), 1))

        for idx, group_val in enumerate(group_values):
            group_df = filtered_df[
                filtered_df[group_variable] == group_val
            ].sort_values(by=x_variable)
            x_vals = group_df[x_variable].values
            label = f"{group_variable}={group_val}"

            axes[0].plot(
                x_vals,
                group_df["ElemPrecision"].values,
                "o-",
                label=label,
                linewidth=2,
                markersize=5,
                color=colors[idx],
            )
            axes[1].plot(
                x_vals,
                group_df["ElemRecall"].values,
                "s-",
                label=label,
                linewidth=2,
                markersize=5,
                color=colors[idx],
            )
            axes[2].plot(
                x_vals,
                group_df["PredCost"].values,
                "^-",
                label=label,
                linewidth=2,
                markersize=5,
                color=colors[idx],
            )

        axes[0].set_xlabel(self._get_label(x_variable), fontsize=12)
        axes[0].set_ylabel("Elem Precision (%)", fontsize=12)
        axes[0].set_title("Element Precision", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=10)
        # grid removed per style requirement

        axes[1].set_xlabel(self._get_label(x_variable), fontsize=12)
        axes[1].set_ylabel("Elem Recall (%)", fontsize=12)
        axes[1].set_title("Element Recall", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=10)
        # grid removed per style requirement

        axes[2].set_xlabel(self._get_label(x_variable), fontsize=12)
        axes[2].set_ylabel("Prediction Cost", fontsize=12)
        axes[2].set_title("Prediction Cost", fontsize=14, fontweight="bold")
        axes[2].legend(fontsize=10)
        # grid removed per style requirement

        title_parts = [f"{k}={v}" for k, v in fixed_params.items()]
        fig.suptitle(
            f"Element-Level Metrics - {x_variable} vs {group_variable} ({', '.join(title_parts)})",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"✓ 图形已保存到: {output_file}")

        if show_plot:
            plt.show()
        else:
            plt.close()


def main():
    """Main function: command line interface examples"""
    # CSV file path
    csv_path = "../result_files/sphere_hashing_cost_results.csv"

    # Create plotter
    plotter = SphereHashingPlotter(csv_path)

    # Create output directory
    output_dir = "figs/sphere_hashing"
    os.makedirs(output_dir, exist_ok=True)

    # print("\n" + "=" * 60)
    # print(
    #     "Example 1: Threshold Analysis (fixed Density=dens6, CoordBits=4, RadiusBits=2, SampleRate=1.0)"
    # )
    # print("=" * 60)
    # plotter.plot_variable_analysis(
    #     variable="Threshold",
    #     fixed_params={
    #         "Density": "dens9",
    #         "CoordBits": 4,
    #         "RadiusBits": 2,
    #         "SampleRate": 1.0,
    #     },
    #     output_file=f"{output_dir}/threshold_analysis.pdf",
    #     show_plot=False,
    # )

    # print("\n" + "=" * 60)
    # print(
    #     "Example 2: CoordBits Analysis (fixed Density=dens6, RadiusBits=2, Threshold=0.125, SampleRate=1.0)"
    # )
    # print("=" * 60)
    # plotter.plot_variable_analysis(
    #     variable="CoordBits",
    #     fixed_params={
    #         "Density": "dens9",
    #         "RadiusBits": 2,
    #         "Threshold": 0.5,
    #         "SampleRate": 1.0,
    #     },
    #     output_file=f"{output_dir}/coordbits_analysis.pdf",
    #     show_plot=False,
    # )

    # print("\n" + "=" * 60)
    # print(
    #     "Example 3: Multi-Density Comparison (Threshold vs Density, fixed CoordBits=4, RadiusBits=2, SampleRate=1.0)"
    # )
    # print("=" * 60)
    # plotter.plot_multi_curves(
    #     x_variable="Threshold",
    #     group_variable="Density",
    #     fixed_params={"CoordBits": 4, "RadiusBits": 2, "SampleRate": 1.0},
    #     output_file=f"{output_dir}/threshold_vs_density.pdf",
    #     show_plot=False,
    # )

    # print("\n" + "=" * 60)
    # print(
    #     "Example 4: RadiusBits Comparison (Threshold vs RadiusBits, fixed Density=dens6, CoordBits=4, SampleRate=1.0)"
    # )
    # print("=" * 60)
    # plotter.plot_multi_curves(
    #     x_variable="Threshold",
    #     group_variable="RadiusBits",
    #     fixed_params={"Density": "dens9", "CoordBits": 4, "SampleRate": 1.0},
    #     output_file=f"{output_dir}/threshold_vs_radiusbits.pdf",
    #     show_plot=False,
    # )

    # print("\n" + "=" * 60)
    # print(
    #     "Example 5: CoordBits vs Density (fixed RadiusBits=2, Threshold=0.5, SampleRate=1.0)"
    # )
    # print("=" * 60)
    # plotter.plot_multi_curves(
    #     x_variable="CoordBits",
    #     group_variable="Density",
    #     fixed_params={"RadiusBits": 2, "Threshold": 0.5, "SampleRate": 1.0},
    #     output_file=f"{output_dir}/coordbits_vs_density.pdf",
    #     show_plot=False,
    # )

    # print("\n" + "=" * 60)
    # print(
    #     "Example 6: RadiusBits vs Density (fixed CoordBits=4, Threshold=0.5, SampleRate=1.0)"
    # )
    # print("=" * 60)
    # plotter.plot_multi_curves(
    #     x_variable="RadiusBits",
    #     group_variable="Density",
    #     fixed_params={"CoordBits": 4, "Threshold": 0.5, "SampleRate": 1.0},
    #     output_file=f"{output_dir}/radiusbits_vs_density.pdf",
    #     show_plot=False,
    # )

    # print("\n" + "=" * 60)
    # print(
    #     "Example 7: Element Metrics - CoordBits vs Density (fixed RadiusBits=2, Threshold=0.5, SampleRate=1.0)"
    # )
    # print("=" * 60)
    # plotter.plot_elem_metrics(
    #     x_variable="CoordBits",
    #     group_variable="Density",
    #     fixed_params={"RadiusBits": 2, "Threshold": 0.5, "SampleRate": 1.0},
    #     output_file=f"{output_dir}/elem_coordbits_vs_density.pdf",
    #     show_plot=False,
    # )

    # print("\n" + "=" * 60)
    # print(
    #     "Example 8: Element Metrics - RadiusBits vs Density (fixed CoordBits=4, Threshold=0.5, SampleRate=1.0)"
    # )
    # print("=" * 60)
    # plotter.plot_elem_metrics(
    #     x_variable="RadiusBits",
    #     group_variable="Density",
    #     fixed_params={"CoordBits": 4, "Threshold": 0.5, "SampleRate": 1.0},
    #     output_file=f"{output_dir}/elem_radiusbits_vs_density.pdf",
    #     show_plot=False,
    # )

    print("\n✅ All figures generated successfully!")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
