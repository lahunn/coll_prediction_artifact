#!/usr/bin/env python3
"""
机器人球体半径分布分析工具

通过调用 robot_sphere_analyzer 或 curobo 库直接加载机器人的球体模型,
分析球体半径的统计分布特征,包括:
- 半径范围 (最小值、最大值)
- 半径分布统计 (均值、中位数、标准差)
- 半径直方图可视化
- 不同连杆的半径特征

使用示例:
    python analyze_sphere_radius.py --robot franka
    python analyze_sphere_radius.py --robot ur5e --bins 20 --save-fig
    python analyze_sphere_radius.py --robot franka --output stats.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# 添加当前目录到路径以导入 robot_sphere_analyzer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from robot_sphere_analyzer import RobotSphereAnalyzer


class SphereRadiusAnalyzer:
    """球体半径分布分析器"""

    def __init__(self, robot_name: str, device: str = "cuda:0"):
        """初始化分析器

        Args:
            robot_name: 机器人名称
            device: 计算设备
        """
        self.robot_name = robot_name
        self.device = device
        self.sphere_analyzer = RobotSphereAnalyzer(robot_name, device)
        self.analysis_result = None

    def collect_radius_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """收集所有球体的半径数据

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]:
                - all_radii: 所有球体的半径数组
                - link_radii: 每个连杆的半径字典 {link_name: radii_array}
        """
        # 获取连杆球体信息
        link_spheres = self.sphere_analyzer.get_link_spheres_info()

        # 收集所有半径
        all_radii = []
        link_radii = {}

        for link_name, spheres in link_spheres.items():
            # spheres 格式: [x, y, z, radius]
            radii = spheres[:, 3]
            all_radii.extend(radii)
            link_radii[link_name] = radii

        return np.array(all_radii), link_radii

    def compute_statistics(self, radii: np.ndarray) -> Dict:
        """计算半径统计信息

        Args:
            radii: 半径数组

        Returns:
            Dict: 统计信息字典
        """
        if len(radii) == 0:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "percentiles": {},
            }

        return {
            "count": len(radii),
            "min": float(np.min(radii)),
            "max": float(np.max(radii)),
            "mean": float(np.mean(radii)),
            "median": float(np.median(radii)),
            "std": float(np.std(radii)),
            "percentiles": {
                "25": float(np.percentile(radii, 25)),
                "50": float(np.percentile(radii, 50)),
                "75": float(np.percentile(radii, 75)),
                "90": float(np.percentile(radii, 90)),
                "95": float(np.percentile(radii, 95)),
                "99": float(np.percentile(radii, 99)),
            },
        }

    def analyze(self) -> Dict:
        """执行完整的半径分析

        Returns:
            Dict: 完整的分析结果
        """
        # 收集数据
        all_radii, link_radii = self.collect_radius_data()

        # 计算全局统计
        global_stats = self.compute_statistics(all_radii)

        # 计算每个连杆的统计
        link_stats = {}
        for link_name, radii in link_radii.items():
            link_stats[link_name] = self.compute_statistics(radii)

        # 查找最大和最小半径的连杆
        max_radius_link = None
        min_radius_link = None
        max_radius_value = None
        min_radius_value = None

        if len(link_radii) > 0:
            max_radius_link = max(link_radii.items(), key=lambda x: np.max(x[1]))
            min_radius_link = min(link_radii.items(), key=lambda x: np.min(x[1]))
            max_radius_value = float(np.max(max_radius_link[1]))
            min_radius_value = float(np.min(min_radius_link[1]))

        self.analysis_result = {
            "robot_name": self.robot_name,
            "device": self.device,
            "global_statistics": global_stats,
            "link_statistics": link_stats,
            "all_radii": all_radii,
            "link_radii": link_radii,
            "extremes": {
                "max_radius_link": max_radius_link[0] if max_radius_link else None,
                "max_radius_value": max_radius_value,
                "min_radius_link": min_radius_link[0] if min_radius_link else None,
                "min_radius_value": min_radius_value,
            },
        }

        return self.analysis_result

    def print_report(self):
        """打印分析报告"""
        if self.analysis_result is None:
            self.analyze()

        result = self.analysis_result
        if result is None:
            print("无法生成分析报告")
            return

        stats = result["global_statistics"]

        print("\n" + "=" * 70)
        print(f"球体半径分布分析报告 - {result['robot_name']}")
        print("=" * 70)

        # 全局统计
        print("\n【全局统计】")
        print(f"  总球体数: {stats['count']}")
        if stats["count"] > 0:
            print(f"  半径范围: [{stats['min']:.4f}, {stats['max']:.4f}] m")
            print(f"  平均半径: {stats['mean']:.4f} m")
            print(f"  中位数: {stats['median']:.4f} m")
            print(f"  标准差: {stats['std']:.4f} m")

            print("\n【百分位数】")
            for p, value in stats["percentiles"].items():
                print(f"  {p}%: {value:.4f} m")

        # 极值信息
        extremes = result["extremes"]
        print("\n【极值信息】")
        if extremes["max_radius_link"]:
            print(
                f"  最大半径: {extremes['max_radius_value']:.4f} m "
                f"(连杆: {extremes['max_radius_link']})"
            )
        if extremes["min_radius_link"]:
            print(
                f"  最小半径: {extremes['min_radius_value']:.4f} m "
                f"(连杆: {extremes['min_radius_link']})"
            )

        # 各连杆统计
        print("\n【各连杆半径统计】")
        link_stats = result["link_statistics"]
        for link_name, lstats in sorted(link_stats.items()):
            print(f"\n  {link_name}:")
            print(f"    球体数: {lstats['count']}")
            if lstats["count"] > 0:
                print(f"    范围: [{lstats['min']:.4f}, {lstats['max']:.4f}] m")
                print(f"    均值: {lstats['mean']:.4f} m")
                print(f"    中位数: {lstats['median']:.4f} m")

        print("\n" + "=" * 70)

    def plot_histogram(
        self, bins: int = 20, save_path: str | None = None, show: bool = True
    ):
        """绘制半径分布直方图

        Args:
            bins: 直方图分桶数
            save_path: 保存路径 (可选)
            show: 是否显示图表
        """
        if self.analysis_result is None:
            self.analyze()

        if self.analysis_result is None:
            print("没有分析结果可以绘制")
            return

        all_radii = self.analysis_result["all_radii"]
        stats = self.analysis_result["global_statistics"]

        if len(all_radii) == 0:
            print("没有球体数据可以绘制")
            return

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 子图1: 全局半径直方图
        ax1.hist(all_radii, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
        ax1.axvline(
            stats["mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {stats['mean']:.4f} m",
        )
        ax1.axvline(
            stats["median"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {stats['median']:.4f} m",
        )
        ax1.set_xlabel("Radius (m)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title(
            f"{self.robot_name} - Sphere Radius Distribution",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 子图2: 各连杆半径箱线图
        if self.analysis_result is None:
            return
        link_radii = self.analysis_result["link_radii"]
        link_names = sorted(link_radii.keys())
        radii_data = [link_radii[name] for name in link_names]

        bp = ax2.boxplot(radii_data, labels=link_names, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        ax2.set_xlabel("Link", fontsize=12)
        ax2.set_ylabel("Radius (m)", fontsize=12)
        ax2.set_title(
            f"{self.robot_name} - Radius Distribution by Link",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(alpha=0.3, axis="y")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\n图表已保存: {save_path}")

        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()

    def save_statistics(self, output_path: str):
        """保存统计信息到JSON文件

        Args:
            output_path: 输出文件路径
        """
        if self.analysis_result is None:
            self.analyze()

        if self.analysis_result is None:
            print("没有分析结果可以保存")
            return

        result = self.analysis_result
        # 准备可序列化的数据
        output_data = {
            "robot_name": result["robot_name"],
            "device": result["device"],
            "global_statistics": result["global_statistics"],
            "link_statistics": result["link_statistics"],
            "extremes": result["extremes"],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n统计信息已保存: {output_path}")

    def get_radius_bins(self, num_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """获取半径分桶信息

        Args:
            num_bins: 分桶数量

        Returns:
            Tuple[np.ndarray, np.ndarray]: (bin_edges, bin_counts)
        """
        if self.analysis_result is None:
            self.analyze()

        if self.analysis_result is None:
            return np.array([]), np.array([])

        result = self.analysis_result
        all_radii = result["all_radii"]
        if len(all_radii) == 0:
            return np.array([]), np.array([])

        counts, edges = np.histogram(all_radii, bins=num_bins)
        return edges, counts


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="机器人球体半径分布分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --robot franka                        # 分析Franka机器人
  %(prog)s --robot ur5e --bins 30                # 使用30个直方图分桶
  %(prog)s --robot franka --save-fig radius.png  # 保存图表
  %(prog)s --robot franka --output stats.json    # 导出统计数据
  %(prog)s --robot franka --no-plot              # 只显示统计,不绘图
        """,
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="franka",
        help="机器人名称 (默认: franka)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备 (默认: cuda:0)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="直方图分桶数 (默认: 20)",
    )
    parser.add_argument(
        "--save-fig",
        type=str,
        metavar="PATH",
        help="保存图表到指定路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help="保存统计信息到JSON文件",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不显示图表,仅输出统计信息",
    )

    args = parser.parse_args()

    try:
        print(f"\n正在分析 {args.robot} 机器人的球体半径分布...")

        # 创建分析器
        analyzer = SphereRadiusAnalyzer(args.robot, args.device)

        # 执行分析
        analyzer.analyze()

        # 打印报告
        analyzer.print_report()

        # 绘制图表
        if not args.no_plot:
            analyzer.plot_histogram(
                bins=args.bins,
                save_path=args.save_fig,
                show=(args.save_fig is None),  # 如果保存则不显示
            )

        # 保存统计信息
        if args.output:
            analyzer.save_statistics(args.output)

        print("\n✅ 分析完成!")

    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
