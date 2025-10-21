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
        output_file='threshold_analysis.png'
    )
    
    # 示例2: 对比不同密度下的表现
    plotter.plot_multi_curves(
        x_variable='Threshold',
        group_variable='Density',
        fixed_params={'CoordBits': 4, 'RadiusBits': 2, 'SampleRate': 1.0},
        output_file='density_comparison.png'
    )
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


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
        self.load_data()
        
    def load_data(self):
        """加载CSV数据"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
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
        filtered_df = self.df.copy()
        
        for param, value in fixed_params.items():
            if param not in filtered_df.columns:
                raise ValueError(f"参数 '{param}' 不存在于数据中")
            filtered_df = filtered_df[filtered_df[param] == value]
        
        if len(filtered_df) == 0:
            raise ValueError(f"没有数据满足条件: {fixed_params}")
            
        print(f"✓ 过滤后数据: {len(filtered_df)} 行")
        return filtered_df
    
    def plot_variable_analysis(self, variable, fixed_params, 
                               output_file=None, show_plot=True):
        """
        绘制指定变量的分析图
        
        Args:
            variable: 作为自变量的参数名 (例如 'Threshold', 'CoordBits')
            fixed_params: 固定的其他参数字典
            output_file: 输出文件路径 (可选)
            show_plot: 是否显示图形
        """
        # 检查变量是否存在
        if variable not in self.df.columns:
            raise ValueError(f"变量 '{variable}' 不存在于数据中")
        
        # 过滤数据
        filtered_df = self.filter_data(**fixed_params)
        
        # 按变量排序
        filtered_df = filtered_df.sort_values(by=variable)
        
        # 提取数据
        x_values = filtered_df[variable].values
        precision = filtered_df['Precision'].values
        recall = filtered_df['Recall'].values
        pred_cost = filtered_df['PredCost'].values
        baseline_cost = filtered_df['BaselineCost'].values
        speedup = filtered_df['Speedup'].values
        
        # 创建图形：3个子图（Precision/Recall、Cost、Speedup）
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Subplot 1: Precision and Recall
        ax1 = axes[0]
        ax1.plot(x_values, precision, 'o-', label='Precision', 
                linewidth=2, markersize=6, color='#2E86AB')
        ax1.plot(x_values, recall, 's-', label='Recall', 
                linewidth=2, markersize=6, color='#A23B72')
        ax1.set_xlabel(self._get_label(variable), fontsize=12)
        ax1.set_ylabel('Percentage (%)', fontsize=12)
        ax1.set_title('Precision & Recall', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Cost (Prediction vs Baseline)
        ax2 = axes[1]
        ax2.plot(x_values, pred_cost, 'o-', label='Prediction Cost', 
                linewidth=2, markersize=6, color='#F18F01')
        ax2.plot(x_values, baseline_cost, 's--', label='Baseline Cost', 
                linewidth=2, markersize=6, color='#C73E1D', alpha=0.7)
        ax2.set_xlabel(self._get_label(variable), fontsize=12)
        ax2.set_ylabel('Computation Cost', fontsize=12)
        ax2.set_title('Cost Comparison', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Speedup
        ax3 = axes[2]
        ax3.plot(x_values, speedup, 'o-', label='Speedup', 
                linewidth=2, markersize=6, color='#06A77D')
        ax3.axhline(y=1.0, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.5, label='Baseline (1x)')
        ax3.set_xlabel(self._get_label(variable), fontsize=12)
        ax3.set_ylabel('Speedup', fontsize=12)
        ax3.set_title('Speedup', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Set overall title
        title_parts = [f"{k}={v}" for k, v in fixed_params.items()]
        fig.suptitle(f'Sphere Hashing Analysis - {variable} Variation ({", ".join(title_parts)})', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ 图形已保存到: {output_file}")
        
        # 显示图形
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _get_label(self, variable):
        """Get English label for variable"""
        labels = {
            'Density': 'Obstacle Density',
            'CoordBits': 'Coordinate Quantization Bits',
            'RadiusBits': 'Radius Quantization Bits',
            'Threshold': 'Collision Threshold (S)',
            'SampleRate': 'Sampling Rate (U)',
            'Precision': 'Precision (%)',
            'Recall': 'Recall (%)',
            'PredCost': 'Prediction Cost',
            'BaselineCost': 'Baseline Cost',
            'Speedup': 'Speedup'
        }
        return labels.get(variable, variable)
    
    def plot_multi_curves(self, x_variable, group_variable, fixed_params,
                         output_file=None, show_plot=True):
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Color mapping (compatible with matplotlib 3.7+)
        try:
            # For matplotlib >= 3.7
            cmap = plt.colormaps['viridis']
        except (AttributeError, KeyError):
            # For older matplotlib versions
            cmap = plt.cm.get_cmap('viridis')
        colors = [cmap(i / max(len(group_values) - 1, 1)) for i in range(len(group_values))]
        
        for idx, group_val in enumerate(group_values):
            group_df = filtered_df[filtered_df[group_variable] == group_val]
            group_df = group_df.sort_values(by=x_variable)
            
            x_values = group_df[x_variable].values
            precision = group_df['Precision'].values
            recall = group_df['Recall'].values
            speedup = group_df['Speedup'].values
            
            label = f"{group_variable}={group_val}"
            
            # Precision & Recall
            axes[0].plot(x_values, precision, 'o-', label=f'{label} (P)', 
                        linewidth=2, markersize=5, color=colors[idx])
            axes[0].plot(x_values, recall, 's--', label=f'{label} (R)', 
                        linewidth=2, markersize=5, color=colors[idx], alpha=0.6)
            
            # Speedup
            axes[1].plot(x_values, speedup, 'o-', label=label, 
                        linewidth=2, markersize=5, color=colors[idx])
            
            # Precision vs Recall
            axes[2].plot(recall, precision, 'o-', label=label, 
                        linewidth=2, markersize=5, color=colors[idx])
        
        # Setup Subplot 1: Precision & Recall
        axes[0].set_xlabel(self._get_label(x_variable), fontsize=12)
        axes[0].set_ylabel('Percentage (%)', fontsize=12)
        axes[0].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=9, ncol=2)
        axes[0].grid(True, alpha=0.3)
        
        # Setup Subplot 2: Speedup
        axes[1].set_xlabel(self._get_label(x_variable), fontsize=12)
        axes[1].set_ylabel('Speedup', fontsize=12)
        axes[1].set_title('Speedup', fontsize=14, fontweight='bold')
        axes[1].axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        # Setup Subplot 3: P-R Curve
        axes[2].set_xlabel('Recall (%)', fontsize=12)
        axes[2].set_ylabel('Precision (%)', fontsize=12)
        axes[2].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)
        
        # Overall title
        title_parts = [f"{k}={v}" for k, v in fixed_params.items()]
        fig.suptitle(f'Sphere Hashing Analysis - {x_variable} vs {group_variable} ({", ".join(title_parts)})', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
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
    
    print("\n" + "="*60)
    print("Example 1: Threshold Analysis (fixed Density=dens6, CoordBits=4, RadiusBits=2, SampleRate=1.0)")
    print("="*60)
    plotter.plot_variable_analysis(
        variable='Threshold',
        fixed_params={
            'Density': 'dens6',
            'CoordBits': 4,
            'RadiusBits': 2,
            'SampleRate': 1.0
        },
        output_file=f"{output_dir}/threshold_analysis.png",
        show_plot=False
    )
    
    print("\n" + "="*60)
    print("Example 2: CoordBits Analysis (fixed Density=dens6, RadiusBits=2, Threshold=0.125, SampleRate=1.0)")
    print("="*60)
    plotter.plot_variable_analysis(
        variable='CoordBits',
        fixed_params={
            'Density': 'dens6',
            'RadiusBits': 2,
            'Threshold': 0.125,
            'SampleRate': 1.0
        },
        output_file=f"{output_dir}/coordbits_analysis.png",
        show_plot=False
    )
    
    print("\n" + "="*60)
    print("Example 3: Multi-Density Comparison (Threshold vs Density, fixed CoordBits=4, RadiusBits=2, SampleRate=1.0)")
    print("="*60)
    plotter.plot_multi_curves(
        x_variable='Threshold',
        group_variable='Density',
        fixed_params={
            'CoordBits': 4,
            'RadiusBits': 2,
            'SampleRate': 1.0
        },
        output_file=f"{output_dir}/threshold_vs_density.png",
        show_plot=False
    )
    
    print("\n" + "="*60)
    print("Example 4: RadiusBits Comparison (Threshold vs RadiusBits, fixed Density=dens6, CoordBits=4, SampleRate=1.0)")
    print("="*60)
    plotter.plot_multi_curves(
        x_variable='Threshold',
        group_variable='RadiusBits',
        fixed_params={
            'Density': 'dens6',
            'CoordBits': 4,
            'SampleRate': 1.0
        },
        output_file=f"{output_dir}/threshold_vs_radiusbits.png",
        show_plot=False
    )
    
    print("\n✅ All figures generated successfully!")
    print(f"📁 Output directory: {output_dir}")


if __name__ == '__main__':
    main()
