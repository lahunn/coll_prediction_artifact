#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate Hashing Cost Analysis Visualization Tool

Features:
1. Load coord_hashing_cost_results.csv file
2. Specify a variable as independent variable with fixed other parameters
3. Plot Precision, Recall and Cost curves against the variable

Supported variables:
- Density: Obstacle density (dens3, dens6, dens9, dens12)
- QuantBits: Coordinate quantization bits (3, 4, 5, 6, 7, 8)
- Threshold: Collision threshold (0.0, 0.03125, 0.0625, ..., 4.0)
- SampleRate: Sampling rate (0.01, 0.05, 0.1, ..., 1.0)

Usage Example:
    from plot_coord_hashing_analysis import CoordHashingPlotter
    
    plotter = CoordHashingPlotter("../result_files/coord_hashing_cost_results.csv")
    
    # Example 1: Analyze Threshold variation
    plotter.plot_variable_analysis(
        variable='Threshold',
        fixed_params={'Density': 'dens6', 'QuantBits': 4, 'SampleRate': 1.0},
        output_file='threshold_analysis.png'
    )
    
    # Example 2: Compare different densities
    plotter.plot_multi_curves(
        x_variable='Threshold',
        group_variable='Density',
        fixed_params={'QuantBits': 4, 'SampleRate': 1.0},
        output_file='density_comparison.png'
    )
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


class CoordHashingPlotter:
    """Coordinate Hashing Results Plotter"""
    
    def __init__(self, csv_path):
        """
        Initialize plotter
        
        Args:
            csv_path: CSV file path
        """
        self.csv_path = csv_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load CSV data"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ Successfully loaded data: {len(self.df)} rows")
        print(f"✓ Columns: {list(self.df.columns)}")
        
    def filter_data(self, **fixed_params):
        """
        Filter data based on fixed parameters
        
        Args:
            **fixed_params: Fixed parameters, e.g. Density='dens6', QuantBits=4
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = self.df.copy()
        
        for param, value in fixed_params.items():
            if param not in filtered_df.columns:
                raise ValueError(f"Parameter '{param}' not found in data")
            filtered_df = filtered_df[filtered_df[param] == value]
        
        if len(filtered_df) == 0:
            raise ValueError(f"No data matches conditions: {fixed_params}")
            
        print(f"✓ Filtered data: {len(filtered_df)} rows")
        return filtered_df
    
    def plot_variable_analysis(self, variable, fixed_params, 
                               output_file=None, show_plot=True):
        """
        Plot analysis for specified variable
        
        Args:
            variable: Variable name as independent variable (e.g. 'Threshold', 'QuantBits')
            fixed_params: Dictionary of fixed parameters
            output_file: Output file path (optional)
            show_plot: Whether to display the plot
        """
        # Check if variable exists
        if variable not in self.df.columns:
            raise ValueError(f"Variable '{variable}' not found in data")
        
        # Filter data
        filtered_df = self.filter_data(**fixed_params)
        
        # Sort by variable
        filtered_df = filtered_df.sort_values(by=variable)
        
        # Extract data
        x_values = filtered_df[variable].values
        precision = filtered_df['Precision'].values
        recall = filtered_df['Recall'].values
        collision_ratio = filtered_df['CollisionRatio'].values
        pred_cost = filtered_df['PredCost'].values
        baseline_cost = filtered_df['BaselineCost'].values
        speedup = filtered_df['Speedup'].values
        
        # Create figure: 4 subplots (Precision/Recall, Cost, Speedup, CollisionRatio)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Subplot 1: Precision and Recall
        ax1 = axes[0, 0]
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
        ax2 = axes[0, 1]
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
        ax3 = axes[1, 0]
        ax3.plot(x_values, speedup, 'o-', label='Speedup', 
                linewidth=2, markersize=6, color='#06A77D')
        ax3.axhline(y=1.0, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.5, label='Baseline (1x)')
        ax3.set_xlabel(self._get_label(variable), fontsize=12)
        ax3.set_ylabel('Speedup', fontsize=12)
        ax3.set_title('Speedup', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Collision Ratio
        ax4 = axes[1, 1]
        ax4.plot(x_values, collision_ratio, 'o-', label='Collision Ratio', 
                linewidth=2, markersize=6, color='#8E44AD')
        ax4.set_xlabel(self._get_label(variable), fontsize=12)
        ax4.set_ylabel('Collision Ratio (%)', fontsize=12)
        ax4.set_title('Collision Ratio', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        # Set overall title
        title_parts = [f"{k}={v}" for k, v in fixed_params.items()]
        fig.suptitle(f'Coordinate Hashing Analysis - {variable} Variation ({", ".join(title_parts)})', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save figure
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved to: {output_file}")
        
        # Display figure
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _get_label(self, variable):
        """Get English label for variable"""
        labels = {
            'Density': 'Obstacle Density',
            'QuantBits': 'Quantization Bits',
            'Threshold': 'Collision Threshold (S)',
            'SampleRate': 'Sampling Rate (U)',
            'Precision': 'Precision (%)',
            'Recall': 'Recall (%)',
            'CollisionRatio': 'Collision Ratio (%)',
            'PredCost': 'Prediction Cost',
            'BaselineCost': 'Baseline Cost',
            'Speedup': 'Speedup'
        }
        return labels.get(variable, variable)
    
    def plot_multi_curves(self, x_variable, group_variable, fixed_params,
                         output_file=None, show_plot=True):
        """
        Plot multiple curves grouped by a variable
        
        Args:
            x_variable: X-axis variable
            group_variable: Grouping variable
            fixed_params: Fixed parameters
            output_file: Output file path
            show_plot: Whether to display the plot
        """
        # Filter data
        filtered_df = self.filter_data(**fixed_params)
        
        # Get unique values of group variable
        group_values = sorted(filtered_df[group_variable].unique())
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
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
            collision_ratio = group_df['CollisionRatio'].values
            
            label = f"{group_variable}={group_val}"
            
            # Precision & Recall
            axes[0, 0].plot(x_values, precision, 'o-', label=f'{label} (P)', 
                        linewidth=2, markersize=5, color=colors[idx])
            axes[0, 0].plot(x_values, recall, 's--', label=f'{label} (R)', 
                        linewidth=2, markersize=5, color=colors[idx], alpha=0.6)
            
            # Speedup
            axes[0, 1].plot(x_values, speedup, 'o-', label=label, 
                        linewidth=2, markersize=5, color=colors[idx])
            
            # Precision vs Recall
            axes[1, 0].plot(recall, precision, 'o-', label=label, 
                        linewidth=2, markersize=5, color=colors[idx])
            
            # Collision Ratio
            axes[1, 1].plot(x_values, collision_ratio, 'o-', label=label,
                        linewidth=2, markersize=5, color=colors[idx])
        
        # Setup Subplot 1: Precision & Recall
        axes[0, 0].set_xlabel(self._get_label(x_variable), fontsize=12)
        axes[0, 0].set_ylabel('Percentage (%)', fontsize=12)
        axes[0, 0].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=9, ncol=2)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Setup Subplot 2: Speedup
        axes[0, 1].set_xlabel(self._get_label(x_variable), fontsize=12)
        axes[0, 1].set_ylabel('Speedup', fontsize=12)
        axes[0, 1].set_title('Speedup', fontsize=14, fontweight='bold')
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Setup Subplot 3: P-R Curve
        axes[1, 0].set_xlabel('Recall (%)', fontsize=12)
        axes[1, 0].set_ylabel('Precision (%)', fontsize=12)
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Setup Subplot 4: Collision Ratio
        axes[1, 1].set_xlabel(self._get_label(x_variable), fontsize=12)
        axes[1, 1].set_ylabel('Collision Ratio (%)', fontsize=12)
        axes[1, 1].set_title('Collision Ratio', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Overall title
        title_parts = [f"{k}={v}" for k, v in fixed_params.items()]
        fig.suptitle(f'Coordinate Hashing Analysis - {x_variable} vs {group_variable} ({", ".join(title_parts)})', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved to: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def main():
    """Main function: command line interface examples"""
    # CSV file path
    csv_path = "../result_files/coord_hashing_cost_results.csv"
    
    # Create plotter
    plotter = CoordHashingPlotter(csv_path)
    
    # Create output directory
    output_dir = "figs/coord_hashing"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Example 1: Threshold Analysis (fixed Density=dens6, QuantBits=4, SampleRate=1.0)")
    print("="*60)
    plotter.plot_variable_analysis(
        variable='Threshold',
        fixed_params={
            'Density': 'dens6',
            'QuantBits': 4,
            'SampleRate': 1.0
        },
        output_file=f"{output_dir}/threshold_analysis.png",
        show_plot=False
    )
    
    print("\n" + "="*60)
    print("Example 2: QuantBits Analysis (fixed Density=dens6, Threshold=0.125, SampleRate=1.0)")
    print("="*60)
    plotter.plot_variable_analysis(
        variable='QuantBits',
        fixed_params={
            'Density': 'dens6',
            'Threshold': 0.125,
            'SampleRate': 1.0
        },
        output_file=f"{output_dir}/quantbits_analysis.png",
        show_plot=False
    )
    
    print("\n" + "="*60)
    print("Example 3: Multi-Density Comparison (Threshold vs Density, fixed QuantBits=4, SampleRate=1.0)")
    print("="*60)
    plotter.plot_multi_curves(
        x_variable='Threshold',
        group_variable='Density',
        fixed_params={
            'QuantBits': 4,
            'SampleRate': 1.0
        },
        output_file=f"{output_dir}/threshold_vs_density.png",
        show_plot=False
    )
    
    print("\n" + "="*60)
    print("Example 4: QuantBits Comparison (Threshold vs QuantBits, fixed Density=dens6, SampleRate=1.0)")
    print("="*60)
    plotter.plot_multi_curves(
        x_variable='Threshold',
        group_variable='QuantBits',
        fixed_params={
            'Density': 'dens6',
            'SampleRate': 1.0
        },
        output_file=f"{output_dir}/threshold_vs_quantbits.png",
        show_plot=False
    )
    
    print("\n✅ All figures generated successfully!")
    print(f"📁 Output directory: {output_dir}")


if __name__ == '__main__':
    main()
