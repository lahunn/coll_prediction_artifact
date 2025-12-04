#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置文件路径
csv_file = "../result_files/sphere_link_comparison_results.csv"
output_dir = "figs"
os.makedirs(output_dir, exist_ok=True)

# 读取数据
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: File {csv_file} not found.")
    exit(1)

# 准备数据容器
difficulties = ["G1", "G2", "G3", "G4", "G5"]
sphere_cycles = []
link_cycles = []

# 提取数据
for diff in difficulties:
    # 获取对应难度和策略的行
    s_row = df[(df["Difficulty"] == diff) & (df["Strategy"] == "sphere_coord")]
    l_row = df[(df["Difficulty"] == diff) & (df["Strategy"] == "link_coord")]
    
    # 提取 Total_Pred_Cycles，如果不存在则为0
    sphere_cycles.append(s_row["Total_Pred_Cycles"].values[0] if not s_row.empty else 0)
    link_cycles.append(l_row["Total_Pred_Cycles"].values[0] if not l_row.empty else 0)

# 绘图配置
x = np.arange(len(difficulties))
width = 0.35

plt.figure(figsize=(10, 6))

# 修改：Link在左，Sphere在右
rects1 = plt.bar(x - width/2, link_cycles, width, label='Link Coord', color='lightcoral')
rects2 = plt.bar(x + width/2, sphere_cycles, width, label='Sphere Coord', color='skyblue')

# 添加标签和标题
plt.xlabel('Difficulty Level')
plt.ylabel('Total Prediction Cycles')
plt.title('Cycle Comparison: Link Coord vs Sphere Coord')
plt.xticks(x, difficulties)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱状图上方显示数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{int(height):,}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

# 计算并显示差距百分比
for i in range(len(difficulties)):
    l_val = link_cycles[i]
    s_val = sphere_cycles[i]
    
    if l_val > 0:
        # 计算相对于Link Coord的变化百分比
        diff_pct = (s_val - l_val) / l_val * 100
        
        # 确定文本位置（在两个柱子中间上方）
        max_height = max(l_val, s_val)
        
        # 格式化文本，正数为红色（增加），负数为绿色（减少 - 性能更好）
        text_color = 'green' if diff_pct < 0 else 'red'
        sign = '+' if diff_pct > 0 else ''
        
        plt.text(x[i], max_height + 10e5, f'{sign}{diff_pct:.1f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color=text_color)

plt.tight_layout()

# 保存图片
output_path = os.path.join(output_dir, "cycle_comparison_sphere_link.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
