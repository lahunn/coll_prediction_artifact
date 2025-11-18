import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import math
import sys
import matplotlib

# --- 全局设置 ---

# 设置matplotlib的字体属性，以确保在PDF和PS文件中正确嵌入字体
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# 定义绘图时使用的全局字体样式
font = {
    "family": "Times New Roman",  # 字体族
    "weight": "normal",  # 字体粗细
    "size": 35,  # 字体大小
}

# --- 数据加载与处理 ---

width = 1  # 定义柱状图的宽度
name = sys.argv[1]  # 从命令行第一个参数获取输入文件名（不含扩展名）

# 从CSV文件加载数据，分隔符为空格
dfnp = pd.read_csv("result_files/" + name + ".csv", header=None, sep=" ")

# 初始化列表，用于存储不同类别的数据
parallel_q = []  # 存储 "CSP" 方法的查询次数
serial_q = []  # 存储串行查询次数（未使用）
cpu_q = []  # 存储 "CSP+CP" 方法的查询次数
oracle_q = []  # 存储 "Oracle" 方法的查询次数

# 按第二列（索引为1）对数据进行排序。这一列通常代表了运动规划查询的某种难度度量（例如路径长度或环境复杂度）。
# 排序后，我们可以将查询从“最简单”到“最困难”进行分组。
dfnp = dfnp.sort_values(1)

# 将排序后的数据分成5个组（bins），以便分析不同难度下的性能
num_entry = len(dfnp.axes[0])  # 获取总查询次数
bins = 5  # 定义要分成的组数
binsize = math.ceil(num_entry / bins)  # 计算每组的大小

# 遍历所有数据行，并根据binsize将它们分配到对应的组中
for i in range(0, len(dfnp.axes[0]), binsize):
    # 为每个新组初始化空列表
    parallel_q.append([])
    serial_q.append([])
    cpu_q.append([])
    oracle_q.append([])
    # 将当前组的数据添加到相应的列表中
    for j in range(i, min(i + binsize, num_entry)):
        parallel_q[-1].append(dfnp.iloc[j, 0])
        serial_q[-1].append(dfnp.iloc[j, 1])
        cpu_q[-1].append(dfnp.iloc[j, 2])
        oracle_q[-1].append(dfnp.iloc[j, 3])

# --- 数据聚合与统计 ---

# 初始化列表，用于存储每个组的平均值（用于绘图）
parallel_f = []
cpu_f = []
oracle_f = []

# 打印总体统计数据：CSP 和 CSP+CP 的平均查询次数，以及相比于CSP的总体计算量减少百分比
print(
    "Overall Mean (CSP):",
    np.mean(dfnp[0]),
    "Overall Mean (CSP+CP):",
    np.mean(dfnp[2]),
    "Overall Reduction %:",
    (np.sum(dfnp[0]) - np.sum(dfnp[2])) / np.sum(dfnp[0]),
)
print(
    "Average computation reduction compared to CSP:  ",
    100 * (np.sum(dfnp[0]) - np.sum(dfnp[2])) / np.sum(dfnp[0]),
)

# 使用第一组（最简单的查询）的CSP平均查询次数作为归一化的基准
scale = np.mean(parallel_q[0])

# 计算每个组的平均查询次数，并使用scale进行归一化
for i, j, k in zip(parallel_q, cpu_q, oracle_q):
    parallel_f.append(np.mean(i) / scale)
    cpu_f.append(np.mean(j) / scale)
    oracle_f.append(np.mean(k) / scale)

# 打印第五组（最困难的查询）的计算量减少百分比
print(
    "Computation reduction compared to CSP For group 5 (more cluttered environments): ",
    100 * (parallel_f[4] - cpu_f[4]) / parallel_f[4],
)

# --- 绘图 ---

fig = plt.figure(figsize=(16, 5))  # 创建一个图形实例
plt.rc("font", **font)  # 应用全局字体设置
ax = fig.add_subplot(1, 1, 1)  # 添加一个子图

# 设置柱状图的x轴位置
group = list(range(0, bins * 4, 4))

# 绘制三组柱状图，分别代表 CSP, CSP+CP, 和 Oracle 方法
ax.bar(group, parallel_f, width, color="tab:orange", label="CSP")
group = [x + 1 for x in group]  # x轴位置右移，为下一组柱子腾出空间
ax.bar(group, cpu_f, width, color="tab:blue", label="CSP+CP")
group = [x + 1 for x in group]
ax.bar(group, oracle_f, width, color="tab:gray", label="Oracle")

# 设置图例
ax.legend(ncol=2)

# 设置x轴刻度和标签
ax.set_xticks([i - 1 for i in group])
lab = ["G" + str(i) for i in range(1, bins + 1)]  # 创建标签 G1, G2, ...
ax.set_xticklabels(lab, rotation=0)

# 设置x轴和y轴的标签
ax.set_xlabel("Groups of motion planning queries")
ax.set_ylabel("#Collision Queries\n(Normalized)")

# 调整布局以防止标签重叠
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# 从命令行第二个参数获取输出文件的前缀，并保存图形
plt.savefig(sys.argv[2] + "_" + name + ".pdf")
