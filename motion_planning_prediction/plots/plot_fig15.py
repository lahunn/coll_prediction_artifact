import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import math
import sys
import matplotlib

# --- 全局设置 ---
sns.set_theme(style="whitegrid")
# 设置matplotlib的字体属性
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 获取 Seaborn 配色
colors = sns.color_palette("deep")

# --- 数据加载与处理 ---
width = 1
name = sys.argv[1]
dfnp = pd.read_csv("result_files/" + name + ".csv", header=None, sep=" ")
dfnp = dfnp.sort_values(1)

parallel_q = []
cpu_q = []
oracle_q = []

num_entry = len(dfnp.axes[0])
bins = 5
binsize = math.ceil(num_entry / bins)

for i in range(0, len(dfnp.axes[0]), binsize):
    parallel_q.append([])
    cpu_q.append([])
    oracle_q.append([])
    for j in range(i, min(i + binsize, num_entry)):
        parallel_q[-1].append(dfnp.iloc[j, 0])
        cpu_q[-1].append(dfnp.iloc[j, 2])
        oracle_q[-1].append(dfnp.iloc[j, 3])

parallel_f = []
cpu_f = []
oracle_f = []

scale = np.mean(parallel_q[0])

for i, j, k in zip(parallel_q, cpu_q, oracle_q):
    parallel_f.append(np.mean(i) / scale)
    cpu_f.append(np.mean(j) / scale)
    oracle_f.append(np.mean(k) / scale)

# --- 绘图 ---
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)

group = list(range(0, bins * 4, 4))

# 使用 Seaborn 颜色
ax.bar(group, parallel_f, width, color=colors[0], label="CSP (基准)")
group = [x + 1 for x in group]
ax.bar(group, cpu_f, width, color=colors[1], label="CSP+CP (预测器)")
group = [x + 1 for x in group]
ax.bar(group, oracle_f, width, color=colors[2], label="Oracle (最优解)")

# 设置图例
ax.legend(loc='upper left')

# 设置x轴刻度和标签
ax.set_xticks([i - 1 for i in group])
lab = ["第" + str(i) + "组" for i in range(1, bins + 1)]
ax.set_xticklabels(lab, rotation=0)

# 设置轴标签
ax.set_xlabel("运动规划问题分组(按碰撞检测请求总数)")
ax.set_ylabel("碰撞检测次数 (归一化)")

plt.tight_layout()
plt.savefig(sys.argv[2] + "_" + name + ".pdf")

