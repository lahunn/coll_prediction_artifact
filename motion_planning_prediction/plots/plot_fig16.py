import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import math
from matplotlib.ticker import MaxNLocator
import pandas as pd
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

fig = plt.figure(figsize=(10, 6))

dfnp = pd.read_csv ("result_files/perf_data.csv",header=None,sep=" ")
print(dfnp)

ncdu=dfnp[0].to_numpy()
cycles=dfnp[1].to_numpy()
coll=dfnp[2].to_numpy()

utlization=( coll*40/(cycles*ncdu))

oocd_area=0.172
oocd_power=41.04
obb_area=0.054
obb_power= 30
copu_area=0.0465 
copu_power=4+11/4

#500 MHz frequency
throughpout=500000000/cycles

total_power= (utlization*ncdu*oocd_power+obb_power+[0,copu_power,0,copu_power,0,copu_power])/1000
total_area= (ncdu*oocd_area+obb_area+[0,copu_area,0,copu_area,0,copu_area])

perfw = throughpout/total_power
perfa = throughpout/total_area

perfw_norm = perfw/perfw[0]
perfa_norm = perfa/perfa[0]
tput_norm = throughpout/throughpout[0]
runtime_norm = 1/tput_norm

ax = fig.add_subplot(1,1,1)
group=list(range(1,7))

# 使用 Seaborn 颜色和中文图例
ax.plot(group, perfw_norm, color=colors[1], label="性能/功耗比", linewidth=0, marker="<", markersize=12)
ax.plot(group, perfa_norm, color=colors[0], label="性能/面积比", linewidth=0, marker="*", markersize=12)
ax.plot(group, runtime_norm, color=colors[2], label="运行时间", linewidth=0, marker="o", markersize=12)

ax.legend(ncol=3)
ax.set_xticks([i for i in group]) 
lab=["基准.1","COPU.1","基准.4","COPU.4","基准.6","COPU.6"]

ax.set_xticklabels(lab, rotation = 20)

ax.set_ylabel("性能指标 (归一化)")
ax.set_xlabel("硬件配置")
ax.set_ylim(0,2.5)

plt.tight_layout()
plt.savefig('perf_area_plot.pdf')

print("perf/W increase for COPU.1, COPU.4, and COPU.6: ",perfw_norm[1]/perfw_norm[0],perfw_norm[3]/perfw_norm[2],perfw_norm[5]/perfw_norm[4])
