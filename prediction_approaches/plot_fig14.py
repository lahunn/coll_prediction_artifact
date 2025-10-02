import random
from tqdm import tqdm
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import math
from matplotlib.ticker import MaxNLocator
import pandas as pd
import sys
import matplotlib

# 设置matplotlib的字体属性，以确保在PDF和PS文件中正确嵌入字体
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 定义绘图时使用的字体样式
font = {
    'family' : 'Times New Roman', # 字体族
    'weight' : 'normal',         # 字体粗细
    'size'   : 35,                # 字体大小
}

def find_sim_cost(R,C,A,N):
    """
    通过蒙特卡洛模拟计算预期的计算成本（运行次数）。

    参数:
    R (float): 真实碰撞率。
    C (float): 预测器的覆盖率（召回率）。
    A (float): 预测器的准确率（精确率）。
    N (int): 检查的总样本数。

    返回:
    float: 10000次模拟运行的平均成本。
    """
    all_runs=0
    # 运行10000次模拟以获得稳定的平均值
    for ex in tqdm(range(0,10000)):
        pred=[] # 存储预测为会碰撞的样本
        non_pred=[] # 存储预测为不会碰撞的样本
        
        # 根据预测器触发的概率，将N个样本分类
        for i in range(0,N):
            # 预测器触发（预测为碰撞）的概率为 R*C/A
            if random.random()<=R*C/A:
                pred.append(1)
            else:
                non_pred.append(1)
        
        coll=0 # 标志位，指示是否检测到碰撞
        runs=0 # 本次模拟的运行次数（成本）

        # 首先检查被预测为会碰撞的样本
        for i in pred:
            runs+=1
            # 如果预测为碰撞，实际发生碰撞的概率为A（准确率）
            if random.random()<=A:
                coll=1
                break
        
        # 如果在预测碰撞的样本中未发现碰撞，则继续检查未被预测的样本
        if coll==0:
            for i in non_pred:
                runs+=1
                # 如果预测为不碰撞，实际发生碰撞的概率为 R*(1-C/A)
                if random.random()<=(R*(1-C/A)):
                    coll=1
                    break
        all_runs+=runs
    # 返回所有模拟的平均运行次数
    return (all_runs/10000)


# 从CSV文件中加载数据，该文件包含中等障碍物密度下的准确率和覆盖率
dflow = pd.read_csv ("result_files/coord_mid_su.csv",header=None,sep=",")
mid_acc2=dflow[0].tolist() # 提取准确率（精确率）
mid_cov2=dflow[1].tolist() # 提取覆盖率（召回率）


mid_cost2=[]
width = 1 # 柱状图的宽度
N=80      # 每次模拟的总样本数
R=0.11    # 环境的真实碰撞率

# 遍历准确率和覆盖率数据，计算每个配置的模拟成本
for i,j in zip(mid_cov2,mid_acc2):
    # 调用函数计算模拟成本
    sim_cost=find_sim_cost(R,i/100,j/100,N)
    mid_cost2.append(sim_cost)

# 对计算成本进行归一化，以便在图上与其他百分比数据进行比较
scale=np.max(mid_cost2) # 找到最大成本作为缩放因子
mid_cost_scale2=[i*100/scale for i in mid_cost2] # 将成本缩放到0-100范围
print(mid_cost_scale2)

# --- 开始绘图 ---
fig = plt.figure(figsize=(16,6)) # 创建一个图形实例
plt.rc('font', **font)  # 应用之前定义的字体设置
ax = fig.add_subplot(1,1,1) # 添加一个子图

# 设置柱状图的x轴位置
group=list(range(0,30,5))

# 绘制代表精确率的蓝色柱状图
ax.bar(group,mid_acc2,width, color="navy",label="Precision (%)")
group= [x +1 for x in group] # x轴位置右移，为下一组柱子腾出空间
# 绘制代表召回率的浅蓝色柱状图
ax.bar(group,mid_cov2,width, color="cornflowerblue",label="Recall (%)")
group= [x +1 for x in group] # x轴位置再次右移
# 绘制代表归一化计算成本的橙色柱状图
ax.bar(group,mid_cost_scale2,width, color="tab:orange",label="Compute (%)")

# 设置图例
ax.legend(ncol=2,fontsize=28)
# 设置x轴刻度和标签
ax.set_xticks([i-1 for i in group]) 
ax.set_xticklabels(["Random","S=0
U=0","S=1/2
U=1","S=1
U=1/2","S=2
U=1/4","S=4
U=1/8"], rotation = 0)
# 设置y轴刻度和标签
ax.set_yticks([0,50,100]) 
ax.set_yticklabels(["0%","50%","100%"], rotation = 0)
# 在图上添加一条垂直灰线，以区分基线和COORD方法
ax.axvline(x = 3.5,ymin=0,ymax=1, color = 'gray',lw=0.5)

# 在图的下方添加文本标签
plt.text(1, -26, "Baseline",ha="center",va="top", fontsize=36,color="tab:blue")
plt.text(16, -26, "COORD Collision prediction",ha="center",va="top", fontsize=36,color="tab:blue")
# 调整布局以防止标签重叠
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# 将最终的图形保存为PDF文件
plt.savefig('fig14_coord_prediction_mid_upfreq.pdf')