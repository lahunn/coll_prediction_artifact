"""
机器人碰撞检测数据生成脚本
用于生成微架构仿真所需的训练/测试数据集

主要功能：
1. 在给定障碍物环境中随机采样机器人姿态
2. 计算每个姿态下各个 link 的 OBB (Oriented Bounding Box) 几何信息  
3. 执行精确的碰撞检测，生成 ground truth 标签
4. 输出 link 级和 pose 级的碰撞检测数据供后续仿真使用

输入参数：
- sys.argv[1]: numqueries (采样姿态数量)
- sys.argv[2]: foldername (环境文件夹)  
- sys.argv[3]: filenumber (环境文件编号)

输出文件：
- obstacles_X_coord.pkl: link 级数据 (qarr, dirarr, yarr)
- obstacles_X_pose.pkl: pose 级数据 (qarr_pose, yarr_pose)
"""

import klampt
from klampt.plan import cspace,robotplanning
from klampt.io import resource
from klampt.model import collide
import klampt.model
from klampt.model import trajectory
import time
import sys
from klampt import vis
import pickle
from typing import NamedTuple
from collections import namedtuple
import numpy as np
import random
import math
from klampt.model.create import primitives
from klampt import WorldModel,Geometry3D

# ========== 几何变换辅助函数 ==========

def give_RT(x):
    """
    从 4x4 变换矩阵中提取旋转矩阵 R 和平移向量 T
    
    Args:
        x: 4x4 齐次变换矩阵
    Returns:
        R: 展平的 3x3 旋转矩阵 (9个元素的列表)
        T: 3x1 平移向量 (3个元素的列表)
    """
    R = []
    T = []
    for j in range(0, 3):
        for i in range(0, 3):
            R.append(x[i][j])  # 按列优先顺序展平旋转矩阵
        T.append(x[j][3])      # 提取平移分量
    return (R, T)

def get_obbRT(R, T):
    """
    将展平的旋转矩阵和平移向量重构为标准的 numpy 数组格式
    
    Args:
        R: 展平的旋转矩阵 (9个元素)
        T: 平移向量 (3个元素)
    Returns:
        newR: 3x3 numpy 旋转矩阵
        newT: 3x1 numpy 平移向量
    """
    newR = np.eye(3)
    newT = np.array(T)
    pointer = 0
    for j in range(0, 3):
        for i in range(0, 3):
            newR[i][j] = R[pointer]  # 重构 3x3 旋转矩阵
            pointer += 1
    return newR, newT

def give_dh(d, r, th, al):
    """
    根据 DH (Denavit-Hartenberg) 参数构建齐次变换矩阵
    用于机器人运动学正向计算
    
    Args:
        d: 连杆偏移 (link offset)
        r: 连杆长度 (link length) 
        th: 关节角 (joint angle)
        al: 连杆扭转角 (link twist)
    Returns:
        new: 4x4 DH 变换矩阵
    """
    new = np.eye(4)
    # DH 变换矩阵的标准公式
    new[0,0] = math.cos(th)
    new[0,1] = -1 * math.cos(al) * math.sin(th)
    new[0,2] = math.sin(al) * math.sin(th)
    new[0,3] = r * math.cos(th)

    new[1,0] = math.sin(th)
    new[1,1] = 1 * math.cos(al) * math.cos(th)
    new[1,2] = -1 * math.sin(al) * math.cos(th)
    new[1,3] = r * math.sin(th)

    new[2,0] = 0
    new[2,1] = math.sin(al)
    new[2,2] = math.cos(al)
    new[2,3] = d
    return new

def get_RT(x):
    """
    从旋转矩阵和平移向量重构 4x4 齐次变换矩阵
    
    Args:
        x: 包含 [R, T] 的元组，R为展平旋转矩阵，T为平移向量
    Returns:
        new: 4x4 齐次变换矩阵
    """
    R = x[0]
    T = x[1]
    new = np.eye(4)
    pointer = 0
    for j in range(0, 3):
        for i in range(0, 3):
            new[i][j] = R[pointer]
            pointer += 1
        new[j][3] = T[j]
    return new

def transform_point(p, R, T):
    """
    应用旋转和平移变换到给定点
    
    Args:
        p: 输入点坐标 (3D 向量)
        R: 3x3 旋转矩阵
        T: 3x1 平移向量
    Returns:
        newT: 变换后的点坐标
    """
    new = np.zeros((3, 1))
    new[:, 0] = p
    newT = np.array(T)
    temp = np.matmul(R, new)
    newT[0] += temp[0, 0]
    newT[1] += temp[1, 0]
    newT[2] += temp[2, 0]
    return newT

def inverse(R):
    """
    从旋转矩阵提取欧拉角 (暂未在主程序中使用)
    """
    x = math.atan(R[1][2] / R[0][2])
    y = math.atan(math.sqrt(R[0][2]**2 + R[1][2]**2) / R[2][2])
    z = math.atan(-1 * R[2][1] / R[2][0])
    return([(x), (y), (z)])

# ========== 核心 OBB 计算函数 ==========

def get_obbs(world, qbase):
    """
    计算给定关节配置下所有 links 的 OBB (Oriented Bounding Box) 信息
    这是整个数据生成的核心函数，为每个 link 计算几何特征和方向编码
    
    Args:
        world: Klampt 世界模型
        qbase: 7-DOF 关节角配置 [q0, q1, q2, q3, q4, q5, q6]
    
    Returns:
        dump_list: 包含所有 links OBB 信息的列表
                  每个元素为 [link_id, 中心坐标, 尺寸, 旋转矩阵, 方向编码]
    """
    
    # ========== 通过 DH 参数计算各 link 的变换矩阵 ==========
    # 基于 KUKA iiwa 机器人的 DH 参数
    dh01 = give_dh(0.1535, 0, 0, 0)                    # base -> link1
    dh1e = give_dh(0.08, 0, qbase[1], 0)               # link1 末端调整

    dh12 = give_dh(0.1185, 0, qbase[1], -1*math.pi/2)  # link1 -> link2  
    dh2e = give_dh(-0.029, 0.206, qbase[2], 0)         # link2 末端调整

    dh23 = give_dh(0.0, 0.41, qbase[2], 0)             # link2 -> link3
    dh3e = give_dh(0.014, -0.08, -1*math.pi/2+qbase[3], -1*math.pi/2)  # link3 末端

    dh34 = give_dh(0.01125, 0.0, qbase[3], -1*math.pi/2)  # link3 -> link4
    # link4 的复杂变换链
    dh4e = give_dh(0.207, 0.0, qbase[4], 0)
    dh4ed = np.matmul(give_dh(0.05, 0.0, 0, 0.61), give_dh(0.0, 0.014, -1*math.pi/2, 0.0))
    dh45 = give_dh(0.207+0.0658, 0.00, qbase[4], math.pi+0.9599)  # link4 -> link5

    # link5 变换
    dh5e = give_dh(-0.028, 0.01968, (math.pi/2), 0.0)
    dh5ed = give_dh(-0.050, 0.01, -qbase[5], 0.0)

    # link6 变换  
    dh56 = np.matmul(give_dh(-0.028, 0.01968, (math.pi/2), 0.0), give_dh(0, 0, -qbase[5], 0))
    dh6e = np.matmul(give_dh(0, 0, -1*math.pi/2, 0.9599), give_dh(-0.0658, -0.0343, math.pi/2, 0))
    dh6ed = give_dh(-0.055, 0, -qbase[6], 0)

    # ========== 累积变换：从基座到各 link 的世界坐标变换 ==========
    fin1e = np.matmul(dh01, dh1e)
    fin2e = np.matmul(dh01, np.matmul(dh12, dh2e))
    fin3e = np.matmul(dh01, np.matmul(dh12, np.matmul(dh23, dh3e)))
    fin4e = np.matmul(dh01, np.matmul(dh12, np.matmul(dh23, np.matmul(dh34, np.matmul(dh4e, dh4ed)))))
    fin5e = np.matmul(dh01, np.matmul(dh12, np.matmul(dh23, np.matmul(dh34, np.matmul(dh45, np.matmul(dh5e, (dh5ed)))))))
    fin6e = np.matmul(dh01, np.matmul(dh12, np.matmul(dh23, np.matmul(dh34, np.matmul(dh45, np.matmul(dh56, np.matmul(dh6e, dh6ed)))))))

    # ========== 各 link 的物理尺寸定义 ==========
    # 每个 link 的包围盒尺寸 [宽度, 深度, 高度] (单位: 米)
    sizelist = [
        [0.10, 0.091, 0.16],    # link1 尺寸
        [0.494, 0.083, 0.053],  # link2 尺寸
        [0.25, 0.064, 0.10],    # link3 尺寸
        [0.072, 0.064, 0.096],  # link4 尺寸
        [0.084, 0.066, 0.1],    # link5 尺寸
        [0.092, 0.116, 0.11],   # link6 尺寸
        [0.084, 0.089, 0.1535]  # link0 (base) 尺寸
    ]
    
    dump_list = []
    namelist = ["l1", "l2", "l3", "l4", "l5", "l6", "l0"]
    finlist = [fin1e, fin2e, fin3e, fin4e, fin5e, fin6e]

    # ========== 处理 link0 (基座) ==========
    s = sizelist[6]  # 基座尺寸
    obbc = np.array([0, 0, s[2]*0.5])  # 基座中心：z 方向偏移半高度
    obbe = np.array([s[0], s[1], s[2]])  # 基座范围
    obbr = np.eye(3)  # 基座旋转矩阵 (单位矩阵，无旋转)
    
    # 计算基座的方向编码
    newpoint = (transform_point(np.array([s[0], s[1], s[2]]), obbr, obbc))
    direction = np.sign(newpoint - obbc)
    if direction[0] < 0:
        direction = direction * -1  # 确保 x 方向为正
    direction = ((direction + 1) / 2)  # 归一化到 [0, 1]
    dirstring = str(int(direction[1])) + str(int(direction[2]))  # y,z 方向编码
    dump_list.append([0, obbc, obbe, obbr, dirstring])

    # ========== 处理 link1-6 ==========
    for i in range(0, 6):
        # 从累积变换矩阵中提取旋转和平移
        R, T = give_RT(finlist[i])
        s = sizelist[i]  # 当前 link 尺寸
        
        # 重构旋转矩阵和平移向量
        obbr, obbc = get_obbRT(R, T)
        
        # 计算方向编码：基于变换后的特定点相对于中心的方向
        newpoint = (transform_point(np.array([s[0], s[1], s[2]]), obbr, obbc))
        direction = np.sign(newpoint - obbc)
        if direction[0] < 0:
            direction = direction * -1  # 标准化 x 方向
        direction = ((direction + 1) / 2)  # 归一化
        dirstring = str(int(direction[1])) + str(int(direction[2]))  # y,z 编码
        
        obbe = np.array([s[0], s[1], s[2]])  # OBB 范围
        dump_list.append([i+1, obbc, obbe, obbr, dirstring])

    return dump_list

# ========== 主程序：数据生成流程 ==========

# 解析命令行参数
foldername = sys.argv[2]   # 环境文件夹路径
filenumber = sys.argv[3]   # 环境文件编号

# ========== 环境和机器人初始化 ==========
# 加载包含障碍物的环境
world = klampt.WorldModel()
world.readFile(foldername + "/obstacles_" + filenumber + ".xml")
print(foldername + "/obstacles_" + filenumber + ".xml")

# 加载无障碍物的机器人模型 (用于可行性检查)
world1 = klampt.WorldModel()
world1.readFile("jaco_collision.xml")

# 初始化碰撞检测器
collider_w = collide.WorldCollider(world)
num_ob = collider_w.world.numTerrains()  # 障碍物数量

# 获取机器人对象
robot = world.robot(0)    # 带障碍物环境中的机器人
robot1 = world1.robot(0)  # 无障碍物环境中的机器人 (用于采样)
qbase = robot.getConfig()  # 初始关节配置

# 设置随机种子确保可重现性
klampt.plan.motionplanning.setRandomSeed(2)

# 创建配置空间用于可行性采样
space = robotplanning.makeSpace(world1, robot1, edgeCheckResolution=0.005)

# ========== 数据存储数组初始化 ==========
numqueries = int(sys.argv[1])  # 需要采样的姿态数量

# link 级数据数组 (总共 7*numqueries 行)
qarr = np.zeros((7*numqueries, 3))     # OBB 中心坐标: [x, y, z]
dirarr = []                           # 方向编码字符串列表
yarr = np.zeros((7*numqueries, 1))     # 碰撞标签: 1=自由, 0=碰撞

# pose 级数据数组 (总共 numqueries 行)  
qarr_pose = np.zeros((numqueries, 7))  # 关节配置: [q0, q1, ..., q6]
yarr_pose = np.zeros((numqueries, 1))  # 整体碰撞标签: 1=整个机器人自由, 0=有碰撞

# 初始化采样
q = space.sample()
robot.setConfig(q)

# ========== 为每个 link 创建独立的碰撞检测器 ==========
# 关键设计：每个 link 的碰撞检测器忽略其他 links，只检测当前 link 与障碍物的碰撞
colliderlist = []
for lid in range(0, 7):
    ignore = []  # 需要忽略的 links 列表
    for igid in range(0, 7):
        if igid == lid:
            continue  # 不忽略当前 link
        ignore.append(robot.link(igid))  # 忽略其他所有 links
    
    # 创建只检测当前 link 的碰撞检测器
    collider = klampt.model.collide.WorldCollider(world, ignore=ignore)
    colliderlist.append(collider)

# ========== 主要数据生成循环 ==========
counter = 0      # 成功采样的姿态计数器
coll_count = 0   # 发生碰撞的姿态计数器

while counter < numqueries:
    feasible = 0
    
    # ========== 采样可行的机器人配置 ==========
    while feasible == 0:
        q = space.sample()  # 随机采样关节配置
        if space.isFeasible(q):  # 检查配置是否在关节限制内且无自碰撞
            feasible = 1
            robot.setConfig(q)  # 设置机器人到采样配置
            obbs = get_obbs(world, q)  # 计算所有 links 的 OBB 信息
            
            # ========== 逐 link 碰撞检测 ==========
            for lid in range(0, 7):
                collider = colliderlist[lid]  # 当前 link 的专用碰撞检测器
                ans = 1  # 默认无碰撞
                
                # 检查当前 link 与所有障碍物的碰撞
                for it in range(0, num_ob):
                    check1 = any(True for _ in collider.robotTerrainCollisions(robot.index, it))
                    if check1:
                        ans = 0  # 发现碰撞
                        break
                
                # 存储当前 link 的数据
                qarr[counter*7 + lid] = obbs[lid][1]    # OBB 中心坐标
                yarr[counter*7 + lid] = ans             # 碰撞标签
                dirarr.append(obbs[lid][4])             # 方向编码
            
            # ========== 整体机器人碰撞检测 ==========
            ans = 1  # 默认整体无碰撞
            for it in range(0, num_ob):
                check1 = any(True for _ in collider_w.robotTerrainCollisions(robot.index, it))
                if check1:
                    coll_count += 1  # 碰撞计数器增加
                    ans = 0          # 标记为碰撞
                    break
            
            # ========== 存储姿态级数据 ==========
            # 对某些关节角进行偏移调整 (可能是为了数据预处理/标准化)
            q[2] += 3.97935067
            q[3] += 4.41568301
            
            qarr_pose[counter] = q    # 存储调整后的关节配置
            yarr_pose[counter] = ans  # 存储整体碰撞标签
            counter += 1              # 成功处理一个姿态

# ========== 结果输出和数据保存 ==========
print("collision count", coll_count, "out of ", numqueries)

# 保存 link 级数据到 coord.pkl
f = open(foldername + "/obstacles_" + filenumber + "_coord.pkl", "wb")
pickle.dump((qarr, dirarr, yarr), f)
f.close()

# 保存 pose 级数据到 pose.pkl  
f = open(foldername + "/obstacles_" + filenumber + "_pose.pkl", "wb")
pickle.dump((qarr_pose, yarr_pose), f)
f.close()

# 可选：可视化调试
# vis.debug(world)