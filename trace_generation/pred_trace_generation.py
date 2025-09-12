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
from klampt.plan import cspace, robotplanning
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
from klampt import WorldModel, Geometry3D
import obb_calculator

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
        T.append(x[j][3])  # 提取平移分量
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
    new[0, 0] = math.cos(th)
    new[0, 1] = -1 * math.cos(al) * math.sin(th)
    new[0, 2] = math.sin(al) * math.sin(th)
    new[0, 3] = r * math.cos(th)

    new[1, 0] = math.sin(th)
    new[1, 1] = 1 * math.cos(al) * math.cos(th)
    new[1, 2] = -1 * math.sin(al) * math.cos(th)
    new[1, 3] = r * math.sin(th)

    new[2, 0] = 0
    new[2, 1] = math.sin(al)
    new[2, 2] = math.cos(al)
    new[2, 3] = d
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
    return ([(x), (y), (z)])


# ========== 核心 OBB 计算函数 ==========


def get_obbs(world, qbase, robot_urdf):
    """
    计算给定关节配置下所有 links 的 OBB (Oriented Bounding Box) 信息
    通用版本：支持任意机器人模型，通过 obb_calculator 模块自动计算 OBB
    
    Args:
        world: Klampt 世界模型
        qbase: 关节角配置 (DOF 数量根据机器人而定)
    
    Returns:
        dump_list: 包含所有 links OBB 信息的列表
                  每个元素为 [link_id, 中心坐标, 尺寸, 旋转矩阵, 方向编码]
    """

    # 获取机器人对象
    robot = world.robot(0)
    robot.setConfig(qbase)  # 设置当前关节配置

    # 获取机器人的 DOF 数量和 link 数量
    num_links = robot.numLinks()

    # 尝试使用 obb_calculator 计算精确的 OBB
    # 检查 obb_calculator 的依赖库是否可用
    deps_available, missing_libs = obb_calculator.check_dependencies()

    if deps_available:
        # 使用精确的 OBB 计算
        print(f"  Using precise OBB calculation for {num_links} links...")

        # 获取机器人的 URDF 路径 (如果可用)
        # 注意：Klampt 中可能没有直接获取 URDF 路径的方法
        # 这里我们回退到几何包围盒方法
        use_precise_obb = False
        robot_urdf_path = None

        # 尝试从机器人文件名推断 URDF 路径
        if hasattr(robot, 'getFilename'):
            robot_file = robot.getFilename()
            if robot_file and robot_file.endswith('.urdf'):
                robot_urdf_path = robot_file
                use_precise_obb = True

        if use_precise_obb and robot_urdf_path:
            # 使用精确的 CoACD + Open3D OBB 计算
            link_obbs = obb_calculator.calculate_link_obbs(robot_urdf_path, verbose=False)

            if link_obbs:
                print(f"  Successfully computed precise OBBs for {len(link_obbs)} links")

                dump_list = []
                for i, obb_info in enumerate(link_obbs):
                    # 提取 OBB 参数
                    obbc = obb_info['position']  # 中心位置
                    obbr = obb_info['rotation_matrix']  # 旋转矩阵
                    obbe = obb_info['extents']  # 尺寸

                    # 计算方向编码
                    dirstring = calculate_direction_encoding(obbe, obbr, obbc)

                    dump_list.append([i, obbc, obbe, obbr, dirstring])

                return dump_list


def calculate_direction_encoding(obbe, obbr, obbc):
    """
    计算方向编码字符串
    
    Args:
        obbe: OBB 尺寸 [x, y, z]
        obbr: 3x3 旋转矩阵
        obbc: OBB 中心位置
        
    Returns:
        dirstring: 2位方向编码字符串
    """
    try:
        # 使用 OBB 尺寸作为参考点
        newpoint = transform_point(obbe, obbr, obbc)
        direction = np.sign(newpoint - obbc)

        if direction[0] < 0:
            direction = direction * -1  # 标准化 x 方向

        direction = ((direction + 1) / 2)  # 归一化到 [0, 1]
        dirstring = str(int(direction[1])) + str(int(direction[2]))  # y,z 编码

        return dirstring

    except Exception as e:
        print(f"    Warning: Direction encoding failed: {e}")
        return "00"  # 默认编码


# ========== 主程序：数据生成流程 ==========

# 解析命令行参数
foldername = sys.argv[2]  # 环境文件夹路径
filenumber = sys.argv[3]  # 环境文件编号

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
robot = world.robot(0)  # 带障碍物环境中的机器人
robot1 = world1.robot(0)  # 无障碍物环境中的机器人 (用于采样)
qbase = robot.getConfig()  # 初始关节配置

# 设置随机种子确保可重现性
klampt.plan.motionplanning.setRandomSeed(2)

# 创建配置空间用于可行性采样
space = robotplanning.makeSpace(world1, robot1, edgeCheckResolution=0.005)

# ========== 数据存储数组初始化 ==========
numqueries = int(sys.argv[1])  # 需要采样的姿态数量

# 获取机器人的 link 数量
robot = world.robot(0)
num_links = robot.numLinks()
num_dofs = robot.numLinks()  # 假设 DOF 数量等于 link 数量，可根据需要调整

print(f"Robot has {num_links} links and {num_dofs} DOFs")

# link 级数据数组 (总共 num_links*numqueries 行)
qarr = np.zeros((num_links * numqueries, 3))  # OBB 中心坐标: [x, y, z]
dirarr = []  # 方向编码字符串列表
yarr = np.zeros((num_links * numqueries, 1))  # 碰撞标签: 1=自由, 0=碰撞

# pose 级数据数组 (总共 numqueries 行)
qarr_pose = np.zeros((numqueries, num_dofs))  # 关节配置: [q0, q1, ..., q_{num_dofs-1}]
yarr_pose = np.zeros((numqueries, 1))  # 整体碰撞标签: 1=整个机器人自由, 0=有碰撞

# 初始化采样
q = space.sample()
robot.setConfig(q)

# ========== 为每个 link 创建独立的碰撞检测器 ==========
# 关键设计：每个 link 的碰撞检测器忽略其他 links，只检测当前 link 与障碍物的碰撞
colliderlist = []
for lid in range(0, num_links):
    ignore = []  # 需要忽略的 links 列表
    for igid in range(0, num_links):
        if igid == lid:
            continue  # 不忽略当前 link
        ignore.append(robot.link(igid))  # 忽略其他所有 links

    # 创建只检测当前 link 的碰撞检测器
    collider = collide.WorldCollider(world, ignore=ignore)
    colliderlist.append(collider)

# ========== 主要数据生成循环 ==========
counter = 0  # 成功采样的姿态计数器
coll_count = 0  # 发生碰撞的姿态计数器

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
            for lid in range(0, num_links):
                collider = colliderlist[lid]  # 当前 link 的专用碰撞检测器
                ans = 1  # 默认无碰撞

                # 检查当前 link 与所有障碍物的碰撞
                for it in range(0, num_ob):
                    check1 = any(True for _ in collider.robotTerrainCollisions(robot.index, it))
                    if check1:
                        ans = 0  # 发现碰撞
                        break

                # 存储当前 link 的数据
                qarr[counter * num_links + lid] = obbs[lid][1]  # OBB 中心坐标
                yarr[counter * num_links + lid] = ans  # 碰撞标签
                dirarr.append(obbs[lid][4])  # 方向编码

            # ========== 整体机器人碰撞检测 ==========
            ans = 1  # 默认整体无碰撞
            for it in range(0, num_ob):
                check1 = any(True for _ in collider_w.robotTerrainCollisions(robot.index, it))
                if check1:
                    coll_count += 1  # 碰撞计数器增加
                    ans = 0  # 标记为碰撞
                    break

            # ========== 存储姿态级数据 ==========
            # 注意：这里移除了原来针对特定机器人的关节角偏移调整
            # 如果需要特定的预处理，可以在这里添加

            qarr_pose[counter] = q  # 存储关节配置
            yarr_pose[counter] = ans  # 存储整体碰撞标签
            counter += 1  # 成功处理一个姿态

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
