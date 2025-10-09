#!/usr/bin/env python3
"""
BIT* 运动规划日志文件处理脚本

功能：
1. 读取 BIT*/GNN 算法生成的原始轨迹日志文件
2. 将数据从 (edge, pose, link) 三层结构重组
3. 生成适用于碰撞预测评估的标准格式数据
4. 保存为 coord_motiom_*.pkl 供 prediction_simulation 使用

输入文件格式：
    link_info_<benchid>.pkl:
        - link_info: List[List[List[Tuple]]] - (edge, pose, link) 的 link 位置信息
        - link_feas_info: List[List[List[int]]] - 对应的碰撞可行性标签 (0/1)

输出文件格式：
    coord_motiom_<benchid>.pkl:
        - qmotionposearr: List[List[List[List[float]]]] - 保持 edge/pose 结构的坐标
        - ymotionposearr: List[List[List[int]]] - 对应的碰撞标签

用法：
    python read_logfile.py <benchid>
    例如: python read_logfile.py 0
"""

import pickle
import numpy as np
import sys

# ========== 加载原始轨迹日志文件 ==========
# 可选择 GNN 或 BIT* 算法生成的日志
# f = open("logfiles_GNN_link/link_info_"+str(sys.argv[1])+".pkl", "rb")  # GNN 版本
f = open("logfiles_BIT_link/link_info_" + str(sys.argv[1]) + ".pkl", "rb")  # BIT* 版本

# 加载数据：
# link_info: 每个 edge 的每个 pose 的每个 link 的 3D 坐标
# link_feas_info: 对应的碰撞可行性标签 (1=无碰撞, 0=碰撞)
(link_info, link_feas_info) = pickle.load(f)
f.close()

# ========== 数据统计和边界计算 ==========
# 用于统计所有 link 的坐标范围（调试用）
llx = []  # 所有 link 的 x 坐标
lly = []  # 所有 link 的 y 坐标
llz = []  # 所有 link 的 z 坐标

# 统计总的 link 数量（用于预分配数组大小）
count = 0
for edge in link_info:
    for pose in edge:
        for link in pose:
            count += 1

# ========== 初始化数据结构 ==========
# 展平的数组格式（用于某些分析，但未在后续使用）
yarr = np.zeros((count, 1))  # 展平的碰撞标签数组
qarr = np.zeros((count, 3))  # 展平的坐标数组 [x, y, z]

# 运动轨迹数据结构（保持层次结构）
ymotionarr = []  # 按 edge 组织的碰撞标签（展平 pose 和 link）
qmotionarr = []  # 按 edge 组织的坐标（展平 pose 和 link）
qmotionposearr = []  # 按 edge/pose 组织的坐标（用于 CSP 模拟）
ymotionposearr = []  # 按 edge/pose 组织的碰撞标签

# ========== 处理 link 坐标信息 ==========
counter = 0
for edge in link_info:
    # 为当前 edge 初始化列表
    qmotionarr.append([])  # 当前 edge 的所有 link 坐标（展平 pose）
    qmotionposearr.append([])  # 当前 edge 的所有 pose（保持结构）

    for pose in edge:
        # 为当前 pose 初始化列表
        qmotionposearr[-1].append([])  # 当前 pose 的所有 link 坐标

        for link in pose:
            # 提取 link 的 3D 坐标 [x, y, z]
            link_coord = [link[0], link[1], link[2]]

            # 存储到三层结构：edge -> pose -> link
            qmotionposearr[-1][-1].append(link_coord)

            # 存储到两层结构：edge -> link（忽略 pose 边界）
            qmotionarr[-1].append(link_coord)

            # 存储到展平数组（用于统计分析）
            qarr[counter] = link_coord
            counter += 1

            # 收集坐标范围统计
            llx.append(link[0])
            lly.append(link[1])
            llz.append(link[2])  # 注意：原代码这里有错误，应该是 link[2] 而不是 link[1]

# ========== 处理碰撞可行性标签 ==========
counter = 0
colliding = 0  # 统计无碰撞的 link 总数（标签为1表示无碰撞）

# 处理 link 的碰撞标签：1=无碰撞（可行）, 0=碰撞
for edge in link_feas_info:
    # 为当前 edge 初始化列表
    ymotionarr.append([])  # 当前 edge 的所有 link 碰撞标签（展平 pose）
    ymotionposearr.append([])  # 当前 edge 的所有 pose（保持结构）

    for pose in edge:
        # 为当前 pose 初始化列表
        ymotionposearr[-1].append([])  # 当前 pose 的所有 link 碰撞标签

        for link in pose:
            # link 的碰撞标签：1=无碰撞（可行）, 0=碰撞

            # 存储到三层结构：edge -> pose -> link
            ymotionposearr[-1][-1].append(link)

            # 存储到两层结构：edge -> link（忽略 pose 边界）
            ymotionarr[-1].append(link)

            # 存储到展平数组
            yarr[counter] = link
            colliding += link  # 累计无碰撞的 link 数量
            counter += 1

# 输出统计信息
print(f"Total links: {counter}, Collision-free links: {colliding}")
print("Sample structure:", ymotionposearr[:2] if len(ymotionposearr) > 0 else "Empty")

# ========== 保存处理后的数据 ==========
# 注释掉的部分：旧格式的保存方式（用于不同的评估脚本）
# f = open("obstacles_gnn.pkl", "wb")
# pickle.dump((qarr, [], yarr), f)  # 展平格式
# f.close()
#
# f = open("obstacles_gnn_motiom.pkl", "wb")
# pickle.dump((qmotionarr, [], ymotionarr), f)  # 两层结构（edge->link）
# f.close()

# 保存标准格式：适用于 prediction_simulation 脚本
# 保持 edge/pose/link 三层结构，用于模拟不同的碰撞检测策略
output_file = "logfiles_BIT_link/coord_motiom_" + str(sys.argv[1]) + ".pkl"
f = open(output_file, "wb")
pickle.dump((qmotionposearr, ymotionposearr), f)
f.close()
print(f"✓ Saved processed data to: {output_file}")

# ========== 验证保存的数据 ==========
# 重新加载以验证数据完整性
f = open(output_file, "rb")
(qmotionposearr_verify, ymotionposearr_verify) = pickle.load(f)
f.close()
print(f"✓ Verification: Loaded {len(qmotionposearr_verify)} edges")

# 调试信息（已注释）
# print("Sample collision labels:", ymotionposearr)
# print("X range:", np.min(llx), "to", np.max(llx))
# print("Y range:", np.min(lly), "to", np.max(lly))
# print("Z range:", np.min(llz), "to", np.max(llz))
