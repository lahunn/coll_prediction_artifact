#!/usr/bin/env python3
"""
统计每个edge中碰撞检查数量并绘制分布图
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加 trace_generation 目录到 Python 路径
sys.path.append('../trace_generation')
from trace_generation.config.ana_parameters import get_robot_params
import simulation_utils as su

# 参数设置
data_folder = "../trace_files/scene_benchmarks/bit_collision_data"
basename = "iiwa_7"
num_benchmarks = 50
robot_name = "iiwa"

print("=== Edge碰撞检查数量统计 ===")
print(f"数据文件夹: {data_folder}")
print(f"基准测试: {basename}")
print(f"基准测试数量: {num_benchmarks}")
print(f"机器人: {robot_name}")
print("=" * 50)

# 获取机器人参数
robot_params = get_robot_params(robot_name)
sphere_num = robot_params["sphere_num"]

# 收集所有edge的碰撞检查数量
coll_check_counts = []

# 循环处理基准测试
benchrange = range(1, num_benchmarks + 1)
for benchid in tqdm(benchrange, desc="处理基准测试"):
    # 加载球体数据
    sphere_link_data, sphere_link_coll_data = (
        su.load_data(basename, benchid, data_folder, collision_model_type="sphere")
    )

    if sphere_link_data is None or sphere_link_coll_data is None:
        continue

    # 处理每条边
    for edge_idx, (edge, edge_coll) in enumerate(
        zip(sphere_link_data, sphere_link_coll_data)
    ):
        if not edge_coll:
            continue

        # 计算此edge中的碰撞检查数量
        # edge_coll是pose列表，每个pose是sphere碰撞结果列表
        num_poses = len(edge_coll)
        num_spheres_per_pose = len(edge_coll[0]) if edge_coll else 0
        total_coll_checks = num_poses * num_spheres_per_pose

        coll_check_counts.append(total_coll_checks)

# 统计分析
if coll_check_counts:
    print("\n统计结果:")
    print(f"  总edge数: {len(coll_check_counts)}")
    print(f"  平均碰撞检查数: {np.mean(coll_check_counts):.1f}")
    print(f"  中位数: {np.median(coll_check_counts):.1f}")
    print(f"  最小值: {np.min(coll_check_counts)}")
    print(f"  最大值: {np.max(coll_check_counts)}")
    print(f"  标准差: {np.std(coll_check_counts):.1f}")

    # 绘制分布图
    plt.figure(figsize=(10, 6))
    plt.hist(coll_check_counts, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Collision Checks per Edge')
    plt.ylabel('Number of Edges')
    plt.title('Distribution of Collision Checks per Edge')
    plt.grid(True, alpha=0.3)

    # 保存图表
    plt.savefig('result_files/edge_coll_check_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n分布图已保存到: result_files/edge_coll_check_distribution.pdf")
else:
    print("未找到有效的碰撞数据")