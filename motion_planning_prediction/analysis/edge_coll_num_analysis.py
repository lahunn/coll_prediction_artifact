#!/usr/bin/env python3
"""
统计每个edge中碰撞检查数量并绘制分布图
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 添加 trace_generation 目录到 Python 路径
sys.path.append("../../trace_generation")
sys.path.append("..")
from trace_generation.config.ana_parameters import get_robot_params
import simulation_utils as su

# 创建result_files目录
os.makedirs("result_files", exist_ok=True)

# 参数设置
data_folder = "../../trace_files/scene_benchmarks/bit_collision_data"
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

# 收集所有edge的碰撞检查数量（sphere和link两种类型）
sphere_coll_check_counts = []
link_coll_check_counts = []
sphere_coll_check_counts_nocoll = []
sphere_coll_check_counts_withcoll = []
link_coll_check_counts_nocoll = []
link_coll_check_counts_withcoll = []

# 循环处理基准测试
benchrange = range(1, num_benchmarks + 1)
for benchid in tqdm(benchrange, desc="处理基准测试"):
    # 加载球体数据
    sphere_link_data, sphere_link_coll_data = su.load_data(
        basename, benchid, data_folder, collision_model_type="sphere"
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

        sphere_coll_check_counts.append(total_coll_checks)

        # 判断是否有碰撞（存在0表示有碰撞）
        has_collision = any(0 in pose for pose in edge_coll)
        if has_collision:
            sphere_coll_check_counts_withcoll.append(total_coll_checks)
        else:
            sphere_coll_check_counts_nocoll.append(total_coll_checks)

    # 加载link数据
    link_data, link_coll_data = su.load_data(
        basename, benchid, data_folder, collision_model_type="link"
    )

    if link_data is None or link_coll_data is None:
        continue

    # 处理每条边
    for edge_idx, (edge, edge_coll) in enumerate(zip(link_data, link_coll_data)):
        if not edge_coll:
            continue

        # 计算此edge中的碰撞检查数量
        # edge_coll是pose列表，每个pose是link碰撞结果列表
        num_poses = len(edge_coll)
        num_links_per_pose = len(edge_coll[0]) if edge_coll else 0
        total_coll_checks = num_poses * num_links_per_pose

        link_coll_check_counts.append(total_coll_checks)

        # 判断是否有碰撞（存在0表示有碰撞）
        has_collision = any(0 in pose for pose in edge_coll)
        if has_collision:
            link_coll_check_counts_withcoll.append(total_coll_checks)
        else:
            link_coll_check_counts_nocoll.append(total_coll_checks)


# 统计分析
def print_stats(
    coll_check_counts, collision_type, nocoll_counts=None, withcoll_counts=None
):
    if coll_check_counts:
        print(f"\n{collision_type}碰撞统计结果:")
        print(f"  总edge数: {len(coll_check_counts)}")
        print(f"  平均碰撞检查数: {np.mean(coll_check_counts):.1f}")
        print(f"  中位数: {np.median(coll_check_counts):.1f}")
        print(f"  最小值: {np.min(coll_check_counts)}")
        print(f"  最大值: {np.max(coll_check_counts)}")
        print(f"  标准差: {np.std(coll_check_counts):.1f}")

        # 分别统计无碰撞和有碰撞edge
        if nocoll_counts is not None and len(nocoll_counts) > 0:
            print(
                f"  无碰撞edge: {len(nocoll_counts)}, 平均检查数: {np.mean(nocoll_counts):.1f}"
            )
        if withcoll_counts is not None and len(withcoll_counts) > 0:
            print(
                f"  有碰撞edge: {len(withcoll_counts)}, 平均检查数: {np.mean(withcoll_counts):.1f}"
            )

        # 绘制分布图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(coll_check_counts, bins=50, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Number of Collision Checks per Edge")
        ax.set_ylabel("Number of Edges")
        ax.set_title(f"Distribution of Collision Checks per Edge ({collision_type})")
        ax.grid(True, alpha=0.3)

        # 保存图表
        filename = (
            f"result_files/edge_coll_check_distribution_{collision_type.lower()}.pdf"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"\n分布图已保存到: {filename}")
    else:
        print(f"\n未找到有效的{collision_type}碰撞数据")


print_stats(
    sphere_coll_check_counts,
    "Sphere",
    sphere_coll_check_counts_nocoll,
    sphere_coll_check_counts_withcoll,
)
print_stats(
    link_coll_check_counts,
    "Link",
    link_coll_check_counts_nocoll,
    link_coll_check_counts_withcoll,
)
