#!/usr/bin/env python3
"""
碰撞检测不一致分析工具

用于分析OBB和球体碰撞检测结果不一致的原因
"""

import sys
import os
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../bit_planning"))
sys.path.insert(0, os.path.dirname(__file__))

from trace_generation.sphere_as.sphere_method import SphereEnv


def analyze_single_inconsistent_config(
    problem_file, config_idx, obb_data_dir, robot_name="franka", visualize=True
):
    """
    详细分析单个不一致的配置
    """
    print(f"分析配置 {config_idx} 在文件 {problem_file}")

    # 加载problem数据
    with open(problem_file, "rb") as f:
        problem_data = pickle.load(f)

    obstacles = problem_data["obstacles"]
    configs = problem_data["configs"]

    if config_idx >= len(configs):
        print(f"配置索引 {config_idx} 超出范围")
        return

    config = configs[config_idx]
    print(f"关节配置: {config}")

    # 加载OBB数据
    basename = os.path.basename(problem_file)
    obb_filename = basename.replace(".pkl", "_obb.pkl")
    obb_filepath = os.path.join(obb_data_dir, obb_filename)

    with open(obb_filepath, "rb") as f:
        obb_data, _ = pickle.load(f)

    obb_edge = obb_data[config_idx]
    obb_config = obb_edge[0]  # 取第一个配置
    obb_collision = any(coll == 0 for coll in obb_config)  # 0表示碰撞

    print(f"OBB碰撞结果: {obb_collision} (links: {obb_config})")

    # 创建球体环境并分析
    sphere_env = SphereEnv(robot_name=robot_name)
    sphere_env.load_obstacles(obstacles)

    # 获取球体数据
    collision, coords, colls = sphere_env.get_sphere_collision_data(config)
    sphere_collision = any(coll == 0 for coll in colls)

    print(f"球体碰撞结果: {sphere_collision} (spheres: {colls})")
    print(f"球体坐标数量: {len(coords)}")

    # 打印详细信息
    print("\n球体详细信息:")
    for i, (coord, coll) in enumerate(zip(coords, colls)):
        print(f"  球体 {i}: 位置({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}), 半径{coord[3]:.3f}, 碰撞:{coll}")

    # 可视化
    if visualize:
        visualize_config_with_obstacles(config, coords, obstacles, obb_collision, sphere_collision)

    sphere_env.cleanup_obstacles()
    sphere_env.close()

    return {
        'config': config,
        'obb_collision': obb_collision,
        'obb_links': obb_config,
        'sphere_collision': sphere_collision,
        'sphere_colls': colls,
        'sphere_coords': coords,
        'obstacles': obstacles
    }


def visualize_config_with_obstacles(config, sphere_coords, obstacles, obb_collision, sphere_collision):
    """
    可视化配置、球体和障碍物
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制障碍物
    for i, (half_extents, base_pos) in enumerate(obstacles):
        # 创建障碍物边界框
        x = [base_pos[0] - half_extents[0], base_pos[0] + half_extents[0]]
        y = [base_pos[1] - half_extents[1], base_pos[1] + half_extents[1]]
        z = [base_pos[2] - half_extents[2], base_pos[2] + half_extents[2]]

        # 绘制线框
        vertices = [(x[0], y[0], z[0]), (x[1], y[0], z[0]), (x[1], y[1], z[0]), (x[0], y[1], z[0]),
                   (x[0], y[0], z[1]), (x[1], y[0], z[1]), (x[1], y[1], z[1]), (x[0], y[1], z[1])]

        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]

        for s, e in edges:
            ax.plot3D([vertices[s][0], vertices[e][0]],
                     [vertices[s][1], vertices[e][1]],
                     [vertices[s][2], vertices[e][2]], color='red', alpha=0.7)

    # 绘制球体
    for i, (x, y, z, r) in enumerate(sphere_coords):
        # 绘制球体表面
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xs = x + r * np.cos(u) * np.sin(v)
        ys = y + r * np.sin(u) * np.sin(v)
        zs = z + r * np.cos(v)
        ax.plot_surface(xs, ys, zs, color='blue', alpha=0.3)

        # 绘制球心
        ax.scatter([x], [y], [z], color='blue')

    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'碰撞检测对比 - OBB:{obb_collision}, Sphere:{sphere_collision}')

    plt.tight_layout()
    plt.show()


def analyze_inconsistency_statistics(problem_file, obb_data_dir, robot_name="franka"):
    """
    分析不一致的统计信息
    """
    print(f"分析文件: {problem_file}")

    # 加载problem数据
    with open(problem_file, "rb") as f:
        problem_data = pickle.load(f)

    obstacles = problem_data["obstacles"]
    configs = problem_data["configs"]

    # 加载OBB数据
    basename = os.path.basename(problem_file)
    obb_filename = basename.replace(".pkl", "_obb.pkl")
    obb_filepath = os.path.join(obb_data_dir, obb_filename)

    with open(obb_filepath, "rb") as f:
        obb_data, _ = pickle.load(f)

    # 创建球体环境
    sphere_env = SphereEnv(robot_name=robot_name)
    sphere_env.load_obstacles(obstacles)

    # 统计
    total_configs = len(configs)
    inconsistent_count = 0
    obb_only_collisions = 0
    sphere_only_collisions = 0

    print("分析配置不一致情况...")

    for i, config in enumerate(configs):
        if i >= len(obb_data):
            break

        # OBB结果
        obb_edge = obb_data[i]
        obb_config = obb_edge[0]
        obb_collision = any(coll == 0 for coll in obb_config)

        # 球体结果
        collision, coords, colls = sphere_env.get_sphere_collision_data(config)
        sphere_collision = any(coll == 0 for coll in colls)

        if obb_collision != sphere_collision:
            inconsistent_count += 1
            if obb_collision and not sphere_collision:
                obb_only_collisions += 1
            elif sphere_collision and not obb_collision:
                sphere_only_collisions += 1

    sphere_env.cleanup_obstacles()
    sphere_env.close()

    print("\n统计结果:")
    print(f"总配置数: {total_configs}")
    print(f"不一致配置数: {inconsistent_count} ({inconsistent_count/total_configs*100:.1f}%)")
    print(f"仅OBB检测到碰撞: {obb_only_collisions}")
    print(f"仅球体检测到碰撞: {sphere_only_collisions}")


def main():
    parser = argparse.ArgumentParser(description="碰撞检测不一致分析工具")

    parser.add_argument("--problem-file", type=str, required=True, help="problem文件路径")
    parser.add_argument("--obb-data-dir", type=str, required=True, help="OBB数据目录")
    parser.add_argument("--robot-name", type=str, default="franka", help="机器人名称")
    parser.add_argument("--config-idx", type=int, help="要分析的配置索引")
    parser.add_argument("--analyze-stats", action="store_true", help="分析统计信息")
    parser.add_argument("--visualize", action="store_true", help="可视化配置")

    args = parser.parse_args()

    if args.analyze_stats:
        analyze_inconsistency_statistics(args.problem_file, args.obb_data_dir, args.robot_name)
    elif args.config_idx is not None:
        analyze_single_inconsistent_config(
            args.problem_file, args.config_idx, args.obb_data_dir,
            args.robot_name, args.visualize
        )
    else:
        print("请指定 --analyze-stats 或 --config-idx")


if __name__ == "__main__":
    main()