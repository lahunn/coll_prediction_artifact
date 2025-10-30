#!/usr/bin/env python3
"""
对比ModularEnv和球体碰撞检测结果

该程序读取obstacle_config_file，
使用ModularEnv和球体模型分别计算碰撞数据，并进行对比。
"""

import sys
import os
import pickle
import argparse

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from sphere_method import SphereEnv


def compare_collision(
    obstacle_config_file,
    collision_data_file,
    robot_name="franka",
    output_file=None,
    benchmark_id=None,
):
    """
    对比OBB和球体碰撞检测结果

    Args:
        obstacle_config_file: 障碍物-配置文件路径
        collision_data_file: OBB碰撞数据文件路径
        robot_name: 机器人名称
        output_file: 球体碰撞数据输出文件路径（可选）
    """
    print(f"加载obstacle_config_file: {obstacle_config_file}")
    with open(obstacle_config_file, "rb") as f:
        obstacle_data = pickle.load(f)
    obstacles = obstacle_data["obstacles"]
    configs = obstacle_data["configs"]

    print(f"加载collision_data_file: {collision_data_file}")
    with open(collision_data_file, "rb") as f:
        obb_data, obb_link_coll_data = pickle.load(f)

    print(f"障碍物数量: {len(obstacles)}")
    print(f"边数量: {len(configs)}")

    # 创建球体环境
    sphere_env = SphereEnv(robot_name=robot_name)
    sphere_env.init_obstacle_bodies(len(obstacles), obstacles)

    inconsistent_count = 0
    sphere_collision_obb_free_count = 0  # sphere碰撞，obb无碰撞
    sphere_free_obb_collision_count = 0  # sphere无碰撞，obb碰撞
    inconsistent_dir = os.path.join(os.path.dirname(__file__), "inconsistent_edge")
    os.makedirs(inconsistent_dir, exist_ok=True)
    inconsistent_edges = []  # 收集所有不一致的edge
    inconsistent_obb_colls = []  # 收集所有不一致edge的OBB碰撞结果
    inconsistent_sphere_colls = []  # 收集所有不一致edge的Sphere碰撞结果
    inconsistent_indices = []  # 收集不一致edge的索引
    if len(configs) > len(obb_link_coll_data):
        print("警告: OBB数据中缺少部分edge")

    # 处理每个edge
    for i, edge_configs in enumerate(configs):
        obb_edge = obb_link_coll_data[i]
        if not obb_edge:
            continue

        edge_sphere_coords = []
        edge_sphere_colls = []

        if len(edge_configs) > len(obb_edge):
            print(f"警告: OBB数据中edge {i}缺少pose")
            continue

        # 初始化edge层面的碰撞标志
        obb_edge_collision = any(
            any(coll == 0 for coll in pose_colls) for pose_colls in obb_edge
        )
        sphere_edge_collision = False

        # 处理edge中的每个pose
        for j, config in enumerate(edge_configs):
            # 获取球体碰撞数据
            collision, coords, colls = sphere_env.get_sphere_collision_data(config)
            edge_sphere_coords.append(coords)
            edge_sphere_colls.append(colls)

            # 更新sphere edge碰撞标志
            if any(coll == 0 for coll in colls):
                sphere_edge_collision = True

        # 在edge层面检查一致性
        if obb_edge_collision != sphere_edge_collision:
            print(
                f"Edge {i}: OBB={obb_edge_collision}, Sphere={sphere_edge_collision} - 不一致!"
            )
            inconsistent_count += 1
            if not obb_edge_collision and sphere_edge_collision:
                sphere_collision_obb_free_count += 1
            elif obb_edge_collision and not sphere_edge_collision:
                sphere_free_obb_collision_count += 1
            inconsistent_edges.append(edge_configs)
            inconsistent_obb_colls.append(obb_edge)
            inconsistent_sphere_colls.append(edge_sphere_colls)
            inconsistent_indices.append(i)
        else:
            # 只有一致时才存储球体数据
            if edge_sphere_coords:
                sphere_env.store_sphere_data(
                    edge_sphere_coords, edge_sphere_colls, is_edge=True
                )

    sphere_env.cleanup_obstacles()
    sphere_env.close()

    print(f"对比完成，发现 {inconsistent_count} 个不一致的配置")
    print(f"  Sphere碰撞但OBB无碰撞: {sphere_collision_obb_free_count}")
    print(f"  Sphere无碰撞但OBB碰撞: {sphere_free_obb_collision_count}")
    # 收集不一致的edge数据
    # inconsistent_data = {
    #     "obstacles": obstacles,
    #     "edge_configs": inconsistent_edges,
    #     "obb_edge_collision": inconsistent_obb_colls,
    #     "sphere_edge_collision": inconsistent_sphere_colls,
    #     "inconsistent_count": inconsistent_count,
    #     "sphere_collision_obb_free_count": sphere_collision_obb_free_count,
    #     "sphere_free_obb_collision_count": sphere_free_obb_collision_count,
    # }
    # # 保存所有不一致的edge到一个文件中
    # if inconsistent_edges:
    #     filename = (
    #         f"inconsistent_edges_{benchmark_id}.pkl"
    #         if benchmark_id is not None
    #         else "inconsistent_edges.pkl"
    #     )
    #     with open(os.path.join(inconsistent_dir, filename), "wb") as f:
    #         pickle.dump(inconsistent_data, f)
    #     print(f"不一致的edge已保存到 {os.path.join(inconsistent_dir, filename)}")
    # 删除不一致的edge并写回源文件
    if inconsistent_count > 0:
        for idx in sorted(inconsistent_indices, reverse=True):
            del obstacle_data["configs"][idx]
            del obb_link_coll_data[idx]
            del obb_data[idx]
        with open(obstacle_config_file, "wb") as f:
            pickle.dump(obstacle_data, f)
        with open(collision_data_file, "wb") as f:
            pickle.dump((obb_data, obb_link_coll_data), f)
        print(f"已从源文件中删除 {inconsistent_count} 个不一致的edge")

    # 保存球体碰撞数据（如果指定了输出文件）
    if output_file:
        sphere_env.save_collision_data(output_file)


def main():
    parser = argparse.ArgumentParser(description="对比OBB和球体碰撞检测结果")

    parser.add_argument(
        "--obstacle-config-file", type=str, required=True, help="障碍物-配置文件路径"
    )
    parser.add_argument(
        "--collision-data-file", type=str, required=True, help="OBB碰撞数据文件路径"
    )
    parser.add_argument("--robot-name", type=str, default="franka", help="机器人名称")
    parser.add_argument("--benchmark-id", type=int, help="基准测试ID，用于命名输出文件")
    parser.add_argument("--output-file", type=str, help="球体碰撞数据输出文件路径")

    args = parser.parse_args()

    compare_collision(
        args.obstacle_config_file,
        args.collision_data_file,
        args.robot_name,
        args.output_file,
        args.benchmark_id,
    )

    return 0


if __name__ == "__main__":
    exit(main())
