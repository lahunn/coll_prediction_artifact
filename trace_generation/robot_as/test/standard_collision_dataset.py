#!/usr/bin/env python3
"""
标准碰撞检测数据集生成程序

该程序生成10个随机环境场景，每个场景包含：
- 随机障碍物
- 100个随机机器人配置
- 每个配置的碰撞检测结果

输出格式：
- standard_collision_dataset.pkl: 包含所有场景的数据
  - scenes: 场景列表
    - obstacles: 障碍物列表
    - configs: 配置列表
    - collision_results: 碰撞结果列表 (每个配置的 (is_free, link_coords, link_colls))
"""

import sys
import os
import pickle
import select

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from modular_env import ModularEnv
from sphere_as.sphere_method import SphereEnv


def generate_standard_collision_dataset(
    robot_file, output_file, num_scenes=10, num_configs_per_scene=100
):
    """
    生成标准碰撞检测数据集

    Args:
        robot_file: 机器人URDF文件路径
        output_file: 输出文件路径
        num_scenes: 场景数量
        num_configs_per_scene: 每个场景的配置数量
    """
    print(
        f"生成标准碰撞检测数据集: {num_scenes} 个场景, 每个场景 {num_configs_per_scene} 个配置"
    )

    # 初始化环境
    env = ModularEnv(robot_file, GUI=False)

    scenes_data = []

    for scene_idx in range(num_scenes):
        print(f"\n生成场景 {scene_idx + 1}/{num_scenes}")

        # 生成随机障碍物
        obstacles = env.generate_random_obstacles(
            num_obstacles=8,
            workspace_range=(-1.0, 1.0),
            voxel_size_range=(0.05, 0.15),
            safe_zone_center=(0.0, 0.0, 0.0),
            safe_zone_radius=0.3,
        )
        print(f"  生成 {len(obstacles)} 个障碍物")

        # 生成随机配置
        configs = env.sample_n_points(num_configs_per_scene)
        print(f"  生成 {len(configs)} 个随机配置")

        # 进行碰撞检测
        collision_results = []
        for i, config in enumerate(configs):
            if i % 20 == 0:
                print(f"    检测配置 {i + 1}/{len(configs)}")

            # 使用 _state_fp_probe 获取详细信息
            is_free, link_coords, link_colls = env._state_fp_probe(config)
            collision_results.append((is_free, link_coords, link_colls))

        # 计算碰撞统计
        free_configs = sum(1 for result in collision_results if result[0])
        collision_rate = 1.0 - (free_configs / len(collision_results))
        print(
            f"  碰撞率: {collision_rate:.4f} ({len(collision_results) - free_configs}/{len(collision_results)})"
        )

        # 保存场景数据
        scene_data = {
            "scene_idx": scene_idx,
            "obstacles": obstacles,
            "configs": configs,
            "collision_results": collision_results,
            "collision_rate": collision_rate,
        }
        scenes_data.append(scene_data)

    # 保存数据集
    dataset = {
        "num_scenes": num_scenes,
        "num_configs_per_scene": num_configs_per_scene,
        "scenes": scenes_data,
    }

    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\n数据集保存到: {output_file}")
    print(f"总场景数: {len(scenes_data)}")
    total_configs = sum(len(scene["configs"]) for scene in scenes_data)
    total_collisions = sum(
        len(scene["configs"])
        - sum(1 for result in scene["collision_results"] if result[0])
        for scene in scenes_data
    )
    overall_collision_rate = (
        total_collisions / total_configs if total_configs > 0 else 0.0
    )
    print(f"总配置数: {total_configs}")
    print(f"总体碰撞率: {overall_collision_rate:.4f}")

    # 关闭环境
    env.close()


def visualize_standard_dataset(dataset_file, robot_file):
    """
    可视化标准碰撞检测数据集

    Args:
        dataset_file: 数据集文件路径
        robot_file: 机器人URDF文件路径
    """
    print(f"加载数据集: {dataset_file}")
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    scenes = dataset["scenes"]
    print(f"数据集包含 {len(scenes)} 个场景")

    # 初始化环境（GUI模式）
    env = ModularEnv(robot_file, GUI=True)

    total_configs = 0
    for scene in scenes:
        total_configs += len(scene["configs"])

    print(f"总配置数: {total_configs}")
    print("按 Enter 键加载下一个配置，按 Ctrl+C 退出")

    config_idx = 0
    for scene_idx, scene in enumerate(scenes):
        print(f"\n场景 {scene_idx + 1}/{len(scenes)}")
        print(f"  障碍物数量: {len(scene['obstacles'])}")
        print(f"  配置数量: {len(scene['configs'])}")
        print(f"  碰撞率: {scene['collision_rate']:.4f}")

        # 加载障碍物
        env.obstacle_manager.load_and_init_obstacles_from_data(scene["obstacles"])
        env.collision_env.load_obstacle_body_ids(env.obstacle_manager.obstacle_body_ids)

        for config_idx_in_scene, (
            config,
            (is_free, link_coords, link_colls),
        ) in enumerate(zip(scene["configs"], scene["collision_results"])):
            print(
                f"\n配置 {config_idx + 1}/{total_configs} (场景 {scene_idx + 1}, 配置 {config_idx_in_scene + 1})"
            )
            print(f"  配置值: {config}")
            print(f"  碰撞结果: {'自由' if is_free else '碰撞'}")
            print(
                f"  碰撞链接数: {sum(1 for coll in link_colls if coll == 0)}/{len(link_colls)}"
            )

            # 设置机器人配置
            env.robot_env.set_config(config)

            # 等待用户输入
            try:
                input("按 Enter 键继续...")
            except KeyboardInterrupt:
                print("\n退出可视化")
                env.close()
                return

            config_idx += 1

    print("可视化完成")
    env.close()


def compare_collision_methods(robot_file, num_obstacles=10, num_configs=5000):
    """
    随机生成障碍物和pose，执行碰撞检测，并与球体的碰撞检测结果进行对比

    Args:
        robot_file: 机器人URDF文件路径
        num_obstacles: 障碍物数量
        num_configs: 配置数量
    """
    print(f"比较碰撞检测方法: {num_obstacles} 个障碍物, {num_configs} 个配置")

    # 初始化OBB环境
    obb_env = ModularEnv(robot_file, GUI=False)

    # 初始化Sphere环境
    sphere_env = SphereEnv(robot_name="franka", SPH_GUI=False)

    # 生成随机障碍物
    obstacles = obb_env.generate_random_obstacles(
        num_obstacles=num_obstacles,
        workspace_range=(-1.0, 1.0),
        voxel_size_range=(0.05, 0.15),
        safe_zone_center=(0.0, 0.0, 0.0),
        safe_zone_radius=0.3,
    )
    print(f"生成 {len(obstacles)} 个障碍物")

    # 初始化Sphere环境的障碍物
    sphere_env.init_obstacle_bodies(len(obstacles), obstacles)

    # 生成随机配置
    configs = obb_env.sample_n_points(num_configs)
    print(f"生成 {len(configs)} 个随机配置")

    inconsistent_count = 0

    for i, config in enumerate(configs):
        if i % 10 == 0:
            print(f"检测配置 {i + 1}/{len(configs)}")

        # OBB碰撞检测
        obb_free, obb_coords, obb_colls = obb_env._state_fp_probe(config)

        # Sphere碰撞检测（包括自碰撞和与障碍物）
        sphere_collision, _, _ = sphere_env.get_sphere_collision_data(config)
        sphere_free = not sphere_collision

        # 比较结果
        if obb_free != sphere_free:
            print(f"配置 {i}: OBB={obb_free}, Sphere={sphere_free} - 不一致!")
            print(
                f"  OBB碰撞链接: {sum(1 for c in obb_colls if c == 0)}/{len(obb_colls)}"
            )
            inconsistent_count += 1

    print(f"比较完成，发现 {inconsistent_count} 个不一致的配置")

    # 关闭环境
    obb_env.close()
    sphere_env.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="标准碰撞检测数据集生成、可视化和比较程序"
    )
    parser.add_argument(
        "--robot-file",
        type=str,
        default="/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf",
        help="机器人URDF文件路径",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="standard_collision_dataset.pkl",
        help="输出文件路径",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=10,
        help="场景数量",
    )
    parser.add_argument(
        "--num-configs",
        type=int,
        default=10,
        help="每个场景的配置数量",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        help="可视化数据集文件路径",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="比较OBB和Sphere碰撞检测方法",
    )

    args = parser.parse_args()

    if args.visualize:
        visualize_standard_dataset(args.visualize, args.robot_file)
    elif args.compare:
        compare_collision_methods(args.robot_file)
    else:
        generate_standard_collision_dataset(
            args.robot_file,
            args.output_file,
            args.num_scenes,
            args.num_configs,
        )


if __name__ == "__main__":
    main()
