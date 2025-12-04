#!/usr/bin/env python3
"""
OBB vs Sphere碰撞检测周期成本多场景评估

评估不同障碍物密度场景下,OBB和Sphere碰撞检测的硬件周期成本
专注于周期成本统计,不进行准确性对比
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trace_generation.core.robot.environment import RobotEnv
from trace_generation.core.collision.obb_detector import OBBCollisionEnv
from trace_generation.core.collision.sphere_detector import SphereEnvGeometric
from trace_generation.core.scene.obstacle_manager import ObstacleManager


def generate_multi_density_scenarios(
    densities,
    workspace_range=(-1.0, 1.0),
    voxel_size_range=(0.05, 0.15),
    safe_zone_radius=0.3,
    seed=None,
):
    """
    生成不同密度的障碍物场景

    Args:
        densities: 障碍物密度列表 (e.g., [3, 6, 9, 12])
        workspace_range: 工作空间范围 (min, max)
        voxel_size_range: 障碍物尺寸范围 (min, max)
        safe_zone_radius: 安全区域半径
        seed: 随机种子

    Returns:
        dict: {scene_name: obstacles}
    """
    if seed is not None:
        np.random.seed(seed)

    scenarios = {}
    for density in densities:
        obstacles = ObstacleManager.generate_random_obstacles(
            num_obstacles=density,
            workspace_range=workspace_range,
            voxel_size_range=voxel_size_range,
            safe_zone_center=(0.0, 0.0, 0.0),
            safe_zone_radius=safe_zone_radius,
        )
        scenarios[f"dens{density}"] = obstacles

    return scenarios


def sample_uniform_pose(robot_env):
    """
    在关节限制内均匀采样一个pose

    Args:
        robot_env: RobotEnv实例

    Returns:
        numpy.ndarray: 采样的关节配置
    """
    return np.random.uniform(robot_env.lower_bounds, robot_env.upper_bounds)


def evaluate_scenario(robot_env, obstacles, num_poses, obb_env, sphere_env, scene_name):
    """
    对单个场景进行评估

    Args:
        robot_env: RobotEnv实例
        obstacles: 障碍物列表
        num_poses: 采样的pose数量
        obb_env: OBBCollisionEnv实例
        sphere_env: SphereEnvGeometric实例
        scene_name: 场景名称

    Returns:
        dict: {
            'obb_unit_cycles': [...],  # 每个单独OBB的周期列表
            'sphere_unit_cycles': [...],  # 每个单独Sphere的周期列表
            'obb_pose_cycles': [...],  # 每个pose的总周期列表
            'sphere_pose_cycles': [...]  # 每个pose的总周期列表
        }
    """
    # 加载障碍物
    obb_env.load_obstacles(obstacles)
    sphere_env.load_obstacles(obstacles)

    # 单元级别的周期数(每个OBB/Sphere)
    obb_unit_cycles_list = []
    sphere_unit_cycles_list = []

    # Pose级别的周期数(每个pose的总和)
    obb_pose_cycles_list = []
    sphere_pose_cycles_list = []

    # 采样poses并评估
    for _ in tqdm(range(num_poses), desc=f"{scene_name} Poses", leave=False):
        # 采样pose
        state = sample_uniform_pose(robot_env)

        # OBB检测 (收集单元级和pose级周期数)
        result_obb = obb_env._get_link_collisions(state)
        if obb_env.return_cycles and len(result_obb) == 3:
            _, _, obb_cycles = result_obb
            # 单元级: 每个link的周期
            obb_unit_cycles_list.extend(obb_cycles)
            # Pose级: 所有link的总周期
            obb_pose_cycles_list.append(sum(obb_cycles))
        else:
            print(f"警告: OBB检测器未返回周期数")

        # Sphere检测 (收集单元级和pose级周期数)
        result_sphere = sphere_env.get_sphere_collision_data(state.tolist())
        if len(result_sphere) == 4:  # return_cycles=True
            _, _, _, sphere_cycles = result_sphere
            # 单元级: 每个sphere的周期
            sphere_unit_cycles_list.extend(sphere_cycles)
            # Pose级: 所有sphere的总周期
            sphere_pose_cycles_list.append(sum(sphere_cycles))
        else:
            print(f"警告: Sphere检测器未返回周期数")

    return {
        "obb_unit_cycles": obb_unit_cycles_list,
        "sphere_unit_cycles": sphere_unit_cycles_list,
        "obb_pose_cycles": obb_pose_cycles_list,
        "sphere_pose_cycles": sphere_pose_cycles_list,
    }


def compute_statistics(cycles_list):
    """
    计算周期数统计

    Args:
        cycles_list: 周期数列表

    Returns:
        dict: 统计信息
    """
    if not cycles_list:
        return {
            "count": 0,
            "total": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "percentile_25": 0,
            "percentile_75": 0,
        }

    return {
        "count": len(cycles_list),
        "total": sum(cycles_list),
        "mean": np.mean(cycles_list),
        "median": np.median(cycles_list),
        "std": np.std(cycles_list),
        "min": np.min(cycles_list),
        "max": np.max(cycles_list),
        "percentile_25": np.percentile(cycles_list, 25),
        "percentile_75": np.percentile(cycles_list, 75),
    }


def analyze_density_impact(all_results):
    """
    分析障碍物密度对性能的影响

    Args:
        all_results: 所有场景的评估结果

    Returns:
        dict: 分析结果
    """
    analysis = {}

    for scene_name, results in all_results.items():
        # 单元级别统计 (每个OBB/Sphere)
        obb_unit_stats = compute_statistics(results["obb_unit_cycles"])
        sphere_unit_stats = compute_statistics(results["sphere_unit_cycles"])

        # Pose级别统计 (每个pose)
        obb_pose_stats = compute_statistics(results["obb_pose_cycles"])
        sphere_pose_stats = compute_statistics(results["sphere_pose_cycles"])

        # 计算加速比 (单元级别)
        unit_speedup_ratio = (
            sphere_unit_stats["mean"] / obb_unit_stats["mean"]
            if obb_unit_stats["mean"] > 0
            else 0
        )

        # 计算加速比 (Pose级别)
        pose_speedup_ratio = (
            sphere_pose_stats["mean"] / obb_pose_stats["mean"]
            if obb_pose_stats["mean"] > 0
            else 0
        )

        analysis[scene_name] = {
            "obb_unit": obb_unit_stats,
            "sphere_unit": sphere_unit_stats,
            "obb_pose": obb_pose_stats,
            "sphere_pose": sphere_pose_stats,
            "unit_speedup_ratio": unit_speedup_ratio,
            "pose_speedup_ratio": pose_speedup_ratio,
        }

    return analysis


def print_summary(analysis, robot_name, num_poses, all_results):
    """
    打印汇总统计表格

    Args:
        analysis: 分析结果
        robot_name: 机器人名称
        num_poses: 每场景的pose数量
        all_results: 所有场景的原始结果数据
    """
    print("\n" + "=" * 80)
    print(f"碰撞检测周期成本多场景对比 (Robot: {robot_name}, Poses/场景: {num_poses})")
    print("=" * 80)

    # 提取障碍物密度列表并排序
    scene_names = sorted(analysis.keys(), key=lambda x: int(x.replace("dens", "")))

    for scene_name in scene_names:
        stats = analysis[scene_name]
        density = scene_name.replace("dens", "")

        print(f"\n场景: {scene_name} ({density}个障碍物)")

        # 单元级别统计 (每个OBB/Sphere)
        print(f"\n  [单元级别] 单个OBB/Sphere的平均周期:")
        print(
            f"    单个OBB  - 平均: {stats['obb_unit']['mean']:.2f} 周期 "
            f"(范围: {stats['obb_unit']['min']:.0f}-{stats['obb_unit']['max']:.0f}, "
            f"标准差: {stats['obb_unit']['std']:.2f})"
        )
        print(
            f"    单个Sphere - 平均: {stats['sphere_unit']['mean']:.2f} 周期 "
            f"(范围: {stats['sphere_unit']['min']:.0f}-{stats['sphere_unit']['max']:.0f}, "
            f"标准差: {stats['sphere_unit']['std']:.2f})"
        )
        print(
            f"    单元加速比: {stats['unit_speedup_ratio']:.2f}x "
            f"({'Sphere更快' if stats['unit_speedup_ratio'] < 1 else 'OBB更快'})"
        )

        # Pose级别统计
        print(f"\n  [Pose级别] 每个pose的总周期:")
        print(
            f"    OBB检测  - 平均: {stats['obb_pose']['mean']:.2f} 周期 "
            f"(范围: {stats['obb_pose']['min']:.0f}-{stats['obb_pose']['max']:.0f})"
        )
        print(
            f"    Sphere检测 - 平均: {stats['sphere_pose']['mean']:.2f} 周期 "
            f"(范围: {stats['sphere_pose']['min']:.0f}-{stats['sphere_pose']['max']:.0f})"
        )
        print(
            f"    Pose加速比: {stats['pose_speedup_ratio']:.2f}x "
            f"({'Sphere更快' if stats['pose_speedup_ratio'] < 1 else 'OBB更快'})"
        )

    # 密度影响分析
    print("\n" + "-" * 80)
    print("密度影响分析:")

    if len(scene_names) >= 2:
        # 计算密度增加对周期数的影响
        first_scene = scene_names[0]
        last_scene = scene_names[-1]

        first_density = int(first_scene.replace("dens", ""))
        last_density = int(last_scene.replace("dens", ""))
        density_increase = last_density - first_density

        # 单元级别的增长
        obb_unit_first = analysis[first_scene]["obb_unit"]["mean"]
        obb_unit_last = analysis[last_scene]["obb_unit"]["mean"]
        obb_unit_increase = (
            ((obb_unit_last - obb_unit_first) / obb_unit_first * 100)
            if obb_unit_first > 0
            else 0
        )

        sphere_unit_first = analysis[first_scene]["sphere_unit"]["mean"]
        sphere_unit_last = analysis[last_scene]["sphere_unit"]["mean"]
        sphere_unit_increase = (
            ((sphere_unit_last - sphere_unit_first) / sphere_unit_first * 100)
            if sphere_unit_first > 0
            else 0
        )

        # Pose级别的增长
        obb_pose_first = analysis[first_scene]["obb_pose"]["mean"]
        obb_pose_last = analysis[last_scene]["obb_pose"]["mean"]
        obb_pose_increase = (
            ((obb_pose_last - obb_pose_first) / obb_pose_first * 100)
            if obb_pose_first > 0
            else 0
        )

        sphere_pose_first = analysis[first_scene]["sphere_pose"]["mean"]
        sphere_pose_last = analysis[last_scene]["sphere_pose"]["mean"]
        sphere_pose_increase = (
            ((sphere_pose_last - sphere_pose_first) / sphere_pose_first * 100)
            if sphere_pose_first > 0
            else 0
        )

        print(f"\n  障碍物从{first_density}增加到{last_density}个:")
        print(f"\n  [单元级别] 单个OBB/Sphere周期增长:")
        print(
            f"    单个OBB周期增长: {obb_unit_increase:.1f}% "
            f"({obb_unit_increase / density_increase:.1f}%/个障碍物)"
        )
        print(
            f"    单个Sphere周期增长: {sphere_unit_increase:.1f}% "
            f"({sphere_unit_increase / density_increase:.1f}%/个障碍物)"
        )

        print(f"\n  [Pose级别] 每个pose总周期增长:")
        print(
            f"    OBB Pose周期增长: {obb_pose_increase:.1f}% "
            f"({obb_pose_increase / density_increase:.1f}%/个障碍物)"
        )
        print(
            f"    Sphere Pose周期增长: {sphere_pose_increase:.1f}% "
            f"({sphere_pose_increase / density_increase:.1f}%/个障碍物)"
        )

    # 合并所有场景的数据进行总体分析
    print("\n" + "-" * 80)
    print("总体平均结果 (合并所有障碍物密度场景):")
    
    # 合并所有场景的cycles数据
    all_obb_unit_cycles = []
    all_sphere_unit_cycles = []
    all_obb_pose_cycles = []
    all_sphere_pose_cycles = []
    
    for scene_name in scene_names:
        all_obb_unit_cycles.extend(all_results[scene_name]["obb_unit_cycles"])
        all_sphere_unit_cycles.extend(all_results[scene_name]["sphere_unit_cycles"])
        all_obb_pose_cycles.extend(all_results[scene_name]["obb_pose_cycles"])
        all_sphere_pose_cycles.extend(all_results[scene_name]["sphere_pose_cycles"])
    
    # 计算总体统计
    overall_obb_unit = compute_statistics(all_obb_unit_cycles)
    overall_sphere_unit = compute_statistics(all_sphere_unit_cycles)
    overall_obb_pose = compute_statistics(all_obb_pose_cycles)
    overall_sphere_pose = compute_statistics(all_sphere_pose_cycles)
    
    # 计算总体加速比
    overall_unit_speedup = (
        overall_sphere_unit["mean"] / overall_obb_unit["mean"]
        if overall_obb_unit["mean"] > 0
        else 0
    )
    overall_pose_speedup = (
        overall_sphere_pose["mean"] / overall_obb_pose["mean"]
        if overall_obb_pose["mean"] > 0
        else 0
    )
    
    print(f"\n  总样本数: {len(all_obb_pose_cycles)} poses, "
          f"{len(all_obb_unit_cycles)} OBB单元, "
          f"{len(all_sphere_unit_cycles)} Sphere单元")
    
    print(f"\n  [单元级别] 单个OBB/Sphere的平均周期:")
    print(f"    单个OBB    - 平均: {overall_obb_unit['mean']:.2f} 周期 "
          f"(中位数: {overall_obb_unit['median']:.2f}, "
          f"标准差: {overall_obb_unit['std']:.2f})")
    print(f"                 范围: {overall_obb_unit['min']:.0f} - {overall_obb_unit['max']:.0f} 周期")
    
    print(f"    单个Sphere - 平均: {overall_sphere_unit['mean']:.2f} 周期 "
          f"(中位数: {overall_sphere_unit['median']:.2f}, "
          f"标准差: {overall_sphere_unit['std']:.2f})")
    print(f"                 范围: {overall_sphere_unit['min']:.0f} - {overall_sphere_unit['max']:.0f} 周期")
    
    print(f"    单元加速比: {overall_unit_speedup:.2f}x "
          f"({'Sphere更快' if overall_unit_speedup < 1 else 'OBB更快'})")
    
    if overall_unit_speedup < 1:
        improvement_pct = (1 - overall_unit_speedup) * 100
        print(f"    性能提升: Sphere比OBB快 {improvement_pct:.1f}%")
    else:
        decline_pct = (overall_unit_speedup - 1) * 100
        print(f"    性能下降: Sphere比OBB慢 {decline_pct:.1f}%")
    
    print(f"\n  [Pose级别] 每个pose的总周期:")
    print(f"    OBB检测    - 平均: {overall_obb_pose['mean']:.2f} 周期 "
          f"(中位数: {overall_obb_pose['median']:.2f}, "
          f"标准差: {overall_obb_pose['std']:.2f})")
    print(f"                 范围: {overall_obb_pose['min']:.0f} - {overall_obb_pose['max']:.0f} 周期")
    
    print(f"    Sphere检测 - 平均: {overall_sphere_pose['mean']:.2f} 周期 "
          f"(中位数: {overall_sphere_pose['median']:.2f}, "
          f"标准差: {overall_sphere_pose['std']:.2f})")
    print(f"                 范围: {overall_sphere_pose['min']:.0f} - {overall_sphere_pose['max']:.0f} 周期")
    
    print(f"    Pose加速比: {overall_pose_speedup:.2f}x "
          f"({'Sphere更快' if overall_pose_speedup < 1 else 'OBB更快'})")
    
    if overall_pose_speedup < 1:
        improvement_pct = (1 - overall_pose_speedup) * 100
        print(f"    性能提升: Sphere比OBB快 {improvement_pct:.1f}%")
    else:
        decline_pct = (overall_pose_speedup - 1) * 100
        print(f"    性能下降: Sphere比OBB慢 {decline_pct:.1f}%")

    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="评估OBB和Sphere碰撞检测的周期成本(多场景)"
    )
    parser.add_argument(
        "--robot-name",
        default="franka",
        help="机器人类型 (franka/iiwa/kinova_gen3/ur5e)",
    )
    parser.add_argument(
        "--num-poses", type=int, default=1000, help="每个场景采样的pose数量"
    )
    parser.add_argument(
        "--obstacle-densities",
        nargs="+",
        type=int,
        default=[3, 6, 9, 12],
        help="障碍物密度列表",
    )
    parser.add_argument(
        "--workspace-range",
        nargs=2,
        type=float,
        default=[-1.0, 1.0],
        help="工作空间范围 (min max)",
    )
    parser.add_argument(
        "--voxel-size-range",
        nargs=2,
        type=float,
        default=[0.05, 0.15],
        help="障碍物尺寸范围 (min max)",
    )
    parser.add_argument(
        "--safe-zone-radius", type=float, default=0.3, help="机器人基座安全区半径"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (可复现性)")

    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print("OBB vs Sphere 碰撞检测周期成本评估")
    print("=" * 80)
    print(f"机器人: {args.robot_name}")
    print(f"每场景pose数: {args.num_poses}")
    print(f"障碍物密度: {args.obstacle_densities}")
    print(f"工作空间: {args.workspace_range}")
    print(f"障碍物尺寸: {args.voxel_size_range}")
    print(f"随机种子: {args.seed}")
    print("=" * 80 + "\n")

    # 1. 生成多场景
    print("生成多密度障碍物场景...")
    scenarios = generate_multi_density_scenarios(
        densities=args.obstacle_densities,
        workspace_range=tuple(args.workspace_range),
        voxel_size_range=tuple(args.voxel_size_range),
        safe_zone_radius=args.safe_zone_radius,
        seed=args.seed,
    )
    print(f"✓ 生成 {len(scenarios)} 个场景\n")

    # 2. 初始化机器人环境
    print(f"初始化机器人环境: {args.robot_name}")
    robot_env = RobotEnv(args.robot_name)
    print(f"✓ 机器人环境已初始化\n")

    # 3. 初始化检测器 (启用周期计数)
    print("初始化碰撞检测器...")
    obb_env = OBBCollisionEnv(args.robot_name, return_cycles=True)
    sphere_env = SphereEnvGeometric(
        robot_env=robot_env, robot_name=args.robot_name, return_cycles=True
    )
    print(f"✓ OBB和Sphere检测器已初始化\n")

    # 4. 对每个场景进行评估
    print("开始评估各场景...")
    all_results = {}

    for scene_name, obstacles in tqdm(scenarios.items(), desc="场景评估"):
        results = evaluate_scenario(
            robot_env, obstacles, args.num_poses, obb_env, sphere_env, scene_name
        )
        all_results[scene_name] = results

        # 实时显示简要统计
        obb_unit_mean = np.mean(results["obb_unit_cycles"])
        sphere_unit_mean = np.mean(results["sphere_unit_cycles"])
        obb_pose_mean = np.mean(results["obb_pose_cycles"])
        sphere_pose_mean = np.mean(results["sphere_pose_cycles"])
        print(
            f"  {scene_name}: "
            f"单元级[OBB={obb_unit_mean:.2f}, Sphere={sphere_unit_mean:.2f}] "
            f"Pose级[OBB={obb_pose_mean:.2f}, Sphere={sphere_pose_mean:.2f}]"
        )

    print("\n✓ 所有场景评估完成\n")

    # 5. 分析结果
    print("分析结果...")
    analysis = analyze_density_impact(all_results)

    # 6. 打印汇总
    print_summary(analysis, args.robot_name, args.num_poses, all_results)

    # 清理
    robot_env.close()
    obb_env.close()
    sphere_env.close()

    print("\n评估完成!")


if __name__ == "__main__":
    main()
