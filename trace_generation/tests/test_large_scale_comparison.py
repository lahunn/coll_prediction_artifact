#!/usr/bin/env python3
"""
大规模对比测试：几何OBB vs PyBullet碰撞检测

测试设置：
- 10个随机障碍物场景
- 每个场景2000个随机机器人姿态
- 统计结果一致性
"""

import numpy as np
import time
import json
from typing import List, Tuple, Dict, Any

from trace_generation.core.collision.obb_detector import OBBCollisionEnv
from trace_generation.core.robot.modular_env import ModularEnv


def generate_random_obstacles(num_obstacles: int, seed: int) -> List[Tuple]:
    """
    生成随机障碍物

    Args:
        num_obstacles: 障碍物数量
        seed: 随机种子

    Returns:
        障碍物列表 [(halfExtents, basePosition), ...]
    """
    np.random.seed(seed)
    obstacles = []

    for _ in range(num_obstacles):
        # 随机半范围：0.05米到0.2米
        half_extents = tuple(np.random.uniform(0.05, 0.2, 3))

        # 随机位置：机器人工作空间附近
        # X: [-0.5, 0.8], Y: [-0.6, 0.6], Z: [0.2, 1.2]
        position = (
            np.random.uniform(-0.5, 0.8),
            np.random.uniform(-0.6, 0.6),
            np.random.uniform(0.2, 1.2),
        )

        obstacles.append((half_extents, position))

    return obstacles


def generate_random_configs(
    num_configs: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray, seed: int
) -> List[np.ndarray]:
    """
    生成随机关节配置

    Args:
        num_configs: 配置数量
        lower_bounds: 关节下限
        upper_bounds: 关节上限
        seed: 随机种子

    Returns:
        配置列表
    """
    np.random.seed(seed)
    configs = []

    for _ in range(num_configs):
        config = np.random.uniform(lower_bounds, upper_bounds)
        configs.append(config)

    return configs


def test_single_scenario(
    scenario_id: int,
    num_obstacles: int,
    num_configs: int,
    obb_env: OBBCollisionEnv,
    pybullet_env: ModularEnv,
) -> Dict[str, Any]:
    """
    测试单个场景

    Args:
        scenario_id: 场景ID
        num_obstacles: 障碍物数量
        num_configs: 配置数量
        obb_env: OBB碰撞检测环境
        pybullet_env: PyBullet碰撞检测环境

    Returns:
        测试结果字典
    """
    print(f"\n{'=' * 80}")
    print(f"场景 {scenario_id + 1}: {num_obstacles}个障碍物, {num_configs}个姿态")
    print(f"{'=' * 80}")

    # 生成随机障碍物
    obstacles = generate_random_obstacles(num_obstacles, seed=scenario_id * 1000)

    # 加载障碍物
    obb_env.load_obstacles(obstacles)
    pybullet_env.obstacle_manager.load_obstacles(obstacles)
    pybullet_env.collision_env.load_obstacle_body_ids(
        pybullet_env.obstacle_manager.obstacle_body_ids
    )

    # 生成随机配置
    lower_bounds = np.array(pybullet_env.robot_env.lower_bounds)
    upper_bounds = np.array(pybullet_env.robot_env.upper_bounds)
    configs = generate_random_configs(
        num_configs, lower_bounds, upper_bounds, seed=scenario_id * 1000 + 1
    )

    # 统计数据
    total_tests = 0
    matches = 0
    obb_collision_count = 0
    obb_free_count = 0
    pybullet_collision_count = 0
    pybullet_free_count = 0

    obb_time_total = 0.0
    pybullet_time_total = 0.0

    # 不匹配案例（只保存前10个）
    mismatches = []

    # 测试每个配置
    start_time = time.time()
    for i, config in enumerate(configs):
        # 进度显示
        if (i + 1) % 500 == 0:
            print(f"  进度: {i + 1}/{num_configs} ({100 * (i + 1) / num_configs:.1f}%)")

        # OBB检测
        obb_start = time.time()
        obb_result = obb_env._state_fp(config)
        obb_time = time.time() - obb_start
        obb_time_total += obb_time

        # PyBullet检测
        pb_start = time.time()
        pb_result = pybullet_env._state_fp(config)
        pb_time = time.time() - pb_start
        pybullet_time_total += pb_time

        # 统计
        total_tests += 1

        if obb_result:
            obb_free_count += 1
        else:
            obb_collision_count += 1

        if pb_result:
            pybullet_free_count += 1
        else:
            pybullet_collision_count += 1

        # 检查是否匹配
        if obb_result == pb_result:
            matches += 1
        else:
            if len(mismatches) < 10:
                mismatches.append(
                    {
                        "config": config.tolist(),
                        "obb_result": bool(obb_result),
                        "pybullet_result": bool(pb_result),
                    }
                )

    elapsed_time = time.time() - start_time

    # 计算统计数据
    accuracy = matches / total_tests if total_tests > 0 else 0.0
    avg_obb_time = obb_time_total / total_tests if total_tests > 0 else 0.0
    avg_pb_time = pybullet_time_total / total_tests if total_tests > 0 else 0.0

    result = {
        "scenario_id": scenario_id,
        "num_obstacles": num_obstacles,
        "total_tests": total_tests,
        "matches": matches,
        "mismatches_count": total_tests - matches,
        "accuracy": accuracy,
        "obb_free": obb_free_count,
        "obb_collision": obb_collision_count,
        "pybullet_free": pybullet_free_count,
        "pybullet_collision": pybullet_collision_count,
        "avg_obb_time": avg_obb_time,
        "avg_pybullet_time": avg_pb_time,
        "total_time": elapsed_time,
        "mismatches_samples": mismatches,
    }

    # 打印场景结果
    print(f"\n场景 {scenario_id + 1} 结果:")
    print(f"  总测试数: {total_tests}")
    print(f"  匹配数: {matches}")
    print(f"  不匹配数: {total_tests - matches}")
    print(f"  准确率: {accuracy * 100:.2f}%")
    print(f"  OBB - 自由: {obb_free_count}, 碰撞: {obb_collision_count}")
    print(f"  PyBullet - 自由: {pybullet_free_count}, 碰撞: {pybullet_collision_count}")
    print(f"  平均OBB时间: {avg_obb_time * 1000:.3f}ms")
    print(f"  平均PyBullet时间: {avg_pb_time * 1000:.3f}ms")
    print(f"  总耗时: {elapsed_time:.2f}秒")

    return result


def main():
    """主函数"""
    print("=" * 80)
    print("大规模碰撞检测对比测试")
    print("=" * 80)
    print("配置: 10个场景 × 2000个姿态 = 20000次测试")
    print("=" * 80)

    robot_name = "franka"
    num_scenarios = 10
    num_configs_per_scenario = 2000

    # 初始化环境（只初始化一次，重复使用）
    print("\n初始化环境...")
    obb_env = OBBCollisionEnv(robot_name)
    pybullet_env = ModularEnv(robot_name, GUI=False, enable_self_collision=False)
    print("✓ 环境初始化完成")

    # 所有场景的结果
    all_results = []

    # 运行所有场景
    for scenario_id in range(num_scenarios):
        # 障碍物数量：3-8个随机
        num_obstacles = np.random.randint(3, 9)

        result = test_single_scenario(
            scenario_id=scenario_id,
            num_obstacles=num_obstacles,
            num_configs=num_configs_per_scenario,
            obb_env=obb_env,
            pybullet_env=pybullet_env,
        )
        all_results.append(result)

    # 汇总统计
    print(f"\n{'=' * 80}")
    print("汇总统计")
    print(f"{'=' * 80}")

    total_tests_all = sum(r["total_tests"] for r in all_results)
    total_matches_all = sum(r["matches"] for r in all_results)
    total_mismatches_all = sum(r["mismatches_count"] for r in all_results)
    overall_accuracy = (
        total_matches_all / total_tests_all if total_tests_all > 0 else 0.0
    )

    print("\n总体结果:")
    print(f"  场景数: {num_scenarios}")
    print(f"  总测试数: {total_tests_all}")
    print(f"  总匹配数: {total_matches_all}")
    print(f"  总不匹配数: {total_mismatches_all}")
    print(f"  总体准确率: {overall_accuracy * 100:.2f}%")
    print(f"  不一致比例: {(1 - overall_accuracy) * 100:.2f}%")

    print("\n各场景准确率:")
    for i, result in enumerate(all_results):
        print(
            f"  场景 {i + 1} ({result['num_obstacles']}障碍物): "
            f"{result['accuracy'] * 100:.2f}% "
            f"({result['matches']}/{result['total_tests']})"
        )

    avg_obb_time = np.mean([r["avg_obb_time"] for r in all_results])
    avg_pb_time = np.mean([r["avg_pybullet_time"] for r in all_results])

    print("\n性能对比:")
    print(f"  OBB平均检测时间: {avg_obb_time * 1000:.3f}ms")
    print(f"  PyBullet平均检测时间: {avg_pb_time * 1000:.3f}ms")
    print(
        f"  速度比: {avg_pb_time / avg_obb_time:.2f}x (OBB更快)"
        if avg_obb_time < avg_pb_time
        else f"  速度比: {avg_obb_time / avg_pb_time:.2f}x (PyBullet更快)"
    )

    # 保存结果到JSON文件
    output_file = "large_scale_collision_comparison_report.json"
    summary = {
        "test_config": {
            "robot_name": robot_name,
            "num_scenarios": num_scenarios,
            "configs_per_scenario": num_configs_per_scenario,
            "total_tests": total_tests_all,
        },
        "overall": {
            "total_matches": total_matches_all,
            "total_mismatches": total_mismatches_all,
            "accuracy": overall_accuracy,
            "inconsistency_rate": 1 - overall_accuracy,
            "avg_obb_time_ms": avg_obb_time * 1000,
            "avg_pybullet_time_ms": avg_pb_time * 1000,
        },
        "scenarios": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ 详细报告已保存到: {output_file}")

    # 清理
    obb_env.close()
    pybullet_env.close()

    print(f"\n{'=' * 80}")
    print("测试完成!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
