#!/usr/bin/env python3
"""
对比测试：几何OBB碰撞检测 vs PyBullet碰撞检测

测试目的：
- 在相同机器人和障碍物环境下对比两种碰撞检测方法的正确性
- 验证几何实现的准确性

测试流程：
1. 初始化相同机器人（franka）
2. 加载相同障碍物配置
3. 在多个随机关节配置下进行碰撞检测
4. 对比两种方法的检测结果
5. 生成详细的对比报告
"""

import numpy as np
import time
import sys
import os
from typing import List, Tuple, Dict, Any

# 设置项目路径（优雅的方式）
from trace_generation.core.collision.obb_detector import OBBCollisionEnv
from trace_generation.core.robot.modular_env import ModularEnv


class CollisionDetectionComparator:
    """
    碰撞检测对比器

    对比几何OBB碰撞检测和PyBullet碰撞检测的结果
    """

    def __init__(self, robot_name: str = "franka"):
        """
        初始化对比器

        Args:
            robot_name: 机器人名称
        """
        self.robot_name = robot_name

        # 初始化两种碰撞检测环境
        print(f"初始化 {robot_name} 机器人环境...")

        # 几何OBB碰撞检测环境
        self.obb_env = OBBCollisionEnv(robot_name)

        # PyBullet碰撞检测环境（通过ModularEnv）
        self.pybullet_env = ModularEnv(
            robot_name, GUI=False, enable_self_collision=False
        )

        print("✓ 环境初始化完成")

    def setup_obstacles(self, obstacles: List[Tuple]):
        """
        设置障碍物

        Args:
            obstacles: 障碍物列表 [(halfExtents, basePosition), ...]
        """
        print(f"设置 {len(obstacles)} 个障碍物...")

        # 为几何环境设置障碍物
        self.obb_env.load_obstacles(obstacles)

        # 为PyBullet环境设置障碍物
        # 注意：ModularEnv需要通过其obstacle_manager来设置障碍物
        self.pybullet_env.obstacle_manager.load_obstacles(obstacles)

        # 更新碰撞环境中的障碍物ID
        self.pybullet_env.collision_env.load_obstacle_body_ids(
            self.pybullet_env.obstacle_manager.obstacle_body_ids
        )

        print("✓ 障碍物设置完成")

    def generate_test_configs(self, num_configs: int = 100) -> List[np.ndarray]:
        """
        生成测试关节配置

        Args:
            num_configs: 配置数量

        Returns:
            配置列表
        """
        configs = []

        # 获取关节限位
        lower_bounds = np.array(self.pybullet_env.robot_env.lower_bounds)
        upper_bounds = np.array(self.pybullet_env.robot_env.upper_bounds)

        print(f"关节限位 - 下限: {lower_bounds}, 上限: {upper_bounds}")

        for i in range(num_configs):
            # 在关节限位范围内随机采样
            config = np.random.uniform(lower_bounds, upper_bounds)
            configs.append(config)

        print(f"✓ 生成 {len(configs)} 个测试配置")
        return configs

    def test_single_config(self, config: np.ndarray) -> Dict[str, Any]:
        """
        测试单个关节配置的碰撞检测

        Args:
            config: 关节配置

        Returns:
            测试结果字典
        """
        result = {
            "config": config.copy(),
            "obb_result": None,
            "pybullet_result": None,
            "match": None,
            "obb_time": 0.0,
            "pybullet_time": 0.0,
        }

        # 测试几何OBB碰撞检测
        start_time = time.time()
        try:
            obb_free = self.obb_env._state_fp(config)
            result["obb_result"] = obb_free
            result["obb_time"] = time.time() - start_time
        except Exception as e:
            print(f"几何OBB检测失败: {e}")
            result["obb_result"] = None
            result["obb_time"] = time.time() - start_time

        # 测试PyBullet碰撞检测
        start_time = time.time()
        try:
            pybullet_free = self.pybullet_env._state_fp(config)
            result["pybullet_result"] = pybullet_free
            result["pybullet_time"] = time.time() - start_time
        except Exception as e:
            print(f"PyBullet检测失败: {e}")
            result["pybullet_result"] = None
            result["pybullet_time"] = time.time() - start_time

        # 判断结果是否匹配
        if result["obb_result"] is not None and result["pybullet_result"] is not None:
            result["match"] = result["obb_result"] == result["pybullet_result"]
        else:
            result["match"] = None

        return result

    def run_comparison_test(
        self, num_configs: int = 100, obstacles: List[Tuple] = None
    ) -> Dict[str, Any]:
        """
        运行对比测试

        Args:
            num_configs: 测试配置数量
            obstacles: 障碍物列表（可选）

        Returns:
            测试报告字典
        """
        print("=" * 80)
        print("碰撞检测对比测试开始")
        print("=" * 80)
        print(f"机器人: {self.robot_name}")
        print(f"测试配置数: {num_configs}")

        # 设置默认障碍物（如果未提供）
        if obstacles is None:
            obstacles = [
                ((0.1, 0.1, 0.1), (0.5, 0.0, 0.5)),  # 小障碍物
                ((0.2, 0.2, 0.2), (0.0, 0.5, 0.3)),  # 中等障碍物
                ((0.15, 0.15, 0.15), (-0.3, -0.3, 0.4)),  # 另一个障碍物
            ]

        # 设置障碍物
        self.setup_obstacles(obstacles)

        # 生成测试配置
        test_configs = self.generate_test_configs(num_configs)

        # 运行测试
        print("\n开始测试...")
        results = []
        match_count = 0
        total_valid_tests = 0

        for i, config in enumerate(test_configs):
            if (i + 1) % 20 == 0:
                print(f"  测试进度: {i + 1}/{num_configs}")

            result = self.test_single_config(config)
            results.append(result)

            if result["match"] is not None:
                total_valid_tests += 1
                if result["match"]:
                    match_count += 1

        # 生成测试报告
        report = self._generate_report(results, match_count, total_valid_tests)
        return report

    def _generate_report(
        self, results: List[Dict], match_count: int, total_valid_tests: int
    ) -> Dict[str, Any]:
        """
        生成测试报告

        Args:
            results: 测试结果列表
            match_count: 匹配的数量
            total_valid_tests: 有效测试数量

        Returns:
            测试报告字典
        """
        print("\n" + "=" * 80)
        print("测试报告")
        print("=" * 80)

        # 基本统计
        total_tests = len(results)
        accuracy = match_count / total_valid_tests if total_valid_tests > 0 else 0.0

        print(f"总测试数: {total_tests}")
        print(f"有效测试数: {total_valid_tests}")
        print(f"匹配数: {match_count}")
        print(f"准确率: {accuracy:.2%}")
        # 性能统计
        obb_times = [r["obb_time"] for r in results if r["obb_time"] > 0]
        pybullet_times = [r["pybullet_time"] for r in results if r["pybullet_time"] > 0]

        if obb_times:
            avg_obb_time = np.mean(obb_times)
            print(f"几何OBB平均检测时间: {avg_obb_time:.4f} 秒")
        if pybullet_times:
            avg_pybullet_time = np.mean(pybullet_times)
            print(f"PyBullet平均检测时间: {avg_pybullet_time:.4f} 秒")

        # 详细结果分析
        obb_free_count = sum(1 for r in results if r["obb_result"] is True)
        obb_collision_count = sum(1 for r in results if r["obb_result"] is False)
        pybullet_free_count = sum(1 for r in results if r["pybullet_result"] is True)
        pybullet_collision_count = sum(
            1 for r in results if r["pybullet_result"] is False
        )

        print("\n检测结果统计:")
        print(f"  几何OBB - 自由: {obb_free_count}, 碰撞: {obb_collision_count}")
        print(
            f"  PyBullet - 自由: {pybullet_free_count}, 碰撞: {pybullet_collision_count}"
        )

        # 不匹配的案例分析
        mismatches = [r for r in results if r["match"] is False]
        if mismatches:
            print(f"\n不匹配案例数: {len(mismatches)}")
            print("前5个不匹配案例:")
            for i, mismatch in enumerate(mismatches[:5]):
                print(f"  案例 {i + 1}:")
                print(f"    配置: {np.array2string(mismatch['config'], precision=3)}")
                print(f"    几何OBB: {'自由' if mismatch['obb_result'] else '碰撞'}")
                print(
                    f"    PyBullet: {'自由' if mismatch['pybullet_result'] else '碰撞'}"
                )

        # 返回详细报告
        report = {
            "robot_name": self.robot_name,
            "total_tests": total_tests,
            "valid_tests": total_valid_tests,
            "matches": match_count,
            "accuracy": accuracy,
            "avg_obb_time": np.mean(obb_times) if obb_times else 0.0,
            "avg_pybullet_time": np.mean(pybullet_times) if pybullet_times else 0.0,
            "obb_free_count": obb_free_count,
            "obb_collision_count": obb_collision_count,
            "pybullet_free_count": pybullet_free_count,
            "pybullet_collision_count": pybullet_collision_count,
            "mismatches": len(mismatches),
            "results": results,
        }

        return report

    def close(self):
        """关闭环境"""
        self.obb_env.close()
        self.pybullet_env.close()


def main():
    """主函数"""
    print("碰撞检测对比测试工具")
    print("=" * 50)

    # 创建对比器
    comparator = CollisionDetectionComparator("franka")

    try:
        # 运行对比测试
        report = comparator.run_comparison_test(
            num_configs=50,  # 测试50个配置
            obstacles=[
                ((0.1, 0.1, 0.1), (0.5, 0.0, 0.5)),  # 小障碍物
                ((0.2, 0.2, 0.2), (0.0, 0.5, 0.3)),  # 中等障碍物
                ((0.15, 0.15, 0.15), (-0.3, -0.3, 0.4)),  # 另一个障碍物
            ],
        )

        # 保存报告（可选）
        import json

        with open("collision_comparison_report.json", "w") as f:
            # 将numpy数组转换为列表以便JSON序列化
            json_report = report.copy()
            json_report["results"] = [
                {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in result.items()
                }
                for result in report["results"]
            ]
            json.dump(json_report, f, indent=2)
        print("\n✓ 测试报告已保存到 collision_comparison_report.json")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        comparator.close()


if __name__ == "__main__":
    main()
