#!/usr/bin/env python3
"""
测试 SphereEnvCurobo 在不同场景下的碰撞检测

测试场景：
- 无障碍物
- 1个障碍物
- 10个障碍物

对于每个场景，测试多个关节配置的世界碰撞和自碰撞。
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from sphere_coll_curobo import SphereEnvCurobo
from curobo.geom.types import WorldConfig


def test_collision_scenarios():
    """测试不同场景下的碰撞检测"""

    # 定义测试关节配置
    joint_configs = {
        "零配置": np.zeros(7),
        "随机配置1": np.random.uniform(-0.5, 0.5, 7),
        "随机配置2": np.random.uniform(-1.0, 1.0, 7),
        "折叠配置": np.array([0, -1.57, 0, -1.57, 0, 1.57, 0]),  # 近似折叠姿态
    }

    # 定义测试场景
    scenarios = [
        {
            "name": "无障碍物",
            "num_obstacles": 0,
            "obstacles": None,
            "world_config": WorldConfig(cuboid=[]),
        },  # 空世界
        {
            "name": "1个障碍物",
            "num_obstacles": 1,
            "obstacles": [([0.1, 0.1, 0.1], [0.5, 0.0, 0.3])],  # 靠近机器人
            "world_config": None,
        },
        {
            "name": "10个障碍物",
            "num_obstacles": 10,
            "obstacles": [
                ([0.05, 0.05, 0.05], [0.3 + i * 0.1, 0.0, 0.2]) for i in range(10)
            ],  # 沿x轴分布
            "world_config": None,
        },
    ]

    for scenario in scenarios:
        print(f"\n=== 测试场景: {scenario['name']} ===")

        # 初始化环境
        env = SphereEnvCurobo(
            robot_name="franka", world_config=scenario.get("world_config")
        )

        # 设置障碍物
        if scenario["num_obstacles"] > 0:
            env.init_obstacle_bodies(scenario["num_obstacles"], scenario["obstacles"])
        else:
            env.init_obstacle_bodies(0)

        for config_name, joint_state in joint_configs.items():
            print(f"\n  -- {config_name}: {joint_state}")

            # 转换为张量
            joint_config = torch.tensor(
                joint_state, dtype=torch.float32, device="cuda:0"
            ).unsqueeze(0)

            # 检查碰撞
            world_collision = env._check_world_collision(joint_config)
            self_collision = env._check_self_collision(joint_config)

            print(f"    世界碰撞: {world_collision}")
            print(f"    自碰撞: {self_collision}")

            # 整体碰撞
            any_collision, _ = env._check_sphere_collision(joint_config)
            print(f"    整体碰撞: {any_collision}")

        # 清理
        env.cleanup_obstacles()
        env.close()


if __name__ == "__main__":
    test_collision_scenarios()
