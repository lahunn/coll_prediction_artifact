#!/usr/bin/env python3
"""
使用CuRobo获取机器人球体模型的简洁脚本
直接调用cuRobo自带方法，输出机器人各个link的球体位置和半径
"""

import torch
import argparse
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


def print_robot_spheres(robot_name: str = "franka", device: str = "cuda:0"):
    """打印机器人球体模型信息"""

    # 1. 初始化设备和加载机器人配置
    tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    config_file = join_path(get_robot_configs_path(), f"{robot_name}.yml")
    robot_cfg = load_yaml(config_file)["robot_cfg"]
    robot_config = RobotConfig.from_dict(robot_cfg, tensor_args)
    cuda_model = CudaRobotModel(robot_config.kinematics)

    print(f"\n=== {robot_name} 机器人球体模型 ===")
    print(f"自由度: {cuda_model.get_dof()}")

    # 2. 显示各连杆球体信息（相对坐标）
    link_name_to_idx_map = cuda_model.kinematics_config.link_name_to_idx_map
    if link_name_to_idx_map is None:
        print("错误: 无法获取连杆信息")
        return

    print(f"碰撞连杆数: {len(link_name_to_idx_map)}")
    print("\n--- 各连杆球体信息（连杆坐标系） ---")

    total_spheres = 0
    for link_name in link_name_to_idx_map.keys():
        try:
            spheres = cuda_model.kinematics_config.get_link_spheres(link_name)
            valid_spheres = spheres[spheres[:, 3] > 0]  # 过滤有效球体

            if valid_spheres.shape[0] > 0:
                print(f"\n{link_name}: {valid_spheres.shape[0]} 个球体")
                for i, (x, y, z, r) in enumerate(valid_spheres):
                    print(f"  球体{i + 1}: 位置[{x:.4f}, {y:.4f}, {z:.4f}] 半径{r:.4f}")
                total_spheres += valid_spheres.shape[0]
        except Exception as e:
            print(f"{link_name}: 获取球体失败 ({e})")

    print(f"\n总球体数: {total_spheres}")

    # 3. 显示默认配置下的世界坐标球体
    print("\n--- 默认配置下球体世界坐标 ---")

    # 获取默认关节配置
    if (
        cuda_model.kinematics_config.cspace is not None
        and hasattr(cuda_model.kinematics_config.cspace, "retract_config")
        and cuda_model.kinematics_config.cspace.retract_config is not None
    ):
        q = cuda_model.kinematics_config.cspace.retract_config.clone().unsqueeze(0)
        config_type = "收起配置"
    else:
        q = torch.zeros((1, cuda_model.get_dof()), **tensor_args.as_torch_dict())
        config_type = "零配置"

    print(f"使用{config_type}: {[f'{x:.3f}' for x in q.squeeze().tolist()]}")

    # 获取世界坐标下的球体位置
    state = cuda_model.get_state(q)
    spheres_world = state.get_link_spheres()[0].cpu().numpy()  # [n_spheres, 4]
    valid_world_spheres = spheres_world[spheres_world[:, 3] > 0]

    print("\n世界坐标下球体 (显示前10个):")
    for i, (x, y, z, r) in enumerate(valid_world_spheres[:10]):
        print(f"  球体{i + 1}: 位置[{x:.4f}, {y:.4f}, {z:.4f}] 半径{r:.4f}")

    print(f"\n总计{len(valid_world_spheres)}个世界坐标球体")


def main():
    parser = argparse.ArgumentParser(description="获取CuRobo机器人球体模型")
    parser.add_argument("--robot", default="franka", help="机器人名称")
    parser.add_argument("--device", default="cuda:0", help="计算设备")

    args = parser.parse_args()

    try:
        print_robot_spheres(args.robot, args.device)
    except Exception as e:
        print(f"错误: {e}")
        print("请检查机器人名称，可用的包括: franka, ur5e, iiwa 等")


if __name__ == "__main__":
    main()
