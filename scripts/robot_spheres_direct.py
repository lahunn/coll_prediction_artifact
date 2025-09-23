#!/usr/bin/env python3
"""
使用CuRobo直接获取机器人球体模型结构
基于CuRobo自带的方法，输出各个link对应的球体位置和半径
代码简洁，不超过100行
"""

import torch
import argparse
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


def get_robot_spheres(robot_name: str = "franka", device: str = "cuda:0"):
    """获取机器人球体模型结构"""

    # 设置计算设备和数据类型
    tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)

    # 加载机器人配置文件
    config_file = join_path(get_robot_configs_path(), f"{robot_name}.yml")
    robot_cfg = load_yaml(config_file)["robot_cfg"]
    robot_config = RobotConfig.from_dict(robot_cfg, tensor_args)

    # 创建CUDA机器人模型
    cuda_model = CudaRobotModel(robot_config.kinematics)

    print(f"\n=== 机器人 {robot_name} 球体模型信息 ===")
    print(f"自由度: {cuda_model.get_dof()}")

    # 获取有球体数据的连杆名称
    link_name_to_idx_map = cuda_model.kinematics_config.link_name_to_idx_map
    if link_name_to_idx_map is not None:
        collision_link_names = list(link_name_to_idx_map.keys())
        print(f"碰撞连杆数: {len(collision_link_names)}")

        # 打印各连杆的球体信息（相对坐标）
        print("\n--- 各连杆球体信息（连杆坐标系） ---")
        total_spheres = 0
        for link_name in collision_link_names:
            try:
                spheres = cuda_model.kinematics_config.get_link_spheres(link_name)
                if spheres.shape[0] > 0:
                    valid_spheres = spheres[spheres[:, 3] > 0]  # 过滤有效球体
                    if valid_spheres.shape[0] > 0:
                        print(f"\n连杆 {link_name}: {valid_spheres.shape[0]} 个球体")
                        for i, sphere in enumerate(valid_spheres):
                            x, y, z, r = sphere[0], sphere[1], sphere[2], sphere[3]
                            print(
                                f"  球体{i + 1}: 位置[{x:.4f}, {y:.4f}, {z:.4f}] 半径{r:.4f}"
                            )
                            total_spheres += 1
            except Exception as e:
                print(f"连杆 {link_name}: 无球体数据 ({e})")

        print(f"\n总有效球体数: {total_spheres}")

    # 获取默认关节配置下的世界坐标球体位置
    print("\n--- 默认配置下球体世界坐标 ---")

    # 使用收起配置或零配置
    if (
        cuda_model.kinematics_config.cspace is not None
        and hasattr(cuda_model.kinematics_config.cspace, "retract_config")
        and cuda_model.kinematics_config.cspace.retract_config is not None
    ):
        q = cuda_model.kinematics_config.cspace.retract_config.clone().unsqueeze(0)
        print("使用收起配置")
    else:
        q = torch.zeros((1, cuda_model.get_dof()), **tensor_args.as_torch_dict())
        print("使用零配置")

    print(f"关节角度: {[f'{x:.3f}' for x in q.squeeze().tolist()]}")

    # 获取运动学状态和球体世界坐标
    state = cuda_model.get_state(q)
    spheres_world = state.get_link_spheres()  # shape: [batch, n_spheres, 4]

    print("\n世界坐标下前10个球体:")
    spheres_data = spheres_world[0].cpu().numpy()  # 取第一个batch

    valid_count = 0
    for i, sphere in enumerate(spheres_data):
        x, y, z, r = sphere[0], sphere[1], sphere[2], sphere[3]
        if r > 0:  # 只显示有效球体
            print(
                f"  球体{valid_count + 1}: 位置[{x:.4f}, {y:.4f}, {z:.4f}] 半径{r:.4f}"
            )
            valid_count += 1
            if valid_count >= 10:  # 只显示前10个
                break

    print(f"\n总计 {(spheres_data[:, 3] > 0).sum()} 个有效球体")


def main():
    parser = argparse.ArgumentParser(description="获取CuRobo机器人球体模型")
    parser.add_argument("--robot", default="franka", help="机器人名称 (默认: franka)")
    parser.add_argument("--device", default="cuda:0", help="计算设备 (默认: cuda:0)")

    args = parser.parse_args()

    try:
        get_robot_spheres(args.robot, args.device)
    except Exception as e:
        print(f"错误: {e}")
        print("请检查机器人名称是否正确，可用的机器人包括: franka, ur5e, iiwa, etc.")


if __name__ == "__main__":
    main()
