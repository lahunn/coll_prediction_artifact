#!/usr/bin/env python3
"""
机器人球体结构分析模块

专门负责从CuRobo获取和分析机器人的球体模型结构信息，
包括连杆坐标系和世界坐标系下的球体信息。
"""

import torch
import argparse
from typing import Dict, Tuple, Optional
import numpy as np
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


class RobotSphereAnalyzer:
    """机器人球体结构分析器"""

    def __init__(self, robot_name: str, device: str = "cuda:0"):
        """初始化分析器

        Args:
            robot_name: 机器人名称
            device: 计算设备
        """
        self.robot_name = robot_name
        self.device = device
        self.cuda_model = self._load_robot_model()

    def _load_robot_model(self) -> CudaRobotModel:
        """加载机器人模型

        Returns:
            CudaRobotModel: 加载的机器人模型
        """
        tensor_args = TensorDeviceType(
            device=torch.device(self.device), dtype=torch.float32
        )
        config_file = join_path(get_robot_configs_path(), f"{self.robot_name}.yml")
        robot_cfg = load_yaml(config_file)["robot_cfg"]
        robot_config = RobotConfig.from_dict(robot_cfg, tensor_args)
        return CudaRobotModel(robot_config.kinematics)

    def get_robot_info(self) -> Dict:
        """获取机器人基本信息

        Returns:
            Dict: 包含机器人基本信息的字典
        """
        return {
            "name": self.robot_name,
            "dof": self.cuda_model.get_dof(),
            "device": self.device,
        }

    def get_link_spheres_info(self) -> Dict[str, np.ndarray]:
        """获取各连杆的球体信息（连杆坐标系）

        Returns:
            Dict[str, np.ndarray]: 连杆名称到球体信息的映射，格式为 [x, y, z, radius]
        """
        link_name_to_idx_map = self.cuda_model.kinematics_config.link_name_to_idx_map
        if link_name_to_idx_map is None:
            return {}

        link_spheres = {}
        for link_name in link_name_to_idx_map.keys():
            try:
                spheres = self.cuda_model.kinematics_config.get_link_spheres(link_name)
                valid_spheres = spheres[spheres[:, 3] > 0]  # 过滤有效球体
                if valid_spheres.shape[0] > 0:
                    link_spheres[link_name] = valid_spheres.cpu().numpy()
            except Exception:
                continue  # 忽略获取失败的连杆

        return link_spheres

    def get_default_joint_config(self) -> Tuple[torch.Tensor, str]:
        """获取默认关节配置

        Returns:
            Tuple[torch.Tensor, str]: (关节配置, 配置类型描述)
        """
        # 获取默认关节配置
        if (
            self.cuda_model.kinematics_config.cspace is not None
            and hasattr(self.cuda_model.kinematics_config.cspace, "retract_config")
            and self.cuda_model.kinematics_config.cspace.retract_config is not None
        ):
            q = self.cuda_model.kinematics_config.cspace.retract_config.clone().unsqueeze(
                0
            )
            config_type = "收起配置"
        else:
            # 获取tensor参数信息
            device = self.cuda_model.kinematics_config.fixed_transforms.device
            dtype = self.cuda_model.kinematics_config.fixed_transforms.dtype
            q = torch.zeros((1, self.cuda_model.get_dof()), device=device, dtype=dtype)
            config_type = "零配置"

        return q, config_type

    def get_world_spheres_with_links(
        self, joint_config: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取球体在世界坐标系下的信息及其连杆对应关系

        Args:
            joint_config: 关节配置，如果为None则使用默认配置

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - world_spheres: 世界坐标系下的球体信息 [n_spheres, 4] (x, y, z, radius)
                - sphere_link_ids: 每个球体对应的连杆ID [n_spheres,]
        """
        if joint_config is None:
            joint_config, _ = self.get_default_joint_config()

        # 确保输入是批次形式
        if hasattr(joint_config, "shape") and len(joint_config.shape) == 1:
            joint_config = joint_config.unsqueeze(0)

        # 计算正向运动学
        state = self.cuda_model.get_state(joint_config)
        spheres_world = state.get_link_spheres()[0].cpu().numpy()

        # 获取球体与连杆的对应关系
        sphere_link_ids = []
        link_name_to_idx_map = self.cuda_model.kinematics_config.link_name_to_idx_map
        if link_name_to_idx_map is not None:
            # 按连杆索引顺序遍历
            for link_name in sorted(
                link_name_to_idx_map.keys(), key=lambda x: link_name_to_idx_map[x]
            ):
                link_idx = link_name_to_idx_map[link_name]

                try:
                    spheres = self.cuda_model.kinematics_config.get_link_spheres(
                        link_name
                    )
                    valid_spheres = spheres[spheres[:, 3] > 0]  # 过滤有效球体
                    num_valid_spheres = valid_spheres.shape[0]

                    if num_valid_spheres > 0:
                        # 为该连杆的所有球体记录连杆ID
                        sphere_link_ids.extend([link_idx] * num_valid_spheres)

                except Exception:
                    continue

        sphere_link_ids = np.array(sphere_link_ids)

        # 过滤有效球体及其对应的连杆ID
        valid_mask = spheres_world[:, 3] > 0
        valid_world_spheres = spheres_world[valid_mask]
        valid_sphere_link_ids = sphere_link_ids

        return valid_world_spheres, valid_sphere_link_ids

    def get_world_spheres(
        self, joint_config: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """获取球体在世界坐标系下的信息（核心返回函数）

        Args:
            joint_config: 关节配置，如果为None则使用默认配置

        Returns:
            np.ndarray: 世界坐标系下的球体信息，形状为 [n_spheres, 4]
                       每行格式为 [x, y, z, radius]
        """
        if joint_config is None:
            joint_config, _ = self.get_default_joint_config()

        # 确保输入是批次形式
        if hasattr(joint_config, "shape") and len(joint_config.shape) == 1:
            joint_config = joint_config.unsqueeze(0)

        # 计算正向运动学
        state = self.cuda_model.get_state(joint_config)
        spheres_world = state.get_link_spheres()[0].cpu().numpy()

        # 过滤有效球体
        valid_world_spheres = spheres_world[spheres_world[:, 3] > 0]
        return valid_world_spheres

    def analyze_spheres(self) -> Dict:
        """分析机器人球体结构

        Returns:
            Dict: 包含完整分析结果的字典
        """
        # 获取基本信息
        robot_info = self.get_robot_info()

        # 获取连杆球体信息
        link_spheres = self.get_link_spheres_info()

        # 获取默认关节配置和世界坐标球体
        joint_config, config_type = self.get_default_joint_config()
        world_spheres = self.get_world_spheres(joint_config)

        # 统计信息
        total_link_spheres = sum(len(spheres) for spheres in link_spheres.values())

        return {
            "robot_info": robot_info,
            "link_spheres": link_spheres,
            "joint_config": joint_config,
            "config_type": config_type,
            "world_spheres": world_spheres,
            "statistics": {
                "total_link_spheres": total_link_spheres,
                "total_world_spheres": len(world_spheres),
                "num_links_with_spheres": len(link_spheres),
            },
        }

    def print_analysis(self):
        """打印分析结果"""
        analysis = self.analyze_spheres()

        # 打印基本信息
        robot_info = analysis["robot_info"]
        print(f"\n=== {robot_info['name']} 机器人球体模型 ===")
        print(f"自由度: {robot_info['dof']}")
        print(f"设备: {robot_info['device']}")

        # 打印连杆球体信息
        link_spheres = analysis["link_spheres"]
        stats = analysis["statistics"]
        print(f"碰撞连杆数: {stats['num_links_with_spheres']}")
        print("\n--- 各连杆球体信息（连杆坐标系） ---")

        for link_name, spheres in link_spheres.items():
            print(f"\n{link_name}: {len(spheres)} 个球体")
            for i, (x, y, z, r) in enumerate(spheres):
                print(f"  球体{i + 1}: 位置[{x:.4f}, {y:.4f}, {z:.4f}] 半径{r:.4f}")

        print(f"\n总球体数: {stats['total_link_spheres']}")

        # 打印世界坐标球体信息
        joint_config = analysis["joint_config"]
        config_type = analysis["config_type"]
        world_spheres = analysis["world_spheres"]

        print("\n--- 默认配置下球体世界坐标 ---")
        print(
            f"使用{config_type}: {[f'{x:.3f}' for x in joint_config.squeeze().tolist()]}"
        )

        print("\n世界坐标下球体 (显示前10个):")
        for i, (x, y, z, r) in enumerate(world_spheres[:10]):
            print(f"  球体{i + 1}: 位置[{x:.4f}, {y:.4f}, {z:.4f}] 半径{r:.4f}")

        print(f"\n总计{len(world_spheres)}个世界坐标球体")


def main():
    """主函数 - 独立的球体结构分析工具"""
    parser = argparse.ArgumentParser(
        description="CuRobo机器人球体结构分析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --robot franka             # 分析Franka机器人球体结构
  %(prog)s --robot ur5e --device cpu # 使用CPU设备分析UR5e
        """,
    )
    parser.add_argument("--robot", default="franka", help="机器人名称")
    parser.add_argument("--device", default="cuda:0", help="计算设备")

    args = parser.parse_args()

    try:
        # 创建分析器并运行分析
        analyzer = RobotSphereAnalyzer(args.robot, args.device)
        analyzer.print_analysis()

        # 获取分析结果供其他模块使用
        analysis_result = analyzer.analyze_spheres()
        print(
            f"\n分析完成，共发现 {analysis_result['statistics']['total_world_spheres']} 个世界坐标球体"
        )

    except Exception as e:
        print(f"分析失败: {e}")
        print("请检查机器人名称和设备配置")


if __name__ == "__main__":
    main()
