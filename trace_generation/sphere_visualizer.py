#!/usr/bin/env python3
"""
球体结构可视化程序

功能：
1. 解析YAML格式的球体配置文件
2. 在PyBullet中可视化球体结构
3. 支持球体颜色、透明度设置
4. 提供交互式界面控制

使用方法：
python sphere_visualizer.py <yaml_file_path>

示例：
python sphere_visualizer.py ../content/configs/robot/spheres/franka.yml
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List


class SphereConfig:
    """球体配置数据结构"""

    def __init__(self, center: List[float], radius: float):
        self.center = np.array(center)
        self.radius = abs(radius)  # 确保半径为正数
        self.is_valid = radius > 0  # 半径 <= 0 的球体被标记为无效

    def __repr__(self):
        return f"Sphere(center={self.center}, radius={self.radius:.3f}, valid={self.is_valid})"


class SphereYAMLParser:
    """YAML球体配置文件解析器"""

    def __init__(self, yaml_path: str):
        self.yaml_path = Path(yaml_path)
        self.collision_spheres = {}
        self.robot_name = "Unknown"
        self.urdf_path = ""

    def parse(self) -> Dict[str, List[SphereConfig]]:
        """解析YAML文件并返回球体配置"""
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # 提取机器人信息
            self.robot_name = data.get("robot", "Unknown Robot")
            self.urdf_path = data.get("urdf_path", "")

            # 解析球体配置
            collision_spheres_data = data.get("collision_spheres", {})

            for link_name, spheres_list in collision_spheres_data.items():
                sphere_configs = []

                for sphere_data in spheres_list:
                    center = sphere_data.get("center", [0, 0, 0])
                    radius = sphere_data.get("radius", 0.01)

                    sphere_config = SphereConfig(center, radius)
                    sphere_configs.append(sphere_config)

                self.collision_spheres[link_name] = sphere_configs

            print(f"成功解析配置文件: {self.yaml_path}")
            print(f"机器人: {self.robot_name}")
            print(f"URDF路径: {self.urdf_path}")
            print(f"解析到 {len(self.collision_spheres)} 个连杆的球体配置")

            return self.collision_spheres

        except Exception as e:
            print(f"解析YAML文件失败: {e}")
            return {}

    def print_summary(self):
        """打印球体配置摘要"""
        print("\n=== 球体配置摘要 ===")
        total_spheres = 0
        valid_spheres = 0

        for link_name, spheres in self.collision_spheres.items():
            valid_count = sum(1 for s in spheres if s.is_valid)
            total_count = len(spheres)

            print(f"{link_name}: {valid_count}/{total_count} 有效球体")
            total_spheres += total_count
            valid_spheres += valid_count

        print(f"总计: {valid_spheres}/{total_spheres} 有效球体")


if __name__ == "__main__":
    print("THIS is not main ")
