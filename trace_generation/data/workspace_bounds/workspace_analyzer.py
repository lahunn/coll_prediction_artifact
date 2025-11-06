#!/usr/bin/env python3
"""
机器人工作空间分析程序

该程序通过分析 robot_method 中加载的机器人，计算机器人的大概工作空间范围。
通过采样不同的关节配置，获取末端执行器的位置分布，从而估算工作空间边界。

输出：x_start, x_end, y_start, y_end, z_start, z_end
其中x,y方向的start和end是对称的。

使用方法:
python workspace_analyzer.py <robot_name> [output_json_file]

示例:
python workspace_analyzer.py franka workspace.json
"""

import numpy as np
import random
import json
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from core.robot.environment import RobotEnv


class WorkspaceAnalyzer:
    """机器人工作空间分析器"""

    def __init__(self, robot_name):
        """
        初始化工作空间分析器

        Args:
            robot_name: 机器人名称
        """
        self.robot_name = robot_name
        self.robot_env = None
        self.joint_limits = []
        self.valid_joints = []

    def load_robot(self):
        """加载机器人模型"""
        try:
            self.robot_env = RobotEnv(self.robot_name)
        except SystemExit:
            return False
        except Exception as e:
            print(f"机器人加载失败: {e}")
            return False

        self._setup_joint_info()
        print(f"成功加载机器人: {self.robot_env.robot_file}")
        print(f"机器人有 {len(self.valid_joints)} 个可动关节")
        return True

    def _setup_joint_info(self):
        """设置关节信息"""
        if self.robot_env is None:
            return

        self.joint_limits = list(
            zip(self.robot_env.lower_bounds.tolist(), self.robot_env.upper_bounds.tolist())
        )
        self.valid_joints = list(self.robot_env.valid_joints)

        for idx, (lower_limit, upper_limit) in zip(self.valid_joints, self.joint_limits):
            print(
                f"  关节 {idx}: [{lower_limit:.3f}, {upper_limit:.3f}]"
            )

    def sample_workspace(self, num_samples=1000):
        """
        通过采样分析工作空间

        Args:
            num_samples: 采样次数

        Returns:
            positions: 末端执行器位置列表 [(x, y, z), ...]
        """
        positions = []

        print(f"开始采样工作空间，总共 {num_samples} 次...")

        for i in range(num_samples):
            # 生成随机关节配置
            joint_config = []
            for lower, upper in self.joint_limits:
                angle = random.uniform(lower, upper)
                joint_config.append(angle)

            # 设置机器人配置
            self.set_robot_config(joint_config)

            # 获取末端执行器位置
            end_effector_pos = self.get_end_effector_position()
            if end_effector_pos is not None:
                positions.append(end_effector_pos)

            # 输出进度
            if (i + 1) % (num_samples // 10) == 0:
                print(f"  已完成 {i + 1}/{num_samples} 次采样")

        print(f"采样完成，获得 {len(positions)} 个有效位置")
        return positions

    def set_robot_config(self, joint_angles):
        """设置机器人关节配置"""
        if self.robot_env is None:
            return

        self.robot_env.set_config(joint_angles)

    def get_end_effector_position(self):
        """获取末端执行器位置"""
        if self.robot_env is None:
            return None

        points = self.robot_env.get_robot_points(None, end_point=True)
        return points[0] if points else None

    def analyze_workspace_bounds(self, positions):
        """
        分析工作空间边界

        Args:
            positions: 位置列表 [(x, y, z), ...]

        Returns:
            dict: 工作空间边界信息
        """
        if not positions:
            print("警告：没有有效的位置数据")
            return None

        positions_array = np.array(positions)

        # 计算各轴的最小值和最大值
        x_min, y_min, z_min = np.min(positions_array, axis=0)
        x_max, y_max, z_max = np.max(positions_array, axis=0)

        # 计算各轴的范围
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # 使对称的范围 (以原点为中心)
        max_xy_range = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

        # 为了安全，稍微扩大工作空间边界
        safety_margin = 0.1  # 10cm安全边距

        workspace_bounds = {
            "x_start": -(max_xy_range + safety_margin),
            "x_end": max_xy_range + safety_margin,
            "y_start": -(max_xy_range + safety_margin),
            "y_end": max_xy_range + safety_margin,
            "z_start": max(z_min - safety_margin, 0.0),  # Z最小值不能小于0
            "z_end": z_max + safety_margin,
            # 额外的统计信息
            "statistics": {
                "num_samples": len(positions),
                "x_range": x_range,
                "y_range": y_range,
                "z_range": z_range,
                "raw_bounds": {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "z_min": z_min,
                    "z_max": z_max,
                },
            },
        }

        return workspace_bounds

    def print_workspace_summary(self, workspace_bounds):
        """打印工作空间摘要"""
        if workspace_bounds is None:
            return

        print("\n=== 工作空间分析结果 ===")
        print(
            f"X 轴范围: {workspace_bounds['x_start']:.3f} 到 {workspace_bounds['x_end']:.3f}"
        )
        print(
            f"Y 轴范围: {workspace_bounds['y_start']:.3f} 到 {workspace_bounds['y_end']:.3f}"
        )
        print(
            f"Z 轴范围: {workspace_bounds['z_start']:.3f} 到 {workspace_bounds['z_end']:.3f}"
        )

        stats = workspace_bounds["statistics"]
        print("\n统计信息:")
        print(f"  采样点数: {stats['num_samples']}")
        print(f"  X 轴实际范围: {stats['x_range']:.3f}m")
        print(f"  Y 轴实际范围: {stats['y_range']:.3f}m")
        print(f"  Z 轴实际范围: {stats['z_range']:.3f}m")

        raw = stats["raw_bounds"]
        print("\n原始边界:")
        print(f"  X: [{raw['x_min']:.3f}, {raw['x_max']:.3f}]")
        print(f"  Y: [{raw['y_min']:.3f}, {raw['y_max']:.3f}]")
        print(f"  Z: [{raw['z_min']:.3f}, {raw['z_max']:.3f}]")

    def save_workspace_bounds(self, workspace_bounds, output_file):
        """保存工作空间边界到JSON文件"""
        try:
            with open(output_file, "w") as f:
                json.dump(workspace_bounds, f, indent=2)
            print(f"\n工作空间边界已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件失败: {e}")

    def disconnect(self):
        """断开PyBullet连接"""
        if self.robot_env is not None:
            self.robot_env.close()
            self.robot_env = None


def load_workspace_bounds(json_file):
    """
    从JSON文件加载工作空间边界

    Args:
        json_file: JSON文件路径

    Returns:
        dict: 工作空间边界信息
    """
    try:
        with open(json_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"加载工作空间文件失败: {e}")
        return None


def main():
    """主程序"""
    if len(sys.argv) < 2:
        print("用法: python workspace_analyzer.py <robot_name> [output_json_file]")
        print("示例: python workspace_analyzer.py franka workspace.json")
        return

    robot_name = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"{robot_name}_workspace.json"

    # 创建工作空间分析器
    analyzer = WorkspaceAnalyzer(robot_name)

    try:
        # 加载机器人
        if not analyzer.load_robot():
            return

        # 采样工作空间
        positions = analyzer.sample_workspace(num_samples=2000)

        # 分析工作空间边界
        workspace_bounds = analyzer.analyze_workspace_bounds(positions)

        # 打印结果
        analyzer.print_workspace_summary(workspace_bounds)

        # 保存结果
        analyzer.save_workspace_bounds(workspace_bounds, output_file)

    finally:
        analyzer.disconnect()


if __name__ == "__main__":
    main()
