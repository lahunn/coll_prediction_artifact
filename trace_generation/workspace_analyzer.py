#!/usr/bin/env python3
"""
机器人工作空间分析程序

该程序通过分析机器人URDF文件，计算机器人的大概工作空间范围。
通过采样不同的关节配置，获取末端执行器的位置分布，从而估算工作空间边界。

输出：x_start, x_end, y_start, y_end, z_start, z_end
其中x,y方向的start和end是对称的。

使用方法:
python workspace_analyzer.py <robot_urdf_path> [output_json_file]

示例:
python workspace_analyzer.py /path/to/robot.urdf workspace.json
"""

import pybullet as p
import numpy as np
import math
import random
import json
import sys
import os


class WorkspaceAnalyzer:
    """机器人工作空间分析器"""

    def __init__(self, robot_urdf_path):
        """
        初始化工作空间分析器

        Args:
            robot_urdf_path: 机器人URDF文件路径
        """
        self.robot_urdf_path = robot_urdf_path
        self.robot_id = None
        self.joint_limits = []
        self.valid_joints = []

        # 连接PyBullet (无GUI模式)
        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)

    def load_robot(self):
        """加载机器人模型"""
        try:
            self.robot_id = p.loadURDF(self.robot_urdf_path, [0, 0, 0])
            self._setup_joint_info()
            print(f"成功加载机器人: {self.robot_urdf_path}")
            print(f"机器人有 {len(self.valid_joints)} 个可动关节")
            return True
        except Exception as e:
            print(f"机器人加载失败: {e}")
            return False

    def _setup_joint_info(self):
        """设置关节信息"""
        if self.robot_id is None:
            return

        num_joints = p.getNumJoints(self.robot_id)
        self.joint_limits = []
        self.valid_joints = []

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:  # 非固定关节
                self.valid_joints.append(i)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]

                # 处理无限制关节
                if lower_limit == 0 and upper_limit == -1:
                    lower_limit, upper_limit = -math.pi, math.pi
                elif lower_limit >= upper_limit:
                    lower_limit, upper_limit = -math.pi, math.pi

                self.joint_limits.append((lower_limit, upper_limit))

                joint_name = joint_info[1].decode("utf-8")
                print(
                    f"  关节 {i} ({joint_name}): [{lower_limit:.3f}, {upper_limit:.3f}]"
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
        if self.robot_id is None:
            return

        for i, angle in enumerate(joint_angles):
            if i < len(self.valid_joints):
                p.resetJointState(self.robot_id, self.valid_joints[i], angle)

    def get_end_effector_position(self):
        """获取末端执行器位置"""
        if self.robot_id is None:
            return None

        num_joints = p.getNumJoints(self.robot_id)
        if num_joints == 0:
            # 如果没有关节，使用基座位置
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            return pos

        # 使用最后一个连杆作为末端执行器
        try:
            link_state = p.getLinkState(self.robot_id, num_joints - 1)
            return link_state[0]  # 世界坐标位置
        except Exception:
            # 如果获取失败，尝试倒数第二个连杆
            link_state = p.getLinkState(self.robot_id, num_joints - 2)
            return link_state[0]

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
        p.disconnect()


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
        print("用法: python workspace_analyzer.py <robot_urdf_path> [output_json_file]")
        print("示例: python workspace_analyzer.py /path/to/robot.urdf workspace.json")
        return

    robot_urdf_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "workspace_bounds.json"

    # 验证URDF文件存在
    if not os.path.exists(robot_urdf_path):
        print(f"错误：URDF文件不存在: {robot_urdf_path}")
        return

    # 创建工作空间分析器
    analyzer = WorkspaceAnalyzer(robot_urdf_path)

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
