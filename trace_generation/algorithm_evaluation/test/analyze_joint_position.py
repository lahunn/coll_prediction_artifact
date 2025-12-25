#!/usr/bin/env python3
"""
分析URDF中prismatic关节的origin和limit关系
"""

import numpy as np


def analyze_prismatic_joint():
    """分析prismatic关节的位置计算"""

    print("=== 分析Prismatic关节位置计算 ===")

    # 从URDF中提取的关节参数
    joint_name = "panda_finger_joint1"

    # origin定义：关节值为0时的初始变换
    origin_xyz = np.array([0, 0, 0.0584])  # xyz="0 0 0.0584"
    origin_rpy = np.array([0, 0, 0])       # rpy="0 0 0"

    # axis定义：关节运动方向
    axis = np.array([0, 1, 0])  # xyz="0 1 0"

    # limit定义：关节运动范围
    lower_limit = 0.0   # lower="0.0"
    upper_limit = 0.04  # upper="0.04"

    print(f"关节名称: {joint_name}")
    print(f"Origin XYZ: {origin_xyz}")
    print(f"Origin RPY: {origin_rpy}")
    print(f"运动轴: {axis}")
    print(f"关节限位: [{lower_limit}, {upper_limit}]")
    print()

    # 计算不同关节值对应的位置
    joint_values = [0.0, 0.01, 0.02, 0.03, 0.04]

    print("关节值 -> 子link相对于父link的位置:")
    print("关节值\t位置变换\t\t最终位置")
    print("-" * 50)

    for joint_val in joint_values:
        # Prismatic关节的位置变换 = origin + joint_val * axis
        transform = origin_xyz + joint_val * axis
        print(f"{joint_val:.2f}\t{transform}\t{transform + np.array([0, 0, 0.0584])}")

    print()
    print("=== 详细解释 ===")
    print("1. Origin (0, 0, 0.0584) 是关节值为0时的基准位置")
    print("2. Axis (0, 1, 0) 表示沿Y轴正方向运动")
    print("3. 关节值表示沿运动轴的位移量")
    print("4. 最终位置 = Origin + 关节值 × Axis")
    print()
    print("因此：")
    print("- 关节值为0.0时：位置 = (0, 0, 0.0584)")
    print("- 关节值为0.04时：位置 = (0, 0.04, 0.0584)")
    print("  相当于在Y轴上从0.0584移动到0.0984")

    # 验证计算
    print()
    print("=== 验证计算 ===")
    joint_0_pos = origin_xyz + 0.0 * axis
    joint_04_pos = origin_xyz + 0.04 * axis

    print(f"关节0.0位置: {joint_0_pos}")
    print(f"关节0.04位置: {joint_04_pos}")
    print(f"位置差值: {joint_04_pos - joint_0_pos}")
    print(f"运动轴方向: {axis}")
    print(f"验证结果: 位置差值 == 0.04 * 运动轴 ? {np.allclose(joint_04_pos - joint_0_pos, 0.04 * axis)}")


if __name__ == "__main__":
    analyze_prismatic_joint()