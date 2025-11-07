#!/usr/bin/env python3
"""通用机器人 OBB 分析工具"""

import sys
import argparse
from pathlib import Path
import numpy as np

from trace_generation.core.robot.environment import robot_urdf_mapping  # noqa: E402
from trace_generation.core.robot.obb_calculator import (  # noqa: E402
    calculate_link_obbs,
    check_dependencies,
    get_default_coacd_params,
)


project_root = Path(__file__).parent.parent.parent.parent


def save_obbs_to_file(obbs, robot_name, output_path):
    """保存 OBB 数据到 Python 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            f'"""{robot_name.upper()} 机器人 OBB 数据（自动生成）"""\nimport numpy as np\n\n'
        )

        # 原始格式
        f.write(f"{robot_name}_obbs = [\n")
        for obb in obbs:
            f.write("    {\n")
            f.write(f'        "link_name": "{obb["link_name"]}",\n')
            f.write(f'        "position": np.array({list(obb["position"])}),\n')
            f.write(f'        "extents": np.array({list(obb["extents"])}),\n')
            f.write(f'        "rotation_matrix": np.{repr(obb["rotation_matrix"])},\n')
            f.write(f'        "volume": {obb["volume"]},\n')
            f.write("    },\n")
        f.write("]\n\n")

        # Transform 格式（兼容 obb_forward_kinematics）
        f.write(f"{robot_name}_obbs_with_transform = [\n")
        for obb in obbs:
            transform = np.eye(4)
            transform[:3, :3] = obb["rotation_matrix"]
            transform[:3, 3] = obb["position"]
            f.write("    {\n")
            f.write(f'        "link_name": "{obb["link_name"]}",\n')
            f.write(f'        "extents": np.array({list(obb["extents"])}),\n')
            f.write(f'        "transform": np.{repr(transform)},\n')
            f.write(f'        "volume": {obb["volume"]},\n')
            f.write("    },\n")
        f.write("]\n\n")

        # 统计信息
        volumes = [obb["volume"] for obb in obbs]
        f.write(f"num_links = {len(obbs)}\n")
        f.write(f"total_volume = {sum(volumes)}\n")
        f.write(f"avg_volume = {np.mean(volumes)}\n")
        f.write(f"max_volume = {max(volumes)}\n")
        f.write(f"min_volume = {min(volumes)}\n")
        f.write(f'max_volume_link = "{obbs[np.argmax(volumes)]["link_name"]}"\n')
        f.write(f'min_volume_link = "{obbs[np.argmin(volumes)]["link_name"]}"\n')


def main():
    parser = argparse.ArgumentParser(
        description="机器人 OBB 分析",
        epilog=f"支持的机器人: {', '.join(sorted(robot_urdf_mapping.keys()))}",
    )
    parser.add_argument("robot_name", help="机器人名称")
    parser.add_argument("--threshold", type=float, default=0.05, help="CoACD 阈值")
    parser.add_argument("--max-convex-hull", type=int, default=32, help="最大凸包数")
    parser.add_argument("--resolution", type=int, default=2000, help="分辨率")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细过程")
    args = parser.parse_args()

    # 检查依赖
    has_libs, missing = check_dependencies()
    if not has_libs:
        print(f"❌ 缺少依赖: {', '.join(missing)}")
        return 1

    # 获取 URDF
    rel_urdf_path = robot_urdf_mapping.get(args.robot_name)
    if not rel_urdf_path:
        print(f"❌ 未找到 '{args.robot_name}'")
        print(f"支持: {', '.join(sorted(robot_urdf_mapping.keys()))}")
        return 1

    urdf_path = project_root / rel_urdf_path
    if not urdf_path.exists():
        print(f"❌ URDF 不存在: {urdf_path}")
        return 1

    # 计算 OBB
    print(f"计算 {args.robot_name.upper()} OBB...")
    coacd_params = get_default_coacd_params()
    coacd_params.update(
        {
            "threshold": args.threshold,
            "max_convex_hull": args.max_convex_hull,
            "resolution": args.resolution,
        }
    )
    obbs = calculate_link_obbs(str(urdf_path), coacd_params, args.verbose)

    # 保存
    output_file = (
        project_root
        / "trace_generation"
        / "core"
        / "robot"
        / "robot_config"
        / f"{args.robot_name}_obbs.py"
    )
    save_obbs_to_file(obbs, args.robot_name, output_file)

    # 统计
    volumes = [obb["volume"] for obb in obbs]
    print(f"\n✓ 完成: {len(obbs)} 个连杆")
    print(f"  总体积: {sum(volumes):.6f} m³")
    print(f"  保存到: {output_file.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
