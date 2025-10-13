#!/usr/bin/env python3
"""
使用 PyBullet 测试和可视化 URDF 及其 OBB

该脚本执行以下操作：
1. 使用 PyBullet 在 GUI 环境中加载指定的 URDF 机器人模型。
2. 调用 obb_calculator.py 为每个连杆计算有向包围盒 (OBB)。
3. 将计算出的 OBB 作为半透明的绿色方块在机器人上进行可视化。
4. 确保 OBB 的位置和姿态与其对应的机器人连杆精确匹配。

这使得用户可以直观地验证 URDF 模型加载的正确性以及 OBB 分解的准确性。

使用方法:
python test_pybullet_obb.py <path_to_your_urdf_file>

例如:
python test_pybullet_obb.py ../data/robots/jaco_7/jaco_7s.urdf
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import argparse
import os
from pathlib import Path

# 确保 obb_calculator.py 在 Python 路径中
try:
    import obb_calculator
    import trimesh
except ImportError as e:
    print(f"错误: 无法导入必要的库: {e}")
    print("请安装: pip install trimesh scipy")
    exit(1)


def rotation_matrix_to_quaternion(m):
    """
    将 3x3 旋转矩阵转换为 [x, y, z, w] 格式的四元数。
    """
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]


def load_link_collision_mesh(urdf_path, link_name, verbose=False):
    """
    从URDF加载指定连杆的碰撞网格
    """
    try:
        urdf_dir = Path(urdf_path).parent

        # 使用yourdfpy加载URDF
        import yourdfpy

        robot = yourdfpy.URDF.load(
            urdf_path,
            load_meshes=True,
            load_collision_meshes=True,
            mesh_dir=str(urdf_dir),
        )

        if link_name not in robot.link_map:
            if verbose:
                print(f"    Link '{link_name}' not found in URDF")
            return None

        link = robot.link_map[link_name]
        if not link.collisions:
            if verbose:
                print(f"    Link '{link_name}' has no collision geometry")
            return None

        # 加载所有碰撞网格
        meshes = []
        for collision in link.collisions:
            if hasattr(collision.geometry, "mesh") and collision.geometry.mesh:
                mesh_path = collision.geometry.mesh.filename

                # 处理相对路径
                if not mesh_path.startswith("/"):
                    resolved_path = urdf_dir / mesh_path
                else:
                    resolved_path = Path(mesh_path)

                if resolved_path.exists():
                    try:
                        loaded_mesh = trimesh.load(str(resolved_path))
                        if isinstance(loaded_mesh, trimesh.Scene):
                            for geom in loaded_mesh.geometry.values():
                                if isinstance(geom, trimesh.Trimesh):
                                    # 应用变换
                                    if (
                                        hasattr(collision, "origin")
                                        and collision.origin is not None
                                    ):
                                        geom.apply_transform(collision.origin)
                                    meshes.append(geom)
                        elif isinstance(loaded_mesh, trimesh.Trimesh):
                            # 应用变换
                            if (
                                hasattr(collision, "origin")
                                and collision.origin is not None
                            ):
                                loaded_mesh.apply_transform(collision.origin)
                            meshes.append(loaded_mesh)
                    except Exception as e:
                        if verbose:
                            print(f"    Failed to load mesh {resolved_path}: {e}")

        if not meshes:
            return None

        # 合并所有网格
        if len(meshes) == 1:
            return meshes[0]
        else:
            return trimesh.util.concatenate(meshes)

    except Exception as e:
        if verbose:
            print(f"    Error loading mesh for link '{link_name}': {e}")
        return None


def point_in_obb(point, obb_center, obb_rotation, obb_extents):
    """
    检查点是否在OBB内部的正确方法

    Args:
        point: 3D点坐标
        obb_center: OBB中心位置
        obb_rotation: OBB旋转矩阵 (3x3)
        obb_extents: OBB尺寸 [长, 宽, 高]

    Returns:
        bool: 点是否在OBB内部
        float: 到OBB边界的最大距离（负值表示内部）
    """
    # 将点变换到OBB局部坐标系
    local_point = np.dot(obb_rotation.T, point - obb_center)

    # 在局部坐标系中进行轴对齐包围盒检测
    half_extents = obb_extents / 2.0
    distances = np.abs(local_point) - half_extents
    max_distance = np.max(distances)

    return max_distance <= 0, max_distance


def validate_obb_coverage(obb_data, mesh, tolerance=0.02, verbose=False):
    """
    验证OBB是否正确覆盖网格

    这个函数使用正确的几何方法验证OBB，不会人为对齐坐标系。
    如果OBB的位置、方向或尺寸有误，会被检测出来。

    Args:
        obb_data: OBB数据字典，包含position, rotation_matrix, extents
        mesh: trimesh对象
        tolerance: 允许的覆盖误差比例 (默认2%)
        verbose: 是否输出详细信息

    Returns:
        dict: 验证结果，包含coverage_ratio, is_valid, details等
    """
    if mesh is None or mesh.is_empty:
        return {
            "is_valid": False,
            "error": "Empty or invalid mesh",
            "coverage_ratio": 0.0,
        }

    try:
        # 获取OBB参数
        obb_center = np.array(obb_data["position"])
        obb_rotation = np.array(obb_data["rotation_matrix"])
        obb_extents = np.array(obb_data["extents"])

        if verbose:
            print(f"    OBB center: {obb_center}")
            print(f"    OBB extents: {obb_extents}")
            print(f"    Mesh vertices: {len(mesh.vertices)}")

        # 获取网格顶点（保持在原始坐标系中）
        mesh_vertices = np.array(mesh.vertices)

        # 使用正确的点-OBB相交检测
        inside_count = 0
        outside_distances = []

        for vertex in mesh_vertices:
            is_inside, distance = point_in_obb(
                vertex, obb_center, obb_rotation, obb_extents
            )
            if is_inside:
                inside_count += 1
            else:
                outside_distances.append(distance)

        total_count = len(mesh_vertices)
        coverage_ratio = inside_count / total_count if total_count > 0 else 0

        # 计算超出距离的统计
        if outside_distances:
            max_outside_distance = max(outside_distances)
            avg_outside_distance = np.mean(outside_distances)
        else:
            max_outside_distance = 0.0
            avg_outside_distance = 0.0

        # 计算相对误差（相对于OBB尺寸）
        obb_scale = np.mean(obb_extents)
        relative_max_error = max_outside_distance / obb_scale if obb_scale > 0 else 0

        # 判断是否有效（覆盖率高且超出距离相对较小）
        is_valid = (coverage_ratio >= 0.95) and (relative_max_error <= tolerance)

        result = {
            "is_valid": is_valid,
            "coverage_ratio": coverage_ratio,
            "inside_count": inside_count,
            "total_count": total_count,
            "max_outside_distance": max_outside_distance,
            "avg_outside_distance": avg_outside_distance,
            "relative_max_error": relative_max_error,
            "obb_scale": obb_scale,
        }

        if verbose:
            print(f"    Coverage ratio: {coverage_ratio:.3f}")
            print(f"    Max outside distance: {max_outside_distance:.4f}")
            print(f"    Relative error: {relative_max_error:.3f}")
            print(f"    Valid: {is_valid}")

        return result

    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Validation error: {str(e)}",
            "coverage_ratio": 0.0,
        }


def validate_all_obbs(urdf_path, link_obbs, verbose=False):
    """
    验证所有OBB的正确性

    Args:
        urdf_path: URDF文件路径
        link_obbs: OBB数据列表
        verbose: 是否输出详细信息

    Returns:
        dict: 验证结果统计
    """
    print("\n" + "=" * 60)
    print("开始验证OBB正确性...")
    print("注意：此验证不会人为对齐坐标系，可以检测出坐标错误")
    print("=" * 60)

    validation_results = {}
    valid_count = 0
    total_tested = 0

    for obb_data in link_obbs:
        link_name = obb_data["link_name"]

        # 跳过虚拟连杆（体积为零的OBB）
        if obb_data["volume"] < 1e-9:
            if verbose:
                print(f"\n验证连杆 '{link_name}': 跳过（虚拟连杆）")
            continue

        print(f"\n验证连杆 '{link_name}':")

        # 加载连杆的碰撞网格
        mesh = load_link_collision_mesh(urdf_path, link_name, verbose)

        if mesh is None:
            print("  ❌ 无法加载网格")
            validation_results[link_name] = {
                "is_valid": False,
                "error": "Cannot load mesh",
            }
            continue

        # 输出诊断信息
        mesh_bounds = mesh.bounds
        mesh_center = np.mean(mesh_bounds, axis=0)
        mesh_size = mesh_bounds[1] - mesh_bounds[0]

        print("  网格信息:")
        print(
            f"    中心位置: [{mesh_center[0]:.3f}, {mesh_center[1]:.3f}, {mesh_center[2]:.3f}]"
        )
        print(f"    尺寸: [{mesh_size[0]:.3f}, {mesh_size[1]:.3f}, {mesh_size[2]:.3f}]")

        print("  OBB信息:")
        print(
            f"    中心位置: [{obb_data['position'][0]:.3f}, {obb_data['position'][1]:.3f}, {obb_data['position'][2]:.3f}]"
        )
        print(
            f"    尺寸: [{obb_data['extents'][0]:.3f}, {obb_data['extents'][1]:.3f}, {obb_data['extents'][2]:.3f}]"
        )

        # 计算中心偏移
        center_offset = np.linalg.norm(np.array(obb_data["position"]) - mesh_center)
        print(f"    中心偏移: {center_offset:.3f}")

        # 验证OBB覆盖
        result = validate_obb_coverage(obb_data, mesh, verbose=verbose)
        validation_results[link_name] = result
        total_tested += 1

        if result["is_valid"]:
            valid_count += 1
            print(
                f"  ✅ OBB有效 (覆盖率: {result['coverage_ratio']:.1%}, 相对误差: {result.get('relative_max_error', 0):.3f})"
            )
        else:
            if "error" in result:
                print(f"  ❌ 验证失败: {result['error']}")
            else:
                print(
                    f"  ❌ OBB无效 (覆盖率: {result['coverage_ratio']:.1%}, 相对误差: {result.get('relative_max_error', 0):.3f})"
                )

                # 提供诊断建议
                if result["coverage_ratio"] < 0.8:
                    print("    💡 建议: OBB可能位置错误或尺寸过小")
                if center_offset > np.mean(mesh_size) * 0.1:
                    print("    💡 建议: OBB中心与网格中心偏移较大，可能存在坐标系问题")

    # 输出统计结果
    print("\n" + "=" * 60)
    print("验证结果统计:")
    print("=" * 60)
    print(f"总计测试连杆: {total_tested}")
    print(f"有效OBB: {valid_count}")
    print(f"无效OBB: {total_tested - valid_count}")
    if total_tested > 0:
        print(f"成功率: {valid_count / total_tested:.1%}")

    # 详细的失败案例报告
    failed_links = [
        name
        for name, result in validation_results.items()
        if not result["is_valid"] and "error" not in result
    ]

    if failed_links:
        print("\n失败的连杆详情:")
        for link_name in failed_links:
            result = validation_results[link_name]
            print(
                f"  - {link_name}: 覆盖率 {result['coverage_ratio']:.1%}, "
                f"相对误差 {result.get('relative_max_error', 0):.3f}"
            )

    return {
        "total_tested": total_tested,
        "valid_count": valid_count,
        "success_rate": valid_count / total_tested if total_tested > 0 else 0,
        "results": validation_results,
    }


def visualize_urdf_and_obbs(urdf_path, enable_validation=True):
    """
    主函数，用于设置环境、加载模型、计算并可视化 OBB。
    """
    # 1. 设置 PyBullet 环境
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=30,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5],
    )

    # 2. 加载机器人模型
    if not os.path.exists(urdf_path):
        print(f"错误: URDF 文件未找到于 '{urdf_path}'")
        p.disconnect()
        return

    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.3], useFixedBase=True)
        print(f"成功加载机器人: {urdf_path}")
    except p.error as e:
        print(f"加载 URDF 失败: {e}")
        p.disconnect()
        return

    # 3. 计算 OBB
    print("正在计算 OBBs...")
    deps_ok, missing = obb_calculator.check_dependencies()
    if not deps_ok:
        print(f"错误: 缺少 OBB 计算器依赖项: {missing}")
        p.disconnect()
        return

    link_obbs = obb_calculator.calculate_link_obbs(
        urdf_path, verbose=False
    )  # 降低详细度
    if not link_obbs:
        print("错误: OBB 计算失败。")
        p.disconnect()
        return
    print(f"成功计算 {len(link_obbs)} 个 OBBs。")

    # 4. 验证OBB正确性（如果启用）
    validation_results = None
    if enable_validation:
        validation_results = validate_all_obbs(urdf_path, link_obbs, verbose=False)

    # 5. 建立 link 名称到 OBB 数据和 PyBullet link 索引的映射
    obb_map = {obb["link_name"]: obb for obb in link_obbs}
    link_name_to_index = {
        p.getJointInfo(robot_id, i)[12].decode("utf-8"): i
        for i in range(p.getNumJoints(robot_id))
    }
    base_name = p.getBodyInfo(robot_id)[0].decode("utf-8")
    link_name_to_index[base_name] = -1  # 基座的 link 索引为 -1

    # 6. 为 OBB 创建可视化实体（根据验证结果着色）
    obb_visual_ids = {}
    for link_name, obb_data in obb_map.items():
        # OBB 的 extents 是全尺寸，PyBullet 需要半尺寸 (halfExtents)
        half_extents = [e / 2.0 for e in obb_data["extents"]]

        # 跳过体积为零的 OBB (通常是虚拟连杆)
        if np.prod(half_extents) < 1e-9:
            continue

        # 根据验证结果选择颜色
        if validation_results and link_name in validation_results["results"]:
            result = validation_results["results"][link_name]
            if result["is_valid"]:
                color = [0, 1, 0, 0.5]  # 绿色 - 有效
            else:
                color = [1, 0, 0, 0.5]  # 红色 - 无效
        else:
            color = [0, 0, 1, 0.5]  # 蓝色 - 未验证

        # 创建 OBB 的可视化形状
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
        )

        # 创建一个多体对象来承载这个可视化形状，稍后我们将移动它
        body_id = p.createMultiBody(
            baseVisualShapeIndex=visual_shape_id, basePosition=[0, 0, 0]
        )
        obb_visual_ids[link_name] = body_id

    print("\n可视化设置完成:")
    print("  🟢 绿色 = 有效的OBB")
    print("  🔴 红色 = 无效的OBB")
    print("  🔵 蓝色 = 未验证的OBB")
    print("在窗口激活时按 Ctrl+C 退出。")

    # 7. 主循环，用于更新 OBB 的位姿以匹配连杆
    try:
        while p.isConnected():
            for link_name, link_index in link_name_to_index.items():
                if link_name not in obb_visual_ids:
                    continue

                # 获取连杆在世界坐标系中的位姿
                if link_index == -1:  # 基座
                    link_pos, link_orn = p.getBasePositionAndOrientation(robot_id)
                else:  # 其他连杆
                    link_state = p.getLinkState(robot_id, link_index)
                    link_pos, link_orn = link_state[0], link_state[1]

                # 获取 OBB 相对于其连杆的局部变换
                obb_data = obb_map[link_name]
                obb_local_pos = obb_data["position"]
                obb_local_orn = rotation_matrix_to_quaternion(
                    obb_data["rotation_matrix"]
                )

                # 组合变换: T_world_obb = T_world_link * T_link_obb
                world_obb_pos, world_obb_orn = p.multiplyTransforms(
                    link_pos, link_orn, obb_local_pos, obb_local_orn
                )

                # 更新 OBB 可视化实体的位姿
                p.resetBasePositionAndOrientation(
                    obb_visual_ids[link_name], world_obb_pos, world_obb_orn
                )

            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        print("\n用户中断，退出可视化。")
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 PyBullet 可视化 URDF 及其计算出的 OBB。"
    )
    parser.add_argument("urdf_file", type=str, help="要测试的 URDF 文件的路径。")
    parser.add_argument(
        "--no-validation", action="store_true", help="禁用OBB正确性验证"
    )
    args = parser.parse_args()

    enable_validation = not args.no_validation
    visualize_urdf_and_obbs(os.path.abspath(args.urdf_file), enable_validation)
