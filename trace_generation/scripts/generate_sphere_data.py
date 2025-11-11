#!/usr/bin/env python3
"""
对比ModularEnv和球体碰撞检测结果

该程序读取obstacle_config_file，
使用ModularEnv和球体模型分别计算碰撞数据，并进行对比。
"""

import pickle
import argparse

# 添加项目路径

from trace_generation.core.robot.environment import RobotEnv


def compare_collision(
    obstacle_config_file,
    collision_data_file,
    robot_name="franka",
    output_file=None,
    benchmark_id=None,
    enable_self_collision=False,
    detector_type="pybullet",
    return_cycles=False,
):
    """
    对比OBB和球体碰撞检测结果

    Args:
        obstacle_config_file: 障碍物-配置文件路径
        collision_data_file: OBB碰撞数据文件路径
        robot_name: 机器人名称
        output_file: 球体碰撞数据输出文件路径（可选）
        benchmark_id: 基准测试ID
        enable_self_collision: 是否启用自碰撞检测
        detector_type: 球体碰撞检测器类型 ("pybullet" 或 "geometric")
        return_cycles: 是否记录周期数（仅geometric支持）
    """
    print(f"加载obstacle_config_file: {obstacle_config_file}")
    with open(obstacle_config_file, "rb") as f:
        obstacle_data = pickle.load(f)
    obstacles = obstacle_data["obstacles"]
    configs = obstacle_data["configs"]

    print(f"加载collision_data_file: {collision_data_file}")
    with open(collision_data_file, "rb") as f:
        obb_data, obb_link_coll_data = pickle.load(f)

    print(f"障碍物数量: {len(obstacles)}")
    print(f"边数量: {len(configs)}")
    print(f"自碰撞检测: {'启用' if enable_self_collision else '禁用'}")
    print(f"检测器类型: {detector_type}")
    if detector_type == "geometric" and return_cycles:
        print("周期计数: 启用")

    # 创建机器人环境
    robot_env = RobotEnv(
        robot_name, OBB_GUI=False, enable_self_collision=enable_self_collision
    )

    # 根据detector_type创建相应的球体环境
    if detector_type == "geometric":
        from trace_generation.core.collision.sphere_detector import SphereEnvGeometric

        sphere_env = SphereEnvGeometric(
            robot_env=robot_env, robot_name=robot_name, return_cycles=return_cycles
        )
    else:  # pybullet
        from trace_generation.core.collision.sphere_method import SphereEnv

        sphere_env = SphereEnv(
            robot_env=robot_env, robot_name=robot_name, SPH_GUI=False
        )

    # 加载障碍物
    sphere_env.load_obstacles(obstacles)

    inconsistent_count = 0
    sphere_collision_obb_free_count = 0  # sphere碰撞，obb无碰撞
    sphere_free_obb_collision_count = 0  # sphere无碰撞，obb碰撞

    if len(configs) > len(obb_link_coll_data):
        print("警告: OBB数据中缺少部分edge")

    # 处理每个edge
    for i, edge_configs in enumerate(configs):
        obb_edge = obb_link_coll_data[i]
        if not obb_edge:
            continue

        edge_sphere_coords = []
        edge_sphere_colls = []
        edge_sphere_cycles = []  # 新增：存储周期数据

        if len(edge_configs) > len(obb_edge):
            print(f"警告: OBB数据中edge {i}缺少pose")
            continue

        # 初始化edge层面的碰撞标志
        obb_edge_collision = any(
            any(coll == 0 for coll in pose_colls) for pose_colls in obb_edge
        )
        sphere_edge_collision = False

        # 处理edge中的每个pose
        for j, config in enumerate(edge_configs):
            # 获取球体碰撞数据
            if detector_type == "geometric" and return_cycles:
                collision, coords, colls, cycles = sphere_env.get_sphere_collision_data(  # pyright: ignore[reportAssignmentType]
                    config
                )  # pyright: ignore[reportAssignmentType]
                edge_sphere_cycles.append(cycles)
            else:
                collision, coords, colls = sphere_env.get_sphere_collision_data(config)  # pyright: ignore[reportAssignmentType]

            edge_sphere_coords.append(coords)
            edge_sphere_colls.append(colls)

            # 更新sphere edge碰撞标志
            if any(coll == 0 for coll in colls):
                sphere_edge_collision = True

        # 在edge层面检查一致性
        if obb_edge_collision != sphere_edge_collision:
            print(
                f"Edge {i}: OBB={obb_edge_collision}, Sphere={sphere_edge_collision} - 不一致!"
            )
            inconsistent_count += 1
            if not obb_edge_collision and sphere_edge_collision:
                sphere_collision_obb_free_count += 1
            elif obb_edge_collision and not sphere_edge_collision:
                sphere_free_obb_collision_count += 1

        # 存储球体数据（无论是否一致）
        if edge_sphere_coords:
            if detector_type == "geometric":
                if return_cycles and edge_sphere_cycles:
                    sphere_env.store_sphere_data(
                        edge_sphere_coords,
                        edge_sphere_colls,
                        cycles=edge_sphere_cycles,  # type: ignore
                        is_edge=True,
                    )
                else:
                    sphere_env.store_sphere_data(
                        edge_sphere_coords, edge_sphere_colls, is_edge=True
                    )
            else:  # pybullet模式不支持cycles参数
                sphere_env.store_sphere_data(
                    edge_sphere_coords, edge_sphere_colls, is_edge=True
                )

    # 清理资源
    sphere_env.cleanup_obstacles()
    sphere_env.close()
    robot_env.close()

    print(f"对比完成，发现 {inconsistent_count} 个不一致的配置")
    print(f"  Sphere碰撞但OBB无碰撞: {sphere_collision_obb_free_count}")
    print(f"  Sphere无碰撞但OBB碰撞: {sphere_free_obb_collision_count}")

    # 保存球体碰撞数据（如果指定了输出文件）
    if output_file:
        sphere_env.save_collision_data(output_file)


def main():
    parser = argparse.ArgumentParser(description="对比OBB和球体碰撞检测结果")

    parser.add_argument(
        "--obstacle-config-file", type=str, required=True, help="障碍物-配置文件路径"
    )
    parser.add_argument(
        "--collision-data-file", type=str, required=True, help="OBB碰撞数据文件路径"
    )
    parser.add_argument("--robot-name", type=str, default="franka", help="机器人名称")
    parser.add_argument("--benchmark-id", type=int, help="基准测试ID，用于命名输出文件")
    parser.add_argument("--output-file", type=str, help="球体碰撞数据输出文件路径")
    parser.add_argument(
        "--enable-self-collision",
        action="store_true",
        help="启用自碰撞检测",
    )
    parser.add_argument(
        "--detector-type",
        type=str,
        default="pybullet",
        choices=["pybullet", "geometric"],
        help="球体碰撞检测器类型: pybullet(默认) 或 geometric",
    )
    parser.add_argument(
        "--return-cycles",
        action="store_true",
        help="是否记录周期数(仅geometric支持)",
    )

    args = parser.parse_args()

    compare_collision(
        args.obstacle_config_file,
        args.collision_data_file,
        args.robot_name,
        args.output_file,
        args.benchmark_id,
        args.enable_self_collision,
        args.detector_type,
        args.return_cycles,
    )

    return 0


if __name__ == "__main__":
    exit(main())
