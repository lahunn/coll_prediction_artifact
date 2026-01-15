"""用于采样预测轨迹碰撞数据集的实用函数集合。

提供一个小型命令行接口和辅助函数，用于创建用于训练和评估的碰撞标注数据集。
脚本会随机采样机器人配置，查询基于 OBB 的链路级碰撞环境（link-level）以及
可选的球近似（sphere-based approximation），并将结果写入下游工具使用的旧版
pickle 格式。

实现依赖于 ``core.robot`` 和 ``core.collision`` 中的共享环境，以保证下游工具
能够使用一致的数据格式。
"""

# python pred_trace_generation.py franka 100 ../trace_files/scene_benchmarks/dens3 1 --seed 0
from __future__ import annotations

import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np

# 将父目录加入路径以便导入（保留原有导入行为）

from trace_generation.core.robot.modular_env import ModularEnv
from trace_generation.core.collision.sphere_method import SphereEnv

# 类型别名，描述采样函数返回的元组格式：
# (链路坐标, 方向字符串列表, 链路标签, 姿态配置数组, 姿态标签,
#  球坐标?, 球半径?, 球标签?)
CollisionArrays = Tuple[
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]


def _sample_uniform(robot_env) -> np.ndarray:
    """在关节限制范围内均匀采样一个关节配置。"""

    return np.random.uniform(robot_env.lower_bounds, robot_env.upper_bounds)


def _format_orientation(quaternion: Sequence[float]) -> str:
    """将四元数方向编码为紧凑的字符串格式。"""

    return ",".join(f"{value:+.3f}" for value in quaternion)


def sample_and_generate_data(
    robot_name: str,
    numqueries: int,
    *,
    include_sphere_data: bool = True,
    obb_gui: bool = False,
    sphere_gui: bool = False,
    obstacle_file: Optional[str] = None,
    enable_self_collision: bool = False,
) -> Tuple[
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """为指定机器人生成带碰撞标签的采样数据。

    参数：
        robot_name: ``RobotEnv`` / ``SphereEnv`` 可识别的机器人标识符。
        numqueries: 要采样的配置数量。
        include_sphere_data: 是否收集球近似的注释数据。
        obb_gui: 是否为 OBB 环境打开 PyBullet GUI。
        sphere_gui: 是否为球近似环境打开 PyBullet GUI。
        obstacle_file: 可选，指向由 ``scene_generator.py`` 生成的障碍物 pickle 文件路径。
        enable_self_collision: 是否启用自碰撞检测（默认 False）。

    返回：
        返回值为元组 ``(qarr, dirarr, yarr, qarr_pose, yarr_pose, qarr_sphere,
        rarr_sphere, yarr_sphere)``，当 ``include_sphere_data`` 为 ``False``
        或机器人没有球近似时，后三个球相关数组将为 ``None``。
    """

    # 创建一个模块化环境，封装 RobotEnv、CollisionEnv 和 ObstacleManager。
    # 该对象统一管理机器人和场景的初始化，并提供下文使用的碰撞查询接口。
    modular_env = ModularEnv(
        robot_name, GUI=obb_gui, enable_self_collision=enable_self_collision
    )
    # 方便使用的快捷引用：具体的机器人环境和碰撞环境
    robot_env = modular_env.robot_env
    collision_env = modular_env.collision_env

    # 确定哪些关节/链路暴露了碰撞几何。如果没有可用的链路，则无法生成链路级数据并终止。
    valid_links = [idx for idx in robot_env.valid_collision_links if idx != -1]
    num_links = len(valid_links)

    if num_links == 0:
        modular_env.close()
        raise RuntimeError(
            f"Robot '{robot_name}' exposes no valid collision links; "
            "cannot build link-level data."
        )

    # 分配输出容器：
    # qarr_pose / yarr_pose：每个样本的完整关节配置及自由/碰撞标记
    # qarr / yarr / dirarr：按链路展平的数组，分别存储链路位置、碰撞标签和方向字符串
    qarr_pose = np.zeros((numqueries, robot_env.config_dim), dtype=np.float32)
    yarr_pose = np.zeros((numqueries, 1), dtype=np.int8)
    qarr = np.zeros((numqueries * num_links, 3), dtype=np.float32)
    yarr = np.zeros((numqueries * num_links, 1), dtype=np.int8)
    dirarr: List[str] = []

    collect_sphere_data = include_sphere_data
    need_sphere_env = collect_sphere_data or sphere_gui
    sphere_env: Optional[SphereEnv] = None
    qarr_sphere: Optional[np.ndarray] = None
    rarr_sphere: Optional[np.ndarray] = None
    yarr_sphere: Optional[np.ndarray] = None
    num_spheres = 0

    # 可选：加载由 scene_generator.py 生成的障碍物描述（pickle），并将其注册到两个管理器中，
    # 以确保 OBB 和球近似的查询都能看到相同的障碍物。
    obstacles = None
    if obstacle_file is not None:
        if not os.path.exists(obstacle_file):
            raise FileNotFoundError(f"Obstacle file not found: {obstacle_file}")
        with open(obstacle_file, "rb") as pf:
            obstacles = pickle.load(pf)

        modular_env.obstacle_manager.load_obstacles(obstacles)
        collision_env.load_obstacles(obstacles)

    if need_sphere_env:
        # 球近似提供每个球体的碰撞信号，为可选功能。仅在需要收集数据或可视化时创建。
        sphere_env = SphereEnv(
            robot_env=robot_env,
            robot_name=robot_name,
            SPH_GUI=sphere_gui,
        )

        # 若上面已加载障碍物，也要将其加载到球环境中
        if obstacles is not None:
            sphere_env.load_obstacles(obstacles)

        # 探测初始状态以确定该机器人模型包含多少球体；若为 0 则禁用球数据收集
        _, initial_coords, _ = sphere_env.get_sphere_collision_data(
            robot_env.init_state
        )
        num_spheres = len(initial_coords)
        if num_spheres == 0 and collect_sphere_data:
            collect_sphere_data = False
        if collect_sphere_data:
            qarr_sphere = np.zeros((numqueries * num_spheres, 3), dtype=np.float32)
            rarr_sphere = np.zeros((numqueries * num_spheres, 1), dtype=np.float32)
            yarr_sphere = np.zeros((numqueries * num_spheres, 1), dtype=np.int8)

    # 循环直到收集到 `numqueries` 个样本。每次采样在关节范围内均匀抽取一个配置，
    # 验证其有效性后会查询 OBB（链路级）以及可选的球近似碰撞结果。
    link_offset = 0
    sphere_offset = 0
    sample_count = 0

    while sample_count < numqueries:
        # 在关节范围内随机抽样一个配置
        state = _sample_uniform(robot_env)

        # 跳过无效配置（例如超出限位）
        if not robot_env._valid_state(state):  # type: ignore[attr-defined]
            continue

        # 执行链路级碰撞查询。返回是否自由的布尔值、每个链路的位姿和每个链路的碰撞标签。
        is_free, link_coords, link_colls = collision_env._point_in_free_space(state)
        # 若返回的链路位姿数量与预期不符（模型不匹配），跳过该样本以保持数组对齐。
        if len(link_coords) != num_links:
            continue

        # 记录完整配置样本及其 OBB 自由/碰撞标签
        qarr_pose[sample_count] = state
        yarr_pose[sample_count] = 1 if is_free else 0

        # 将每个链路的输出展平并写入链路级数组
        for pose, coll_value in zip(link_coords, link_colls):
            position = pose[:3]
            orientation = pose[3:]
            qarr[link_offset] = position
            yarr[link_offset] = coll_value
            dirarr.append(_format_orientation(orientation))
            link_offset += 1

        # 若启用球近似数据收集，则查询球环境
        sphere_coords = None
        sphere_colls = None
        if sphere_env is not None:
            _, sphere_coords, sphere_colls = sphere_env.get_sphere_collision_data(
                state.tolist()
            )

        # 验证并追加每个球体的输出（如需）
        if (
            collect_sphere_data
            and sphere_coords is not None
            and sphere_colls is not None
        ):
            # 若本次样本的球体数量与预期不符，则回滚本次插入的链路数据并跳过该样本，
            # 以保持数据对齐。
            if len(sphere_coords) != num_spheres:
                link_offset -= num_links
                del dirarr[-num_links:]
                continue

            for coord, coll_value in zip(sphere_coords, sphere_colls):
                qarr_sphere[sphere_offset] = coord[:3]  # type: ignore[index]
                rarr_sphere[sphere_offset] = coord[3]  # type: ignore[index]
                yarr_sphere[sphere_offset] = coll_value  # type: ignore[index]
                sphere_offset += 1

        # 本次样本接受，计数器加一
        sample_count += 1

    # 关闭环境（确保 PyBullet 客户端被正确释放）并返回所有数组。返回的元组遵循上面定义的 `CollisionArrays` 类型别名。
    modular_env.close()
    if sphere_env is not None:
        sphere_env.close()

    return (
        qarr,
        dirarr,
        yarr,
        qarr_pose,
        yarr_pose,
        qarr_sphere,
        rarr_sphere,
        yarr_sphere,
    )


def save_results(
    foldername: str,
    filenumber: str,
    qarr: np.ndarray,
    dirarr: List[str],
    yarr: np.ndarray,
    qarr_pose: np.ndarray,
    yarr_pose: np.ndarray,
    qarr_sphere: Optional[np.ndarray] = None,
    rarr_sphere: Optional[np.ndarray] = None,
    yarr_sphere: Optional[np.ndarray] = None,
) -> None:
    """将生成的数组保存为旧版 ``*_coord.pkl`` 格式。

    本函数会在名为 ``<foldername>_rs`` 的文件夹中写入三份 pickle（coord、pose、sphere）。
    仅当存在球相关数组时才会创建 sphere 文件。
    """

    output_folder = f"{foldername}_rs"
    os.makedirs(output_folder, exist_ok=True)

    with open(
        os.path.join(output_folder, f"obstacles_{filenumber}_coord.pkl"), "wb"
    ) as f:
        pickle.dump((qarr, dirarr, yarr), f)

    with open(
        os.path.join(output_folder, f"obstacles_{filenumber}_pose.pkl"), "wb"
    ) as f:
        pickle.dump((qarr_pose, yarr_pose), f)

    if qarr_sphere is not None and rarr_sphere is not None and yarr_sphere is not None:
        with open(
            os.path.join(output_folder, f"obstacles_{filenumber}_sphere.pkl"), "wb"
        ) as f:
            pickle.dump((qarr_sphere, rarr_sphere, yarr_sphere), f)


def main():
    """命令行入口，使用简化的采样辅助函数。

    使用示例：
      python pred_trace_generation.py franka 100 outdir 1 --obstacle-file obs.pkl

    脚本会将结果保存到 ``<foldername>_rs`` 目录下，文件名使用提供的 filenumber 作为后缀。
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate collision samples using RobotEnv/SphereEnv."
    )
    parser.add_argument("robot_name", help="Robot identifier understood by RobotEnv")
    parser.add_argument("numqueries", type=int, help="Number of configurations")
    parser.add_argument(
        "foldername",
        help="Output folder prefix; results saved under '<foldername>_rs'",
    )
    parser.add_argument(
        "filenumber",
        help="Output file suffix, matching legacy pickled dataset naming",
    )
    parser.add_argument(
        "--no-sphere",
        action="store_true",
        help="Skip sphere approximation annotations",
    )
    parser.add_argument(
        "--obb-vis",
        action="store_true",
        help="Open PyBullet GUI for the robot/OBB environment",
    )
    parser.add_argument(
        "--sphere-vis",
        action="store_true",
        help="Open PyBullet GUI for the sphere approximation environment",
    )
    parser.add_argument(
        "--obstacle-file",
        help="Optional path to a pickled obstacle description generated by scene_generator.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional NumPy random seed for reproducibility",
    )
    parser.add_argument(
        "--enable-self-collision",
        action="store_true",
        help="Enable self-collision detection for the robot",
    )

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    (
        qarr,
        dirarr,
        yarr,
        qarr_pose,
        yarr_pose,
        qarr_sphere,
        rarr_sphere,
        yarr_sphere,
    ) = sample_and_generate_data(
        robot_name=args.robot_name,
        numqueries=args.numqueries,
        include_sphere_data=not args.no_sphere,
        obb_gui=args.obb_vis,
        sphere_gui=args.sphere_vis,
        obstacle_file=args.obstacle_file,
        enable_self_collision=args.enable_self_collision,
    )

    save_results(
        foldername=args.foldername,
        filenumber=args.filenumber,
        qarr=qarr,
        dirarr=dirarr,
        yarr=yarr,
        qarr_pose=qarr_pose,
        yarr_pose=yarr_pose,
        qarr_sphere=qarr_sphere,
        rarr_sphere=rarr_sphere,
        yarr_sphere=yarr_sphere,
    )

    obb_free_count = int(yarr_pose.sum())
    obb_colliding_count = args.numqueries - obb_free_count

    print(
        f"Saved {args.numqueries} samples for '{args.robot_name}' into {args.foldername}_rs"
    )
    print(f"  OBB method: free={obb_free_count}, colliding={obb_colliding_count}")

    if yarr_sphere is not None:
        sphere_free_count = int(
            yarr_sphere.reshape(args.numqueries, -1).all(axis=1).sum()
        )
        sphere_colliding_count = args.numqueries - sphere_free_count
        print(
            f"  Sphere method: free={sphere_free_count}, colliding={sphere_colliding_count}"
        )


if __name__ == "__main__":
    main()
