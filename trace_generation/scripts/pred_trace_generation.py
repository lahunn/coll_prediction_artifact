"""Utility helpers for sampling prediction-trace collision datasets.

The implementation relies on the shared environments in ``core.robot`` and
``core.collision`` so downstream tools can consume a consistent data format.
"""

# python pred_trace_generation.py franka 100 ../trace_files/scene_benchmarks/dens3 1 --seed 0
from __future__ import annotations

import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Add parent directory to path to allow imports

from trace_generation.core.robot.modular_env import ModularEnv
from trace_generation.core.collision.sphere_method import SphereEnv

CollisionArrays = Tuple[
    np.ndarray,
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]

def _sample_uniform(robot_env) -> np.ndarray:
    """Sample a joint configuration uniformly within the joint limits."""

    return np.random.uniform(robot_env.lower_bounds, robot_env.upper_bounds)

def _format_orientation(quaternion: Sequence[float]) -> str:
    """Encode orientation quaternions as compact strings."""

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
) -> CollisionArrays:
    """Generate collision-labelled samples for the given robot.

    Args:
        robot_name: Identifier understood by ``RobotEnv`` / ``SphereEnv``.
        numqueries: Number of configurations to sample.
        include_sphere_data: Whether to gather sphere-level annotations.
        obb_gui: Open the PyBullet GUI for the robot/OBB environment.
        sphere_gui: Open the PyBullet GUI for the sphere approximation.
        obstacle_file: Optional path to a pickled obstacle description produced by
            ``scene_generator.py``.
        enable_self_collision: Whether to enable self-collision detection (default: False).

    Returns:
        ``(qarr, dirarr, yarr, qarr_pose, yarr_pose, qarr_sphere,
        rarr_sphere, yarr_sphere)``. The three sphere arrays are ``None`` when
        ``include_sphere_data`` is ``False`` or when the robot exposes no sphere
        approximation.
    """

    modular_env = ModularEnv(
        robot_name, GUI=obb_gui, enable_self_collision=enable_self_collision
    )
    robot_env = modular_env.robot_env
    collision_env = modular_env.collision_env

    valid_links = [idx for idx in robot_env.valid_collision_links if idx != -1]
    num_links = len(valid_links)

    if num_links == 0:
        modular_env.close()
        raise RuntimeError(
            f"Robot '{robot_name}' exposes no valid collision links; "
            "cannot build link-level data."
        )

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

    obstacles = None
    if obstacle_file is not None:
        if not os.path.exists(obstacle_file):
            raise FileNotFoundError(f"Obstacle file not found: {obstacle_file}")
        with open(obstacle_file, "rb") as pf:
            obstacles = pickle.load(pf)

        modular_env.obstacle_manager.load_obstacles(obstacles)
        collision_env.load_obstacle_body_ids( # type: ignore
            modular_env.obstacle_manager.obstacle_body_ids
        )

    if need_sphere_env:
        sphere_env = SphereEnv(
            robot_env=robot_env,
            robot_name=robot_name,
            SPH_GUI=sphere_gui,
        )

        # 如果有障碍物，也加载到 sphere_env 中
        if obstacles is not None:
            sphere_env.load_obstacles(obstacles)

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

    link_offset = 0
    sphere_offset = 0
    sample_count = 0

    while sample_count < numqueries:
        state = _sample_uniform(robot_env)

        if not robot_env._valid_state(state):  # type: ignore[attr-defined]
            continue

        is_free, link_coords, link_colls = collision_env._point_in_free_space(state)
        if len(link_coords) != num_links:
            continue

        qarr_pose[sample_count] = state
        yarr_pose[sample_count] = 1 if is_free else 0

        for pose, coll_value in zip(link_coords, link_colls):
            position = pose[:3]
            orientation = pose[3:]
            qarr[link_offset] = position
            yarr[link_offset] = coll_value
            dirarr.append(_format_orientation(orientation))
            link_offset += 1

        sphere_coords = None
        sphere_colls = None
        if sphere_env is not None:
            _, sphere_coords, sphere_colls = sphere_env.get_sphere_collision_data(
                state.tolist()
            )

        if (
            collect_sphere_data
            and sphere_coords is not None
            and sphere_colls is not None
        ):
            if len(sphere_coords) != num_spheres:
                link_offset -= num_links
                del dirarr[-num_links:]
                continue

            for coord, coll_value in zip(sphere_coords, sphere_colls):
                qarr_sphere[sphere_offset] = coord[:3]  # type: ignore[index]
                rarr_sphere[sphere_offset] = coord[3]  # type: ignore[index]
                yarr_sphere[sphere_offset] = coll_value  # type: ignore[index]
                sphere_offset += 1

        sample_count += 1

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
    """Persist generated arrays to the legacy ``*_coord.pkl`` format."""

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
    """CLI entry point using the simplified sampling helpers."""

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
