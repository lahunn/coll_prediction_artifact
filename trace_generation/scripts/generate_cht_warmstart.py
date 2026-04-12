#!/usr/bin/env python3
"""Generate per-problem warm-start packages for the collision history table.

The script reads a ``problems.pkl`` file produced by the dataset pipeline,
samples random spheres inside each problem workspace, labels each sample with
geometric sphere-vs-AABB collision checks, and accumulates the result into the
same hash-keyed memory format used by the runtime CHT.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trace_generation.core.collision.geometric_collision_detection import AABB, Sphere, sphere_aabb
from trace_generation.core.robot.sphere_analyzer import RobotSphereAnalyzer
from motion_planning_prediction.simulation_core.collision_prediction import initialize_cht, update_collision_dict
from motion_planning_prediction.simulation_core.hash_utils import calculate_bins_from_workspace, compute_hash_keyy


def load_problems(problems_path: str) -> List[Tuple]:
    with open(problems_path, "rb") as f:
        problems = pickle.load(f)
    if not isinstance(problems, list):
        raise TypeError(f"Expected problems.pkl to contain a list, got {type(problems)!r}")
    return problems


def obstacles_to_aabbs(obstacles: Sequence[Tuple]) -> List[AABB]:
    aabbs: List[AABB] = []
    for half_extents, base_position in obstacles:
        hx, hy, hz = half_extents
        cx, cy, cz = base_position
        aabbs.append(
            AABB(
                min_x=float(cx - hx),
                min_y=float(cy - hy),
                min_z=float(cz - hz),
                max_x=float(cx + hx),
                max_y=float(cy + hy),
                max_z=float(cz + hz),
            )
        )
    return aabbs


def sample_sphere_centers(num_samples: int, bins: Sequence[np.ndarray]) -> np.ndarray:
    x_min, x_max = float(bins[0][0]), float(bins[0][-1])
    y_min, y_max = float(bins[1][0]), float(bins[1][-1])
    z_min, z_max = float(bins[2][0]), float(bins[2][-1])
    xs = np.random.uniform(x_min, x_max, size=num_samples)
    ys = np.random.uniform(y_min, y_max, size=num_samples)
    zs = np.random.uniform(z_min, z_max, size=num_samples)
    return np.stack([xs, ys, zs], axis=1)


def get_robot_radius_pool(robot_name: str) -> np.ndarray:
    analyzer = RobotSphereAnalyzer(robot_name, device="cuda:0")
    world_spheres = analyzer.get_world_spheres()
    if world_spheres.size == 0:
        raise RuntimeError(f"No robot spheres found for {robot_name}")
    radii = world_spheres[:, 3]
    return np.asarray(radii, dtype=np.float64)


def generate_warmstart_memory(
    obstacles: Sequence[Tuple],
    bins: Sequence[np.ndarray],
    radii_pool: np.ndarray,
    samples_per_problem: int,
) -> Tuple[dict, dict]:
    memory = initialize_cht()
    stats = {
        "samples": 0,
        "collision_samples": 0,
        "safe_samples": 0,
        "unique_keys": 0,
    }

    aabbs = obstacles_to_aabbs(obstacles)
    centers = sample_sphere_centers(samples_per_problem, bins)
    sampled_radii = np.random.choice(radii_pool, size=samples_per_problem, replace=True)

    for center, radius in zip(centers, sampled_radii):
        sphere = Sphere(float(center[0]), float(center[1]), float(center[2]), float(radius))
        is_collision = any(sphere_aabb(sphere, aabb)[0] == 0 for aabb in aabbs)
        hash_key = compute_hash_keyy([sphere.x, sphere.y, sphere.z, sphere.r], bins)
        is_free = 0 if is_collision else 1
        memory = update_collision_dict(memory, hash_key, is_free, sample_rate=1.0)

        stats["samples"] += 1
        if is_collision:
            stats["collision_samples"] += 1
        else:
            stats["safe_samples"] += 1

    stats["unique_keys"] = len(memory)
    return memory, stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-problem warm-start packages for CHT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--problems-pkl", required=True, help="path to problems.pkl")
    parser.add_argument("--output-dir", required=True, help="directory for warm-start pkls")
    parser.add_argument("--basename", required=True, help="benchmark basename, e.g. iiwa_7")
    parser.add_argument("--samples-per-problem", type=int, default=1000, help="random sphere samples per problem")
    parser.add_argument("--quant-bits", type=int, default=4, help="quantization bits used by runtime hash")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing output files")

    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    robot_name = args.basename.split("_")[0]
    bins = calculate_bins_from_workspace(robot_name, args.quant_bits)
    radii_pool = get_robot_radius_pool(robot_name)
    problems = load_problems(args.problems_pkl)

    print(f"Loaded {len(problems)} problems from {args.problems_pkl}")
    print(f"Using robot {robot_name} with {len(radii_pool)} sphere radii")

    for problem_idx, problem in enumerate(problems, start=1):
        if len(problem) != 4:
            raise ValueError(f"Problem #{problem_idx} must be a 4-tuple, got length {len(problem)}")

        obstacles, start, goal, path = problem
        output_name = f"{args.basename}_{problem_idx:04d}_warmstart.pkl"
        output_path = os.path.join(args.output_dir, output_name)

        if os.path.exists(output_path) and not args.overwrite:
            print(f"[{problem_idx:04d}] skip existing {output_name}")
            continue

        memory, stats = generate_warmstart_memory(
            obstacles=obstacles,
            bins=bins,
            radii_pool=radii_pool,
            samples_per_problem=args.samples_per_problem,
        )

        package = {
            "basename": args.basename,
            "robot_name": robot_name,
            "problem_idx": problem_idx,
            "samples_per_problem": args.samples_per_problem,
            "quant_bits": args.quant_bits,
            "memory": memory,
            "stats": stats,
        }

        with open(output_path, "wb") as f:
            pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            f"[{problem_idx:04d}] saved {output_name}: "
            f"samples={stats['samples']}, collision={stats['collision_samples']}, "
            f"safe={stats['safe_samples']}, unique_keys={stats['unique_keys']}"
        )


if __name__ == "__main__":
    main()