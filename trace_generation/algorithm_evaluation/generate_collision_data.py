#!/usr/bin/env python3
"""
生成碰撞数据脚本

功能:
1) 读入指定 problems 文件 (pickle, 列表: (obstacles, start, goal, path))
2) 遍历指定范围内的问题, 使用指定 planner (bit_star 或 gnnmp) 分别在
   link 与 sphere 碰撞模型环境中进行规划
3) 每次规划后保存:
   - 障碍物-配置对 到 ../../trace_files/bit_traces
   - link 碰撞检测数据 到 ../../trace_files/scene_benchmarks/bit_collision_data
   - sphere 碰撞检测数据 到 ../../trace_files/scene_benchmarks/bit_collision_data

备注:
- 命名规则与 generate_problem_dataset.py 保持一致
- 代码尽量简单, 模块化拆分
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Sequence, Tuple, Optional

import numpy as np

from trace_generation.core.robot.modular_env import ModularEnv
from trace_generation.bit_planning.algorithm.bit_star import BITStar
from trace_generation.bit_planning.algorithm.gnnmp import GNNPlanner


def ensure_dirs(pair_dir: str, collision_dir: str, level: Optional[str] = None, planner: Optional[str] = None) -> None:
    if level:
        pair_dir = os.path.join(pair_dir, level)
        collision_dir = os.path.join(collision_dir, level)
    if planner:
        pair_dir = os.path.join(pair_dir, planner)
        collision_dir = os.path.join(collision_dir, planner)
    os.makedirs(pair_dir, exist_ok=True)
    os.makedirs(collision_dir, exist_ok=True)


def load_problems(
    path: str,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def reconstruct_path(
    edges: dict, start: np.ndarray, goal: np.ndarray
) -> Optional[List[np.ndarray]]:
    from collections import deque

    def to_tuple(state: np.ndarray) -> Tuple[float, ...]:
        return tuple(state.flatten())

    path = deque([goal])
    current = to_tuple(goal)
    start_tuple = to_tuple(start)

    for _ in range(10000):
        if current not in edges:
            return None
        parent = edges[current]
        if parent == start_tuple:
            path.appendleft(np.array(parent))
            break
        path.appendleft(np.array(parent))
        current = parent
    else:
        return None

    return list(path)


def build_env(
    robot_name: str, collision_model: str, visualize: bool, enable_self_collision: bool
) -> ModularEnv:
    return ModularEnv(
        robot_name,
        map_file=None,
        GUI=visualize,
        collision_model_type=collision_model,
        enable_self_collision=enable_self_collision,
    )


def run_bit_star(env: ModularEnv, max_time: float) -> float:
    planner = BITStar(env)
    _, edges, _, cost, _, _ = planner.plan(
        pathLengthLimit=float("inf"), time_budget=max_time
    )
    return float(cost)


def run_gnnmp(env: ModularEnv, model_key: str, t_max: int = 1000) -> float:
    if GNNPlanner is None:
        raise RuntimeError("GNNPlanner 未可用，请确保依赖安装或改用 --planner bit_star")
    planner = GNNPlanner(environment=env, model_key=model_key)
    result = planner.plan(t_max=t_max)
    cost = float(result.get("path_cost", float("inf")))
    return cost


def save_pair(pair_dir: str, base: str, index: int, obstacles, configs, level: Optional[str] = None, planner: Optional[str] = None) -> None:
    if level:
        pair_dir = os.path.join(pair_dir, level)
    if planner:
        pair_dir = os.path.join(pair_dir, planner)
    pair_filename = f"{base}_{index:04d}.pkl"
    pair_filepath = os.path.join(pair_dir, pair_filename)
    payload = {"obstacles": obstacles, "configs": list(configs)}
    with open(pair_filepath, "wb") as f:
        pickle.dump(payload, f)
    print(f"  保存障碍物-配置对到目录: {pair_filepath}")

def save_collision(
    env: ModularEnv, collision_dir: str, base: str, index: int, suffix: str, level: Optional[str] = None, planner: Optional[str] = None
) -> None:
    if level:
        collision_dir = os.path.join(collision_dir, level)
    if planner:
        collision_dir = os.path.join(collision_dir, planner)
    coll_filename = f"{base}_{index:04d}_{suffix}.pkl"
    coll_filepath = os.path.join(collision_dir, coll_filename)
    env.collision_env.data_manager.save_collision_data(coll_filepath)
    print(f"  保存碰撞数据到目录: {coll_filepath}")


def process_one_problem(
    idx1: int,
    problem: Tuple[np.ndarray, np.ndarray, np.ndarray, Sequence],
    robot_name: str,
    planner: str,
    pair_dir: str,
    collision_dir: str,
    visualize: bool,
    enable_self_collision: bool,
    max_time: float,
    gnn_model_key: Optional[str],
    level: Optional[str] = None,
    planner_subdir: Optional[str] = None,
) -> None:
    obstacles, start, goal, _ = problem

    # link 环境s
    env_link = build_env(robot_name, "link", visualize, enable_self_collision)
    env_link.collision_env.config_list = []
    env_link.collision_env.data_manager.reset()
    env_link.load_obstacles(obstacles)
    env_link.init_state = np.array(start)
    env_link.goal_state = np.array(goal)

    # sphere 环境
    env_sphere = build_env(robot_name, "sphere", False, enable_self_collision)
    env_sphere.collision_env.config_list = []
    env_sphere.collision_env.data_manager.reset()
    env_sphere.load_obstacles(obstacles)
    env_sphere.init_state = np.array(start)
    env_sphere.goal_state = np.array(goal)

    # 命名基底
    base = f"{robot_name}_{env_link.config_dim}"

    # 规划: link
    if planner == "bit_star":
        _link_cost = run_bit_star(env_link, max_time)
    elif planner == "gnnmp":
        if not gnn_model_key:
            raise ValueError("使用 gnnmp 需要提供 --model-key")
        _link_cost = run_gnnmp(env_link, gnn_model_key)
    else:
        raise ValueError("planner 仅支持 bit_star 或 gnnmp")

    # 保存障碍物-配置对（以 link 环境记录的 configs 为准）
    save_pair(pair_dir, base, idx1, obstacles, env_link.collision_env.config_list, level, planner_subdir)

    # 保存 link 碰撞数据
    save_collision(env_link, collision_dir, base, idx1, "link", level, planner_subdir)

    # 规划: sphere
    if planner == "bit_star":
        _sphere_cost = run_bit_star(env_sphere, max_time)
    else:
        _sphere_cost = run_gnnmp(env_sphere, gnn_model_key or "")

    # 保存 sphere 碰撞数据
    save_collision(env_sphere, collision_dir, base, idx1, "sphere", level, planner_subdir)

    # 关闭环境
    try:
        env_link.close()
    except Exception:
        pass
    try:
        env_sphere.close()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="根据 problems 生成碰撞数据")
    p.add_argument("--problems-file", required=True, type=str, help="problems.pkl 路径")
    p.add_argument(
        "--robot-name", required=True, type=str, help="机器人名 (如 kuka_iiwa)"
    )
    p.add_argument("--planner", choices=["bit_star", "gnnmp"], default="bit_star")
    p.add_argument(
        "--model-key", type=str, default=None, help="GNN 模型键 (仅 gnnmp 需要)"
    )
    p.add_argument(
        "--start-index", type=int, default=1, help="起始 problem 索引(1-based, 包含)"
    )
    p.add_argument(
        "--end-index", type=int, default=None, help="结束 problem 索引(1-based, 包含)"
    )
    p.add_argument(
        "--level", type=str, default=None, help="难度级别子目录 (如 G1, G2 等)"
    )
    p.add_argument("--pair-dir", type=str, default="../../trace_files/bit_traces")
    p.add_argument(
        "--collision-dir",
        type=str,
        default="../../trace_files/scene_benchmarks/bit_collision_data",
    )
    p.add_argument("--max-time", type=float, default=60.0, help="BIT* 最大规划时间(秒)")
    p.add_argument("--t-max", type=int, default=1000, help="GNN 探索最大采样数 (gnnmp)")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--enable-self-collision", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ensure_dirs(args.pair_dir, args.collision_dir, args.level, args.planner)

    problems = load_problems(args.problems_file)
    total = len(problems)

    start_idx = max(1, int(args.start_index))
    end_idx = int(args.end_index) if args.end_index is not None else total
    end_idx = min(end_idx, total)

    # 预先检查 GNN 依赖
    if args.planner == "gnnmp" and GNNPlanner is None:
        raise RuntimeError("未找到 GNNPlanner，请检查依赖或改用 --planner bit_star")

    # 遍历并处理
    for i in range(start_idx, end_idx + 1):
        problem = problems[i - 1]
        process_one_problem(
            idx1=i,
            problem=problem,
            robot_name=args.robot_name,
            planner=args.planner,
            pair_dir=args.pair_dir,
            collision_dir=args.collision_dir,
            visualize=args.visualize,
            enable_self_collision=args.enable_self_collision,
            max_time=args.max_time,
            gnn_model_key=args.model_key,
            level=args.level,
            planner_subdir=args.planner,
        )
        print(f"✓ 完成 problem {i}/{total}")

    print("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
