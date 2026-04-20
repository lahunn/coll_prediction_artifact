#!/usr/bin/env python3
"""
运行 GNNMP 在内置的 KUKA-7 问题集上并根据碰撞检测次数重新划分难度级别

流程:
  - 使用 str2name 加载 env 和已训练模型（内置会加载 maze_files/kukas_7_4000.pkl）
  - 对每个问题调用 env.init_new_problem(index)
  - 重置碰撞统计并运行 GNNMP.plan()
  - 收集每个问题的碰撞检测次数 (c_explore + c_smooth)
  - 按 20/40/60/80/100 百分位重新划分为 G1..G5
  - 将每个难度级别的问题列表保存到指定输出目录 (每个 level 的 problems.pkl) 以及保存 metadata.csv

示例:
  python generate_gnn_dataset.py --start 0 --end 4000 --batch 50 --t_max 1000

备注:
  本脚本主要用于离线评估与数据分层，输出可用于训练与基准测试。

"""

import argparse
import os
import pickle
import csv
import numpy as np
import torch
from tqdm import tqdm
import re

from str2name import str2name
from algorithm.gnnmp import GNNMP, path_cost
from algorithm.bit_star import BITStar
from trace_generation.utils.config import set_random_seed


def partition_by_collision_counts(
    records, percentiles=(20, 40, 60, 80, 100), levels=None
):
    """根据碰撞检查次数的分位数将记录划分为难度等级

    简化：使用 numpy.digitize 将值放入由前 4 个分位数定义的 5 个箱中，代码更简洁且易扩展。

    Args:
        records: list of dict, each dict 包含至少字段 'index' 和 'checks'
        percentiles: tuple of percentiles 用于分界（默认包含 100）
        levels: 可选的等级列表，默认为 ['G1'..'G5']

    Returns:
        index_to_level: dict mapping index -> level
        level_counters: dict level -> count
        quantiles: numpy array of quantile值
    """
    if levels is None:
        levels = ["G1", "G2", "G3", "G4", "G5"]

    checks_array = np.array([r["checks"] for r in records])
    # 使用前四个分位点作为 bin 边界 (最后一个 100% 用于上界)
    all_quantiles = np.percentile(checks_array, percentiles)
    bins = all_quantiles[:-1]

    index_to_level = {}
    level_counters = {lvl: 0 for lvl in levels}

    for r in records:
        chk = r["checks"]
        idx = np.digitize(chk, bins, right=True)  # 返回 0..4
        lvl = levels[int(idx)]
        index_to_level[r["index"]] = lvl
        level_counters[lvl] += 1

    return index_to_level, level_counters, all_quantiles


def setup_planner(
    seed, batch, t_max, collision_model_type="link", planner_choice: str = "gnnmp"
):
    """创建 ModularEnv 并根据选择返回对应的 planner（GNNMP 或 BITStar）。

    Args:
        planner_choice: 'gnnmp' or 'bit_star'

    Returns:
        env, planner, device
    """
    # 构造数据集路径并创建 env（支持选择碰撞模型）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "maze_files", "kukas_7_3000.pkl")

    # 延迟导入以避免循环依赖
    from trace_generation.core.robot.modular_env import ModularEnv

    env = ModularEnv(
        robot_name="iiwa", map_file=data_path, collision_model_type=collision_model_type
    )

    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if planner_choice == "gnnmp":
        # 使用 str2name 创建并加载模型，同时 reuse 我们创建的 env（str2name 会尊重传入的 env）
        _, model, _, model_s, _, _ = str2name(
            "kuka7", get_data=True, load=True, env=env
        )

        planner = GNNMP(
            env=env,
            model_explore=model,
            model_smooth=model_s,
            batch=batch,
            t_max=t_max,
            device=device,
        )
        return env, planner, device

    elif planner_choice == "bit_star":
        # BITStar 的构造参数：environment, maxIter=..., plot_flag=False, batch_size=..., T=...
        planner = BITStar(environment=env, batch_size=batch, T=t_max, plot_flag=False)

        # NOTE: BITStar.plan 已被统一为返回 dict，包含字段：
        #       'success', 'path', 'edges', 'collision_checks', 'total_time', 'cost', 'n_samples'
        return env, planner, device

    else:
        raise ValueError(f"Unknown planner_choice: {planner_choice}")


def save_current_problem_data(
    env, idx, collision_model, dry_run=False, trace_dir=None, collision_dir=None
):
    """保存当前问题的障碍物-配置对和碰撞数据到 trace_files 中

    This simplified version assumes `trace_dir` and `collision_dir` are provided
    by the caller (computed in `main`). It does not attempt any fallback.
    """
    obstacle_config_dir = os.path.abspath(trace_dir)
    collision_data_dir = os.path.abspath(collision_dir)
    os.makedirs(obstacle_config_dir, exist_ok=True)
    os.makedirs(collision_data_dir, exist_ok=True)

    base_filename = f"{env.robot_env.robot_name}_{env.config_dim}"

    pair_filename = f"{base_filename}_{idx:04d}.pkl"
    pair_filepath = os.path.join(obstacle_config_dir, pair_filename)
    obstacles = env.problem_manager.get_current_problem()["obstacles"]
    configs = env.collision_env.config_list.copy()
    obstacle_config_pair = {
        "obstacles": obstacles,
        "configs": configs,
    }

    if dry_run:
        print(f"DRY-RUN: would save pair file to {pair_filepath}")
    else:
        try:
            with open(pair_filepath, "wb") as f:
                pickle.dump(obstacle_config_pair, f)
        except Exception as e:
            print(f"Failed to save pair file {pair_filepath}: {e}")

    coll_filename = f"{base_filename}_{idx:04d}_{collision_model}.pkl"
    coll_filepath = os.path.join(collision_data_dir, coll_filename)

    if dry_run:
        print(f"DRY-RUN: would save collision data to {coll_filepath}")
    else:
        try:
            env.collision_env.data_manager.save_collision_data(coll_filepath)
            print("edge counts stored:", len(configs))
        except Exception as e:
            print(f"Failed to save collision data {coll_filepath}: {e}")


def evaluate_problems(
    env,
    planner,
    indexes,
    use_tqdm=False,
    smooth=True,
    save_on_success=True,
    dry_run=False,
    collision_model="link",
    trace_dir=None,
    collision_dir=None,
    bit_time_budget=10.0,
):
    """在给定索引范围上运行 planner，返回 records 列表

    如果 save_on_success 为 True, 在每个成功的问题上保存配对和碰撞数据。
    """
    it = tqdm(indexes) if use_tqdm else indexes
    records = []
    is_bit_star = planner.__class__.__name__ == "BITStar"
    # 提前保存构造参数，避免在循环内持有旧实例引用
    for idx in it:
        try:
            env.init_new_problem(idx)
        except Exception as e:
            print(f"Skip index {idx}: init_new_problem failed: {e}")
            continue

        # reset collision stats and data (clear previous run's stored configs)
        try:
            env.collision_env.detector.reset()
            env.collision_env.data_manager.reset()
            env.collision_env.config_list = []
        except Exception:
            print(f"Warning: failed to reset collision env for index {idx}")
            exit(1)
        if not is_bit_star:
            # GNNMP 调用 plan()
            res = planner.plan(smooth=smooth)
        else:
            # BITStar 在构造时会缓存 start/goal，
            # 每个问题用局部变量新建实例，使旧实例在本轮结束后能被 GC 回收。
            bit_planner = BITStar(
                environment=env,
                plot_flag=False,
            )
            res = bit_planner.plan(
                pathLengthLimit=1.2, time_budget=bit_time_budget, refine_time_budget=200
            )
            del bit_planner

        checks = int(res.get("collision_checks", 0))
        success = bool(res.get("success", False))
        runtime = float(res.get("total_time", 0.0))
        cost = float(res.get("cost", 0.0)) if success else float("inf")

        records.append(
            {
                "index": idx,
                "checks": checks,
                "success": success,
                "time": runtime,
                "cost": cost,
            }
        )

        # 如果成功并要求保存, 保存当前问题的数据
        if success and save_on_success:
            save_current_problem_data(
                env,
                idx,
                collision_model,
                dry_run=dry_run,
                trace_dir=trace_dir,
                collision_dir=collision_dir,
            )

        if use_tqdm:
            it.set_description(f"Idx {idx} checks={checks} success={success}")

    return records


def move_files_for_levels(
    records,
    index_to_level,
    env,
    levels,
    out_root,
    dry_run=False,
    trace_dir=None,
    collision_dir=None,
):
    """将现存的 pair 文件和 collision 文件移动到对应难度目录并重命名。

    优化：先扫描 collision_dir 构建索引映射，避免为每个 record 重复遍历目录（从 O(n^2) 改为 O(n)).
    支持 dry_run（仅打印，不实际移动）。
    """
    # Simplified: assume trace_dir and collision_dir are provided and valid
    source_pair_dir = os.path.abspath(trace_dir)
    collision_dir = os.path.abspath(collision_dir)

    # 创建每级目标目录
    for lvl in levels:
        os.makedirs(os.path.join(out_root, lvl), exist_ok=True)
        os.makedirs(os.path.join(source_pair_dir, lvl), exist_ok=True)
        os.makedirs(os.path.join(collision_dir, lvl), exist_ok=True)

    level_move_counters = {lvl: 0 for lvl in levels}
    base_filename = f"{env.robot_env.robot_name}_{env.config_dim}"

    # 先扫描 collision_dir，建立 index->(link_fname, sphere_fname) 映射
    collision_map = {}
    pattern = re.compile(rf"^{re.escape(base_filename)}_(\d{{4}})_(link|sphere)\.pkl$")
    for fname in os.listdir(collision_dir):
        m = pattern.match(fname)
        if not m:
            continue
        idx_str, kind = m.group(1), m.group(2)
        idx_int = int(idx_str)
        ent = collision_map.setdefault(idx_int, {})
        ent[kind] = fname

    for r in records:
        idx = r["index"]
        lvl = index_to_level.get(idx)
        if lvl is None:
            continue

        level_move_counters[lvl] += 1
        new_idx = level_move_counters[lvl]

        # 移动 pair 文件
        orig_pair = os.path.join(source_pair_dir, f"{base_filename}_{idx:04d}.pkl")
        if os.path.exists(orig_pair):
            dst_pair = os.path.join(
                source_pair_dir, lvl, f"{base_filename}_{new_idx:04d}.pkl"
            )
            if dry_run:
                print(f"DRY-RUN: would move {orig_pair} -> {dst_pair}")
            else:
                try:
                    os.replace(orig_pair, dst_pair)
                except Exception as e:
                    print(f"Failed to move pair file {orig_pair} -> {dst_pair}: {e}")

        # 使用预扫描的映射查找 collision 文件
        ent = collision_map.get(idx, {})
        if "link" in ent:
            src_link = os.path.join(collision_dir, ent["link"])
            dst_link = os.path.join(
                collision_dir, lvl, f"{base_filename}_{new_idx:04d}_link.pkl"
            )
            if dry_run:
                print(f"DRY-RUN: would move {src_link} -> {dst_link}")
            else:
                try:
                    os.replace(src_link, dst_link)
                except Exception as e:
                    print(
                        f"Failed to move collision link file {src_link} -> {dst_link}: {e}"
                    )

        if "sphere" in ent:
            src_sphere = os.path.join(collision_dir, ent["sphere"])
            dst_sphere = os.path.join(
                collision_dir, lvl, f"{base_filename}_{new_idx:04d}_sphere.pkl"
            )
            if dry_run:
                print(f"DRY-RUN: would move {src_sphere} -> {dst_sphere}")
            else:
                try:
                    os.replace(src_sphere, dst_sphere)
                except Exception as e:
                    print(
                        f"Failed to move collision sphere file {src_sphere} -> {dst_sphere}: {e}"
                    )


def save_problems_by_level(problems_by_level, out_root):
    for lvl, plist in problems_by_level.items():
        lvl_dir = os.path.join(out_root, lvl)
        os.makedirs(lvl_dir, exist_ok=True)
        problems_file = os.path.join(lvl_dir, "problems.pkl")
        with open(problems_file, "wb") as f:
            pickle.dump(plist, f)


def save_metadata_csv(records, index_to_level, out_root):
    metadata_csv = os.path.join(out_root, "metadata.csv")
    with open(metadata_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["index", "checks", "success", "time", "cost", "level"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "index": r["index"],
                    "checks": r["checks"],
                    "success": r["success"],
                    "time": r["time"],
                    "cost": r["cost"],
                    "level": index_to_level.get(r["index"], ""),
                }
            )


def redistribute_problems_by_difficulty(
    env, records, out_root, dry_run=False, trace_dir=None, collision_dir=None
):
    """根据记录 (records) 的碰撞检测次数分位数将问题划分为 G1..G5，
    并将相应的 pair 文件与 collision 数据移动到 out_root/{Gx}，同时保存 per-level problems.pkl 和 metadata.csv。

    参数:
        env: ModularEnv (用于读取 problem 列表及文件名模式)
        records: list of dict with fields: index, checks, success, time, cost
        out_root: 目标根目录
        dry_run: 若为 True 则仅打印计划的移动，不执行
    """
    levels = ["G1", "G2", "G3", "G4", "G5"]

    # 计算分位数并映射 index -> level
    index_to_level, level_counters, quantiles = partition_by_collision_counts(
        records, levels=levels
    )

    print("Collision check statistics:")
    checks_array = np.array([r["checks"] for r in records])
    print(
        f"  min: {checks_array.min()}, max: {checks_array.max()}, mean: {checks_array.mean():.2f}, median: {np.median(checks_array):.2f}"
    )
    print(f"  20/40/60/80/100 percentiles: {quantiles}")

    # 组织每级 problems（从 env 获取问题对象，若失败则使用 record）
    problems_by_level = {lvl: [] for lvl in levels}
    for r in records:
        lvl = index_to_level[r["index"]]
        try:
            prob = env.problem_manager.problems[r["index"]]
            problems_by_level[lvl].append(prob)
        except Exception:
            problems_by_level[lvl].append(r)

    # 保存 per-level problems.pkl
    save_problems_by_level(problems_by_level, out_root)

    # 移动与重命名 pair & collision 文件
    move_files_for_levels(
        records,
        index_to_level,
        env,
        levels,
        out_root,
        dry_run=dry_run,
        trace_dir=trace_dir,
        collision_dir=collision_dir,
    )

    # 保存 metadata.csv
    save_metadata_csv(records, index_to_level, out_root)

    print("\nRedistribution complete:")
    for lvl in levels:
        print(
            f"  {lvl}: {level_counters[lvl]} problems -> saved to {os.path.join(out_root, lvl)}"
        )

    print(f"Metadata saved to: {os.path.join(out_root, 'metadata.csv')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GNNMP on KUKA-7 and redistribute by collision counts"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive). If omitted uses full dataset",
    )
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--t_max", type=int, default=1000)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=10.0,
        help="Per-problem wall-clock time budget (seconds) for BITStar.plan",
    )
    parser.add_argument(
        "--no-smooth", action="store_true", help="Disable path smoothing"
    )
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")
    parser.add_argument(
        "--out",
        type=str,
        default="../../trace_files/gnn_kuka7_difficulty",
        help="Output root directory to save per-level problems and metadata",
    )
    parser.add_argument(
        "--collision-model",
        type=str,
        choices=["link", "sphere"],
        default="link",
        help="Collision model type to use when creating env (link or sphere)",
    )
    parser.add_argument(
        "--planner",
        type=str,
        choices=["gnnmp", "bit_star"],
        default="gnnmp",
        help="Planner to evaluate: 'gnnmp' (default) or 'bit_star'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, don't actually move files; only print planned moves",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure trace and collision directories based on CLI or planner choice
    if args.planner == "bit_star":
        trace_dir = os.path.abspath(
            os.path.join("..", "..", "trace_files", "bit_traces")
        )
    else:
        trace_dir = os.path.abspath(
            os.path.join("..", "..", "trace_files", "gnn_traces")
        )

    if args.planner == "bit_star":
        collision_dir = os.path.abspath(
            os.path.join(
                "..", "..", "trace_files", "scene_benchmarks", "bit_collision_data"
            )
        )
    else:
        collision_dir = os.path.abspath(
            os.path.join(
                "..", "..", "trace_files", "scene_benchmarks", "gnn_collision_data"
            )
        )

    env, planner, device = setup_planner(
        args.seed,
        args.batch,
        args.t_max,
        collision_model_type=args.collision_model,
        planner_choice=args.planner,
    )
    print(
        f"Device: {device}, collision_model: {args.collision_model}, planner: {args.planner}"
    )

    total_problems = env.problem_manager.get_problem_count()
    start_idx = args.start
    end_idx = args.end if args.end is not None else total_problems
    end_idx = min(end_idx, total_problems)
    indexes = list(range(start_idx, end_idx))

    # 运行评估
    records = evaluate_problems(
        env,
        planner,
        indexes,
        use_tqdm=args.tqdm,
        smooth=not args.no_smooth,
        save_on_success=True,
        dry_run=args.dry_run,
        collision_model=args.collision_model,
        trace_dir=trace_dir,
        collision_dir=collision_dir,
        bit_time_budget=args.time_budget,
    )

    if len(records) == 0:
        print("No records collected. Exiting.")
        return 1

    # 使用统一的 redistributor 函数来划分与搬迁数据
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    redistribute_problems_by_difficulty(
        env,
        records,
        out_root,
        dry_run=args.dry_run,
        trace_dir=trace_dir,
        collision_dir=collision_dir,
    )

    return 0


if __name__ == "__main__":
    exit(main())
