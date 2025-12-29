#!/usr/bin/env python
"""
One-click runner for GNNMP evaluation on KUKA-7 using the GNNMP class.
Example:
  .venv/bin/python eval_kuka7.py --start 2000 --end 2005 --seed 1234
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm as tqdm_lib

# Ensure project root is on path when running from anywhere
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from algorithm.gnnmp import GNNMP, path_cost  # noqa: E402
from str2name import str2name  # noqa: E402
from trace_generation.utils.config import set_random_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GNNMP on KUKA-7")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--start", type=int, default=2000, help="Start index (inclusive)"
    )
    parser.add_argument("--end", type=int, default=2003, help="End index (exclusive)")
    parser.add_argument(
        "--no-smooth", action="store_true", help="Disable path smoothing"
    )
    parser.add_argument("--tqdm", action="store_true", help="Show tqdm progress bar")
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    # 1. Load Environment and Models using str2name (which loads weights automatically)
    env_str = "kuka7"
    # print(f"DEBUG: Calling str2name with {env_str}...")
    # Note: str2name returns: env, model_explore, model_explore_path, model_smooth, model_smooth_path
    env, model, _, model_s, _ = str2name(env_str, load=True)
    # print("DEBUG: str2name returned.")

    indexes = np.arange(args.start, args.end)
    smooth = not args.no_smooth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Env: {env}")
    print(f"Indexes: {indexes.tolist()}")
    print(f"Smooth: {smooth}")

    # 2. Initialize GNNMP Planner
    # print("DEBUG: Initializing GNNMP Planner...")
    planner = GNNMP(
        env=env,
        model_explore=model,
        model_smooth=model_s if smooth else None,
        batch=50,
        t_max=1000,
        device=device,
    )
    # print("DEBUG: GNNMP Planner initialized.")

    results = []

    pbar = tqdm_lib(indexes) if args.tqdm else indexes

    for index in pbar:
        # Initialize the specific problem instance
        # print(f"DEBUG: Initializing problem index {index}...")
        env.init_new_problem(index)
        # print(f"DEBUG: Problem {index} initialized.")

        # 3. Execute Plan
        # print(f"DEBUG: Starting plan for index {index}...")
        res = planner.plan(smooth=smooth)
        # print(f"DEBUG: Plan finished for index {index}. Success: {res['success']}")

        results.append(res)

        if args.tqdm:
            pbar.set_description(
                f"Success: {res['success']}, Cost: {path_cost(res['smooth_path']):.2f}"
            )

    # 4. Aggregated Stats
    n_success = sum([r["success"] for r in results])
    if n_success > 0:
        avg_cost = np.mean(
            [path_cost(r["smooth_path"]) for r in results if r["success"]]
        )
        avg_time = np.mean([r["total_time"] for r in results if r["success"]])
        avg_col_checks = np.mean(
            [r["c_explore"] + r["c_smooth"] for r in results if r["success"]]
        )
    else:
        avg_cost = 0.0
        avg_time = 0.0
        avg_col_checks = 0.0

    print("\n--- Evaluation Summary ---")
    print(
        f"Success Rate: {n_success}/{len(indexes)} ({n_success / len(indexes) * 100:.1f}%)"
    )
    print(f"Avg Cost: {avg_cost:.2f}")
    print(f"Avg Time: {avg_time:.2f}s")
    print(f"Avg Collision Checks: {avg_col_checks:.1f}")


if __name__ == "__main__":
    main()
