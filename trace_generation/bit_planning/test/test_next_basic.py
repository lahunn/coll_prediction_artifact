import argparse
import os
import sys
import pickle
import torch
import numpy as np

# Ensure local packages import correctly: add repo root and bit_planning dir
THIS_DIR = os.path.dirname(__file__)
BIT_PLANNING_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
for p in (REPO_ROOT, BIT_PLANNING_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from trace_generation.core.robot.modular_env import ModularEnv
from trace_generation.bit_planning.algorithm.next_planner import NEXTPlanner, NEXTConfig
from trace_generation.bit_planning.next_model.model3D import Model3D


def load_problem(file_path, index):
    with open(file_path, "rb") as f:
        problems = pickle.load(f)
    obstacles, start, goal, _ = problems[index]
    return obstacles, start, goal


def prepare_env(robot_name, map_file, problem_index):
    """Create and initialize ModularEnv, directly loading problem data"""
    obstacles, start, goal = load_problem(map_file, problem_index)
    
    env = ModularEnv(
        robot_name, 
        map_file=None, 
        GUI=False, 
        collision_model_type="link"
    )
    
    # Load obstacles
    env.obstacle_manager.load_obstacles(obstacles)
    env.collision_env.load_obstacles(obstacles)
    
    # Set start and goal states
    env.robot_env.init_state = start
    env.robot_env.goal_state = goal
    env.init_state = tuple(start)
    env.goal_state = tuple(goal)
    
    # Load problem data to support init_new_problem if needed
    env.problem_manager.load_problems(map_file)
    
    return env


def run_once(robot_name, map_file, problem_index, model=None, config=None):
    """Run NEXT planner once on a problem"""
    path = None
    env = prepare_env(robot_name, map_file, problem_index)
    
    # Debug: check environment initialization
    print(f"Robot config_dim: {env.robot_env.config_dim}")
    print(f"Init state: {env.init_state}")
    print(f"Goal state: {env.goal_state}")
    print(f"Obstacles count: {len(env.obstacle_manager.obstacles) if hasattr(env, 'obstacle_manager') else 0}")
    
    # Check if initial and goal states are in free space
    init_free = env._state_fp(np.array(env.init_state))
    goal_free = env._state_fp(np.array(env.goal_state))
    print(f"Init state in free space: {init_free}")
    print(f"Goal state in free space: {goal_free}")
    
    if not init_free or not goal_free:
        print("WARNING: Initial or goal state is in collision!")
    
    # Create planner with default or custom config
    if config is None:
        config = NEXTConfig(
            T=500,
            g_explore_eps=0.1,
            model_eps=0.05,
            c=1.0,
            verbose=True
        )
    
    planner = NEXTPlanner(env, model=model, config=config)
    print(f"\nNEXTPlanner initialized with config:")
    print(f"  T_max: {config.T}")
    print(f"  g_explore_eps: {config.g_explore_eps}")
    print(f"  model_eps: {config.model_eps}")
    print(f"  Model available: {model is not None}")
    
    # Run planning
    success = planner.plan(stop_when_success=True)
    
    # Get results
    print(f"\n{'='*50}")
    print(f"NEXT Planner Results")
    print(f"{'='*50}")
    print(f"Success: {success}")
    print(f"Iterations: {planner.iterations}")
    print(f"Planning time: {planner.planning_time:.3f}s")
    print(f"Collision checks: {planner.collision_checks}")
    
    if success:
        path, path_costs = planner.get_path()
        print(f"Path length: {len(path) if path else 0}")
        print(f"Path cost: {planner.path_cost:.3f}")
        print(f"Solution found at iteration: {planner.goal_reach_iteration}")
    else:
        print(f"Planning failed after {planner.iterations} iterations")
        print(f"Explored {len(planner.search_tree.states) if planner.search_tree else 0} states")
    
    print(f"{'='*50}\n")
    
    return {
        "success": success,
        "iterations": planner.iterations,
        "planning_time": planner.planning_time,
        "collision_checks": planner.collision_checks,
        "path_length": len(path) if success and path else None,
        "path_cost": planner.path_cost if success else None,
        "goal_reach_iteration": planner.goal_reach_iteration,
    }


def run_batch(robot_name, map_file, num_problems, model=None, config=None):
    """Run NEXT planner on multiple problems"""
    results = []
    
    for i in range(num_problems):
        print(f"\n{'#'*60}")
        print(f"Problem {i+1}/{num_problems}")
        print(f"{'#'*60}\n")
        
        try:
            result = run_once(robot_name, map_file, i, model=model, config=config)
            results.append(result)
        except Exception as e:
            print(f"ERROR on problem {i}: {str(e)}")
            results.append({"success": False, "error": str(e)})
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Batch Summary ({num_problems} problems)")
    print(f"{'='*60}")
    
    successes = sum(1 for r in results if r.get("success", False))
    print(f"Success rate: {successes}/{num_problems} ({100*successes/num_problems:.1f}%)")
    
    if successes > 0:
        avg_time = np.mean([r["planning_time"] for r in results if r.get("success")])
        avg_checks = np.mean([r["collision_checks"] for r in results if r.get("success")])
        print(f"Avg planning time (successful): {avg_time:.3f}s")
        print(f"Avg collision checks (successful): {avg_checks:.0f}")
    
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Basic NEXTPlanner test on saved problems.")
    parser.add_argument("--map-file", default="trace_generation/bit_planning/maze_files/iiwa_7_50.pkl",
                        help="Path to problem pickle file")
    parser.add_argument("--robot-name", default="iiwa", help="Robot name")
    parser.add_argument("--index", type=int, default=0, help="Problem index (single problem)")
    parser.add_argument("--batch", type=int, default=None, help="Number of problems to run (batch mode)")
    parser.add_argument("--T-max", type=int, default=500, help="Maximum iterations")
    parser.add_argument("--explore-eps", type=float, default=0.1, 
                        help="Probability for RRT-like exploration")
    parser.add_argument("--model-eps", type=float, default=0.05,
                        help="Probability for goal-biased heuristic")
    args = parser.parse_args()

    np.random.seed(0)
    
    # Create config with command-line parameters
    config = NEXTConfig(
        T=args.T_max,
        g_explore_eps=args.explore_eps,
        model_eps=args.model_eps,
        verbose=True
    )
    
    if args.batch is not None:
        # Batch mode: run multiple problems
        results = run_batch(args.robot_name, args.map_file, args.batch, model=None, config=config)
    else:
        # Single problem mode
        run_once(args.robot_name, args.map_file, args.index, model=None, config=config)


if __name__ == "__main__":
    main()
