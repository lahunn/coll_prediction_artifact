import argparse
import os
import sys
import pickle
import numpy as np

# Ensure local packages import correctly: add repo root and bit_planning dir
THIS_DIR = os.path.dirname(__file__)
BIT_PLANNING_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
for p in (REPO_ROOT, BIT_PLANNING_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from trace_generation.core.robot.modular_env import ModularEnv
from trace_generation.bit_planning.algorithm.gnnmp import GNNPlanner


def load_problem(file_path, index):
    with open(file_path, "rb") as f:
        problems = pickle.load(f)
    obstacles, start, goal, _ = problems[index]
    return obstacles, start, goal


def prepare_env(robot_name, map_file, problem_index):
    """创建并初始化ModularEnv，直接加载问题数据"""
    obstacles, start, goal = load_problem(map_file, problem_index)
    
    env = ModularEnv(
        robot_name, 
        map_file=None, 
        GUI=False, 
        collision_model_type="link"
    )
    
    # 加载障碍物
    env.obstacle_manager.load_obstacles(obstacles)
    env.collision_env.load_obstacles(obstacles)
    
    # 设置起点和终点
    env.robot_env.init_state = start
    env.robot_env.goal_state = goal
    env.init_state = tuple(start)
    env.goal_state = tuple(goal)
    
    # 加载问题数据以支持init_new_problem
    env.problem_manager.load_problems(map_file)
    
    return env


def run_once(robot_name, map_file, problem_index, model_key):
    env = prepare_env(robot_name, map_file, problem_index)
    
    # 调试：检查环境初始化
    print(f"Robot config_dim: {env.robot_env.config_dim}")
    print(f"Init state: {env.init_state}")
    print(f"Goal state: {env.goal_state}")
    print(f"Obstacles count: {len(env.obstacle_manager.obstacles) if hasattr(env, 'obstacle_manager') else 0}")
    
    # 检查初始状态和目标状态是否在自由空间中
    init_free = env._state_fp(np.array(env.init_state))
    goal_free = env._state_fp(np.array(env.goal_state))
    print(f"Init state in free space: {init_free}")
    print(f"Goal state in free space: {goal_free}")
    
    if not init_free or not goal_free:
        print("WARNING: Initial or goal state is in collision!")
    
    planner = GNNPlanner(env, model_key=model_key)
    print(f"Model loaded: {type(planner.model).__name__}")
    print(f"Smoother loaded: {type(planner.smoother).__name__}")
    
    result = planner.plan(problem_index=problem_index, batch=200, t_max=500, k=30)
    
    print(f"\n=== Planning Results ===")
    print(f"Success: {result['success']}")
    print(f"Path length: {len(result['path'])}")
    print(f"Cost: {result['path_cost']:.3f}")
    print(f"Collision checks: {result['collision_checks']}")
    print(f"Time: {result['total_time']:.3f}s")
    print(f"Explored states: {len(result['explored'])}")
    
    if not result['success']:
        print(f"Planning failed - explored {len(result['explored'])} states but didn't reach goal")


def main():
    parser = argparse.ArgumentParser(description="Basic GNNPlanner test on saved iiwa problems.")
    parser.add_argument("--map-file", default="trace_generation/bit_planning/maze_files/iiwa_7_50.pkl")
    parser.add_argument("--robot-name", default="iiwa")
    parser.add_argument("--model-key", default="kuka7")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(0)
    run_once(args.robot_name, args.map_file, args.index, args.model_key)


if __name__ == "__main__":
    main()
