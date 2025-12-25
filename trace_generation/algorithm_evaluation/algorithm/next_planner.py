"""
NEXTPlanner: Object-oriented wrapper for NEXT motion planning algorithm.

This module provides a class-based interface for the NEXT (Neural-guided
Extended Tree Search) algorithm, following the design pattern of BITStar.

Author: Adapted from NEXT_plan function
Date: 2025
"""

import numpy as np
import torch
import pickle
import os
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from time import time
from environment.timer import Timer
from .search_tree import (
    SearchTree,
    insert_new_state,
    compute_w,
    rewire_to,
    set_cost,
    update_collision_checks,
)


@dataclass
class NEXTConfig:
    """Configuration for NEXT planner."""

    T: int = 100  # Maximum number of samples
    g_explore_eps: float = 0.1  # Probability for RRT-like global exploration
    model_eps: float = 0.05  # Probability for goal-biased heuristic
    UCB_type: str = "kde"  # Type of UCB ('kde' or 'GP')
    c: float = 1.0  # Hyperparameter for exploration-exploitation balance
    neighbor_r: Optional[float] = None  # Radius for rewiring
    obs_cost: float = 2.0  # Cost for obstacle
    verbose: bool = True  # Print debug information


class NEXTPlanner:
    """
    NEXT (Neural-guided Extended Tree Search) motion planning algorithm.

    This class encapsulates the NEXT algorithm with a similar structure to BITStar,
    providing a clean interface for planning and evaluation.

    Attributes:
        env: The planning environment (contains map, initial/goal states, collision checker)
        model: Neural network model for guiding tree expansion (optional)
        config: Planning configuration
        search_tree: Current search tree
        success: Whether a path was found
        iterations: Number of iterations performed
        planning_time: Total planning time
    """

    def __init__(
        self,
        env,
        model: Optional[Any] = None,
        config: Optional[NEXTConfig] = None,
        timer: Optional[Timer] = None,
    ):
        """
        Initialize NEXT planner.

        Args:
            env: Planning environment with collision checking capabilities
            model: Neural network model for guiding expansion (optional)
            config: Planning configuration (uses default if None)
            timer: External timer for performance profiling (optional)
        """
        self.env = env
        self.model = model
        self.config = config if config is not None else NEXTConfig()
        self.timer = timer if timer is not None else Timer()

        # Search tree (initialized in setup_planning)
        self.search_tree: Optional[SearchTree] = None

        # Planning state
        self.success = False
        self.iterations = 0
        self.planning_time = 0.0

        # Statistics
        self.collision_checks = 0
        self.goal_reach_iteration = None
        self.path_cost = float("inf")
        self.path_length = float("inf")

        # Logging
        self.edge_full_info = []
        self.edge_coll_full_info = []
        self.problem_index = None

        if self.config.verbose:
            print(
                f"[NEXTPlanner] Initialized with config: T={self.config.T}, "
                f"g_explore_eps={self.config.g_explore_eps}, "
                f"model_eps={self.config.model_eps}"
            )

    def setup_planning(self) -> None:
        """
        Initialize the search tree and prepare for planning.

        This method must be called before plan().
        """
        if self.config.verbose:
            print("[NEXTPlanner] Setting up planning...")

        # Initialize search tree with starting configuration
        self.search_tree = SearchTree(
            env=self.env, root=self.env.init_state, model=self.model, dim=self.env.config_dim
        )

        # Reset logging
        self.edge_full_info = []
        self.edge_coll_full_info = []
        self.success = False
        self.iterations = 0

        if self.config.verbose:
            print("[NEXTPlanner] Setup completed")

    def plan(self, stop_when_success: bool = True) -> bool:
        """
        Execute the NEXT planning algorithm.

        Args:
            stop_when_success: Terminate once a solution is found

        Returns:
            Boolean indicating if a path was found
        """
        if self.search_tree is None:
            self.setup_planning()

        # Type guard to ensure search_tree is not None
        assert self.search_tree is not None

        start_time = time()

        if self.config.verbose:
            print(f"[NEXTPlanner] Starting planning with T_max={self.config.T}...")

        for i in range(self.config.T):
            self.iterations = i

            # Goal-biased heuristic
            if np.random.rand() < self.config.model_eps:
                leaf_state, parent_idx, _, no_collision, done, einfo, ecollinfo = (
                    self._global_explore(sample_state=self.env.goal_state)
                )
                self.edge_full_info.append(einfo)
                self.edge_coll_full_info.append(ecollinfo)
                self.success = self.success or done
                expanded_by_rrt = True

            # RRT-like global exploration
            elif np.random.rand() < self.config.g_explore_eps:
                leaf_state, parent_idx, _, no_collision, done, einfo, ecollinfo = (
                    self._global_explore()
                )
                self.edge_full_info.append(einfo)
                self.edge_coll_full_info.append(ecollinfo)
                self.success = self.success or done
                expanded_by_rrt = True

            # Guided selection and expansion
            else:
                idx = self._select()
                assert self.search_tree.freesp[idx]

                parent_idx = idx
                leaf_state, _, no_collision, done, einfo, ecollinfo = (
                    self._guided_expand(parent_idx)
                )
                self.edge_full_info.append(einfo)
                self.edge_coll_full_info.append(ecollinfo)
                self.success = self.success or done
                expanded_by_rrt = False

            # Insert new state and rewire
            leaf_id = insert_new_state(
                self.env,
                self.search_tree,
                leaf_state,
                self.model,
                parent_idx,
                no_collision,
                done,
                expanded_by_rrt=expanded_by_rrt,
            )

            # Local rewiring optimization
            tefullinfo, tecollfullinfo = self._rewire_last()
            self.edge_full_info.extend(tefullinfo)
            self.edge_coll_full_info.extend(tecollfullinfo)

            if self.success and stop_when_success:
                self.goal_reach_iteration = i
                if self.config.verbose:
                    print(f"[NEXTPlanner] Solution found at iteration {i}")
                break

        # Update statistics
        self.planning_time = time() - start_time
        self.collision_checks = self.env.collision_check_count()

        # Calculate path metrics if success
        if self.success and self.search_tree is not None:
            path, path_cost = self.search_tree.path()
            self.path_length = len(path)
            self.path_cost = self._calculate_path_cost(path)

        if self.config.verbose:
            print(f"[NEXTPlanner] Planning completed in {self.planning_time:.3f}s")
            print(
                f"[NEXTPlanner] Success: {self.success}, Iterations: {self.iterations}"
            )
            print(f"[NEXTPlanner] Collision checks: {self.collision_checks}")

        return self.success

    def _select(self) -> int:
        """
        Select a state for expansion using UCB-based strategy.

        Returns:
            Index of selected state in search tree
        """
        # Type guard to ensure search_tree is not None
        assert self.search_tree is not None

        self.timer.start()
        scores = []
        for i in range(self.search_tree.non_terminal_states.shape[0]):
            idx = self.search_tree.non_terminal_idxes[i]
            Q = self.search_tree.state_values[idx]
            U = np.sqrt(np.log(self.search_tree.w_sum) / self.search_tree.w[idx])
            scores.append(Q + self.config.c * U)

        self.timer.finish(Timer.HEAP)
        return self.search_tree.non_terminal_idxes[np.argmax(scores)]

    def _global_explore(self, sample_state: Optional[np.ndarray] = None) -> Tuple:
        """
        One step of RRT-like global exploration.

        Args:
            sample_state: Pre-sampled state (if None, sample uniformly)

        Returns:
            Tuple of (new_state, parent_idx, action, no_collision, done, einfo, ecollinfo)
        """
        # Type guard to ensure search_tree is not None
        assert self.search_tree is not None

        non_terminal_states = self.search_tree.non_terminal_states

        # Sample uniformly in the configuration space
        if sample_state is None:
            sample_state = self.env.uniform_sample()

        # Type guard to ensure sample_state is not None
        assert sample_state is not None

        # Find nearest state in tree
        dists = self.env.distance(non_terminal_states, sample_state)
        nearest_idx, min_dist = np.argmin(dists), np.min(dists)

        # Steer towards sampled state
        new_state = self._rrt_steer(
            sample_state, non_terminal_states[nearest_idx], min_dist
        )

        # Collision check
        new_state, action, no_collision, done, einfo, ecollinfo = self.env.step_probe(
            state=non_terminal_states[nearest_idx], new_state=new_state
        )

        return (
            new_state,
            self.search_tree.non_terminal_idxes[nearest_idx],
            action,
            no_collision,
            done,
            einfo,
            ecollinfo,
        )

    @torch.no_grad()
    def _guided_expand(self, parent_idx: int, k: int = 10) -> Tuple:
        """
        Model-guided expansion from a selected state.

        Args:
            parent_idx: Index of parent state
            k: Number of candidate actions to generate

        Returns:
            Tuple of (new_state, action, no_collision, done, einfo, ecollinfo)
        """
        # Type guard to ensure search_tree is not None
        assert self.search_tree is not None

        if self.model is None:
            raise RuntimeError("Model required for guided expansion but not provided")

        state = np.array(self.search_tree.states[parent_idx])

        # Get candidate actions from model
        self.timer.start()
        candidate_actions = self.model.policy(state=state, k=k)[0]
        self.timer.finish(Timer.GPU)

        candidates = []
        for i in range(k):
            action = candidate_actions[i]
            new_state, _ = self.env.step_probe(
                state=state, action=action, check_collision=False
            )
            candidates.append(new_state)

        # Select best candidate using model-based value + exploration bonus
        if k > 1:
            scores = []
            self.timer.start()
            Qs = self.model.pred_value(np.array(candidates))
            self.timer.finish(Timer.GPU)

            for i in range(k):
                Q = Qs[i]
                w = compute_w(self.env, self.search_tree, state=candidates[i])
                U = np.sqrt(np.log(self.search_tree.w_sum) / w)
                scores.append(Q + self.config.c * U)

            new_state = candidates[np.argmax(scores)]
        else:
            new_state = candidates[0]

        # Collision check for best candidate
        new_state, action, no_collision, done, einfo, ecollinfo = self.env.step_probe(
            state=state, new_state=new_state
        )

        return new_state, action, no_collision, done, einfo, ecollinfo

    def _rrt_steer(
        self, sample_state: np.ndarray, nearest: np.ndarray, dist: float
    ) -> np.ndarray:
        """
        Steer from nearest state towards sampled state.

        Args:
            sample_state: Target state
            nearest: Nearest state in tree
            dist: Distance to sample state

        Returns:
            Steered state
        """
        if dist < self.env.RRT_EPS:
            return sample_state

        ratio = self.env.RRT_EPS / dist
        return self.env.interpolate(nearest, sample_state, ratio)

    def _rewire_last(self) -> Tuple[list, list]:
        """
        Local rewiring optimization for the latest added state.

        Returns:
            Tuple of (edge_full_info, edge_coll_full_info)
        """
        edge_full_info = []
        edge_coll_full_info = []

        # Type guard to ensure search_tree is not None
        assert self.search_tree is not None

        if self.config.neighbor_r is None:
            neighbor_r = self.env.RRT_EPS * 3
        else:
            neighbor_r = self.config.neighbor_r

        cur_tree = self.search_tree.states[:-1]
        new_state = self.search_tree.states[-1]
        nearest_parent = self.search_tree.parents[-1]
        # Type guard to ensure nearest_parent is not None
        assert nearest_parent is not None
        nearest = int(nearest_parent)
        freesp = self.search_tree.freesp

        # Return if latest point is in collision
        if not self.search_tree.freesp[-1]:
            set_cost(self.search_tree, -1, self.config.obs_cost)
            update_collision_checks(self.search_tree, self.env.collision_check_count)
            return edge_full_info, edge_coll_full_info

        # Find locally optimal parent
        dists = self.env.distance(cur_tree, new_state)
        near = np.where(dists < neighbor_r)[0]

        min_cost = dists[nearest] + self.search_tree.costs[nearest]
        min_j = nearest

        for j in near:
            if not freesp[j]:
                continue

            cost_new = dists[j] + self.search_tree.costs[j]
            if cost_new < min_cost:
                _, _, no_collision, done, einfo, ecollinfo = self.env.step_probe(
                    state=cur_tree[j], new_state=new_state
                )
                edge_full_info.append(einfo)
                edge_coll_full_info.append(ecollinfo)

                if no_collision:
                    min_cost, min_j = cost_new, j

        # Rewire to optimal parent
        rewire_to(self.search_tree, -1, min_j)
        set_cost(self.search_tree, -1, min_cost)

        # Rewire neighbors
        for j in near:
            cost_new = min_cost + dists[j]
            if cost_new < self.search_tree.costs[j]:
                _, _, no_collision, done, einfo, ecollinfo = self.env.step_probe(
                    state=cur_tree[j], new_state=new_state
                )
                edge_full_info.append(einfo)
                edge_coll_full_info.append(ecollinfo)

                if no_collision:
                    set_cost(self.search_tree, j, cost_new)
                    rewire_to(self.search_tree, j, len(self.search_tree.states) - 1)

        update_collision_checks(self.search_tree, self.env.collision_check_count)

        return edge_full_info, edge_coll_full_info

    def _calculate_path_cost(self, path: list) -> float:
        """Calculate total cost of a path."""
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.env.distance(path[i], path[i + 1])
        return cost

    # ==================== Evaluation Methods ====================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.

        Returns:
            Dictionary with metrics:
            - success: Whether a path was found
            - iterations: Number of iterations
            - planning_time: Total planning time
            - collision_checks: Number of collision checks
            - path_length: Number of nodes in path
            - path_cost: Total path cost
            - goal_reach_iteration: Iteration when solution was found
        """
        return {
            "success": self.success,
            "iterations": self.iterations,
            "planning_time": self.planning_time,
            "collision_checks": self.collision_checks,
            "path_length": self.path_length if self.success else None,
            "path_cost": self.path_cost if self.success else None,
            "goal_reach_iteration": self.goal_reach_iteration,
        }

    def get_path(self) -> Tuple[Optional[list], Optional[list]]:
        """
        Get the planned path and its cost breakdown.

        Returns:
            Tuple of (path, path_costs) or (None, None) if no path found
        """
        if self.success and self.search_tree is not None:
            return self.search_tree.path()
        return None, None

    def get_search_tree(self) -> Optional[SearchTree]:
        """Get the underlying search tree."""
        return self.search_tree

    def save_results(
        self, problem_index: int, output_dir: str = "logfiles_NEXT_link"
    ) -> None:
        """
        Save planning results and debug information.

        Args:
            problem_index: Index of the problem being solved
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        log_path = os.path.join(output_dir, f"link_info_{problem_index}.pkl")

        with open(log_path, "wb") as f:
            pickle.dump((self.edge_full_info, self.edge_coll_full_info), f)

        if self.config.verbose:
            print(f"[NEXTPlanner] Results saved to {log_path}")

    def get_statistics(self) -> str:
        """Get formatted statistics string."""
        stats = self.get_metrics()

        lines = [
            "=" * 50,
            "NEXT Planner Statistics",
            "=" * 50,
            f"Success: {stats['success']}",
            f"Iterations: {stats['iterations']}",
            f"Planning Time: {stats['planning_time']:.3f}s",
            f"Collision Checks: {stats['collision_checks']}",
        ]

        if stats["success"]:
            lines.extend(
                [
                    f"Path Length: {stats['path_length']}",
                    f"Path Cost: {stats['path_cost']:.3f}",
                    f"Solution found at iteration: {stats['goal_reach_iteration']}",
                ]
            )
        else:
            lines.append("No solution found!")

        lines.append("=" * 50)

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset planner for new planning problem."""
        self.search_tree = None
        self.success = False
        self.iterations = 0
        self.planning_time = 0.0
        self.collision_checks = 0
        self.goal_reach_iteration = None
        self.path_cost = float("inf")
        self.path_length = float("inf")
        self.edge_full_info = []
        self.edge_coll_full_info = []
