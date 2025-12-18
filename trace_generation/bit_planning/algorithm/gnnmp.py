import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_sparse import coalesce

from config import set_random_seed
from smoother import model_smooth, joint_smoother
from str2model import str2model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def path_cost(path):
    """Calculate total Euclidean path length."""
    path = np.array(path)
    cost = 0
    for i in range(len(path) - 1):
        cost += np.linalg.norm(path[i + 1] - path[i])
    return cost


def to_np(tensor):
    """Convert torch tensor to numpy array."""
    return tensor.data.cpu().numpy()


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def obs_data(env, free, collided):
    """Package obstacle and sampled config data for GNN input."""
    data = DotDict({
        'free': torch.FloatTensor(np.array(free)).to(device),
        'collided': torch.FloatTensor(np.array(collided))[:len(free)].to(device),
        'obstacles': torch.FloatTensor(env.obstacles).to(device),
    })
    return data


def create_data(free, collided, env, k):
    """Construct torch_geometric Data object for GNN policy network."""
    data = Data(goal=torch.FloatTensor(env.goal_state))
    data.v = torch.cat((torch.FloatTensor(np.array(free)),
                        torch.FloatTensor(np.array(collided))), dim=0)
    data.labels = torch.zeros(len(data.v), 3)
    data.labels[:len(free), 0] = 1
    data.labels[len(free):, 1] = 1
    data.labels[1, 2] = 1
    k1 = int(np.ceil(k * np.log(len(free)) / np.log(100)))
    edge_index = knn_graph(torch.FloatTensor(data.v), k=k1, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_free = knn_graph(torch.FloatTensor(data.v[:len(free)]), k=k1, loop=True)
    edge_index = torch.cat((edge_index, edge_index_free, edge_index_free.flip(0)), dim=-1)
    data.edge_index, _ = coalesce(edge_index, None, len(data.v), len(data.v))
    return data


@torch.no_grad()
def explore(env, model, model_s, pbindex=2000, smooth=True, batch=500, t_max=1000, k=30, smoother='model', loop=5):
    """
    GNN-guided exploration to find collision-free path.
    
    Returns dict with success, paths, collision counts, timing, and search trace.
    """
    c0 = env.collision_check_count
    t0 = time.time()
    forward = 0
    
    success = False
    path, smooth_path = [], []
    n_batch = batch
    free, collided, link_info, link_feas_info = env.sample_n_points_probe(n_batch, need_negative=True)
    collided = collided[:len(free)]
    free = [env.init_state] + [env.goal_state] + list(free)
    
    explored = [0]
    explored_edges = [[0, 0]]
    costs = {0: 0.}
    prev = {0: 0}

    data = create_data(free, collided, env, k)

    while not success and (len(free) - 2) <= t_max:
        t1 = time.time()
        policy = model(**data.to(device).to_dict(), **obs_data(env, free, collided), loop=loop)
        policy = policy.cpu()
        forward += time.time() - t1

        policy[torch.arange(len(data.v)), torch.arange(len(data.v))] = 0
        policy[:, explored] = 0
        policy[:, data.labels[:, 1] == 1] = 0
        policy[data.labels[:, 1] == 1, :] = 0
        policy[np.array(explored_edges).reshape(2, -1)] = 0
        
        while policy[explored, :].sum() != 0:
            agent = policy[
                np.array(explored)[torch.where(policy[explored, :] != 0)[0]], 
                torch.where(policy[explored, :] != 0)[1]
            ].argmax()

            end_a, end_b = torch.where(policy[explored, :] != 0)[0][agent], \
                           torch.where(policy[explored, :] != 0)[1][agent]
            end_a, end_b = int(end_a), int(end_b)
            end_a = explored[end_a]
            explored_edges.extend([[end_a, end_b], [end_b, end_a]])
            
            edgefeas, sample_info, sample_feas_info = env._edge_fp_probe(to_np(data.v[end_a]), to_np(data.v[end_b]))
            link_info.append(sample_info)
            link_feas_info.append(sample_feas_info)
            
            if edgefeas:
                explored.append(end_b)
                costs[end_b] = costs[end_a] + float(np.linalg.norm(to_np(data.v[end_a]) - to_np(data.v[end_b])))
                prev[end_b] = end_a

                policy[:, end_b] = 0
                if env.in_goal_region(to_np(data.v[end_b])):
                    success = True
                    path = [end_b]
                    node = end_b
                    while node != 0:
                        path.append(prev[node])
                        node = prev[node]
                    path.reverse()
                    break
            else:
                policy[end_a, end_b] = 0
                policy[end_b, end_a] = 0

        if not success:
            if not smooth:
                return []
            if (n_batch + len(free) - 2) > t_max:
                break
            new_free, new_collided, sample_info, sample_feas_info = env.sample_n_points_probe(n_batch, need_negative=True)
            link_info += sample_info
            link_feas_info += sample_feas_info
            free = free + list(new_free)
            collided = collided + list(new_collided)
            collided = collided[:len(free)]
            data = create_data(free, collided, env, k)

    c_explore = env.collision_check_count - c0
    c1 = env.collision_check_count
    t1 = time.time()
    
    if success and smooth:
        path = list(data.v[path].data.cpu().numpy())
        if smoother == 'model':
            smooth_path, edge_info, edge_feas_info = model_smooth(model_s, free, collided, path, env)
            link_info += edge_info
            link_feas_info += edge_feas_info
        elif smoother == 'oracle':
            smooth_path = joint_smoother(path, env, iter=5)
        else:
            smooth_path = path
    
    c_smooth = env.collision_check_count - c1
    
    os.makedirs("logfiles_GNN_2D", exist_ok=True)
    with open(f"logfiles_GNN_2D/link_info_{pbindex}.pkl", "wb") as f:
        pickle.dump((link_info, link_feas_info), f)
    
    if smooth:
        total_time = time.time()
        return {
            'c_explore': c_explore,
            'c_smooth': c_smooth,
            'data': data,
            'explored': explored,
            'forward': forward,
            'total': total_time - t0,
            'total_explore': t1 - t0,
            'success': success,
            't0': t0,
            'path': path,
            'smooth_path': smooth_path,
            'explored_edges': explored_edges
        }
    else:
        return list(data.v[path].data.cpu().numpy()), free, collided


def _load_models(model_key: str, device: torch.device) -> Dict[str, Any]:
    """Load exploration and smoothing models by key using existing helpers."""
    model, model_path, smoother, smoother_path = str2model(model_key)
    model.load_state_dict(torch.load(model_path, map_location=device))
    smoother.load_state_dict(torch.load(smoother_path, map_location=device))
    model.to(device).eval()
    smoother.to(device).eval()
    return {"model": model, "smoother": smoother}


class GNNPlanner:
    """Planner wrapper that exposes a BIT*-like plan() entry for GNN inference."""

    def __init__(
        self,
        environment: Any,
        model: Optional[torch.nn.Module] = None,
        smoother: Optional[torch.nn.Module] = None,
        model_key: Optional[str] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.env = environment
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.seed = seed
        self.model_key = model_key

        if model is None or smoother is None:
            if model_key is None:
                raise ValueError("Provide either models or a model_key to load them.")
            models = _load_models(model_key, self.device)
            model = models["model"]
            smoother = models["smoother"]
        else:
            model.to(self.device).eval()
            smoother.to(self.device).eval()

        self.model = model
        self.smoother = smoother
        self.best_path: List[Any] = []
        self.latest_result: Optional[Dict[str, Any]] = None

    def plan(
        self,
        path_length_limit: Optional[float] = None,
        problem_index: Optional[int] = None,
        smooth: bool = True,
        batch: int = 500,
        t_max: int = 1000,
        k: int = 30,
        loop: int = 5,
    ) -> Dict[str, Any]:
        """
        Run a single planning episode using the GNN explorer and smoother.

        Args:
            path_length_limit: Optional upper bound for accepting a solution.
            problem_index: Problem id used to reset the environment before planning.
            smooth: Whether to call the learned smoother after exploration.
            batch: Number of samples per expansion batch.
            t_max: Maximum number of sampled states during exploration.
            k: k-NN value used when building the policy graph.
            loop: Message-passing iterations used by the GNN policy.

        Returns:
            Dictionary with success flag, paths, collision stats, timing,
            and search trace.
        """
        if self.seed is not None:
            set_random_seed(self.seed)
        if problem_index is not None:
            self.env.init_new_problem(problem_index)

        start_wall = time.time()
        result = cast(
            Dict[str, Any],
            explore(
                self.env,
                self.model,
                self.smoother,
                pbindex=problem_index if problem_index is not None else 0,
                smooth=smooth,
                batch=batch,
                t_max=t_max,
                k=k,
                loop=loop,
            ),
        )
        wall_time = time.time() - start_wall

        if smooth:
            chosen_path = result.get("smooth_path", result.get("path", []))
        else:
            chosen_path = result.get("path", [])
        self.best_path = chosen_path

        collision_checks_explore = result.get("c_explore", 0)
        collision_checks_smooth = result.get("c_smooth", 0)
        collision_checks_total = collision_checks_explore + collision_checks_smooth
        total_cost = path_cost(chosen_path) if result.get("success") else float("inf")

        if path_length_limit is not None and total_cost > path_length_limit:
            result["success"] = False

        summary: Dict[str, Any] = {
            "success": result.get("success", False),
            "path": chosen_path,
            "raw_path": result.get("path", []),
            "smooth_path": result.get("smooth_path", chosen_path),
            "collision_checks": collision_checks_total,
            "collision_checks_explore": collision_checks_explore,
            "collision_checks_smooth": collision_checks_smooth,
            "explored": result.get("explored", []),
            "explored_edges": result.get("explored_edges", []),
            "forward_time": result.get("forward", 0.0),
            "total_time": result.get("total", wall_time),
            "total_explore_time": result.get("total_explore", 0.0),
            "wall_time": wall_time,
            "path_cost": total_cost,
        }

        self.latest_result = summary
        return summary

    def get_best_path(self) -> List[Any]:
        return self.best_path
