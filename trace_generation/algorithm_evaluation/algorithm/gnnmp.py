import torch
import numpy as np
from torch_geometric.data import Data
from torch_sparse import coalesce
from torch_geometric.nn import knn_graph
from time import time
from smoother import model_smooth, joint_smoother
from environment.timer import Timer


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def path_cost(path):
    path = np.array(path)
    cost = 0
    for i in range(0, len(path) - 1):
        cost += np.linalg.norm(path[i + 1] - path[i])
    return cost


def to_np(tensor):
    return tensor.data.cpu().numpy()


class GNNMP:
    def __init__(
        self,
        env,
        model_explore,
        model_smooth=None,
        batch=500,
        t_max=1000,
        k=30,
        loop=5,
        device=None,
    ):
        """
        Initialize the GNN-Motion-Planning wrapper.

        Args:
            env: The environment object (e.g., KukaEnv, MazeEnv).
            model_explore: The trained exploration GNN model.
            model_smooth: The trained smoothing GNN model (optional).
            batch: Number of points to sample per iteration.
            t_max: Maximum number of nodes in the graph (budget).
            k: K for KNN graph construction.
            loop: Number of message passing loops in the GNN.
            device: torch.device (cpu or cuda). If None, auto-detects.
        """
        self.env = env
        self.model = model_explore
        self.model_s = model_smooth
        self.batch = batch
        self.t_max = t_max
        self.k = k
        self.loop = loop

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()
        if self.model_s:
            self.model_s.to(self.device)
            self.model_s.eval()

    def _obs_data(self, free, collided):
        data = DotDict(
            {
                "free": torch.FloatTensor(np.array(free)).to(self.device),
                "collided": torch.FloatTensor(np.array(collided))[: len(free)].to(
                    self.device
                ),
                "obstacles": torch.FloatTensor(
                    np.array(self.env.obstacle_manager.obstacles)
                ).to(self.device),
            }
        )
        return data

    def _create_data(self, free, collided):
        data = Data(goal=torch.FloatTensor(self.env.goal_state))
        data.v = torch.cat(
            (torch.FloatTensor(np.array(free)), torch.FloatTensor(np.array(collided))),
            dim=0,
        )

        # create labels
        data.labels = torch.zeros(len(data.v), 3)
        data.labels[: len(free), 0] = 1
        data.labels[len(free) :, 1] = 1
        data.labels[1, 2] = 1

        k1 = int(np.ceil(self.k * np.log(len(free)) / np.log(100)))
        edge_index = knn_graph(torch.FloatTensor(data.v), k=k1, loop=True)
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
        edge_index_free = knn_graph(
            torch.FloatTensor(data.v[: len(free)]), k=k1, loop=True
        )
        edge_index = torch.cat(
            (edge_index, edge_index_free, edge_index_free.flip(0)), dim=-1
        )
        data.edge_index, _ = coalesce(edge_index, None, len(data.v), len(data.v))
        return data

    @torch.no_grad()
    def plan(self, smooth=True, smoother="model"):
        """
        Execute the GNNMP planning pipeline.

        Args:
            smooth (bool): Whether to perform path smoothing.
            smoother (str): 'model' for GNN smoother, 'oracle' for joint_smoother.

        Returns:
            dict: Contains result statistics and paths.
        """
        c0 = self.env.collision_check_count()
        t0 = time()
        forward = 0

        success = False
        path, smooth_path = [], []
        n_batch = self.batch

        # Initial sampling
        free, collided = self.env.sample_n_points(n_batch, need_negative=True)
        collided = collided[: len(free)]
        free = [self.env.init_state] + [self.env.goal_state] + list(free)

        explored = [0]
        explored_edges = [[0, 0]]
        costs = {0: 0.0}
        prev = {0: 0}

        data = self._create_data(free, collided)

        # 防止在无法采到新的 free 点时陷入无限重采样，设置最大重试次数
        resample_attempts = 0
        max_resample_attempts = 2

        while not success and (len(free) - 2) <= self.t_max:
            # print("DEBUG: GNNMP iteration with graph size:", len(free))
            t1 = time()
            # GNN Inference
            policy = self.model(
                **data.to(self.device).to_dict(),
                **self._obs_data(free, collided),
                loop=self.loop,
            )
            policy = policy.cpu()
            inference_time = time() - t1
            forward += inference_time

            # Masking logic
            policy[torch.arange(len(data.v)), torch.arange(len(data.v))] = 0
            policy[:, explored] = 0
            policy[:, data.labels[:, 1] == 1] = 0
            policy[data.labels[:, 1] == 1, :] = 0

            # Mask explored edges using correct coordinate indexing
            if len(explored_edges) > 0:
                ee = np.array(explored_edges)
                policy[ee[:, 0], ee[:, 1]] = 0

            success = False
            # Greedy search on policy
            while policy[explored, :].sum() != 0:
                agent = policy[
                    np.array(explored)[torch.where(policy[explored, :] != 0)[0]],
                    torch.where(policy[explored, :] != 0)[1],
                ].argmax()

                end_a, end_b = (
                    torch.where(policy[explored, :] != 0)[0][agent],
                    torch.where(policy[explored, :] != 0)[1][agent],
                )
                end_a, end_b = int(end_a), int(end_b)
                end_a = explored[end_a]
                explored_edges.extend([[end_a, end_b], [end_b, end_a]])

                # Collision check edge
                if self.env._edge_fp(to_np(data.v[end_a]), to_np(data.v[end_b])):
                    explored.append(end_b)
                    costs[end_b] = costs[end_a] + np.linalg.norm(
                        to_np(data.v[end_a]) - to_np(data.v[end_b])
                    )
                    prev[end_b] = end_a

                    policy[:, end_b] = 0
                    if self.env.in_goal_region(to_np(data.v[end_b])):
                        success = True
                        cost = costs[end_b]
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
                if (n_batch + len(free) - 2) > self.t_max:
                    break

                # Resample and rebuild graph
                new_free, new_collided = self.env.sample_n_points(
                    n_batch, need_negative=True
                )
                # 如果没有采到新的 free 点，累积重试次数，超过阈值则放弃
                if len(new_free) == 0:
                    resample_attempts += 1
                    if resample_attempts >= max_resample_attempts:
                        print("WARNING: max resample attempts reached, aborting search")
                        break
                else:
                    resample_attempts = 0

                free = free + list(new_free)
                collided = collided + list(new_collided)
                collided = collided[: len(free)]

                data = self._create_data(free, collided)

        c_explore = self.env.collision_check_count() - c0
        c1 = self.env.collision_check_count()
        t1 = time()

        if success and smooth:
            path = list(data.v[path].data.cpu().numpy())
            if smoother == "model" and self.model_s:
                smooth_path = model_smooth(self.model_s, free, collided, path, self.env)
            elif smoother == "oracle":
                smooth_path = joint_smoother(path, self.env, iter=5)
            else:
                smooth_path = path
        elif success:
            path = list(data.v[path].data.cpu().numpy())
            smooth_path = path

        c_smooth = self.env.collision_check_count() - c1
        total_time = time()

        return {
            "success": success,
            "path": path,
            "smooth_path": smooth_path,
            "c_explore": c_explore,
            "c_smooth": c_smooth,
            "total_time": total_time - t0,
            "explore_time": t1 - t0,
            "graph_size": len(free),
        }
