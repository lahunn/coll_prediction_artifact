import numpy as np
import pickle

class MazeLoader:
    def __init__(self, map_file):
        with open(map_file, "rb") as f:
            self.problems = pickle.load(f)
        self.maps = {}
        self.order = list(range(len(self.problems)))
        self.episode_i = 0

    def init_new_problem(self, index=None):
        if index is None:
            self.index = self.episode_i
        else:
            self.index = index
        obstacles, start, goal, path = self.problems[self.index]
        self.episode_i += 1
        self.episode_i = self.episode_i % len(self.order)
        return obstacles, start, goal, path

    def get_problem(self, width=15, index=None):
        if index is None:
            problem = {
                "map": np.array(self.obs_map(width)[1]).astype(float),
                "init_state": self.init_state,
                "goal_state": self.goal_state,
            }
            self.maps[self.index] = problem
            return problem
        else:
            return self.maps[index]

    def obs_map(self, num):
        resolution = 2.0 / (num - 1)
        grid_pos = [np.linspace(-1.0, 1.0, num=num) for i in range(3)]
        points_pos = np.meshgrid(*grid_pos)
        points_pos = np.concatenate(
            (
                points_pos[0].reshape(-1, 1),
                points_pos[1].reshape(-1, 1),
                points_pos[2].reshape(-1, 1),
            ),
            axis=-1,
        )
        points_obs = np.zeros(points_pos.shape[0]).astype(bool)
        for obstacle in self.obstacles:
            obstacle_size, obstacle_base = obstacle
            limit_low, limit_high = (
                obstacle_base - obstacle_size,
                obstacle_base + obstacle_size,
            )
            bools = []
            for i in range(3):
                obs_mask = np.zeros(num).astype(bool)
                obs_mask[
                    max(int((limit_low[i] + 1) / resolution), 0) : min(
                        (1 + int((limit_high[i] + 1) / resolution)),
                        1 + int(2.0 / resolution),
                    )
                ] = True
                bools.append(obs_mask)
            current_obs = np.meshgrid(*bools)
            current_obs = np.concatenate(
                (
                    current_obs[0].reshape(-1, 1),
                    current_obs[1].reshape(-1, 1),
                    current_obs[2].reshape(-1, 1),
                ),
                axis=-1,
            )
            points_obs = np.logical_or(points_obs, np.all(current_obs, axis=-1))
        return points_pos.reshape((num, num, num, -1)), points_obs.reshape((num, num, num))
