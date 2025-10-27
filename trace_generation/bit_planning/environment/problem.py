import numpy as np
import pickle
import os


class ProblemManager:
    """
    问题管理类，负责加载和管理规划问题的数据

    主要功能：
    - 从文件加载问题数据集
    - 初始化新问题（设置障碍物、起点、终点）
    - 提供问题描述信息
    """

    def __init__(self, map_file=None):
        """
        初始化问题管理器

        Args:
            map_file: 问题数据集文件路径（可选）
        """
        self.map_file = map_file
        self.problems = None
        self.current_problem = None
        self.maps = {}

        if map_file:
            self.load_problems(map_file)

    def load_problems(self, map_file):
        """
        从pickle文件加载问题数据集

        Args:
            map_file: pickle文件路径

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
        """
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"问题文件不存在: {map_file}")

        try:
            with open(map_file, "rb") as f:
                self.problems = pickle.load(f)
        except Exception as e:
            raise ValueError(f"加载问题文件失败: {e}")

        if not isinstance(self.problems, list):
            raise ValueError("问题数据格式错误：期望列表类型")

        print(f"成功加载 {len(self.problems)} 个问题")

    def init_new_problem(self, index, obstacle_manager=None):
        """
        初始化新问题

        Args:
            index: 问题索引
            obstacle_manager: 障碍物管理器实例（可选，用于初始化障碍物）

        Returns:
            dict: 问题描述，包含 'obstacles', 'start', 'goal', 'path'

        Raises:
            ValueError: 问题索引无效或数据格式错误
        """
        if self.problems is None:
            raise ValueError("未加载问题数据，请先调用 load_problems()")

        if index >= len(self.problems):
            raise ValueError(f"问题索引超出范围: {index} >= {len(self.problems)}")

        # 加载问题数据
        problem_data = self.problems[index]
        if len(problem_data) < 4:
            raise ValueError(f"问题 {index} 数据格式错误：期望至少4个元素")

        obstacles, start, goal, path = problem_data[:4]

        # 验证数据格式
        if not isinstance(obstacles, list):
            raise ValueError(f"问题 {index} 障碍物数据格式错误")

        for i, obs in enumerate(obstacles):
            if not isinstance(obs, tuple) or len(obs) != 2:
                raise ValueError(
                    f"问题 {index} 障碍物 {i} 格式错误：期望 (halfExtents, position)"
                )

        # 转换为numpy数组
        try:
            start = np.array(start, dtype=np.float64)
            goal = np.array(goal, dtype=np.float64)
            path = [np.array(p, dtype=np.float64) for p in path] if path else []
        except (ValueError, TypeError) as e:
            raise ValueError(f"问题 {index} 配置数据类型错误: {e}")

        # 初始化障碍物管理器（如果提供）
        if obstacle_manager:
            obstacle_manager.load_and_init_obstacles_from_data(obstacles)

        # 保存当前问题
        self.current_problem = {
            "obstacles": obstacles,
            "start": start,
            "goal": goal,
            "path": path,
            "index": index,
        }

        return self.current_problem.copy()

    def get_problem(self, width=15, index=None):
        """
        获取问题描述

        Args:
            width: 地图宽度（用于生成体素地图）
            index: 问题索引（如果为None，使用当前问题）

        Returns:
            dict: 问题描述，包含 'map', 'init_state', 'goal_state'

        Raises:
            ValueError: 未初始化问题或索引无效
        """
        if index is not None:
            # 获取缓存的问题
            if index in self.maps:
                return self.maps[index]
            else:
                # 重新初始化问题
                problem = self.init_new_problem(index)
                problem_desc = self._create_problem_description(problem, width)
                self.maps[index] = problem_desc
                return problem_desc
        else:
            # 使用当前问题
            if self.current_problem is None:
                raise ValueError("未初始化问题，请先调用 init_new_problem()")

            current_index = self.current_problem["index"]
            if current_index in self.maps:
                return self.maps[current_index]
            else:
                problem_desc = self._create_problem_description(
                    self.current_problem, width
                )
                self.maps[current_index] = problem_desc
                return problem_desc

    def _create_problem_description(self, problem, width):
        """
        创建问题描述字典

        Args:
            problem: 问题数据字典
            width: 地图宽度

        Returns:
            dict: 问题描述
        """
        # 生成体素地图（如果需要）
        _, obs_map = self._generate_obs_map(problem["obstacles"], width)

        return {
            "map": obs_map.astype(float),
            "init_state": problem["start"],
            "goal_state": problem["goal"],
        }

    def _generate_obs_map(self, obstacles, num):
        """
        生成障碍物的体素网格地图

        Args:
            obstacles: 障碍物列表
            num: 网格分辨率

        Returns:
            np.ndarray: 体素地图 (num, num, num)
        """
        if not obstacles:
            # 生成空的网格
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
            points_obs = np.zeros((num, num, num)).astype(bool)
            return points_pos.reshape((num, num, num, -1)), points_obs

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

        for obstacle in obstacles:
            obstacle_size, obstacle_base = obstacle
            limit_low, limit_high = (
                obstacle_base - obstacle_size,
                obstacle_base + obstacle_size,
            )
            limit_low[2], limit_high[2] = (
                limit_low[2] - 0.4,
                limit_high[2] - 0.4,
            )  # translate the point
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

        return points_pos.reshape((num, num, num, -1)), points_obs.reshape(
            (num, num, num)
        )

    def get_current_problem(self):
        """
        获取当前问题

        Returns:
            dict or None: 当前问题数据
        """
        return self.current_problem.copy() if self.current_problem else None

    def get_problem_count(self):
        """
        获取问题总数

        Returns:
            int: 问题数量
        """
        return len(self.problems) if self.problems else 0

    def reset(self):
        """
        重置问题管理器状态
        """
        self.current_problem = None
        self.maps.clear()
