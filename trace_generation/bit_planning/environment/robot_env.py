import numpy as np
import pybullet as p
import pybullet_data
import pickle
from time import time
from environment.timer import Timer


class RobotEnv:
    """
    Interface class for maze environment
    """

    RRT_EPS = 0.25
    voxel_r = 0.1

    def __init__(
        self,
        GUI=False,
        robot_file="kuka_iiwa/model_0.urdf",
        map_file="maze_files/kukas_7_3000.pkl",
        z_offset=0.0,
        config_output_file=None,
    ):
        # print("Initializing environment...")

        self.dim = 3
        self.robot_file = robot_file
        self.z_offset = z_offset
        self.config_output_file = config_output_file

        self.collision_check_count = 0
        self.collision_time = 0

        self.maps = {}
        self.episode_i = 0
        self.collision_point = None

        if GUI:
            p.connect(
                p.GUI,
                options="--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0",
            )
        else:
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition=[0, 0, 0.1])

        with open(map_file, "rb") as f:
            self.problems = pickle.load(f)

        self.timer = Timer()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robotId = p.loadURDF(
            robot_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True
        )
        p.performCollisionDetection()

        self.config_dim = p.getNumJoints(self.robotId)

        self.pose_range = [
            (
                p.getJointInfo(self.robotId, jointId)[8],
                p.getJointInfo(self.robotId, jointId)[9],
            )
            for jointId in range(p.getNumJoints(self.robotId))
        ]
        self.bound = np.array(self.pose_range).T.reshape(-1)

        self.robotEndEffectorIndex = self.config_dim - 1

        p.setGravity(0, 0, -10)

        self.order = list(range(len(self.problems)))
        self.episode_i = 0

    def __str__(self):
        """返回环境的字符串表示，格式为'机器人名称_自由度dof'，用于标识不同的机器人配置"""
        # 从robot_file提取机器人名称
        robot_name = self.robot_file.split("/")[-1].split(".")[0]
        return f"{robot_name}_{self.config_dim}dof"

    def init_new_problem(self, index=None):
        """
        初始化一个新的路径规划问题

        Args:
            index: 问题索引,如果为None则使用当前episode索引

        Returns:
            包含地图、起始状态和目标状态的问题字典
        """
        # 设置问题索引
        if index is None:
            self.index = self.episode_i
        else:
            self.index = index

        # 启动计时器
        self.timer.start()

        # 从问题集中获取障碍物、起点、终点和路径
        obstacles, start, goal, path = self.problems[index]

        # 更新episode索引,循环使用问题集
        self.episode_i += 1
        self.episode_i = (self.episode_i) % len(self.order)

        # 重置碰撞检测计数器和计时
        self.collision_check_count = 0
        self.collision_time = 0

        # 重置仿真环境
        p.resetSimulation()
        # 加载机器人模型
        self.robotId = p.loadURDF(
            self.robot_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True
        )
        # 执行碰撞检测
        p.performCollisionDetection()

        # 重置碰撞点
        self.collision_point = None

        # 设置当前问题的障碍物、起始状态、目标状态和路径
        self.obstacles = obstacles
        self.init_state = start
        self.goal_state = goal
        self.path = path

        # 在仿真环境中创建所有障碍物体素
        for halfExtents, basePosition in obstacles:
            self.create_voxel(halfExtents, basePosition)

        # 结束计时
        self.timer.finish(Timer.CREATE)

        # 返回问题描述
        return self.get_problem()

    def set_random_init_goal(self):
        """随机采样两个无碰撞且不重合的配置作为起点和终点，用于生成随机规划问题"""
        while True:
            points = self.sample_n_points(n=2)
            init, goal = points[0], points[1]
            if np.sum(np.abs(init - goal)) != 0:
                break
        self.init_state, self.goal_state = init, goal

    def aug_path(self):
        """对路径进行增密处理，在路径点之间按RRT_EPS步长插值，生成更平滑的轨迹用于可视化或执行"""
        result = [self.init_state]
        path = np.array(self.path)
        agent = np.array(path[0])
        next_index = 1
        while next_index < len(path):
            if np.linalg.norm(self.path[next_index] - agent) <= self.RRT_EPS:
                agent = path[next_index]
                next_index += 1
            else:
                agent = agent + self.RRT_EPS * (
                    path[next_index] - agent
                ) / np.linalg.norm(path[next_index] - agent)

            result.append(np.array(agent))
        return result

    def get_problem(self, width=15, index=None):
        """返回当前规划问题的字典（包含障碍物地图、起点、终点），如果指定index则从缓存中获取"""
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
        """生成3D网格地图，将工作空间离散化为num×num×num的体素网格，标记每个体素是否被障碍物占据"""
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
            limit_low[2], limit_high[2] = (
                limit_low[2] + self.z_offset,
                limit_high[2] + self.z_offset,
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

    def get_robot_points(self, config, end_point=True):
        """根据给定关节配置获取机器人连杆位置，end_point=True时只返回末端执行器位置，否则返回所有连杆位置"""
        points = []
        for i in range(p.getNumJoints(self.robotId)):
            p.resetJointState(self.robotId, i, config[i])
        if end_point:
            point = p.getLinkState(self.robotId, self.robotEndEffectorIndex)[0]
            point = (point[0], point[1], point[2] + self.z_offset)
            return point
        for effector in range(self.robotEndEffectorIndex + 1):
            point = p.getLinkState(self.robotId, effector)[0]
            point = (point[0], point[1], point[2] + self.z_offset)
            points.append(point)
        return points

    def create_voxel(self, halfExtents, basePosition):
        """在PyBullet仿真环境中创建一个长方体障碍物体素，包含碰撞形状和随机颜色的可视化形状"""
        groundColId = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        groundVisID = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            rgbaColor=np.random.uniform(0, 1, size=3).tolist() + [0.8],
            specularColor=[0.4, 0.4, 0],
            halfExtents=halfExtents,
        )
        groundId = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=groundColId,
            baseVisualShapeIndex=groundVisID,
            basePosition=basePosition,
        )
        return groundId

    def sample_n_points(self, n, need_negative=False):
        """采样n个无碰撞的配置点，need_negative=True时同时返回采样过程中遇到的碰撞配置（用于生成负样本）"""
        if need_negative:
            negative = []
        samples = []
        for i in range(n):
            while True:
                sample = self.uniform_sample()
                # print("sampling")
                if self._state_fp(sample):
                    samples.append(sample)
                    break
                elif need_negative:
                    negative.append(sample)
        if not need_negative:
            return samples
        else:
            return samples, negative

    def uniform_sample(self, n=1):
        """在配置空间的关节限位范围内均匀随机采样n个配置，不进行碰撞检测"""
        self.timer.start()
        sample = np.random.uniform(
            np.array(self.pose_range)[:, 0],
            np.array(self.pose_range)[:, 1],
            size=(n, self.config_dim),
        )
        if n == 1:
            self.timer.finish(Timer.SAMPLE)
            return sample.reshape(-1)
        else:
            self.timer.finish(Timer.SAMPLE)
            return sample

    def distance(self, from_state, to_state):
        """计算两个配置之间的欧几里得距离（L2范数），自动将目标配置裁剪到关节限位范围内"""
        to_state = np.maximum(to_state, np.array(self.pose_range)[:, 0])
        to_state = np.minimum(to_state, np.array(self.pose_range)[:, 1])
        diff = np.abs(to_state - from_state)

        return np.sqrt(np.sum(diff**2, axis=-1))

    def interpolate(self, from_state, to_state, ratio):
        """在两个配置之间进行线性插值，ratio∈[0,1]表示插值比例，结果自动裁剪到关节限位范围"""
        diff = to_state - from_state

        new_state = from_state + diff * ratio
        new_state = np.maximum(new_state, np.array(self.pose_range)[:, 0])
        new_state = np.minimum(new_state, np.array(self.pose_range)[:, 1])

        return new_state

    def in_goal_region(self, state):
        """判断给定配置是否到达目标区域（距离目标小于RRT_EPS且无碰撞），用于路径规划的终止条件"""
        return self.distance(state, self.goal_state) < self.RRT_EPS and self._state_fp(
            state
        )

    def step(self, state, action=None, new_state=None, check_collision=True):
        """执行一步状态转移（state+action或直接到new_state），返回新状态、实际动作、是否无碰撞、是否到达目标"""
        print("in step")
        # must specify either action or new_state
        if action is not None:
            new_state = state + action

        new_state = np.maximum(new_state, np.array(self.pose_range)[:, 0])
        new_state = np.minimum(new_state, np.array(self.pose_range)[:, 1])

        action = new_state - state

        if not check_collision:
            return new_state, action
        done = False
        no_collision = self._edge_fp(state, new_state)
        if no_collision and self.in_goal_region(new_state):
            done = True

        return new_state, action, no_collision, done

    def set_config(self, c, robotId=None):
        """设置指定机器人的关节配置并执行碰撞检测更新，用于可视化或状态设置"""
        if robotId is None:
            robotId = self.robotId
        for i in range(p.getNumJoints(robotId)):
            p.resetJointState(robotId, i, c[i])
        p.performCollisionDetection()

    def plot(self, path, make_gif=False):
        """可视化路径执行过程，沿路径显示半透明机器人姿态和末端轨迹红线，make_gif=True时捕获图像帧"""
        path = np.array(path)

        p.resetSimulation()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)

        for halfExtents, basePosition in self.obstacles:
            self.create_voxel(halfExtents, basePosition)

        self.robotId = p.loadURDF(
            self.robot_file,
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_IGNORE_COLLISION_SHAPES,
        )
        self.pose_range = [
            (
                p.getJointInfo(self.robotId, jointId)[8],
                p.getJointInfo(self.robotId, jointId)[9],
            )
            for jointId in range(p.getNumJoints(self.robotId))
        ]
        self.bound = np.array(self.pose_range).T.reshape(-1)

        self.set_config(path[0])

        target_robotId = p.loadURDF(
            self.robot_file,
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_IGNORE_COLLISION_SHAPES,
        )
        self.set_config(path[-1], target_robotId)

        prev_pos = p.getLinkState(self.robotId, self.robotEndEffectorIndex)[0]
        final_pos = p.getLinkState(target_robotId, self.robotEndEffectorIndex)[0]

        p.setGravity(0, 0, -10)
        p.stepSimulation()

        gifs = []
        current_state_idx = 0

        while True:
            current_state = path[current_state_idx]
            disp = path[current_state_idx + 1] - path[current_state_idx]

            d = self.distance(path[current_state_idx], path[current_state_idx + 1])

            new_robot = p.loadURDF(
                self.robot_file,
                [0, 0, 0],
                [0, 0, 0, 1],
                useFixedBase=True,
                flags=p.URDF_IGNORE_COLLISION_SHAPES,
            )
            for data in p.getVisualShapeData(new_robot):
                color = list(data[-1])
                color[-1] = 0.5
                p.changeVisualShape(new_robot, data[1], rgbaColor=color)

            K = int(np.ceil(d / 0.2))
            for k in range(0, K):
                c = path[current_state_idx] + k * 1.0 / K * disp
                self.set_config(c, new_robot)
                new_pos = p.getLinkState(new_robot, self.robotEndEffectorIndex)[0]
                p.addUserDebugLine(prev_pos, new_pos, [1, 0, 0], 10, 0)
                prev_pos = new_pos
                p.loadURDF(
                    "sphere2red.urdf",
                    new_pos,
                    globalScaling=0.05,
                    flags=p.URDF_IGNORE_COLLISION_SHAPES,
                )
                if make_gif:
                    gifs.append(
                        p.getCameraImage(
                            width=1080,
                            height=720,
                            lightDirection=[0, 0, -1],
                            shadow=0,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                        )[2]
                    )

            current_state_idx += 1
            if current_state_idx == len(path) - 1:
                self.set_config(path[-1], new_robot)
                p.addUserDebugLine(prev_pos, final_pos, [1, 0, 0], 10, 0)
                p.loadURDF(
                    "sphere2red.urdf",
                    final_pos,
                    globalScaling=0.05,
                    flags=p.URDF_IGNORE_COLLISION_SHAPES,
                )
                break

        return gifs

    # =====================internal collision check module=======================

    def _valid_state(self, state):
        """检查配置是否在关节限位范围内（内部方法）"""
        return (state >= np.array(self.pose_range)[:, 0]).all() and (
            state <= np.array(self.pose_range)[:, 1]
        ).all()

    def _point_in_free_space(self, state):
        """检查单个配置是否无碰撞（内部方法），使用PyBullet碰撞检测，统计检测次数和耗时"""
        # print("here")
        if self.config_output_file is not None:
            with open(self.config_output_file, "a") as f:
                f.write(" ".join(map(str, state)) + "\n")
        t0 = time()
        if not self._valid_state(state):
            return False
        # print(state)
        for i in range(p.getNumJoints(self.robotId)):
            p.resetJointState(self.robotId, i, state[i])
        # print(len(p.getContactPoints(self.robotId,self.robotEndEffectorIndex)))
        p.performCollisionDetection()
        if len(p.getContactPoints(self.robotId)) == 0:
            self.collision_check_count += 1
            self.collision_time += time() - t0
            return True
        else:
            self.collision_point = state
            self.collision_check_count += 1
            self.collision_time += time() - t0
            # print("False")
            return False

    def _state_fp(self, state):
        """点碰撞检测的公开接口，调用_point_in_free_space并记录计时"""
        self.timer.start()
        free = self._point_in_free_space(state)
        self.timer.finish(Timer.VERTEX_CHECK)
        return free

    def _iterative_check_segment(self, left, right):
        """递归二分检查线段碰撞（未使用的备用方法），在两点距离大于0.1时递归检查中点"""
        print("iterative")
        if np.sum(np.abs(left - left)) > 0.1:
            mid = (left + right) / 2.0
            self.k += 1
            if not self._state_fp(mid):
                self.collision_point = mid
                return False
            return self._iterative_check_segment(
                left, mid
            ) and self._iterative_check_segment(mid, right)

        return True

    def _edge_fp(self, state, new_state):
        """边碰撞检测，在两个配置之间按RRT_EPS步长插值检查，返回整条路径是否无碰撞"""
        # print("in edge fp")
        self.timer.start()
        self.k = 0
        assert state.size == new_state.size

        collanswer = True
        if not self._valid_state(state) or not self._valid_state(new_state):
            self.timer.finish(Timer.EDGE_CHECK)
            return False
        # print("2")
        if not self._point_in_free_space(state) or not self._point_in_free_space(
            new_state
        ):
            self.timer.finish(Timer.EDGE_CHECK)
            collanswer = False
            # return False

        disp = new_state - state

        d = self.distance(state, new_state)
        # print(d,self.RRT_EPS)
        K = int(d / self.RRT_EPS)
        # print("Edge test start",K)
        for k in range(0, K):
            c = state + k * 1.0 / K * disp
            if not self._point_in_free_space(c):
                # self.timer.finish(Timer.EDGE_CHECK)
                collanswer = False
                # print("here")
                # return False
        # print(K)
        # print("Edge test",K,collanswer)
        self.timer.finish(Timer.EDGE_CHECK)
        return collanswer  # True
