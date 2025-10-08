import numpy as np
import math
import heapq
import time
from time import time
from environment.timer import Timer
import pickle

INF = float("inf")


class BITStar:
    def __init__(
        self,
        environment,
        maxIter=5,
        plot_flag=False,
        batch_size=200,
        T=1000,
        sampling=None,
        timer=None,
    ):
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer

        self.env = environment

        start, goal, bounds = (
            tuple(environment.init_state),
            tuple(environment.goal_state),
            environment.bound,
        )

        self.start = start
        self.goal = goal
        # print(start,goal)
        self.bounds = bounds
        self.bounds = np.array(self.bounds).reshape((2, -1)).T
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]
        self.dimension = environment.config_dim

        # This is the tree
        self.vertices = []
        self.edges = dict()  # key = point，value = parent
        self.g_scores = dict()

        self.samples = []
        self.vertex_queue = []
        self.edge_queue = []
        self.old_vertices = set()

        self.maxIter = maxIter
        self.r = INF
        self.batch_size = batch_size
        self.T, self.T_max = 0, T
        self.eta = 1.1  # tunable parameter
        self.obj_radius = 1
        self.resolution = 3

        # the parameters for informed sampling
        self.c_min = self.distance(self.start, self.goal)
        self.center_point = None
        self.C = None

        # whether plot the middle planning process
        self.plot_planning_process = plot_flag

        if sampling is None:
            self.sampling = self.informed_sample
            print("informed sampling")
        else:
            self.sampling = sampling

        self.n_collision_points = 0
        self.n_free_points = 2
        print("init of BITStart done")

    def setup_planning(self):
        # add goal to the samples
        self.samples.append(self.goal)
        self.g_scores[self.goal] = INF

        # add start to the tree
        self.vertices.append(self.start)
        self.g_scores[self.start] = 0

        # Computing the sampling space
        self.informed_sample_init()
        radius_constant = self.radius_init()

        return radius_constant

    def radius_init(self):
        from scipy import special

        # Hypersphere radius calculation
        n = self.dimension
        unit_ball_volume = np.pi ** (n / 2.0) / special.gamma(n / 2.0 + 1)
        volume = (
            np.abs(np.prod(self.ranges))
            * self.n_free_points
            / (self.n_collision_points + self.n_free_points)
        )
        gamma = (1.0 + 1.0 / n) * volume / unit_ball_volume
        radius_constant = 2 * self.eta * (gamma ** (1.0 / n))
        return radius_constant

    def informed_sample_init(self):
        self.center_point = np.array(
            [(self.start[i] + self.goal[i]) / 2.0 for i in range(self.dimension)]
        )
        a_1 = (np.array(self.goal) - np.array(self.start)) / self.c_min
        id1_t = np.array([1.0] * self.dimension)
        M = np.dot(a_1.reshape((-1, 1)), id1_t.reshape((1, -1)))
        U, S, Vh = np.linalg.svd(M, 1, 1)
        self.C = np.dot(
            np.dot(
                U,
                np.diag(
                    [1] * (self.dimension - 1)
                    + [np.linalg.det(U) * np.linalg.det(np.transpose(Vh))]
                ),
            ),
            Vh,
        )

    def sample_unit_ball(self):
        u = np.random.normal(
            0, 1, self.dimension
        )  # an array of d normally distributed random variables
        norm = np.sum(u**2) ** (0.5)
        r = np.random.random() ** (1.0 / self.dimension)
        x = r * u / norm
        return x

    def informed_sample(self, c_best, sample_num, vertices):
        edge_info = []
        edge_info_coll = []
        if c_best < float("inf"):
            c_b = math.sqrt(c_best**2 - self.c_min**2) / 2.0
            r = [c_best / 2.0] + [c_b] * (self.dimension - 1)
            L = np.diag(r)
        sample_array = []
        cur_num = 0
        while cur_num < sample_num:
            if c_best < float("inf"):
                x_ball = self.sample_unit_ball()
                random_point = tuple(
                    np.dot(np.dot(self.C, L), x_ball) + self.center_point
                )
            else:
                random_point = self.get_random_point()
            # print("sampling in informed sample")
            feas, linkinfo, linkinfo_coll = self.is_point_free(random_point)
            edge_info.append([linkinfo])
            edge_info_coll.append([linkinfo_coll])
            if feas:
                sample_array.append(random_point)
                cur_num += 1

        return sample_array, edge_info, edge_info_coll

    def get_random_point(self):
        point = self.bounds[:, 0] + np.random.random(self.dimension) * self.ranges
        return tuple(point)

    def is_point_free(self, point):
        result, info, coll = self.env._state_fp_probe(np.array(point))
        if result:
            self.n_free_points += 1
        else:
            self.n_collision_points += 1
        return result, info, coll

    def is_edge_free(self, edge):
        result, info, coll = self.env._edge_fp_probe(
            np.array(edge[0]), np.array(edge[1])
        )
        # self.T += self.env.k
        return result, info, coll

    def get_g_score(self, point):
        # gT(x)
        if point == self.start:
            return 0
        if point not in self.edges:
            return INF
        else:
            return self.g_scores.get(point)

    def get_f_score(self, point):
        # f^(x)
        return self.heuristic_cost(self.start, point) + self.heuristic_cost(
            point, self.goal
        )

    def actual_edge_cost(self, point1, point2):
        # c(x1,x2)
        feas, info, coll = self.is_edge_free([point1, point2])
        if not feas:
            return INF, info, coll
        return self.distance(point1, point2), info, coll

    def heuristic_cost(self, point1, point2):
        # Euler distance as the heuristic distance
        return self.distance(point1, point2)

    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def get_edge_value(self, edge):
        # sort value for edge
        return (
            self.get_g_score(edge[0])
            + self.heuristic_cost(edge[0], edge[1])
            + self.heuristic_cost(edge[1], self.goal)
        )

    def get_point_value(self, point):
        # sort value for point
        return self.get_g_score(point) + self.heuristic_cost(point, self.goal)

    def bestVertexQueueValue(self):
        if not self.vertex_queue:
            return INF
        else:
            return self.vertex_queue[0][0]

    def bestEdgeQueueValue(self):
        if not self.edge_queue:
            return INF
        else:
            return self.edge_queue[0][0]

    def prune_edge(self, c_best):
        edge_array = list(self.edges.items())
        for point, parent in edge_array:
            if self.get_f_score(point) > c_best or self.get_f_score(parent) > c_best:
                self.edges.pop(point)

    def prune(self, c_best):
        self.samples = [
            point for point in self.samples if self.get_f_score(point) < c_best
        ]
        self.prune_edge(c_best)
        vertices_temp = []
        for point in self.vertices:
            if self.get_f_score(point) <= c_best:
                if self.get_g_score(point) == INF:
                    self.samples.append(point)
                else:
                    vertices_temp.append(point)
        self.vertices = vertices_temp

    def expand_vertex(self, point):
        self.timer.start()

        # get the nearest value in vertex for every one in samples where difference is less than the radius
        neigbors_sample = []
        for sample in self.samples:
            if self.distance(point, sample) <= self.r:
                neigbors_sample.append(sample)

        self.timer.finish(Timer.NN)

        self.timer.start()

        # add an edge to the edge queue is the path might improve the solution
        for neighbor in neigbors_sample:
            estimated_f_score = (
                self.heuristic_cost(self.start, point)
                + self.heuristic_cost(point, neighbor)
                + self.heuristic_cost(neighbor, self.goal)
            )
            if estimated_f_score < self.g_scores[self.goal]:
                heapq.heappush(
                    self.edge_queue,
                    (self.get_edge_value((point, neighbor)), (point, neighbor)),
                )

        # add the vertex to the edge queue
        if point not in self.old_vertices:
            neigbors_vertex = []
            for ver in self.vertices:
                if self.distance(point, ver) <= self.r:
                    neigbors_vertex.append(ver)
            for neighbor in neigbors_vertex:
                if neighbor not in self.edges or point != self.edges.get(neighbor):
                    estimated_f_score = (
                        self.heuristic_cost(self.start, point)
                        + self.heuristic_cost(point, neighbor)
                        + self.heuristic_cost(neighbor, self.goal)
                    )
                    if estimated_f_score < self.g_scores[self.goal]:
                        estimated_g_score = self.get_g_score(
                            point
                        ) + self.heuristic_cost(point, neighbor)
                        if estimated_g_score < self.get_g_score(neighbor):
                            heapq.heappush(
                                self.edge_queue,
                                (
                                    self.get_edge_value((point, neighbor)),
                                    (point, neighbor),
                                ),
                            )

        self.timer.finish(Timer.EXPAND)

    def get_best_path(self):
        path = []
        if self.g_scores[self.goal] != INF:
            path.append(self.goal)
            point = self.goal
            while point != self.start:
                point = self.edges[point]
                path.append(point)
            path.reverse()
        return path

    def path_length_calculate(self, path):
        path_length = 0
        for i in range(len(path) - 1):
            path_length += self.distance(path[i], path[i + 1])
        return path_length

    def plan(
        self,
        pathLengthLimit,
        problemindex=2000,
        refine_time_budget=None,
        time_budget=None,
    ):
        """
        BIT*算法的主规划函数，执行批量增量树搜索以找到最优路径

        参数:
            pathLengthLimit: 路径长度限制，当找到的路径长度小于此值时可提前终止
            problemindex: 问题索引，用于保存边信息文件的命名
            refine_time_budget: 精化时间预算(秒)，找到可行路径后继续优化的最小时间
            time_budget: 总时间预算(秒)，整个规划过程的最大时间限制

        返回:
            tuple: (采样点集, 边字典, 碰撞检测次数, 最终路径代价, 采样总数, 规划用时)

        工作流程:
            1. 初始化规划环境和时间记录
            2. 迭代执行批量采样和树扩展
            3. 维护顶点队列和边队列进行最优搜索
            4. 保存边信息和碰撞信息到pickle文件
            5. 返回规划结果
        """
        # 记录初始碰撞检测次数，用于计算本次规划的碰撞检测开销
        collision_checks = self.env.collision_check_count
        print("collision_checks", collision_checks)

        # 设置时间预算的默认值
        if time_budget is None:
            time_budget = INF  # 无限时间预算
        if refine_time_budget is None:
            refine_time_budget = 10  # 默认精化时间10秒

        # 初始化规划环境：添加起点和终点，设置采样空间
        print("Before planning setup")
        self.setup_planning()
        init_time = time()  # 记录规划开始时间

        # BIT*主循环：在采样预算和时间预算内迭代
        while self.T < self.T_max and (time() - init_time < time_budget):
            # 当两个队列都为空时，进行新一轮批量采样
            if not self.vertex_queue and not self.edge_queue:
                # 获取当前最优路径代价，用于informed sampling
                c_best = self.g_scores[self.goal]

                # 剪枝：移除不可能改进当前解的顶点和边
                self.prune(c_best)

                # 批量采样：生成新样本点，并记录采样过程的边信息
                sample_temp, edge_info_full, edge_infocoll_full = self.sampling(
                    c_best, self.batch_size, self.vertices
                )
                # print(edge_info_full)
                self.samples.extend(sample_temp)
                self.T += self.batch_size  # 更新总采样数

                self.timer.start()
                # 记录旧顶点集，用于区分新旧顶点
                self.old_vertices = set(self.vertices)

                # 构建顶点优先队列：按照f值(g+h)排序
                self.vertex_queue = [
                    (self.get_point_value(point), point) for point in self.vertices
                ]
                heapq.heapify(self.vertex_queue)  # 转换为最小堆

                # 动态更新连接半径r：随着样本数增加而减小
                q = len(self.vertices) + len(self.samples)
                self.r = self.radius_init() * (
                    (math.log(q) / q) ** (1.0 / self.dimension)
                )
                self.timer.finish(Timer.HEAP)

            # 扩展顶点：当顶点队列最优值优于边队列时，优先扩展顶点
            try:
                while self.bestVertexQueueValue() <= self.bestEdgeQueueValue():
                    self.timer.start()
                    _, point = heapq.heappop(self.vertex_queue)  # 取出最优顶点
                    self.timer.finish(Timer.HEAP)
                    self.expand_vertex(point)  # 扩展该顶点，生成新的候选边
            except Exception as e:
                # 如果两个队列都为空，进入下一轮采样
                if (not self.edge_queue) and (not self.vertex_queue):
                    continue
                else:
                    raise e

            # 从边队列中取出最优边进行评估
            best_edge_value, bestEdge = heapq.heappop(self.edge_queue)

            # 检查该边是否可能改进当前解
            if best_edge_value < self.g_scores[self.goal]:
                # 计算边的实际代价（进行碰撞检测）
                # print("actual cost of edge",bestEdge[0],bestEdge[1])
                actual_cost_of_edge, edgeinfo, edgeinfo_coll = self.actual_edge_cost(
                    bestEdge[0], bestEdge[1]
                )
                # 保存边的检测信息，用于后续数据分析
                # print(edgeinfo)
                edge_info_full.append(edgeinfo)
                edge_infocoll_full.append(edgeinfo_coll)

                self.timer.start()
                # 计算通过该边的实际f值
                actual_f_edge = (
                    self.heuristic_cost(self.start, bestEdge[0])
                    + actual_cost_of_edge
                    + self.heuristic_cost(bestEdge[1], self.goal)
                )

                # 如果该边确实能改进解，则更新树结构
                if actual_f_edge < self.g_scores[self.goal]:
                    # 计算目标点的实际g值
                    actual_g_score_of_point = (
                        self.get_g_score(bestEdge[0]) + actual_cost_of_edge
                    )

                    # 如果找到了更优路径，则更新
                    if actual_g_score_of_point < self.get_g_score(bestEdge[1]):
                        self.g_scores[bestEdge[1]] = actual_g_score_of_point
                        self.edges[bestEdge[1]] = bestEdge[0]  # 更新父节点

                        # 如果目标点是新顶点，将其从样本中移到树中
                        if bestEdge[1] not in self.vertices:
                            self.samples.remove(bestEdge[1])
                            self.vertices.append(bestEdge[1])
                            heapq.heappush(
                                self.vertex_queue,
                                (self.get_point_value(bestEdge[1]), bestEdge[1]),
                            )

                        # 清理边队列：移除不再有效的边
                        # 保留的边需满足：终点不是bestEdge[1] 或 估计代价小于起点g值
                        self.edge_queue = [
                            item
                            for item in self.edge_queue
                            if item[1][1] != bestEdge[1]
                            or self.get_g_score(item[1][0])
                            + self.heuristic_cost(item[1][0], item[1][1])
                            < self.get_g_score(item[1][0])
                        ]
                        heapq.heapify(
                            self.edge_queue
                        )  # 重建优先队列（删除元素后需要重新堆化）

                self.timer.finish(Timer.HEAP)

            else:
                # 最优边无法改进当前解，清空队列进入下一轮采样
                self.vertex_queue = []
                self.edge_queue = []

            # 提前终止条件：找到满足长度限制的路径且达到精化时间预算
            if self.g_scores[self.goal] < pathLengthLimit and (
                time() - init_time > refine_time_budget
            ):
                break

        # 保存规划过程中的边信息到pickle文件
        # 包含两部分：边的详细信息(edgeinfo)和碰撞检测信息(edgeinfo_coll)
        f = open("logfiles_BIT_link/link_info_" + str(problemindex) + ".pkl", "wb")
        # print(edge_infocoll_full)
        pickle.dump((edge_info_full, edge_infocoll_full), f)
        f.close()

        # 返回规划结果
        return (
            self.samples,  # 所有采样点
            self.edges,  # 树的边字典(child->parent)
            self.env.collision_check_count - collision_checks,  # 本次规划的碰撞检测次数
            self.g_scores[self.goal],  # 最终路径代价
            self.T,  # 总采样数
            time() - init_time,  # 规划用时(秒)
        )


if __name__ == "__main__":
    from utils.plot import plot_edges
    from config import set_random_seed
    from environment import MazeEnv
    from tqdm import tqdm

    solutions = []

    environment = MazeEnv(dim=2)

    def sample_empty_points(env):
        while True:
            point = np.random.uniform(-1, 1, 2)
            if env._state_fp(point):
                return point

    for _ in tqdm(range(3000)):
        pb = environment.init_new_problem()
        set_random_seed(1234)

        cur_time = time.time()

        BIT = BITStar(environment)
        nodes, edges, collision, success, n_samples = BIT.plan(INF)

        solutions.append((nodes, edges, collision, success, n_samples))

        plot_edges(set(nodes) | set(edges.keys()), edges, environment.get_problem())

    print("hello")
