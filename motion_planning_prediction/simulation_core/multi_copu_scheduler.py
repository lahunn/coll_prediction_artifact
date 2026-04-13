"""
多COPU系统调度器模块
"""

from collections import deque
from .constants import NUM_OOCDS, DEFAULT_QCOLL_LEN, DEFAULT_QNONCOLL_LEN
from .copu_module import COPUModule
from .cht_access_scheduler import CHT_AccessScheduler
from .data_preprocessing import allocate_edge_data_to_copus


class MultiCOPU_Scheduler:
    """
    多COPU系统调度器

    职责：
    - 管理多个COPU模块
    - 协调CHT访问
    - 同步各COPU的进度
    - 收集全局结果
    - 动态分配任务（Edge）给COPU组
    """

    def __init__(
        self,
        num_copus,
        num_oocds=NUM_OOCDS,
        cht_size=4096,
        qcoll_size=DEFAULT_QCOLL_LEN,
        qnoncoll_size=DEFAULT_QNONCOLL_LEN,
        enable_conflict_check=True,
        cht_type="dual_port",
        copus_per_edge=None,
        num_predictions=1,
        **cht_kwargs,
    ):
        self.num_copus = num_copus
        self.copus_per_edge = copus_per_edge if copus_per_edge else num_copus
        self.num_groups = max(1, num_copus // self.copus_per_edge)
        self.num_predictions = num_predictions

        # 创建共享的CHT调度器
        self.cht_scheduler = CHT_AccessScheduler(
            num_copus,
            cht_size,
            enable_conflict_check=enable_conflict_check,
            cht_type=cht_type,
            **cht_kwargs,
        )

        # 创建COPU模块
        self.copus = [
            COPUModule(
                copu_id=i,
                num_oocds=num_oocds,
                qcoll_size=qcoll_size,
                qnoncoll_size=qnoncoll_size,
                cht_scheduler=self.cht_scheduler,
                num_predictions=num_predictions,
            )
            for i in range(num_copus)
        ]

        self.cycle = 0
        self.global_coll_found = False

        # 任务管理
        self.edge_queue = deque()
        self.edge_results = {}  # edge_idx -> result ('collision', 'safe')

        # group_status 结构变更：每个组维护一个prediction状态列表
        # [{'edge_idx': -1, 'finished': [False, False, ...]} for _ in range(num_predictions)]
        self.group_status = [
            [
                {
                    "edge_idx": -1,
                    "finished": [False] * (self.copus_per_edge),
                    "assign_cycle": 0,
                    "active_cycle": None,
                    "first_dispatch_cycle": None,
                }
                for _ in range(num_predictions)
            ]
            for _ in range(self.num_groups)
        ]

        # 等待周期统计：任务进入队列到首次CDU执行
        self.total_wait_cycles = 0
        self.total_wait_samples = 0
        self.total_dead_ratio_sum = 0.0
        self.total_dead_ratio_samples = 0

        # 原始数据存储
        self.all_data = []
        self.all_coll = []
        self.all_cycles = []

    def set_benchmark_data(self, all_data, all_coll, all_cycles):
        """
        设置benchmark数据并初始化任务队列
        """
        self.all_data = all_data
        self.all_coll = all_coll
        self.all_cycles = all_cycles
        self.edge_queue = deque(range(len(all_data)))
        self.edge_results = {}
        self.total_wait_cycles = 0
        self.total_wait_samples = 0
        self.total_dead_ratio_sum = 0.0
        self.total_dead_ratio_samples = 0

        # 重置组状态
        for i in range(self.num_groups):
            self.group_status[i] = [
                {
                    "edge_idx": -1,
                    "finished": [False] * self.copus_per_edge,
                    "assign_cycle": 0,
                    "active_cycle": None,
                    "first_dispatch_cycle": None,
                }
                for _ in range(self.num_predictions)
            ]

        # 重置所有COPU
        for copu in self.copus:
            copu.reset_task()

    def load_warmstart_package(self, warmstart_package):
        """将warm-start包注入到底层CHT调度器。"""
        self.cht_scheduler.load_warmstart_package(warmstart_package)

    def _has_active_tasks(self):
        """检查是否还有活跃任务"""
        for group in self.group_status:
            for prediction_status in group:
                if prediction_status["edge_idx"] != -1:
                    return True
        return False

    def simulate(self, bins, threshold, sample_rate):
        """
        执行多COPU协同仿真（动态调度）

        Returns:
            results dict with global metrics and per-COPU stats
        """
        # 预加载阶段：尝试填满所有组的所有prediction
        # 关键：先按prediction分组，再按group分配，确保同一prediction的edges并行执行
        for prediction_idx in range(self.num_predictions):
            for group_id in range(self.num_groups):
                if self.edge_queue:
                    edge_idx = self.edge_queue.popleft()
                    self._assign_edge_to_group(group_id, edge_idx, prediction_idx)
        # 主仿真循环
        while True:
            # 1. 每个COPU执行一步
            any_copu_active = False
            for copu in self.copus:
                finished = copu.step(bins, threshold, sample_rate)
                any_copu_active = not finished or any_copu_active
                if finished:
                    # 标记该COPU的任务完成
                    group_id = copu.copu_id // self.copus_per_edge
                    copu_idx_in_group = copu.copu_id % self.copus_per_edge
                    active_idx = copu.active_idx
                    self.group_status[group_id][active_idx]["finished"][
                        copu_idx_in_group
                    ] = True

            # 1.5 记录每个活跃edge首次被CDU执行的周期
            self._update_first_dispatch_cycles()

            # 2. 检查组任务完成情况和碰撞情况
            for group_id in range(self.num_groups):
                self._check_group_status(group_id)

            # 3. 推进CHT调度器
            self.cht_scheduler.advance_cycle()

            # 如果所有COPU都空闲且队列为空，则退出
            if (
                not any_copu_active
                and not self._has_active_tasks()
                and not self.edge_queue
            ):
                break

            self.cycle += 1

            # Check cycle synchronization
            # for copu in self.copus:
            #     assert copu.cycle == self.cycle, (
            #         f"COPU {copu.copu_id} cycle mismatch: {copu.cycle} != {self.cycle}"
            #     )
            # assert self.cht_scheduler.cht.current_cycle == self.cycle, (
            #     f"CHT cycle mismatch: {self.cht_scheduler.cht.current_cycle} != {self.cycle}"
            # )

        # 收集统计
        # 检查是否有任何edge发生了碰撞
        self.global_coll_found = any(
            res == "collision" for res in self.edge_results.values()
        )

        results = {
            "total_cycles": self.cycle,
            "collision_found": self.global_coll_found,
            "copus": [copu.get_stats() for copu in self.copus],
            "cht_stats": self.cht_scheduler.get_stats(),
            "edge_results": self.edge_results,
            "total_wait_cycles": self.total_wait_cycles,
            "total_wait_samples": self.total_wait_samples,
            "avg_wait_cycles": (
                self.total_wait_cycles / self.total_wait_samples
                if self.total_wait_samples > 0
                else 0.0
            ),
            "total_dead_ratio_sum": self.total_dead_ratio_sum,
            "total_dead_ratio_samples": self.total_dead_ratio_samples,
            "dead_avg_ratio": (
                (self.total_dead_ratio_sum / self.total_dead_ratio_samples * 100.0)
                if self.total_dead_ratio_samples > 0
                else 0.0
            ),
        }

        return results

    def _assign_edge_to_group(self, group_id, edge_idx, prediction_idx):
        """将edge分配给指定COPU组的指定prediction"""
        edge_data = self.all_data[edge_idx]
        edge_coll = self.all_coll[edge_idx]
        edge_cycle = self.all_cycles[edge_idx] if self.all_cycles else None

        sub_coords, sub_colls, sub_cycles = allocate_edge_data_to_copus(
            edge_data, edge_coll, edge_cycle, self.copus_per_edge
        )

        start_copu = group_id * self.copus_per_edge
        # 获取组内COPU的当前active_idx (假设组内所有COPU同步)
        current_active_idx = self.copus[start_copu].active_idx

        for i in range(self.copus_per_edge):
            copu_id = start_copu + i
            self.copus[copu_id].load_data(
                sub_coords[i], sub_colls[i], sub_cycles[i], prediction_idx, edge_idx
            )

        self.group_status[group_id][prediction_idx] = {
            "edge_idx": edge_idx,
            "finished": [False] * self.copus_per_edge,
            "assign_cycle": self.cycle,
            "active_cycle": self.cycle if prediction_idx == current_active_idx else None,
            "first_dispatch_cycle": None,
        }

    def _finish_task(self, group_id, prediction_idx, group_copus):
        """完成一个任务的处理：重置状态、停止COPU任务、重置OOCD、加载新任务"""
        task_state = self.group_status[group_id][prediction_idx]
        # 使用 active_cycle 代替 assign_cycle 进行统计，更准确反映活跃时的等待情况
        active_cycle = task_state.get("active_cycle")
        if active_cycle is None:
            active_cycle = task_state.get("assign_cycle", self.cycle)

        first_dispatch_cycle = task_state.get("first_dispatch_cycle")
        if first_dispatch_cycle is None:
            wait_cycles = 0
        else:
            wait_cycles = max(0, first_dispatch_cycle - active_cycle)
        self.total_wait_cycles += wait_cycles
        self.total_wait_samples += 1

        # 真实 per-edge dead ratio：按任务从激活到完成的周期归一化
        task_cycles = max(1, self.cycle - active_cycle + 1)
        dead_ratio = wait_cycles / task_cycles
        self.total_dead_ratio_sum += dead_ratio
        self.total_dead_ratio_samples += 1

        # 重置该prediction状态
        self.group_status[group_id][prediction_idx] = {
            "edge_idx": -1,
            "finished": [False] * self.copus_per_edge,
            "assign_cycle": 0,
            "active_cycle": None,
            "first_dispatch_cycle": None,
        }

        # 停止组内所有COPU在该prediction的任务
        for c in group_copus:
            c.reset_prediction(prediction_idx)

        # 重置所有OOCD状态
        for c in group_copus:
            for oocd in c.oocds:
                oocd.reset()

        # 加载新任务到该prediction (如果队列不为空)
        if self.edge_queue:
            new_edge_idx = self.edge_queue.popleft()
            self._assign_edge_to_group(group_id, new_edge_idx, prediction_idx)

        # 切换该组所有COPU到下一个prediction
        new_active_idx = (prediction_idx + 1) % self.num_predictions
        for c in group_copus:
            c.active_idx = new_active_idx

        # 标记新切换到的任务为 active (开始记录其活跃时间)
        if self.group_status[group_id][new_active_idx]["edge_idx"] != -1:
            self.group_status[group_id][new_active_idx]["active_cycle"] = self.cycle

    def _update_first_dispatch_cycles(self):
        """记录每个活跃edge首次有CDU开始执行的周期。"""
        for group_id in range(self.num_groups):
            for prediction_idx in range(self.num_predictions):
                status = self.group_status[group_id][prediction_idx]
                if status["edge_idx"] == -1 or status["first_dispatch_cycle"] is not None:
                    continue

                start_copu = group_id * self.copus_per_edge
                group_copus = self.copus[start_copu : start_copu + self.copus_per_edge]

                dispatched = False
                for copu in group_copus:
                    for oocd in copu.oocds:
                        if oocd.busy == 1 and oocd.free_cycle > self.cycle:
                            dispatched = True
                            break
                    if dispatched:
                        break

                if dispatched:
                    status["first_dispatch_cycle"] = self.cycle

    def _check_group_status(self, group_id):
        """检查组内任务是否完成或发现碰撞，并管理active_idx"""
        start_copu = group_id * self.copus_per_edge
        group_copus = self.copus[start_copu : start_copu + self.copus_per_edge]
        active_idx = group_copus[0].active_idx
        edge_idx = self.group_status[group_id][active_idx]["edge_idx"]

        if edge_idx == -1:
            return  # 该prediction未分配任务

        # 检查碰撞
        if any(copu.coll_found for copu in group_copus):
            self.edge_results[edge_idx] = "collision"
            self._finish_task(group_id, active_idx, group_copus)
            return

        # 检查完成
        if all(self.group_status[group_id][active_idx]["finished"]):
            self.edge_results[edge_idx] = "safe"
            self._finish_task(group_id, active_idx, group_copus)
