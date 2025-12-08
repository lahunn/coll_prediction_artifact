"""
多COPU系统调度器模块
"""

from collections import deque
from .constants import NUM_OOCDS
from .copu_module import COPUModule
from .cht_access_scheduler import CHT_AccessScheduler
import simulation_utils as su


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
                {"edge_idx": -1, "finished": [False] * (self.copus_per_edge)}
                for _ in range(num_predictions)
            ]
            for _ in range(self.num_groups)
        ]

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

        # 重置组状态
        for i in range(self.num_groups):
            self.group_status[i] = [
                {"edge_idx": -1, "finished": [False] * self.copus_per_edge}
                for _ in range(self.num_predictions)
            ]

        # 重置所有COPU
        for copu in self.copus:
            copu.reset_task()

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
        for group_id in range(self.num_groups):
            for prediction_idx in range(self.num_predictions):
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
            "cht_stats": self.cht_scheduler.cht.get_stats(),
            "edge_results": self.edge_results,
        }

        return results

    def _assign_edge_to_group(self, group_id, edge_idx, prediction_idx):
        """将edge分配给指定COPU组的指定prediction"""
        edge_data = self.all_data[edge_idx]
        edge_coll = self.all_coll[edge_idx]
        edge_cycle = self.all_cycles[edge_idx] if self.all_cycles else None

        sub_coords, sub_colls, sub_cycles = su.allocate_edge_data_to_copus(
            edge_data, edge_coll, edge_cycle, self.copus_per_edge
        )

        start_copu = group_id * self.copus_per_edge
        for i in range(self.copus_per_edge):
            copu_id = start_copu + i
            self.copus[copu_id].load_data(
                sub_coords[i], sub_colls[i], sub_cycles[i], prediction_idx, edge_idx
            )

        self.group_status[group_id][prediction_idx] = {
            "edge_idx": edge_idx,
            "finished": [False] * self.copus_per_edge,
        }

    def _finish_prediction(self, group_id, prediction_idx, group_copus):
        """完成一个prediction的处理：重置状态、停止COPU任务、加载新任务"""
        print(
            f"current cycle: {self.cycle} - current edge idx: {self.group_status[group_id][prediction_idx]['edge_idx']}"
        )
        # 重置该prediction状态
        self.group_status[group_id][prediction_idx] = {
            "edge_idx": -1,
            "finished": [False] * self.copus_per_edge,
        }

        # 停止组内所有COPU在该prediction的任务
        for c in group_copus:
            c.reset_prediction(prediction_idx)

        # 加载新任务到该prediction (如果队列不为空)
        if self.edge_queue:
            new_edge_idx = self.edge_queue.popleft()
            self._assign_edge_to_group(group_id, new_edge_idx, prediction_idx)
        # 切换该组所有COPU到下一个prediction
        for c in group_copus:
            c.active_idx = (c.active_idx + 1) % self.num_predictions

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
            self._finish_prediction(group_id, active_idx, group_copus)
            return

        # 检查完成
        if all(self.group_status[group_id][active_idx]["finished"]):
            self.edge_results[edge_idx] = "safe"
            self._finish_prediction(group_id, active_idx, group_copus)
