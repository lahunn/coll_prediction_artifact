"""
全局OOCD池调度器 - 支持多prediction共享OOCD资源

核心概念：
- OOCD池：全局共享的OOCD资源，任意prediction可申请
- 配额限制：每个prediction最多占用 max_oocd_per_pred 个OOCD
- 顺序加载：edges按顺序分配给predictions，完成后动态加载下一edge
- 派发策略：qcoll优先，qnoncoll兜底，按配额限制
"""

from collections import deque

# from typing import Optional
from .data_structures import Prediction, OOCDSlot, OOCDOwnershipMap
from .constants import (
    NUM_OOCDS,
    DEFAULT_QCOLL_LEN,
    DEFAULT_QNONCOLL_LEN,
    DEFAULT_CYCLE_CHECK,
)
from .cht_access_scheduler import CHT_AccessScheduler
from .collision_prediction import predict_next_config, submit_cht_write
from .data_preprocessing import csp_rearrange


class GlobalOOCDScheduler:
    """
    全局OOCD池调度器

    将多个predictions的任务共享到全局OOCD池，支持动态edge加载和配额管理。
    """

    def __init__(
        self,
        num_oocds=NUM_OOCDS,
        num_predictions=4,
        max_oocd_per_pred=2,
        cht_size=4096,
        enable_conflict_check=True,
        cht_type="dual_port",
        qcoll_size=DEFAULT_QCOLL_LEN,
        qnoncoll_size=DEFAULT_QNONCOLL_LEN,
        cycle_check=DEFAULT_CYCLE_CHECK,
        **cht_kwargs,
    ):
        self.num_oocds = num_oocds
        self.num_predictions = num_predictions
        self.qcoll_size = qcoll_size
        self.qnoncoll_size = qnoncoll_size
        self.cycle_check = cycle_check

        # 计算每个prediction的配额
        self.max_oocd_per_pred = max_oocd_per_pred

        # 创建共享CHT调度器
        self.cht_scheduler = CHT_AccessScheduler(
            num_predictions,
            cht_size,
            enable_conflict_check=enable_conflict_check,
            cht_type=cht_type,
            **cht_kwargs,
        )

        # 全局OOCD池
        self.oocd_pool = [OOCDSlot() for _ in range(num_oocds)]

        # OOCD所有权双向映射
        self.oocd_ownership = OOCDOwnershipMap(num_oocds)

        # Predictions数组
        self.predictions = [
            Prediction(qcoll_size, qnoncoll_size) for _ in range(num_predictions)
        ]
        # 全局状态
        self.cycle = 0
        self.global_coll_found = False
        # 边数据队列
        self.edge_queue = deque()
        self.all_data = []
        self.all_coll = []
        self.edge_results = {}  # edge_idx -> 'collision' or 'safe'

        # 统计
        self.total_query_count = 0.0
        self.oocd_cycles = 0  # 被占用的OOCD周期总数

    def set_benchmark_data(self, all_data, all_coll):
        """设置基准数据并初始化任务队列"""
        self.all_data = all_data
        self.all_coll = all_coll
        self.edge_queue = deque(range(len(all_data)))
        self.edge_results = {}

        # 重置全局状态
        self.cycle = 0
        self.global_coll_found = False
        self.total_query_count = 0.0
        self.oocd_cycles = 0

        # 重置OOCD池
        self.oocd_pool = [OOCDSlot() for _ in range(self.num_oocds)]

        # 重置OOCD所有权映射
        self.oocd_ownership.reset(self.num_oocds)

        # 重置predictions
        for i in range(self.num_predictions):
            self.predictions[i].linklist = []
            self.predictions[i].linklist_coll = []
            self.predictions[i].qcoll.clear()
            self.predictions[i].qnoncoll.clear()
            self.predictions[i].edge_idx = -1

        # 预加载前num_predictions个edges
        for pred_id in range(self.num_predictions):
            if self.edge_queue:
                edge_idx = self.edge_queue.popleft()
                self._load_edge_to_prediction(pred_id, edge_idx)

    def _load_edge_to_prediction(self, pred_id, edge_idx):
        """将一条edge装入指定prediction"""
        if edge_idx >= len(self.all_data):
            return False

        edge_data = self.all_data[edge_idx]
        edge_coll = self.all_coll[edge_idx]

        flat_data, flat_coll = csp_rearrange(edge_data, edge_coll, groupsize=8)

        # 装入prediction
        self.predictions[pred_id].linklist = flat_data
        self.predictions[pred_id].linklist_coll = flat_coll
        self.predictions[pred_id].edge_idx = edge_idx

        # 清空队列
        self.predictions[pred_id].qcoll.clear()
        self.predictions[pred_id].qnoncoll.clear()

        return True

    def _is_prediction_finished(self, pred_id):
        """检查一个prediction是否已完成（linklist/队列清空 + 无在飞OOCD 或者 碰撞检测结果为碰撞）"""
        pred = self.predictions[pred_id]

        # 条件1：检测到碰撞
        if pred.collision_detected:
            return True

        # 条件2：linklist/队列清空 + 无在飞OOCD
        has_linklist = len(pred.linklist) > 0
        has_qcoll = len(pred.qcoll) > 0
        has_qnoncoll = len(pred.qnoncoll) > 0

        # 检查该prediction的OOCD
        has_flying_oocd = any(
            self.oocd_pool[oocd_id].busy
            for oocd_id in self.oocd_ownership.get_oocds(pred_id)
        )

        return not (has_linklist or has_qcoll or has_qnoncoll or has_flying_oocd)

    def _get_free_oocd_for_pred(self, pred_id):
        """为prediction申请一个OOCD（优先返回已有空闲、次之申请新的）"""
        # 获取该pred已占用的OOCD
        pred_oocds = self.oocd_ownership.get_oocds(pred_id)

        # 首先尝试返回已分配但空闲的OOCD
        for oocd_id in pred_oocds:
            if not self.oocd_pool[oocd_id].busy:
                return oocd_id

        # 如果已占用数达到配额，返回-1
        if len(pred_oocds) >= self.max_oocd_per_pred:
            return -1

        # 从未分配的OOCD中申请新的
        free_oocds = self.oocd_ownership.get_oocds(-1)
        for oocd_id in free_oocds:
            if not self.oocd_pool[oocd_id].busy:
                self.oocd_ownership.assign(oocd_id, pred_id)
                return oocd_id

        return -1  # 无空闲OOCD

    def simulate(self, bins, threshold, sample_rate):
        """
        执行全局OOCD池仿真

        Returns:
            dict: 包含总周期、查询数、利用率、CHT统计、edge结果等
        """
        while True:
            # 1. 处理OOCD完成与CHT更新
            self._complete_oocd_tasks(sample_rate)

            # 2. 派发任务给OOCD
            self._dispatch_queued_tasks()

            # 3. 预测新配置入队
            self._enqueue_predictions(bins, threshold)

            # 4. 检查各prediction完成情况，动态加载edge
            self._check_and_load_edges()

            # 5. 检查全局收敛
            if self._is_globally_finished():
                break

            # 推进全局周期
            self.cht_scheduler.advance_cycle()
            self.cycle += 1

        # 统计计算
        oocd_utilization = (
            self.oocd_cycles / (self.cycle * self.num_oocds) if self.cycle > 0 else 0.0
        )

        # 检查是否有任何edge发生碰撞
        self.global_coll_found = any(
            res == "collision" for res in self.edge_results.values()
        )

        return {
            "total_cycles": self.cycle,
            "total_queries": self.total_query_count,
            "oocd_utilization": oocd_utilization,
            "collision_found": self.global_coll_found,
            "cht_stats": self.cht_scheduler.cht.get_stats(),
            "edge_results": self.edge_results,
        }

    def _complete_oocd_tasks(self, sample_rate):
        """处理OOCD完成与CHT更新"""

        for oocd_id, oocd in enumerate(self.oocd_pool):
            if not oocd.busy:
                continue
            self.oocd_cycles += 1
            if oocd.free_cycle <= self.cycle:
                # OOCD任务完成
                pred_id = self.oocd_ownership.get_owner(oocd_id)
                self.total_query_count += 1

                # 如果检测到碰撞（result == 0），更新prediction的碰撞状态
                if oocd.result == 0:
                    self.predictions[pred_id].collision_detected = True

                # 直接写回CHT（碰撞历史存储在CHT中）
                submit_cht_write(
                    self.cht_scheduler, pred_id, oocd.hash_key, oocd.result, sample_rate
                )

                # 释放OOCD
                oocd.busy = False

    def _dispatch_queued_tasks(self):
        """派发队列中的任务给OOCD"""
        for pred_id in range(self.num_predictions):
            pred = self.predictions[pred_id]

            # qcoll优先
            if len(pred.qcoll) > 0:
                while len(pred.qcoll) > 0:
                    task = pred.qcoll[0]
                    hash_key = task[0]
                    result = task[1]

                    oocd_id = self._get_free_oocd_for_pred(pred_id)
                    if oocd_id == -1:
                        break  # 该pred无空闲配额

                    # 分配任务（使用默认cycle_check）
                    oocd = self.oocd_pool[oocd_id]
                    oocd.hash_key = hash_key
                    oocd.result = result
                    oocd.busy = True
                    oocd.free_cycle = self.cycle + self.cycle_check

                    pred.qcoll.popleft()

            # qnoncoll兜底
            elif len(pred.linklist) == 0 or len(pred.qnoncoll) == self.qnoncoll_size:
                while len(pred.qnoncoll) > 0:  # 无空闲OOCD则自动退出
                    task = pred.qnoncoll[0]
                    hash_key = task[0]
                    result = task[1]

                    oocd_id = self._get_free_oocd_for_pred(pred_id)
                    if oocd_id == -1:
                        break  # 该pred无空闲配额

                    # 分配任务（使用默认cycle_check）
                    oocd = self.oocd_pool[oocd_id]
                    oocd.hash_key = hash_key
                    oocd.result = result
                    oocd.busy = True
                    oocd.free_cycle = self.cycle + self.cycle_check

                    pred.qnoncoll.popleft()

    def _enqueue_predictions(self, bins, threshold):
        """预测新配置入队"""
        for pred_id in range(self.num_predictions):
            pred = self.predictions[pred_id]
            # 调用predict_next_config，传入pred_id
            # 该函数会从CHT读取并填入队列
            predict_next_config(
                pred.linklist,
                pred.linklist_coll,
                pred.qcoll,
                pred.qnoncoll,
                bins,
                threshold,
                self.cht_scheduler,
                self.qcoll_size,
                self.qnoncoll_size,
            )

    def _check_and_load_edges(self):
        """检查predictions完成情况，动态加载下一edge"""
        for pred_id in range(self.num_predictions):
            if self._is_prediction_finished(pred_id):
                edge_idx = self.predictions[pred_id].edge_idx
                # 记录edge结果（基于prediction的碰撞检测结果）
                if edge_idx >= 0:
                    has_collision = self.predictions[pred_id].collision_detected
                    self.edge_results[edge_idx] = (
                        "collision" if has_collision else "safe"
                    )

                # 释放该prediction的所有OOCD
                for oocd_id in list(self.oocd_ownership.get_oocds(pred_id)):
                    self.oocd_pool[oocd_id].reset()
                    self.oocd_ownership.release(oocd_id)

                # 重置prediction
                self.predictions[pred_id].reset()

                # 加载下一edge
                if self.edge_queue:
                    # 从队列中取出下一个edge并加载
                    edge_idx = self.edge_queue.popleft()
                    self._load_edge_to_prediction(pred_id, edge_idx)

    def _is_globally_finished(self):
        """检查全局是否完成（所有edges处理完毕，所有predictions空）"""
        # 检查是否所有edge都已加载（队列为空）
        all_edges_loaded = len(self.edge_queue) == 0

        # 检查所有predictions是否都空
        all_preds_empty = all(
            self._is_prediction_finished(pred_id)
            for pred_id in range(self.num_predictions)
        )

        return all_edges_loaded and all_preds_empty

    def get_stats(self):
        """返回仿真统计"""
        return {
            "total_cycles": self.cycle,
            "total_queries": self.total_query_count,
            "oocd_utilization": (
                self.oocd_cycles / (self.cycle * self.num_oocds)
                if self.cycle > 0
                else 0.0
            ),
            "collision_found": self.global_coll_found,
        }
