"""
COPU (Collision Prediction Unit) 模块
"""

from .data_structures import OOCDState, Prediction
from .constants import (
    NUM_OOCDS,
    DEFAULT_QCOLL_LEN,
    DEFAULT_QNONCOLL_LEN,
    DEFAULT_CYCLE_CHECK,
)
from .oocd_processor import (
    process_oocd_completion,
    dispatch_new_tasks,
)
from .collision_prediction import (
    predict_next_config,
)


class COPUModule:
    """
    碰撞预测单元 (COPU) 模块
    """

    def __init__(
        self,
        copu_id,
        num_oocds=NUM_OOCDS,
        qcoll_size=DEFAULT_QCOLL_LEN,
        qnoncoll_size=DEFAULT_QNONCOLL_LEN,
        cycle_check=DEFAULT_CYCLE_CHECK,
        cht_scheduler=None,
        num_predictions=1,
    ):
        self.copu_id = copu_id
        self.num_oocds = num_oocds
        self.qcoll_size = qcoll_size
        self.qnoncoll_size = qnoncoll_size
        self.cycle_check = cycle_check
        self.cht_scheduler = cht_scheduler
        self.num_predictions = num_predictions

        self.oocds = [OOCDState() for _ in range(num_oocds)]

        self.predictions = [
            Prediction(qcoll_size, qnoncoll_size) for _ in range(num_predictions)
        ]
        self.active_idx = 0

        self.cycle = 0
        self.coll_found = False
        self.everything_free = False
        self.edge_idx = -1

        self.query_count = 0
        self.oocd_cycles = 0

        self.initial_task_count = 0
        self.conservation_violations = []

    def load_data(
        self, linklist, linklist_coll, linklist_cycles, prediction_idx=0, edge_idx=-1
    ):
        """从外部加载待处理配置数据"""
        if prediction_idx >= self.num_predictions:
            raise ValueError(f"prediction index {prediction_idx} out of range")

        self.edge_idx = edge_idx

        pred = self.predictions[prediction_idx]
        pred.linklist = list(linklist)
        pred.linklist_coll = list(linklist_coll)
        # linklist_cycles is not stored in Prediction currently, keeping it here for now if needed
        # or we can add it to Prediction class if it's used per-prediction
        self.linklist_cycles = list(linklist_cycles)
        self.initial_task_count = len(linklist)

        # 重置任务相关状态
        self.coll_found = False
        self.everything_free = False
        pred.qcoll.clear()
        pred.qnoncoll.clear()

        # 重置OOCD状态()
        for oocd in self.oocds:
            oocd.reset()

    def reset_prediction(self, prediction_idx):
        """重置指定prediction的状态"""
        if prediction_idx >= self.num_predictions:
            return
        pred = self.predictions[prediction_idx]
        pred.linklist = []
        pred.linklist_coll = []
        pred.qcoll.clear()
        pred.qnoncoll.clear()

        # 如果重置的是当前活动的prediction，也需要重置全局状态
        if prediction_idx == self.active_idx:
            self.coll_found = False
            self.everything_free = True

    def reset_task(self):
        """强制重置所有任务状态（用于中断任务）"""
        for i in range(self.num_predictions):
            self.reset_prediction(i)
        self.linklist_cycles = []
        self.coll_found = False
        self.everything_free = True  # 标记为空闲
        self.edge_idx = -1

    def step(self, bins, threshold, sample_rate):
        """执行一个仿真周期"""
        active_pred = self.predictions[self.active_idx]

        # 1. 处理OOCD完成和CHT更新
        oocd_cycles_delta, self.query_count, self.coll_found = process_oocd_completion(
            self.oocds,
            self.cycle,
            self.query_count,
            self.coll_found,
            self.cht_scheduler,
            self.copu_id,
            sample_rate,
            self.num_oocds,
        )

        # 2. 分派新任务
        dispatch_new_tasks(
            self.oocds,
            active_pred.qcoll,
            active_pred.qnoncoll,
            active_pred.linklist,
            self.cycle,
            self.cycle_check,
            self.num_oocds,
            self.qnoncoll_size,
        )

        # 3. 预测新配置
        for pred in self.predictions:
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

        # 4. 检查是否所有任务都完成
        finished = (
            len(active_pred.linklist) == 0
            and not active_pred.qcoll
            and not active_pred.qnoncoll
            and not any(oocd.free_cycle > self.cycle for oocd in self.oocds)
        )

        self.everything_free = finished

        # 推进周期
        self.cycle += 1
        self.oocd_cycles += oocd_cycles_delta

        return finished

    def get_stats(self):
        """返回仿真统计"""
        return {
            "copu_id": self.copu_id,
            "total_cycles": self.cycle,
            "total_queries": self.query_count,
            "coll_found": self.coll_found,
            "oocd_utilization": (
                self.oocd_cycles / (self.cycle * self.num_oocds)
                if self.cycle > 0
                else 0.0
            ),
        }
