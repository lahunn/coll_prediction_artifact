"""
COPU (Collision Prediction Unit) 模块
"""

from collections import deque
from .data_structures import OOCDState
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
    ):
        self.copu_id = copu_id
        self.num_oocds = num_oocds
        self.qcoll_size = qcoll_size
        self.qnoncoll_size = qnoncoll_size
        self.cycle_check = cycle_check
        self.cht_scheduler = cht_scheduler

        self.oocds = [
            OOCDState(hash_key=0, result=0, busy=0, free_cycle=0)
            for _ in range(num_oocds)
        ]
        self.qcoll = deque(maxlen=qcoll_size)
        self.qnoncoll = deque(maxlen=qnoncoll_size)

        self.linklist = []
        self.linklist_coll = []
        self.linklist_cycles = []

        self.cycle = 0
        self.coll_found = False
        self.everything_free = False

        self.query_count = 0
        self.oocd_cycles = 0
        self.first_two_running = 0
        self.first_two_checked = 0

        self.initial_task_count = 0
        self.conservation_violations = []

    def load_data(self, linklist, linklist_coll, linklist_cycles):
        """从外部加载待处理配置数据"""
        self.linklist = list(linklist)
        self.linklist_coll = list(linklist_coll)
        self.linklist_cycles = list(linklist_cycles)
        self.initial_task_count = len(linklist)

        # 重置任务相关状态
        self.coll_found = False
        self.everything_free = False
        self.qcoll.clear()
        self.qnoncoll.clear()

        # 重置OOCD状态()
        for oocd in self.oocds:
            oocd.reset()

    def reset_task(self):
        """强制重置任务状态（用于中断任务）"""
        self.linklist = []
        self.linklist_coll = []
        self.linklist_cycles = []
        self.coll_found = False
        self.everything_free = True  # 标记为空闲
        self.qcoll.clear()
        self.qnoncoll.clear()

    def step(self, bins, threshold, sample_rate):
        """执行一个仿真周期"""
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
            self.qcoll,
            self.qnoncoll,
            self.linklist,
            self.cycle,
            self.first_two_running,
            self.first_two_checked,
            self.cycle_check,
            self.num_oocds,
            self.qnoncoll_size,
        )

        # 3. 预测新配置
        predict_next_config(
            self.linklist,
            self.linklist_coll,
            self.qcoll,
            self.qnoncoll,
            bins,
            threshold,
            self.cht_scheduler,
            self.qcoll_size,
            self.qnoncoll_size,
        )

        # 4. 检查终止条件
        if (
            len(self.linklist) == 0
            and not any(oocd.free_cycle > self.cycle for oocd in self.oocds)
            and not self.qnoncoll
            and not self.qcoll
        ):
            self.everything_free = True

        # 推进周期
        self.cycle += 1
        self.oocd_cycles += oocd_cycles_delta
        continue_sim = not self.coll_found and not self.everything_free
        return continue_sim, self.cycle

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
