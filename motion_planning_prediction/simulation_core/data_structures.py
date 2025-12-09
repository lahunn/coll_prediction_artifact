"""
Data structures for collision detection simulation.
"""

from collections import deque, defaultdict


class OOCDOwnershipMap:
    """双向映射：OOCD <-> Prediction 的所有权关系

    支持：
    - 通过 oocd_id 查询其所属的 pred_id
    - 通过 pred_id 查询其拥有的所有 oocd_id
    - O(1) 分配和释放操作
    """

    def __init__(self, num_oocds):
        # oocd_id -> pred_id 的映射（-1 表示未分配）
        self._oocd_to_pred = [-1] * num_oocds
        # pred_id -> set of oocd_ids 的映射
        self._pred_to_oocds = defaultdict(set)
        # 未分配的 oocd 集合
        self._pred_to_oocds[-1] = set(range(num_oocds))

    def get_owner(self, oocd_id):
        """获取 OOCD 的所有者 pred_id（-1 表示未分配）"""
        return self._oocd_to_pred[oocd_id]

    def get_oocds(self, pred_id):
        """获取 prediction 拥有的所有 oocd_id 集合"""
        return self._pred_to_oocds[pred_id]

    def assign(self, oocd_id, pred_id):
        """将 oocd_id 分配给 pred_id"""
        old_owner = self._oocd_to_pred[oocd_id]
        if old_owner == pred_id:
            return  # 已经分配给该 pred

        # 从旧所有者移除
        self._pred_to_oocds[old_owner].discard(oocd_id)

        # 分配给新所有者
        self._oocd_to_pred[oocd_id] = pred_id
        self._pred_to_oocds[pred_id].add(oocd_id)

    def release(self, oocd_id):
        """释放 oocd_id（设为未分配状态）"""
        self.assign(oocd_id, -1)

    def release_all_for_pred(self, pred_id):
        """释放某个 prediction 的所有 OOCD"""
        oocd_ids = list(self._pred_to_oocds[pred_id])  # 复制一份，避免迭代时修改
        for oocd_id in oocd_ids:
            self.release(oocd_id)

    def reset(self, num_oocds):
        """重置所有映射关系"""
        self._oocd_to_pred = [-1] * num_oocds
        self._pred_to_oocds.clear()
        self._pred_to_oocds[-1] = set(range(num_oocds))


# Class for OOCD state
class OOCDState:
    def __init__(self, hash_key="", result=1, busy=0, free_cycle=0):
        self.hash_key = hash_key
        self.result = result
        self.busy = busy
        self.free_cycle = free_cycle

    def reset(self):
        self.hash_key = ""
        self.result = 1
        self.busy = 0
        self.free_cycle = 0


class OOCDSlot(OOCDState):
    """全局OOCD池中的槽位信息"""

    def __init__(self):
        super().__init__()


class OOCDStatePreemptive:
    def __init__(self, hash_key, result, busy, free_cycle, task_type):
        self.hash_key = hash_key
        self.result = result
        self.busy = busy
        self.free_cycle = free_cycle
        self.task_type = task_type

    def reset(self):
        self.hash_key = ""
        self.result = 1
        self.busy = 0
        self.free_cycle = 0
        self.task_type = 0


class Prediction:
    def __init__(self, qcoll_len, qnoncoll_len):
        self.qcoll = deque(maxlen=qcoll_len)
        self.qnoncoll = deque(maxlen=qnoncoll_len)
        self.linklist = []
        self.linklist_coll = []
        self.edge_idx = -1  # 当前处理的edge索引
        self.collision_detected = False  # 碰撞检测结果

    def reset(self):
        """重置prediction状态"""
        self.qcoll.clear()
        self.qnoncoll.clear()
        self.linklist = []
        self.linklist_coll = []
        self.edge_idx = -1
        self.collision_detected = False
