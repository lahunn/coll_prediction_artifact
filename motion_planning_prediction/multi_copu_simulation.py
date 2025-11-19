"""
多COPU碰撞预测硬件加速器仿真框架

实现四个核心类：
1. DualPortSRAM_CHT - 双端口SRAM形式的碰撞历史表
2. COPUModule - 单个碰撞预测单元（包含OOCD阵列和队列）
3. CHT_AccessScheduler - CHT访问调度器（管理多COPU访问约束）
4. MultiCOPU_Scheduler - 多COPU系统调度器（协调全局仿真）

硬件约束建模：
- CHT读端口：每周期最多2个并发读
- CHT写端口：写操作不与读操作并行
- 数据来源：直接从pickle文件加载预生成的配置数据（无OBB生成）
"""

import numpy as np
import random
from collections import deque, namedtuple

# ======================== Constants ========================
NUM_OOCDS = 7
MAX_COLLISION_COUNT = 15
DEFAULT_QNONCOLL_LEN = 56
DEFAULT_QCOLL_LEN = 8
DEFAULT_CYCLE_CHECK = 15
CHT_DEFAULT_SIZE = 4096
ONE_CYCLE_DELAY = 1

# Named tuple for OOCD state
OOCDState = namedtuple("OOCDState", ["hash_key", "result", "busy", "free_cycle"])


# ======================== Helper Functions ========================
def reutrn_keyy(code):
    """Creates a hash key from a quantized code."""
    bitsize = len(code)
    keyy = ""
    for j in range(0, bitsize):
        if code[j] < 10:
            keyy = keyy + "0"
        keyy = keyy + str(code[j])
    return keyy


def update_collision_dict(colldict, hash_key, is_free, sample_rate):
    """Updates the collision history dictionary."""
    if hash_key in colldict:
        if (
            is_free == 1
            and random.random() <= sample_rate
            and colldict[hash_key][is_free] < MAX_COLLISION_COUNT
        ):
            colldict[hash_key][is_free] += 1
        elif is_free == 0 and colldict[hash_key][is_free] < MAX_COLLISION_COUNT:
            colldict[hash_key][is_free] += 1
    else:
        colldict[hash_key] = [0, 0]
        if (
            is_free == 1
            and random.random() <= sample_rate
            and colldict[hash_key][is_free] < MAX_COLLISION_COUNT
        ):
            colldict[hash_key][is_free] += 1
        elif is_free == 0 and colldict[hash_key][is_free] < MAX_COLLISION_COUNT:
            colldict[hash_key][is_free] += 1
    return colldict


def predict_collision(colldict, hash_key, threshold):
    """Predicts collision based on the history dictionary."""
    if hash_key in colldict:
        if colldict[hash_key][0] > colldict[hash_key][1] * threshold:
            return True
        else:
            return False
    else:
        return False


# ======================== DualPortSRAM_CHT ========================
class DualPortSRAM_CHT:
    """
    双端口SRAM形式的碰撞历史表 (CHT)

    硬件约束（真正的双端口SRAM）：
    - 同一周期内最多2个总并发操作（不区分读写类型）
    - 允许读读、读写、写写并行
    - 每个操作占用1个时钟周期
    - 每个条目：[COLL_count, NONCOLL_count] (各4-bit饱和)
    """

    def __init__(self, size=CHT_DEFAULT_SIZE, enable_conflict_check=True):
        self.size = size
        self.enable_conflict_check = enable_conflict_check
        self.memory = {}  # {hash_key: [COLL, NONCOLL]}

        # 周期级别的访问调度（统一的待决请求列表）
        self.current_cycle = 0
        self.pending_accesses = []  # [{completion_cycle, op_type, hash_key, [data/delta]}]
        # op_type: 'read' 或 'write'

        # 统计信息
        self.read_count = 0
        self.write_count = 0
        self.conflicts = 0  # 超过2个并发操作的次数

    def read_request(self, copu_id, hash_key, cycle):
        """
        提交读请求。

        Returns:
            (result, completion_cycle): 数据和完成周期
        """
        # 检查总并发操作约束（不区分读写）
        num_pending = len(self.pending_accesses)

        if num_pending < 2:
            # 有可用端口，本周期可完成
            completion_cycle = cycle + ONE_CYCLE_DELAY
        else:
            # 端口满了，需要排队
            if self.enable_conflict_check:
                wait_cycles = num_pending // 2
                completion_cycle = cycle + wait_cycles + ONE_CYCLE_DELAY
                self.conflicts += 1
            else:
                completion_cycle = cycle + ONE_CYCLE_DELAY

        # 不存储copu_id，由CHT_AccessScheduler按hash_key去重
        self.pending_accesses.append(
            {
                "completion_cycle": completion_cycle,
                "op_type": "read",
                "hash_key": hash_key,
                "result": self.memory.get(hash_key, [0, 0]),
            }
        )
        self.read_count += 1

        # 返回当前数据（在完成周期时可用）
        data = self.memory.get(hash_key, [0, 0])
        return data, completion_cycle

    def write_request(self, copu_id, hash_key, delta_coll, delta_noncoll, cycle):
        """
        提交写请求。

        约束：同一周期最多2个总操作（读或写）
        """
        # 根据pending_accesses长度计算完成周期
        num_pending = len(self.pending_accesses)

        if num_pending < 2:
            # 有可用端口，本周期可完成
            completion_cycle = cycle + ONE_CYCLE_DELAY
        else:
            # 端口满了，需要排队
            if self.enable_conflict_check:
                wait_cycles = num_pending // 2
                completion_cycle = cycle + wait_cycles + ONE_CYCLE_DELAY
                self.conflicts += 1
            else:
                completion_cycle = cycle + ONE_CYCLE_DELAY

        # 不存储copu_id，由CHT_AccessScheduler按hash_key去重和合并
        self.pending_accesses.append(
            {
                "completion_cycle": completion_cycle,
                "op_type": "write",
                "hash_key": hash_key,
                "delta_coll": delta_coll,
                "delta_noncoll": delta_noncoll,
            }
        )
        self.write_count += 1

        return completion_cycle

    def advance_cycle(self):
        """推进一个周期，执行所有就绪的写操作"""
        cycle = self.current_cycle

        # 执行本周期的所有写操作
        for access in self.pending_accesses:
            if access["completion_cycle"] == cycle and access["op_type"] == "write":
                hash_key = access["hash_key"]
                delta_c = access["delta_coll"]
                delta_n = access["delta_noncoll"]
                if hash_key not in self.memory:
                    self.memory[hash_key] = [0, 0]
                # 4-bit饱和计数器（最大值15）
                self.memory[hash_key][0] = min(
                    self.memory[hash_key][0] + delta_c, MAX_COLLISION_COUNT
                )
                self.memory[hash_key][1] = min(
                    self.memory[hash_key][1] + delta_n, MAX_COLLISION_COUNT
                )

        # 清理已完成的请求
        self.pending_accesses = [
            access
            for access in self.pending_accesses
            if access["completion_cycle"] > cycle
        ]

        self.current_cycle += 1

    def reset(self):
        """重置CHT内容（不重置统计信息）"""
        self.memory = {}

    def get_stats(self):
        """返回CHT访问统计"""
        total_accesses = self.read_count + self.write_count
        conflict_rate = self.conflicts / total_accesses if total_accesses > 0 else 0.0
        return {
            "total_reads": self.read_count,
            "total_writes": self.write_count,
            "total_conflicts": self.conflicts,
            "conflict_rate": conflict_rate,
            "entries_used": len(self.memory),
        }


# ======================== COPUModule ========================
class COPUModule:
    """
    碰撞预测单元 (COPU) 模块

    包含：
    - Collision Predictor（查询CHT）
    - Query Dispatcher + 优先级队列（QCOLL, QNONCOLL）
    - OOCD (Out-of-Order Collision Detector) 阵列
    - CHT更新单元

    支持两种模式：
    1. 单COPU模式（cht_scheduler=None）：使用本地CHT字典
    2. 多COPU模式（cht_scheduler!=None）：通过调度器访问共享CHT
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

        # CHT访问接口（多COPU环境中的共享资源）
        self.cht_scheduler = cht_scheduler

        # COPU内部状态
        self.oocds = [
            OOCDState(hash_key=0, result=0, busy=0, free_cycle=0)
            for _ in range(num_oocds)
        ]
        self.qcoll = deque(maxlen=qcoll_size)
        self.qnoncoll = deque(maxlen=qnoncoll_size)

        # 待处理配置（从load_data加载）
        self.linklist = []
        self.linklist_coll = []
        self.linklist_cycles = []

        # 仿真状态
        self.cycle = 0
        self.coll_found = False
        self.everything_free = False

        # 本地CHT副本（单COPU模式）
        self.local_colldict = {}

        # 统计
        self.query_count = 0
        self.oocd_cycles = 0
        self.first_two_running = 0
        self.first_two_checked = 0

        # 守恒律追踪：初始任务总数（在load_data时设置）
        self.initial_task_count = 0
        self.conservation_violations = []  # 记录守恒律违反

    def load_data(self, linklist, linklist_coll, linklist_cycles):
        """从外部加载待处理配置数据"""
        self.linklist = list(linklist)
        self.linklist_coll = list(linklist_coll)
        self.linklist_cycles = list(linklist_cycles)
        # 记录初始任务总数（用于守恒律验证）
        self.initial_task_count = len(linklist)

    def check_conservation_law(self):
        """
        检查守恒律是否被违反
        守恒律: query_count + len(linklist) + len(qcoll) + len(qnoncoll) + num_busy_oocds = initial_task_count

        Returns:
            violation: 违反量（0表示守恒，>0表示系统中增生了任务，<0表示任务丢失）
        """
        num_busy_oocds = sum(1 for oocd in self.oocds if oocd.busy == 1)
        conservation_sum = (
            self.query_count
            + len(self.linklist)
            + len(self.qcoll)
            + len(self.qnoncoll)
            + num_busy_oocds
        )
        violation = conservation_sum - self.initial_task_count
        if violation != 0:
            self.conservation_violations.append(
                {
                    "cycle": self.cycle,
                    "violation": violation,
                    "query_count": self.query_count,
                    "linklist": len(self.linklist),
                    "qcoll": len(self.qcoll),
                    "qnoncoll": len(self.qnoncoll),
                    "busy_oocds": num_busy_oocds,
                    "sum": conservation_sum,
                }
            )
        return violation

    def set_collision_history_table(self, colldict):
        """设置本地CHT（单COPU环境）"""
        if self.cht_scheduler is None:
            self.local_colldict = colldict

    def step(self, bins, threshold, sample_rate):
        """
        执行一个仿真周期

        Returns:
            (continue_simulation, query_count_this_cycle)
        """

        # --- 步骤1: 处理OOCD完成和CHT更新 ---
        dequeued_this_cycle = False
        oocd_cycles_delta = sum(1 for oocd in self.oocds if oocd.busy == 1)

        for oocd_id in range(self.num_oocds):
            oocd = self.oocds[oocd_id]
            if oocd.busy == 1 and oocd.free_cycle <= self.cycle:
                self.query_count += 1
                self.oocds[oocd_id] = OOCDState(
                    hash_key="",
                    result=1,
                    busy=0,
                    free_cycle=self.cycle,
                )
                if oocd.result == 0:  # 碰撞
                    self.coll_found = True

                # 更新CHT
                if self.cht_scheduler is not None:
                    # 多COPU环境：通过调度器更新
                    delta_coll = 1 if oocd.result == 0 else 0
                    delta_noncoll = 1 if oocd.result == 1 else 0
                    self.cht_scheduler.submit_write(
                        self.copu_id, oocd.hash_key, delta_coll, delta_noncoll
                    )
                else:
                    # 单COPU环境：直接更新本地字典
                    self.local_colldict = update_collision_dict(
                        self.local_colldict, oocd.hash_key, oocd.result, sample_rate
                    )

            # --- 步骤2: 分派新任务 ---
            if oocd.free_cycle <= self.cycle and not dequeued_this_cycle:
                if len(self.qcoll) > 0 and self.first_two_checked < self.cycle:
                    self.first_two_running += 1
                    if self.first_two_running == 1:
                        self.first_two_checked = self.cycle + self.cycle_check

                    task = self.qcoll[0]
                    self.oocds[oocd_id] = OOCDState(
                        hash_key=task[0],
                        result=task[1],
                        busy=1,
                        free_cycle=self.cycle + self.cycle_check,
                    )
                    self.qcoll.popleft()
                    dequeued_this_cycle = True

                elif (
                    len(self.qnoncoll) == self.qnoncoll_size
                    or (len(self.linklist) == 0 and len(self.qnoncoll) > 0)
                ) and self.first_two_checked < self.cycle:
                    task = self.qnoncoll[0]
                    self.oocds[oocd_id] = OOCDState(
                        hash_key=task[0],
                        result=task[1],
                        busy=1,
                        free_cycle=self.cycle + self.cycle_check,
                    )
                    self.qnoncoll.popleft()
                    dequeued_this_cycle = True

                else:
                    self.oocds[oocd_id] = OOCDState(
                        hash_key=0, result=0, busy=0, free_cycle=0
                    )
            self.check_conservation_law()
        # --- 步骤3: 预测新配置 ---
        if len(self.linklist) > 0:
            link = self.linklist[0]
            linkcoll = self.linklist_coll[0]

            code_quant = np.digitize(link, bins, right=True)
            keyy = reutrn_keyy(code_quant)

            # 查询CHT
            if self.cht_scheduler is not None:
                # 多COPU环境：通过调度器读取
                # 自动去重和重试：多次查询同一keyy会自动去重
                is_ready, data = self.cht_scheduler.get_read_result(keyy)

                # 只有当结果就绪时才进行预测和入队
                if is_ready:
                    coll_count, noncoll_count = data
                    is_collision_predicted = coll_count > noncoll_count * threshold

                    # 入队
                    if is_collision_predicted:
                        if len(self.qcoll) < self.qcoll_size:
                            self.qcoll.append([keyy, linkcoll])
                            del self.linklist[0]
                            del self.linklist_coll[0]
                    else:
                        if len(self.qnoncoll) < self.qnoncoll_size:
                            self.qnoncoll.append([keyy, linkcoll])
                            del self.linklist[0]
                            del self.linklist_coll[0]
                # 如果结果未就绪，本周期跳过该配置
                # 下周期自动重试（get_read_result会查询同一keyy的待决请求）
            else:
                # 单COPU环境：直接读取本地字典
                if keyy in self.local_colldict:
                    coll_count = self.local_colldict[keyy][0]
                    noncoll_count = self.local_colldict[keyy][1]
                else:
                    coll_count, noncoll_count = 0, 0

                is_collision_predicted = coll_count > noncoll_count * threshold

                # 入队
                if is_collision_predicted:
                    if len(self.qcoll) < self.qcoll_size:
                        self.qcoll.append([keyy, linkcoll])
                        del self.linklist[0]
                        del self.linklist_coll[0]
                else:
                    if len(self.qnoncoll) < self.qnoncoll_size:
                        self.qnoncoll.append([keyy, linkcoll])
                        del self.linklist[0]
                        del self.linklist_coll[0]
        self.check_conservation_law()
        # --- 步骤4: 检查终止条件 ---
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


# ======================== CHT_AccessScheduler ========================
class CHT_AccessScheduler:
    """
    CHT访问调度器，管理多个COPU对共享CHT的访问

    设计原则：
    - 按hash_key（访问地址）而非request_id管理待决请求
    - 同一地址的多个读请求自动去重
    - 同一地址的多个写请求自动合并delta
    - 简化COPU端的请求追踪逻辑
    """

    def __init__(
        self, num_copus, cht_size=CHT_DEFAULT_SIZE, enable_conflict_check=True
    ):
        self.num_copus = num_copus
        self.cht = DualPortSRAM_CHT(
            size=cht_size, enable_conflict_check=enable_conflict_check
        )
        self.current_cycle = 0

    def get_read_result(self, hash_key):
        """
        查询hash_key的读结果，自动管理pending状态
        首次查询时发起读请求，后续查询相同地址会自动去重

        Returns:
            (is_ready, data)
        """
        # 在pending_accesses中查找该hash_key的读请求
        for access in self.cht.pending_accesses:
            if access["op_type"] == "read" and access["hash_key"] == hash_key:
                # 找到待决的读请求，检查是否就绪
                if self.current_cycle >= access["completion_cycle"]:
                    return True, access["result"]
                else:
                    return False, None

        # 没有待决请求，发起新的读请求
        data, comp_cycle = self.cht.read_request(0, hash_key, self.current_cycle)
        if len(self.cht.pending_accesses) <= 2 and ONE_CYCLE_DELAY == 0:
            return True, data
        else:
            return False, None

    def submit_write(self, copu_id, hash_key, delta_coll, delta_noncoll):
        """
        提交写请求，相同hash_key的多个写自动合并

        Args:
            copu_id: 发起方COPU ID（用于SRAM层统计）
            hash_key: 写入地址
            delta_coll: COLL计数增量
            delta_noncoll: NONCOLL计数增量
        """
        # 在pending_accesses中查找该hash_key的写请求
        for access in self.cht.pending_accesses:
            if access["op_type"] == "write" and access["hash_key"] == hash_key:
                # 找到待决写请求，直接合并delta
                access["delta_coll"] += delta_coll
                access["delta_noncoll"] += delta_noncoll
                return

        # 没有待决请求，发起新的写请求
        self.cht.write_request(
            copu_id, hash_key, delta_coll, delta_noncoll, self.current_cycle
        )

    def advance_cycle(self):
        """推进周期：清理已完成请求，由DualPortSRAM_CHT执行写操作"""
        # 推进SRAM周期（内部执行所有就绪的写操作并清理已完成请求）
        self.cht.advance_cycle()
        self.current_cycle += 1


# ======================== MultiCOPU_Scheduler ========================
class MultiCOPU_Scheduler:
    """
    多COPU系统调度器

    职责：
    - 管理多个COPU模块
    - 协调CHT访问
    - 同步各COPU的进度
    - 收集全局结果
    """

    def __init__(
        self,
        num_copus,
        num_oocds=NUM_OOCDS,
        cht_size=CHT_DEFAULT_SIZE,
        enable_conflict_check=True,
    ):
        self.num_copus = num_copus

        # 创建共享的CHT调度器
        self.cht_scheduler = CHT_AccessScheduler(
            num_copus, cht_size, enable_conflict_check=enable_conflict_check
        )

        # 创建COPU模块
        self.copus = [
            COPUModule(
                copu_id=i,
                num_oocds=num_oocds,
                cht_scheduler=self.cht_scheduler,
            )
            for i in range(num_copus)
        ]

        self.cycle = 0
        self.global_coll_found = False

    def load_data_for_all_copus(self, data_list, coll_list, cycles_list):
        """
        为所有COPU加载任务（数据已预先分割）

        Args:
            data_list: list of data for each COPU
            coll_list: list of collision flags for each COPU
            cycles_list: list of cycles for each COPU
        """
        for copu_id, copu in enumerate(self.copus):
            copu.load_data(data_list[copu_id], coll_list[copu_id], cycles_list[copu_id])

    def simulate(self, bins, threshold, sample_rate, max_cycles=100000):
        """
        执行多COPU协同仿真

        Returns:
            results dict with global metrics and per-COPU stats
        """
        cycle_limit_reached = False

        while not self.global_coll_found and not cycle_limit_reached:
            if self.cycle >= max_cycles:
                cycle_limit_reached = True
                break

            # 每个COPU执行一步
            any_copu_active = False
            for copu in self.copus:
                continue_sim, _ = copu.step(bins, threshold, sample_rate)
                if continue_sim:
                    any_copu_active = True
                if copu.coll_found:
                    self.global_coll_found = True

            # 推进CHT调度器
            self.cht_scheduler.advance_cycle()

            # 如果没有任何COPU需要继续，退出
            if not any_copu_active:
                break

            self.cycle += 1

        # 收集统计
        results = {
            "total_cycles": self.cycle,
            "collision_found": self.global_coll_found,
            "cycle_limit_reached": cycle_limit_reached,
            "copus": [copu.get_stats() for copu in self.copus],
            "cht_stats": self.cht_scheduler.cht.get_stats(),
        }

        return results


# ======================== Utility Functions ========================
def _generate_recursive_reorder(num_poses, step_size):
    """
    生成递归式重排顺序（保持固定步长，只对组序列进行递归二分重排）。

    step_size =8 , 重排序后= [0,8,16,24,4,12,20,28,2,10,18,26,6,14,22,30,1,9,17,25,...]

    Returns:
        重排后的pose索引列表
    """
    # 步骤1：对组号进行递归二分重排
    group_count = min(step_size, (num_poses + step_size - 1) // step_size)
    group_order = _recursive_binary_reorder(group_count)

    # 步骤2：按重排后的组号顺序生成pose列表
    reorder = []
    for group_id in group_order:
        pose_idx = group_id
        while pose_idx < num_poses:
            reorder.append(pose_idx)
            pose_idx += step_size

    return reorder


def _recursive_binary_reorder(n):
    """
    将 [0,1,2,...,n-1] 按递归二分方式重排（使用位反转）。

    算法：对每个索引进行位反转，生成递归二分的访问顺序

    例如 n=4 (2位):
    - 0 (00) -> 00 = 0
    - 1 (01) -> 10 = 2
    - 2 (10) -> 01 = 1
    - 3 (11) -> 11 = 3
    结果：[0,2,1,3]

    Args:
        n: 数字个数

    Returns:
        重排后的索引列表
    """
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]

    # 计算需要的位数
    num_bits = 0
    temp = n - 1
    while temp > 0:
        num_bits += 1
        temp >>= 1

    reorder = []
    for i in range(n):
        # 对i进行位反转
        reversed_i = 0
        for bit in range(num_bits):
            reversed_i = (reversed_i << 1) | ((i >> bit) & 1)
        reorder.append(reversed_i)

    return reorder


def _allocate_edge_data_to_copus(
    edge_coords,
    edge_flags,
    edge_cycles,
    num_copus,
    use_recursive_reorder=True,
    step_size=8,
):
    """
    将单条edge的pose数据按轮转方式分配给所有COPU。

    算法：
    1. 可选：对poses按递归二分方式重排，降低空间相邻性
    2. 按轮转方式将poses分配给各COPU

    Args:
        edge_coords: 该edge的pose坐标数据 List[List[coords]]
        edge_flags: 该edge的碰撞标志 List[List[flags]]
        edge_cycles: 该edge的周期数据 List[List[cycles]] 或 None
        num_copus: COPU总数
        use_recursive_reorder: 是否使用递归二分重排 (默认False，使用顺序)
        step_size: 递归重排的步长 (默认8，仅当use_recursive_reorder=True时有效)

    Returns:
        (copus_coords, copus_flags, copus_cycles):
            copus_coords[i], copus_flags[i], copus_cycles[i] 为第i个COPU的展平后数据
            List[List[coords]], List[List[flags]], List[List[cycles]]
    """
    num_poses = len(edge_coords)

    # 步骤1：确定pose顺序（可选递归重排）
    if use_recursive_reorder:
        reorder = _generate_recursive_reorder(num_poses, step_size)
    else:
        reorder = list(range(num_poses))

    # 步骤2：初始化所有COPU的数据列表
    copus_coords = [[] for _ in range(num_copus)]
    copus_flags = [[] for _ in range(num_copus)]
    copus_cycles = [[] for _ in range(num_copus)]

    # 步骤3：按轮转方式将poses分配给各COPU
    for reordered_idx, original_pose_idx in enumerate(reorder):
        copu_id = reordered_idx % num_copus
        pose_coords = edge_coords[original_pose_idx]  # List[link_coord]
        pose_flags = edge_flags[original_pose_idx]  # List[link_flag]

        # 展平link数据
        copus_coords[copu_id].extend(pose_coords)
        copus_flags[copu_id].extend(pose_flags)

        # 处理周期数据
        if edge_cycles is not None:
            pose_cycles = edge_cycles[original_pose_idx]
            copus_cycles[copu_id].extend(pose_cycles)
        else:
            copus_cycles[copu_id].extend([40 for _ in range(len(pose_coords))])

    return copus_coords, copus_flags, copus_cycles


def analyze_multi_copu_performance(results):
    """
    分析多COPU系统的性能指标

    Key metrics:
    1. System Throughput: total_queries / total_cycles
    2. COPU Utilization: query_count / (total_cycles * num_oocds)
    3. CHT Conflict Rate: conflicts / total_accesses
    4. Load Balance: std(copu_query_counts) / mean(copu_query_counts)
    """
    cht_stats = results["cht_stats"]
    copu_stats = results["copus"]

    # KPI 1: 系统吞吐量
    total_queries = sum(c["total_queries"] for c in copu_stats)
    system_throughput = total_queries / max(1, results["total_cycles"])

    # KPI 2: COPU平均利用率
    avg_utilization = sum(c["oocd_utilization"] for c in copu_stats) / len(copu_stats)

    # KPI 3: CHT冲突率
    cht_conflict_rate = cht_stats["conflict_rate"]

    # KPI 4: 负载平衡
    query_counts = [c["total_queries"] for c in copu_stats]
    if len(query_counts) > 1:
        load_balance = np.std(query_counts) / (np.mean(query_counts) + 1e-6)
    else:
        load_balance = 0.0

    return {
        "system_throughput": system_throughput,
        "avg_copu_utilization": avg_utilization,
        "cht_conflict_rate": cht_conflict_rate,
        "load_balance_variance": load_balance,
        "total_cycles": results["total_cycles"],
        "total_queries": total_queries,
        "num_copus": len(copu_stats),
        "per_copu_queries": query_counts,
    }
