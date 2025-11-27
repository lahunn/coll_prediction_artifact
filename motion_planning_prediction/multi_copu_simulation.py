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

from collections import deque, namedtuple
import numpy as np
import simulation_utils as su

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

    def _calculate_completion_cycle(self, cycle):
        """计算操作完成周期，处理端口冲突"""
        num_pending = len(self.pending_accesses)
        if num_pending < 2:
            return cycle + ONE_CYCLE_DELAY

        if self.enable_conflict_check:
            wait_cycles = num_pending // 2
            self.conflicts += 1
            return cycle + wait_cycles + ONE_CYCLE_DELAY
        else:
            return cycle + ONE_CYCLE_DELAY

    def read_request(self, copu_id, hash_key, cycle):
        """
        提交读请求。
        Returns: (result, completion_cycle)
        """
        completion_cycle = self._calculate_completion_cycle(cycle)

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
        return self.memory.get(hash_key, [0, 0]), completion_cycle

    def write_request(self, copu_id, hash_key, delta_coll, delta_noncoll, cycle):
        """
        提交写请求。
        Returns: completion_cycle
        """
        completion_cycle = self._calculate_completion_cycle(cycle)

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


# ======================== MultiBankSRAM_CHT ========================
class MultiBankSRAM_CHT:
    """
    多Bank单端口SRAM形式的碰撞历史表 (CHT)

    硬件约束：
    - 划分为多个Bank (默认8个)
    - 每个Bank为单端口SRAM (每周期最多1个访问)
    - 不同Bank可并行访问
    - Bank选择策略可配置 (默认使用XYZ坐标的低位)
    """

    def __init__(
        self,
        size=CHT_DEFAULT_SIZE,
        num_banks=8,
        bank_config=None,
        enable_conflict_check=True,
    ):
        self.size = size
        self.num_banks = num_banks
        self.enable_conflict_check = enable_conflict_check
        # 默认配置: 使用第0, 1, 2维度的第0位作为Bank选择位
        # 格式: [(dim_index, bit_index), ...]
        self.bank_config = bank_config if bank_config else [(0, 0), (1, 0), (2, 0)]

        self.memory = {}  # {hash_key: [COLL, NONCOLL]}
        self.current_cycle = 0
        self.pending_accesses = []  # [{completion_cycle, op_type, hash_key, bank_id, ...}]

        # 维护每个Bank的当前待处理请求数，用于快速计算延迟
        self.bank_pending_counts = [0] * num_banks

        # 统计信息
        self.read_count = 0
        self.write_count = 0
        self.bank_conflicts = 0  # 因Bank忙碌而产生的等待次数
        self.bank_access_counts = [0] * num_banks

    def _get_bank_id(self, hash_key):
        """
        根据hash_key的二进制位计算Bank ID
        hash_key是二进制字符串 (e.g., "001101010010")
        bank_config定义了使用哪些bit位来计算Bank ID
        """
        # 从hash_key长度推断quant_bits: hash_key长度 = num_dims * quant_bits = 3 * quant_bits
        hash_key_len = len(hash_key)
        quant_bits = hash_key_len // 3 if hash_key_len > 0 else 0

        bank_id = 0
        for i, (dim, bit) in enumerate(self.bank_config):
            # 计算该维度对应的bit位在二进制字符串中的位置
            # 例如: quant_bits=4时，维度0占bit 0-3，维度1占bit 4-7，维度2占bit 8-11
            # bit=0是最低位，对应位置 dim*quant_bits + (quant_bits-1)
            bit_pos = dim * quant_bits + (quant_bits - 1 - bit)
            if bit_pos < len(hash_key):
                # 从hash_key中直接提取该bit位的值
                bit_val = int(hash_key[bit_pos])
                bank_id |= bit_val << i

        return bank_id % self.num_banks

    def _calculate_completion_cycle(self, cycle, bank_id):
        """计算操作完成周期，处理Bank冲突"""
        # 获取该Bank当前的待处理请求数
        pending_for_bank = self.bank_pending_counts[bank_id]

        if self.enable_conflict_check:
            # 双端口SRAM: 每周期最多2个并行访问
            wait_cycles = pending_for_bank // 2
            if pending_for_bank >= 2:
                self.bank_conflicts += 1
            return cycle + wait_cycles + ONE_CYCLE_DELAY
        else:
            return cycle + ONE_CYCLE_DELAY

    def read_request(self, copu_id, hash_key, cycle):
        """
        提交读请求
        """
        bank_id = self._get_bank_id(hash_key)
        self.bank_access_counts[bank_id] += 1

        completion_cycle = self._calculate_completion_cycle(cycle, bank_id)

        self.pending_accesses.append(
            {
                "completion_cycle": completion_cycle,
                "op_type": "read",
                "hash_key": hash_key,
                "bank_id": bank_id,
                "result": self.memory.get(hash_key, [0, 0]),
            }
        )
        self.bank_pending_counts[bank_id] += 1
        self.read_count += 1

        return self.memory.get(hash_key, [0, 0]), completion_cycle

    def write_request(self, copu_id, hash_key, delta_coll, delta_noncoll, cycle):
        """
        提交写请求
        """
        bank_id = self._get_bank_id(hash_key)
        self.bank_access_counts[bank_id] += 1

        completion_cycle = self._calculate_completion_cycle(cycle, bank_id)

        self.pending_accesses.append(
            {
                "completion_cycle": completion_cycle,
                "op_type": "write",
                "hash_key": hash_key,
                "bank_id": bank_id,
                "delta_coll": delta_coll,
                "delta_noncoll": delta_noncoll,
            }
        )
        self.bank_pending_counts[bank_id] += 1
        self.write_count += 1

        return completion_cycle

    def advance_cycle(self):
        """推进一个周期"""
        cycle = self.current_cycle

        # 分离已完成和未完成的请求
        completed = []
        remaining = []

        for access in self.pending_accesses:
            if access["completion_cycle"] <= cycle:
                completed.append(access)
            else:
                remaining.append(access)

        # 处理已完成的写操作
        for access in completed:
            if access["op_type"] == "write":
                hash_key = access["hash_key"]
                delta_c = access["delta_coll"]
                delta_n = access["delta_noncoll"]

                if hash_key not in self.memory:
                    self.memory[hash_key] = [0, 0]

                self.memory[hash_key][0] = min(
                    self.memory[hash_key][0] + delta_c, MAX_COLLISION_COUNT
                )
                self.memory[hash_key][1] = min(
                    self.memory[hash_key][1] + delta_n, MAX_COLLISION_COUNT
                )

            # 请求完成，减少对应Bank的排队计数
            # 注意：这里减少计数是为了让后续请求的延迟计算正确
            # 实际上，当一个请求完成时，它占用的那个时隙就过去了，
            # 但我们在calculate_completion_cycle中使用的是当前的队列深度。
            # 这种简单的计数方法在稳态下是近似正确的。
            bank_id = access["bank_id"]
            if self.bank_pending_counts[bank_id] > 0:
                self.bank_pending_counts[bank_id] -= 1

        self.pending_accesses = remaining
        self.current_cycle += 1

    def reset(self):
        self.memory = {}
        self.pending_accesses = []
        self.bank_pending_counts = [0] * self.num_banks

    def get_stats(self):
        total_accesses = self.read_count + self.write_count
        conflict_rate = (
            self.bank_conflicts / total_accesses if total_accesses > 0 else 0.0
        )

        # 计算负载均衡度 (方差)
        if self.num_banks > 0:
            avg_access = sum(self.bank_access_counts) / self.num_banks
            variance = (
                sum((x - avg_access) ** 2 for x in self.bank_access_counts)
                / self.num_banks
            )
            load_balance_std = variance**0.5
        else:
            load_balance_std = 0

        return {
            "total_reads": self.read_count,
            "total_writes": self.write_count,
            "total_conflicts": self.bank_conflicts,
            "conflict_rate": conflict_rate,
            "entries_used": len(self.memory),
            "bank_access_counts": self.bank_access_counts,
            "load_balance_std": load_balance_std,
            "bank_config": self.bank_config,
        }


# ======================== COPUModule ========================
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
        self.local_colldict = {}

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

    def check_conservation_law(self):
        """检查守恒律是否被违反"""
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

    def _process_oocd_completion(self, sample_rate):
        """处理OOCD完成和CHT更新"""
        oocd_cycles_delta = 0
        for oocd_id in range(self.num_oocds):
            oocd = self.oocds[oocd_id]
            if oocd.busy == 1:
                oocd_cycles_delta += 1
                if oocd.free_cycle <= self.cycle:
                    self.query_count += 1
                    self.oocds[oocd_id] = OOCDState(
                        hash_key="", result=1, busy=0, free_cycle=self.cycle
                    )
                    if oocd.result == 0:  # 碰撞
                        self.coll_found = True

                    # 更新CHT
                    if self.cht_scheduler is not None:
                        delta_coll = 1 if oocd.result == 0 else 0
                        delta_noncoll = 1 if oocd.result == 1 else 0
                        self.cht_scheduler.submit_write(
                            self.copu_id, oocd.hash_key, delta_coll, delta_noncoll
                        )
                    else:
                        self.local_colldict = su.update_collision_dict(
                            self.local_colldict, oocd.hash_key, oocd.result, sample_rate
                        )
        return oocd_cycles_delta

    def _dispatch_new_tasks(self):
        """分派新任务给空闲的OOCD"""
        dequeued_this_cycle = False
        for oocd_id in range(self.num_oocds):
            oocd = self.oocds[oocd_id]
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
                    # 保持空闲状态
                    if oocd.busy == 0:  # 已经是空闲状态，无需重复赋值
                        pass
                    else:
                        self.oocds[oocd_id] = OOCDState(
                            hash_key=0, result=0, busy=0, free_cycle=0
                        )

    def _predict_next_config(self, bins, threshold):
        """预测下一个配置并入队"""
        if len(self.linklist) > 0:
            link = self.linklist[0]
            linkcoll = self.linklist_coll[0]

            keyy = su.compute_hash_keyy(link, bins)

            # 查询CHT
            is_ready = False
            coll_count, noncoll_count = 0, 0

            if self.cht_scheduler is not None:
                is_ready, data = self.cht_scheduler.get_read_result(keyy)
                if is_ready:
                    coll_count, noncoll_count = data
            else:
                is_ready = True
                if keyy in self.local_colldict:
                    coll_count = self.local_colldict[keyy][0]
                    noncoll_count = self.local_colldict[keyy][1]
                else:
                    coll_count, noncoll_count = 0, 0

            if is_ready:
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

    def step(self, bins, threshold, sample_rate):
        """执行一个仿真周期"""
        # 1. 处理OOCD完成和CHT更新
        oocd_cycles_delta = self._process_oocd_completion(sample_rate)

        # 2. 分派新任务
        self._dispatch_new_tasks()
        self.check_conservation_law()

        # 3. 预测新配置
        self._predict_next_config(bins, threshold)
        self.check_conservation_law()

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
        self,
        num_copus,
        cht_size=CHT_DEFAULT_SIZE,
        enable_conflict_check=True,
        cht_type="dual_port",
        **cht_kwargs,
    ):
        self.num_copus = num_copus
        # 根据 cht_type 字符串选择 CHT 类
        cht_classes = {
            "dual_port": DualPortSRAM_CHT,
            "multi_bank": MultiBankSRAM_CHT,
        }
        cht_class = cht_classes.get(cht_type, DualPortSRAM_CHT)
        self.cht = cht_class(
            size=cht_size, enable_conflict_check=enable_conflict_check, **cht_kwargs
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
        cht_type="dual_port",
        **cht_kwargs,
    ):
        self.num_copus = num_copus

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


def analyze_multi_copu_performance(results):
    """
    分析多COPU系统的性能指标

    Args:
        results (dict): MultiCOPU_Scheduler.simulate 返回的结果字典

    Returns:
        dict: 包含吞吐量、利用率、CHT冲突率、负载均衡等指标
    """
    cht_stats = results["cht_stats"]
    copu_stats = results["copus"]

    # KPI 1: 系统吞吐量
    total_queries = sum(c["total_queries"] for c in copu_stats)
    system_throughput = total_queries / max(1, results["total_cycles"])

    # KPI 2: COPU平均利用率
    avg_utilization = (
        sum(c["oocd_utilization"] for c in copu_stats) / len(copu_stats)
        if len(copu_stats) > 0
        else 0.0
    )

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
