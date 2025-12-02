"""
CHT (Collision History Table) 模块

实现碰撞历史表的硬件仿真，包括双端口SRAM和多Bank SRAM两种实现。
"""

# import numpy as np

# ======================== Constants ========================
MAX_COLLISION_COUNT = 15
CHT_DEFAULT_SIZE = 4096
ONE_CYCLE_DELAY = 1


# ======================== ConfigurableCHT ========================


class ConfigurableCHT:
    """
    可配置的碰撞历史表 (CHT)
    支持双端口SRAM (num_banks=1, ports_per_bank=2) 和多Bank单端口SRAM (num_banks>1, ports_per_bank=1)
    """

    def __init__(
        self,
        size=CHT_DEFAULT_SIZE,
        num_banks=1,
        ports_per_bank=2,
        bank_config=None,
        enable_conflict_check=True,
    ):
        self.size = size
        self.num_banks = num_banks
        self.ports_per_bank = ports_per_bank
        self.enable_conflict_check = enable_conflict_check
        self.bank_config = bank_config if bank_config else [0, 1, 2]

        self.memory = {}  # {hash_key: [COLL, NONCOLL]}

        self.current_cycle = 0
        self.pending_accesses = []  # [{completion_cycle, op_type, hash_key, bank_id, ...}]

        if num_banks > 1:
            self.bank_pending_counts = [0] * num_banks

        # 统计信息
        self.read_count = 0
        self.write_count = 0
        self.conflicts = 0  # 冲突次数
        self.bank_access_counts = [0] * num_banks if num_banks > 1 else [0]

    def _get_bank_id(self, hash_key):
        """
        根据hash_key的二进制位计算Bank ID

        hash_key使用bit interleaving格式：bit0_dim0, bit0_dim1, bit0_dim2, bit1_dim0, ...
        """
        if self.num_banks == 1:
            return 0
        bank_id = 0
        for i, bit_pos in enumerate(self.bank_config):
            if bit_pos < len(hash_key):
                bit_val = int(hash_key[bit_pos])
                bank_id |= bit_val << i

        return bank_id % self.num_banks

    def _calculate_completion_cycle(self, cycle, bank_id):
        """计算操作完成周期，处理端口冲突"""
        if self.num_banks == 1:
            num_pending = len(self.pending_accesses)
            if not self.enable_conflict_check:
                return cycle + ONE_CYCLE_DELAY

            wait_cycles = num_pending // self.ports_per_bank
            if num_pending >= self.ports_per_bank:
                self.conflicts += 1
            return cycle + wait_cycles + ONE_CYCLE_DELAY
        else:
            pending_for_bank = self.bank_pending_counts[bank_id]
            if not self.enable_conflict_check:
                return cycle + ONE_CYCLE_DELAY

            wait_cycles = pending_for_bank // self.ports_per_bank
            if pending_for_bank >= self.ports_per_bank:
                self.conflicts += 1
            return cycle + wait_cycles + ONE_CYCLE_DELAY

    def read_request(self, copu_id, hash_key, cycle):
        """
        提交读请求。
        Returns: (result, completion_cycle)
        """
        bank_id = self._get_bank_id(hash_key)

        if self.num_banks > 1:
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

        if self.num_banks > 1:
            self.bank_pending_counts[bank_id] += 1

        self.read_count += 1
        return self.memory.get(hash_key, [0, 0]), completion_cycle

    def write_request(self, copu_id, hash_key, delta_coll, delta_noncoll, cycle):
        """
        提交写请求。
        Returns: completion_cycle
        """
        bank_id = self._get_bank_id(hash_key)

        if self.num_banks > 1:
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

        if self.num_banks > 1:
            self.bank_pending_counts[bank_id] += 1

        self.write_count += 1
        return completion_cycle

    def advance_cycle(self):
        """推进一个周期，执行所有就绪的写操作"""
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

            bank_id = access["bank_id"]
            if self.num_banks > 1 and self.bank_pending_counts[bank_id] > 0:
                self.bank_pending_counts[bank_id] -= 1

        self.pending_accesses = remaining
        self.current_cycle += 1

    def reset(self):
        """重置CHT内容（不重置统计信息）"""
        self.memory = {}
        self.pending_accesses = []
        if self.num_banks > 1:
            self.bank_pending_counts = [0] * self.num_banks

    def get_stats(self):
        """返回CHT访问统计"""
        total_accesses = self.read_count + self.write_count
        conflict_rate = self.conflicts / total_accesses if total_accesses > 0 else 0.0

        if self.num_banks == 1:
            return {
                "total_reads": self.read_count,
                "total_writes": self.write_count,
                "total_conflicts": self.conflicts,
                "conflict_rate": conflict_rate,
                "entries_used": len(self.memory),
            }
        else:
            avg_access = sum(self.bank_access_counts) / self.num_banks
            variance = (
                sum((x - avg_access) ** 2 for x in self.bank_access_counts)
                / self.num_banks
            )
            load_balance_std = variance**0.5

            return {
                "total_reads": self.read_count,
                "total_writes": self.write_count,
                "total_conflicts": self.conflicts,
                "conflict_rate": conflict_rate,
                "entries_used": len(self.memory),
                "bank_access_counts": self.bank_access_counts,
                "load_balance_std": load_balance_std,
                "bank_config": self.bank_config,
            }


# ======================== DualPortSRAM_CHT ========================


class DualPortSRAM_CHT(ConfigurableCHT):
    def __init__(self, size=CHT_DEFAULT_SIZE, enable_conflict_check=True):
        super().__init__(
            size=size,
            num_banks=1,
            ports_per_bank=2,
            enable_conflict_check=enable_conflict_check,
        )


# ======================== MultiBankSRAM_CHT ========================


class MultiBankSRAM_CHT(ConfigurableCHT):
    def __init__(
        self,
        size=CHT_DEFAULT_SIZE,
        num_banks=8,
        bank_config=None,
        enable_conflict_check=True,
    ):
        super().__init__(
            size=size,
            num_banks=num_banks,
            ports_per_bank=2,
            bank_config=bank_config,
            enable_conflict_check=enable_conflict_check,
        )
