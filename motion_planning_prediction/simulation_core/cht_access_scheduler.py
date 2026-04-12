"""
CHT访问调度器模块
"""

from copy import deepcopy

from .cht import DualPortSRAM_CHT, MultiBankSRAM_CHT


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
        cht_size=4096,
        enable_conflict_check=True,
        cht_type="dual_port",
        **cht_kwargs,
    ):
        self.num_copus = num_copus
        self.cht_type = cht_type
        # 根据 cht_type 字符串选择 CHT 类
        cht_classes = {
            "dual_port": DualPortSRAM_CHT,
            "multi_bank": MultiBankSRAM_CHT,
            "distri_dual_port": DualPortSRAM_CHT,
            "distri_multi_bank": MultiBankSRAM_CHT,
        }
        if cht_type not in cht_classes:
            raise ValueError(f"Unsupported cht_type: {cht_type}")

        cht_class = cht_classes[cht_type]
        self.distributed_mode = cht_type.startswith("distri_")
        if self.distributed_mode:
            self.chts = [
                cht_class(
                    size=cht_size,
                    enable_conflict_check=enable_conflict_check,
                    **cht_kwargs,
                )
                for _ in range(num_copus)
            ]
            # 兼容旧代码中对 self.cht 的访问
            self.cht = self.chts[0]
        else:
            self.cht = cht_class(
                size=cht_size, enable_conflict_check=enable_conflict_check, **cht_kwargs
            )
            self.chts = [self.cht]
        self.current_cycle = 0

    def _resolve_cht(self, copu_id):
        if not self.distributed_mode:
            return self.cht
        if copu_id < 0 or copu_id >= self.num_copus:
            raise ValueError(f"Invalid copu_id for distributed CHT: {copu_id}")
        return self.chts[copu_id]

    def get_read_result(self, hash_key, copu_id=0):
        """
        查询hash_key的读结果，自动管理pending状态
        首次查询时发起读请求，后续查询相同地址会自动去重

        Returns:
            (is_ready, data)
        """
        target_cht = self._resolve_cht(copu_id)

        # 在pending_accesses中查找该hash_key的读请求
        for access in target_cht.pending_accesses:
            if access["op_type"] == "read" and access["hash_key"] == hash_key:
                # 找到待决的读请求，检查是否就绪
                if self.current_cycle >= access["completion_cycle"]:
                    return True, access["result"]
                else:
                    return False, None

        # 没有待决请求，发起新的读请求
        data, comp_cycle = target_cht.read_request(copu_id, hash_key, self.current_cycle)
        if comp_cycle <= self.current_cycle:
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
        if not self.distributed_mode:
            target_cht = self.cht

            # 在pending_accesses中查找该hash_key的写请求
            for access in target_cht.pending_accesses:
                if access["op_type"] == "write" and access["hash_key"] == hash_key:
                    # 找到待决写请求，直接合并delta
                    access["delta_coll"] += delta_coll
                    access["delta_noncoll"] += delta_noncoll
                    return

            # 没有待决请求，发起新的写请求
            target_cht.write_request(
                copu_id, hash_key, delta_coll, delta_noncoll, self.current_cycle
            )
            return

        # distributed 模式：广播到每个 COPU 专属 CHT
        for cht in self.chts:
            for access in cht.pending_accesses:
                if access["op_type"] == "write" and access["hash_key"] == hash_key:
                    access["delta_coll"] += delta_coll
                    access["delta_noncoll"] += delta_noncoll
                    break
            else:
                cht.write_request(
                    copu_id, hash_key, delta_coll, delta_noncoll, self.current_cycle
                )

    def load_warmstart_package(self, warmstart_package):
        """加载warm-start包到当前CHT实例。

        支持两种输入：
        - 直接传入 memory 字典
        - 传入包含 ``memory`` 字段的完整包
        """
        if warmstart_package is None:
            return

        memory = warmstart_package.get("memory", warmstart_package)
        if not isinstance(memory, dict):
            raise TypeError("warmstart_package must be a dict or contain a memory dict")

        for cht in self.chts:
            cht.memory = deepcopy(memory)
            cht.pending_accesses = []
            cht.bank_pending_counts = [0] * cht.num_banks
            cht.current_cycle = 0
            cht.read_count = 0
            cht.write_count = 0
            cht.conflicts = 0

        self.current_cycle = 0

    def advance_cycle(self):
        """推进周期：清理已完成请求，由DualPortSRAM_CHT执行写操作"""
        # 推进SRAM周期（内部执行所有就绪的写操作并清理已完成请求）
        for cht in self.chts:
            cht.advance_cycle()
        self.current_cycle += 1

    def get_stats(self):
        """返回统一格式的CHT统计；distributed模式下聚合所有实例。"""
        if not self.distributed_mode:
            return self.cht.get_stats()

        total_reads = 0
        total_writes = 0
        total_conflicts = 0
        total_entries = 0
        for cht in self.chts:
            cht_stats = cht.get_stats()
            total_reads += cht_stats.get("total_reads", 0)
            total_writes += cht_stats.get("total_writes", 0)
            total_conflicts += cht_stats.get("total_conflicts", 0)
            total_entries += cht_stats.get("entries_used", 0)

        total_accesses = total_reads + total_writes
        return {
            "total_reads": total_reads,
            "total_writes": total_writes,
            "total_conflicts": total_conflicts,
            "conflict_rate": (
                total_conflicts / total_accesses if total_accesses > 0 else 0.0
            ),
            "entries_used": total_entries,
        }
