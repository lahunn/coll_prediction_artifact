"""
CHT访问调度器模块
"""

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
        if len(self.cht.pending_accesses) <= 2 and 1 == 0:  # ONE_CYCLE_DELAY == 0
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