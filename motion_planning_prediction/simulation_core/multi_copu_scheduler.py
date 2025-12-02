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
        **cht_kwargs,
    ):
        self.num_copus = num_copus
        self.copus_per_edge = copus_per_edge if copus_per_edge else num_copus
        self.num_groups = max(1, num_copus // self.copus_per_edge)

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
        
        # 任务管理
        self.edge_queue = deque()
        self.edge_results = {}  # edge_idx -> result ('collision', 'safe')
        self.group_status = [{'state': 'idle', 'edge_idx': -1} for _ in range(self.num_groups)]
        
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
            self.group_status[i] = {'state': 'idle', 'edge_idx': -1}
            
        # 重置所有COPU
        for copu in self.copus:
            copu.reset_task()

    def simulate(self, bins, threshold, sample_rate):
        """
        执行多COPU协同仿真（动态调度）

        Returns:
            results dict with global metrics and per-COPU stats
        """
        while self.edge_queue or any(g['state'] == 'busy' for g in self.group_status):
            # 1. 分配任务给空闲组
            for group_id in range(self.num_groups):
                if self.group_status[group_id]['state'] == 'idle' and self.edge_queue:
                    edge_idx = self.edge_queue.popleft()
                    self._assign_edge_to_group(group_id, edge_idx)

            # 2. 每个COPU执行一步
            any_copu_active = False
            for copu in self.copus:
                # 只有当COPU有任务时才执行step
                if not copu.everything_free:
                    continue_sim, _ = copu.step(bins, threshold, sample_rate)
                    if continue_sim:
                        any_copu_active = True
            
            # 3. 检查组任务完成情况
            for group_id in range(self.num_groups):
                if self.group_status[group_id]['state'] == 'busy':
                    self._check_group_status(group_id)

            # 4. 推进CHT调度器
            self.cht_scheduler.advance_cycle()

            # 如果所有COPU都空闲且队列为空，则退出（由while条件保证，但这里可以作为双重检查）
            if not any_copu_active and not self.edge_queue and all(g['state'] == 'idle' for g in self.group_status):
                break

            self.cycle += 1

        # 收集统计
        # 检查是否有任何edge发生了碰撞
        self.global_coll_found = any(res == 'collision' for res in self.edge_results.values())
        
        results = {
            "total_cycles": self.cycle,
            "collision_found": self.global_coll_found,
            "copus": [copu.get_stats() for copu in self.copus],
            "cht_stats": self.cht_scheduler.cht.get_stats(),
            "edge_results": self.edge_results
        }

        return results

    def _assign_edge_to_group(self, group_id, edge_idx):
        """将edge分配给指定COPU组"""
        edge_data = self.all_data[edge_idx]
        edge_coll = self.all_coll[edge_idx]
        edge_cycle = self.all_cycles[edge_idx] if self.all_cycles else None
        
        sub_coords, sub_colls, sub_cycles = su.allocate_edge_data_to_copus(
            edge_data, edge_coll, edge_cycle, self.copus_per_edge
        )
        
        start_copu = group_id * self.copus_per_edge
        for i in range(self.copus_per_edge):
            copu_id = start_copu + i
            self.copus[copu_id].load_data(sub_coords[i], sub_colls[i], sub_cycles[i])
            
        self.group_status[group_id] = {'state': 'busy', 'edge_idx': edge_idx}

    def _check_group_status(self, group_id):
        """检查组内任务是否完成或发现碰撞"""
        start_copu = group_id * self.copus_per_edge
        group_copus = self.copus[start_copu : start_copu + self.copus_per_edge]
        
        # 检查碰撞
        if any(c.coll_found for c in group_copus):
            # 发现碰撞！
            edge_idx = self.group_status[group_id]['edge_idx']
            self.edge_results[edge_idx] = 'collision'
            self.group_status[group_id] = {'state': 'idle', 'edge_idx': -1}
            
            # 立即停止组内所有COPU的任务
            for c in group_copus:
                c.reset_task()
                
        # 检查完成（所有COPU都空闲）
        elif all(c.everything_free for c in group_copus):
            edge_idx = self.group_status[group_id]['edge_idx']
            self.edge_results[edge_idx] = 'safe'
            self.group_status[group_id] = {'state': 'idle', 'edge_idx': -1}
            # 任务已完成，无需额外重置，因为everything_free已经是True
            # 但为了保险起见，可以调用reset_task清理队列
            for c in group_copus:
                c.reset_task()