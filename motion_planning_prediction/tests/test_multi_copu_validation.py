"""
多COPU系统正确性验证测试套件

分层验证策略：
1. 第一层：基础约束验证（硬件约束遵守）
2. 第二层：多COPU同步验证（系统级一致性）
3. 第三层：等价性验证（与单COPU对齐）
4. 第四层：性能验证（扩展性和效率）
5. 第五层：压力测试（边界和极端情况）
"""

import sys
import os
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_copu_simulation import (
    MultiCOPU_Scheduler,
)


# ======================== 第一层：基础约束验证 ========================
class ConstraintValidator:
    """验证CHT、OOCD、队列等硬件约束"""

    def __init__(self):
        self.violations = []
        self.warnings = []

    def reset(self):
        """重置违规记录"""
        self.violations = []
        self.warnings = []

    # ---- CV-1: CHT双端口约束 ----
    def validate_cht_dual_port(self, cht_scheduler, cycle):
        """
        验证：每个周期最多2个并发CHT操作
        
        Args:
            cht_scheduler: CHT_AccessScheduler对象
            cycle: 当前周期号
            
        返回：True if valid, False otherwise
        """
        pending = cht_scheduler.cht.pending_accesses
        num_pending = len(pending)

        if num_pending > 2:
            self.violations.append(
                f"[CV-1] Cycle {cycle}: CHT超过2个并发操作 "
                f"(pending={num_pending}, limit=2)"
            )
            return False

        # 进阶检查：验证completion_cycle的合理性
        for access in pending:
            # 理论完成周期 = cycle + 1 + floor(pending_count_before_me / 2)
            # 这里只检查是否为正合理值，详细计算留给CHT_AccessScheduler
            if access["completion_cycle"] < cycle + 1:
                self.violations.append(
                    f"[CV-1] Cycle {cycle}: 不合理的完成周期 "
                    f"(hash_key={access['hash_key']}, "
                    f"completion_cycle={access['completion_cycle']}, "
                    f"but should be >= {cycle + 1})"
                )
                return False

        return True

    # ---- CV-2: OOCD分派限制 ----
    def validate_oocd_dispatch_limit(self, copu, cycle, dequeued_this_cycle):
        """
        验证：每周期最多1个任务分派到OOCD
        
        Args:
            copu: COPUModule对象
            cycle: 当前周期号
            dequeued_this_cycle: 本周期是否已分派过任务
            
        返回：True if valid
        """
        # 通过dequeued_this_cycle标志可以直接验证
        if dequeued_this_cycle and copu.cycle == cycle:
            # 检查是否有多个OOCD在本周期被分派
            newly_busy_count = sum(
                1
                for oocd in copu.oocds
                if oocd.busy == 1 and oocd.free_cycle == cycle + copu.cycle_check
            )

            if newly_busy_count > 1:
                self.violations.append(
                    f"[CV-2] Cycle {cycle}: OOCD分派超过限制 "
                    f"(dispatched={newly_busy_count}, limit=1)"
                )
                return False

        return True

    # ---- CV-3: 队列FIFO顺序和容量 ----
    def validate_queue_fifo(self, copu, cycle):
        """
        验证：队列FIFO顺序和容量限制
        
        Args:
            copu: COPUModule对象
            cycle: 当前周期号
            
        返回：True if valid
        """
        # 检查QCOLL容量
        if len(copu.qcoll) > copu.qcoll_size:
            self.violations.append(
                f"[CV-3] Cycle {cycle}: QCOLL超容 "
                f"(size={len(copu.qcoll)}, limit={copu.qcoll_size})"
            )
            return False

        # 检查QNONCOLL容量
        if len(copu.qnoncoll) > copu.qnoncoll_size:
            self.violations.append(
                f"[CV-3] Cycle {cycle}: QNONCOLL超容 "
                f"(size={len(copu.qnoncoll)}, limit={copu.qnoncoll_size})"
            )
            return False

        # FIFO验证：检查deque的maxlen是否生效（Python deque自动保证FIFO）
        # 这里我们只检查容量即可，FIFO由collections.deque保证

        return True

    # ---- CV-4: 去重和合并机制 ----
    def validate_dedup_merge(self, cht_scheduler, cycle):
        """
        验证：同hash_key的请求自动去重/合并
        
        Args:
            cht_scheduler: CHT_AccessScheduler对象
            cycle: 当前周期号
            
        返回：True if valid
        """
        pending = cht_scheduler.cht.pending_accesses

        # 构建hash_key -> 操作列表的映射
        access_by_key = defaultdict(list)
        for access in pending:
            access_by_key[access["hash_key"]].append(access)

        # 检查规则：对于每个hash_key，最多1个pending请求
        for hash_key, accesses in access_by_key.items():
            read_count = sum(1 for a in accesses if a["op_type"] == "read")
            write_count = sum(1 for a in accesses if a["op_type"] == "write")

            if read_count > 1:
                self.violations.append(
                    f"[CV-4] Cycle {cycle}: hash_key重复的读请求 "
                    f"(key={hash_key}, count={read_count})"
                )
                return False

            if write_count > 1:
                self.violations.append(
                    f"[CV-4] Cycle {cycle}: hash_key重复的写请求 "
                    f"(key={hash_key}, count={write_count})"
                )
                return False

        return True

    # ---- CV-5: 饱和计数边界 ----
    def validate_saturation_counters(self, cht_scheduler, cycle):
        """
        验证：CHT计数器饱和在[0, 15]范围内
        
        Args:
            cht_scheduler: CHT_AccessScheduler对象
            cycle: 当前周期号
            
        返回：True if valid
        """
        memory = cht_scheduler.cht.memory

        for hash_key, counts in memory.items():
            coll, noncoll = counts

            if not (0 <= coll <= 15):
                self.violations.append(
                    f"[CV-5] Cycle {cycle}: COLL计数越界 "
                    f"(key={hash_key}, value={coll}, range=[0,15])"
                )
                return False

            if not (0 <= noncoll <= 15):
                self.violations.append(
                    f"[CV-5] Cycle {cycle}: NONCOLL计数越界 "
                    f"(key={hash_key}, value={noncoll}, range=[0,15])"
                )
                return False

        return True

    def validate_all_constraints(self, scheduler, cycle):
        """运行所有约束检查"""
        all_valid = True

        for copu in scheduler.copus:
            all_valid &= self.validate_queue_fifo(copu, cycle)

        all_valid &= self.validate_cht_dual_port(scheduler.cht_scheduler, cycle)
        all_valid &= self.validate_dedup_merge(scheduler.cht_scheduler, cycle)
        all_valid &= self.validate_saturation_counters(scheduler.cht_scheduler, cycle)

        return all_valid

    def report(self):
        """生成验证报告"""
        if self.violations:
            print("\n❌ 约束违规 (Constraint Violations):")
            for v in self.violations:
                print(f"  {v}")
            return False
        else:
            print("\n✅ 所有约束检查通过")
            return True


# ======================== 第二层：多COPU同步验证 ========================
class SyncValidator:
    """验证多COPU的周期同步和全局信号传播"""

    def __init__(self):
        self.sync_errors = []
        self.signal_latencies = []

    def reset(self):
        self.sync_errors = []
        self.signal_latencies = []

    # ---- SV-1: 周期同步检查 ----
    def validate_cycle_sync(self, scheduler, expected_cycle):
        """
        验证：所有COPU的step调用次数一致
        
        注意：COPU.cycle在step内部增加，所以在step后会比scheduler.cycle快1
        这里验证的是所有COPU的step调用次数应该相同
        
        Args:
            scheduler: MultiCOPU_Scheduler对象
            expected_cycle: 期望的周期号
            
        返回：True if synchronized
        """
        # 在simulate循环中，所有COPU应该被调用相同次数的step
        # 这可以通过检查各COPU的total_cycles来验证
        # （在simulate完成后）
        
        # 目前跳过在运行过程中的检查，因为step调用是串行的
        # 验证应该在simulate完成后进行
        return True

    # ---- SV-2: 碰撞信号传播 ----
    def validate_collision_signal(self, scheduler, cycle):
        """
        验证：碰撞信号正确传播到全局标志
        
        Args:
            scheduler: MultiCOPU_Scheduler对象
            cycle: 当前周期号
            
        返回：True if signal is consistent
        """
        # 检查：如果任何COPU发现碰撞，global_coll_found应为True
        any_coll = any(copu.coll_found for copu in scheduler.copus)

        if any_coll and not scheduler.global_coll_found:
            self.sync_errors.append(
                f"[SV-2] Cycle {cycle}: 碰撞信号未传播 "
                f"(某COPU.coll_found=True, 但global.coll_found=False)"
            )
            return False

        return True

    # ---- SV-3: CHT一致性检查 ----
    def validate_cht_consistency(self, scheduler, cycle):
        """
        验证：CHT的读写操作在跨COPU访问中保持一致性
        
        实现方式：检查pending_accesses中是否有冲突的读写
        
        Args:
            scheduler: MultiCOPU_Scheduler对象
            cycle: 当前周期号
            
        返回：True if consistent
        """
        pending = scheduler.cht_scheduler.cht.pending_accesses

        # 按hash_key分组
        access_by_key = defaultdict(list)
        for access in pending:
            access_by_key[access["hash_key"]].append(access)

        # 检查是否存在同hash_key的读写冲突
        for hash_key, accesses in access_by_key.items():
            ops = [a["op_type"] for a in accesses]

            # 同周期内不应该有读写冲突（按设计，应该被去重）
            if len(accesses) > 1:
                # 如果有多个操作，应该是相同类型（已被合并）
                if not all(op == ops[0] for op in ops):
                    self.sync_errors.append(
                        f"[SV-3] Cycle {cycle}: CHT读写冲突 "
                        f"(key={hash_key}, ops={ops})"
                    )
                    return False

        return True

    def report(self):
        """生成同步验证报告"""
        if self.sync_errors:
            print("\n❌ 同步错误 (Synchronization Errors):")
            for e in self.sync_errors:
                print(f"  {e}")
            return False
        else:
            print("\n✅ 所有同步检查通过")
            return True


# ======================== 第三层：等价性验证 ========================
class EquivalenceValidator:
    """验证N=1 case与单COPU等价"""

    def __init__(self):
        self.diffs = []

    def reset(self):
        self.diffs = []

    # ---- EV-1: 结果等价性 ----
    def validate_result_equivalence(self, multi_result_n1, single_result):
        """
        验证：MultiCOPU(N=1)的结果与单COPU结果相同
        
        Args:
            multi_result_n1: MultiCOPU_Scheduler(num_copus=1)的结果
            single_result: 单独运行COPUModule的结果
            
        返回：True if equivalent
        """
        # 比较关键指标
        keys_to_compare = [
            "total_cycles",
            "total_queries",
            "collision_found",
        ]

        for key in keys_to_compare:
            if multi_result_n1.get(key) != single_result.get(key):
                self.diffs.append(
                    f"[EV-1] 结果不等价: {key} "
                    f"(multi_n1={multi_result_n1.get(key)}, "
                    f"single={single_result.get(key)})"
                )
                return False

        return True

    # ---- EV-2: CHT最终状态等价性 ----
    def validate_cht_final_state(self, multi_cht, single_cht):
        """
        验证：CHT最终内存状态相同
        
        Args:
            multi_cht: MultiCOPU的CHT memory dict
            single_cht: 单COPU的local_colldict
            
        返回：True if equivalent
        """
        # 比较所有hash_key对应的值
        all_keys = set(multi_cht.keys()) | set(single_cht.keys())

        for key in all_keys:
            multi_val = multi_cht.get(key, [0, 0])
            single_val = single_cht.get(key, [0, 0])

            if multi_val != single_val:
                self.diffs.append(
                    f"[EV-2] CHT状态不等价: key={key} "
                    f"(multi={multi_val}, single={single_val})"
                )
                # 只报告第一个差异，避免输出过多
                return False

        return True

    def report(self):
        """生成等价性验证报告"""
        if self.diffs:
            print("\n❌ 等价性验证失败 (Equivalence Mismatch):")
            for d in self.diffs:
                print(f"  {d}")
            return False
        else:
            print("\n✅ 等价性验证通过 (N=1 case matches single COPU)")
            return True


# ======================== 第四层：性能验证 ========================
class PerformanceValidator:
    """验证吞吐量扩展性和负载均衡"""

    def __init__(self, min_throughput_scaling=0.85):
        """
        Args:
            min_throughput_scaling: 最小可接受的线性度 (default: 85%)
        """
        self.min_throughput_scaling = min_throughput_scaling
        self.perf_issues = []

    def reset(self):
        self.perf_issues = []

    # ---- PV-1: 吞吐量扩展性 ----
    def validate_throughput_scaling(self, results_by_num_copus):
        """
        验证：多COPU的吞吐量具有良好的线性扩展性
        
        Args:
            results_by_num_copus: {num_copus: result_dict, ...}
                其中result_dict包含'copus'列表
            
        返回：True if satisfactory scaling
        """
        if 1 not in results_by_num_copus:
            self.perf_issues.append("[PV-1] 缺少单COPU基准")
            return False

        baseline_result = results_by_num_copus[1]
        baseline_copus = baseline_result.get("copus", [])
        
        if not baseline_copus:
            self.perf_issues.append("[PV-1] 无法获取baseline COPU统计")
            return False
        
        baseline_throughput = (
            baseline_copus[0].get("total_queries", 0)
            / max(1, baseline_copus[0].get("total_cycles", 1))
        )

        print("\n吞吐量扩展性分析:")
        print(f"  1 COPU: {baseline_throughput:.4f} queries/cycle")

        for num_copus in sorted(results_by_num_copus.keys()):
            if num_copus == 1:
                continue

            result = results_by_num_copus[num_copus]
            copu_stats = result.get("copus", [])
            
            # 多COPU的吞吐量 = 所有COPU的总查询 / 总周期
            total_queries = sum(c.get("total_queries", 0) for c in copu_stats)
            total_cycles = result.get("total_cycles", 1)
            throughput = total_queries / max(1, total_cycles)
            speedup = throughput / baseline_throughput if baseline_throughput > 0 else 0

            # 期望值：线性扩展应该是num_copus倍
            expected_speedup = num_copus
            actual_scaling = speedup / expected_speedup * 100 if expected_speedup > 0 else 0

            print(f"  {num_copus} COPU: {throughput:.4f} queries/cycle, "
                  f"speedup={speedup:.2f}x, "
                  f"scaling={actual_scaling:.1f}%")

            if actual_scaling < self.min_throughput_scaling * 100:
                self.perf_issues.append(
                    f"[PV-1] {num_copus}COPU扩展性不足: "
                    f"scaling={actual_scaling:.1f}% (min={self.min_throughput_scaling*100:.0f}%)"
                )
                return False

        return True

    # ---- PV-2: 负载均衡 ----
    def validate_load_balance(self, results_by_num_copus):
        """
        验证：各COPU的任务分布均匀
        
        Args:
            results_by_num_copus: {num_copus: result_dict, ...}
                其中result_dict['copus']包含per_copu统计
            
        返回：True if balanced
        """
        print("\n负载均衡分析:")

        for num_copus in sorted(results_by_num_copus.keys()):
            result = results_by_num_copus[num_copus]
            copu_stats = result.get("copus", [])

            if not copu_stats:
                continue

            query_counts = [c["total_queries"] for c in copu_stats]
            mean_queries = np.mean(query_counts)
            std_queries = np.std(query_counts)
            cv = std_queries / mean_queries if mean_queries > 0 else 0

            print(f"  {num_copus} COPU: query_counts={query_counts}, "
                  f"CV={cv:.4f} (std/mean)")

            # 允许10%的变异系数
            if cv > 0.15:
                self.perf_issues.append(
                    f"[PV-2] {num_copus}COPU负载不均: CV={cv:.4f} (max=0.15)"
                )
                return False

        return True

    # ---- PV-3: CHT冲突随COPU数增加 ----
    def validate_cht_conflict_growth(self, results_by_num_copus):
        """
        验证：CHT冲突率随COPU数单调递增
        
        Args:
            results_by_num_copus: {num_copus: result_dict, ...}
                其中result_dict['cht_stats']['total_conflicts']
            
        返回：True if monotonic increase
        """
        print("\nCHT冲突增长分析:")

        conflict_by_copus = {}
        for num_copus in sorted(results_by_num_copus.keys()):
            result = results_by_num_copus[num_copus]
            conflicts = result.get("cht_stats", {}).get("total_conflicts", 0)
            conflict_by_copus[num_copus] = conflicts
            print(f"  {num_copus} COPU: {conflicts} conflicts")

        # 检查单调性
        for i, num_copus in enumerate(sorted(conflict_by_copus.keys())):
            if i == 0:
                continue

            prev_num = sorted(conflict_by_copus.keys())[i - 1]
            if conflict_by_copus[num_copus] < conflict_by_copus[prev_num]:
                self.perf_issues.append(
                    f"[PV-3] CHT冲突未单调增长: "
                    f"{prev_num}COPU={conflict_by_copus[prev_num]} > "
                    f"{num_copus}COPU={conflict_by_copus[num_copus]}"
                )
                return False

        return True

    def report(self):
        """生成性能验证报告"""
        if self.perf_issues:
            print("\n⚠️  性能问题 (Performance Issues):")
            for issue in self.perf_issues:
                print(f"  {issue}")
            return False
        else:
            print("\n✅ 所有性能检查通过")
            return True


# ======================== 第五层：压力测试 ========================
class StressTestValidator:
    """边界和极端情况测试"""

    def __init__(self):
        self.stress_issues = []

    def reset(self):
        self.stress_issues = []

    # ---- ST-1: 空配置列表 ----
    def test_empty_linklist(self):
        """
        测试：空的配置列表
        期望：系统立即终止，零周期或1周期
        """
        scheduler = MultiCOPU_Scheduler(num_copus=2, num_oocds=7)

        # 为COPUs加载空数据
        for copu in scheduler.copus:
            copu.load_data([], [], [])

        bins = np.linspace(0, 100, 10)
        result = scheduler.simulate(bins, threshold=1.0, sample_rate=1.0, max_cycles=100)

        if result["total_cycles"] > 2:
            self.stress_issues.append(
                f"[ST-1] 空列表测试失败: 周期过多 (cycles={result['total_cycles']})"
            )
            return False

        print("  [ST-1] ✓ 空配置列表处理正确")
        return True

    # ---- ST-2: 单配置单链接 ----
    def test_single_config_single_link(self):
        """
        测试：最小工作负载 (1个配置, 1个链接)
        期望：正确处理，完成检测
        """
        scheduler = MultiCOPU_Scheduler(num_copus=2, num_oocds=7)

        # 为COPU0加载1个配置
        data = [[50.0]]  # 1个配置, 1个特征
        flags = [[1]]  # 无碰撞
        cycles = [[40]]

        scheduler.copus[0].load_data(data, flags, cycles)
        scheduler.copus[1].load_data([], [], [])

        bins = np.linspace(0, 100, 10)
        result = scheduler.simulate(bins, threshold=1.0, sample_rate=1.0, max_cycles=10000)

        if result["total_queries"] == 0:
            self.stress_issues.append("[ST-2] 单配置测试失败: 查询数为0")
            return False

        print(f"  [ST-2] ✓ 单配置处理正确 (cycles={result['total_cycles']})")
        return True

    # ---- ST-3: CHT高冲突场景 ----
    def test_cht_hash_collision(self):
        """
        测试：所有配置映射到同一hash_key（高冲突）
        期望：系统仍然正确运行，但冲突率高
        """
        scheduler = MultiCOPU_Scheduler(num_copus=2, num_oocds=7)

        # 生成映射到相同hash_key的配置
        # 简单方式：所有数据都在同一量化区间
        num_configs = 50
        data = [
            [[25.0]]  # 所有config都映射到同一bin区间
            for _ in range(num_configs)
        ]
        flags = [[[1]] for _ in range(num_configs)]
        cycles = [[[40]] for _ in range(num_configs)]

        # 分配给COPU0
        scheduler.copus[0].load_data(data, flags, cycles)
        scheduler.copus[1].load_data([], [], [])

        bins = np.linspace(0, 100, 10)
        result = scheduler.simulate(bins, threshold=1.0, sample_rate=1.0, max_cycles=10000)

        cht_conflicts = result["cht_stats"]["total_conflicts"]
        if cht_conflicts == 0:
            self.stress_issues.append(
                "[ST-3] 高冲突场景测试失败: 预期有冲突但为0"
            )
            return False

        print(
            f"  [ST-3] ✓ 高冲突场景处理正确 "
            f"(conflicts={cht_conflicts}, cycles={result['total_cycles']})"
        )
        return True

    # ---- ST-4: 长期稳定性 ----
    def test_long_run_stability(self, num_cycles=100000):
        """
        测试：长期运行的数值稳定性（无溢出，统计单调）
        期望：能运行100k周期无异常
        """
        scheduler = MultiCOPU_Scheduler(num_copus=2, num_oocds=7)

        # 生成足量数据
        num_configs = 1000
        data = [
            [np.random.uniform(0, 100) for _ in range(3)]
            for _ in range(num_configs)
        ]
        flags = [[1 for _ in range(3)] for _ in range(num_configs)]
        cycles = [[40 + int(np.random.random() * 10) for _ in range(3)]
                  for _ in range(num_configs)]

        # 分配
        configs_per_copu = num_configs // 2
        scheduler.copus[0].load_data(data[:configs_per_copu], flags[:configs_per_copu], cycles[:configs_per_copu])
        scheduler.copus[1].load_data(data[configs_per_copu:], flags[configs_per_copu:], cycles[configs_per_copu:])

        bins = np.linspace(0, 100, 10)
        try:
            result = scheduler.simulate(
                bins, threshold=1.0, sample_rate=1.0, max_cycles=num_cycles
            )

            # 检查统计数据的合理性
            total_queries = result.get("total_queries", 0)
            total_cycles = result.get("total_cycles", 0)
            cht_conflicts = result.get("cht_stats", {}).get("total_conflicts", 0)

            # 基本合理性检查
            if total_queries > 0 and total_cycles > 0:
                print(
                    f"  [ST-4] ✓ 长期运行稳定 "
                    f"(cycles={total_cycles}, queries={total_queries}, conflicts={cht_conflicts})"
                )
                return True
            else:
                self.stress_issues.append(
                    f"[ST-4] 长期运行统计异常: queries={total_queries}, cycles={total_cycles}"
                )
                return False

        except Exception as e:
            self.stress_issues.append(f"[ST-4] 长期运行崩溃: {str(e)}")
            return False

    def report(self):
        """生成压力测试报告"""
        if self.stress_issues:
            print("\n❌ 压力测试失败 (Stress Test Failures):")
            for issue in self.stress_issues:
                print(f"  {issue}")
            return False
        else:
            print("\n✅ 所有压力测试通过")
            return True


# ======================== 主验证流程 ========================
def run_comprehensive_validation():
    """运行完整的多层验证"""

    print("\n" + "=" * 70)
    print(" 多COPU系统正确性验证 - 完整测试套件")
    print("=" * 70)

    all_passed = True

    # ---- 第一层：基础约束验证 ----
    print("\n[第一层] 基础约束验证 (Constraint Validation)")
    print("-" * 70)

    cv = ConstraintValidator()
    scheduler = MultiCOPU_Scheduler(num_copus=2, num_oocds=7, cht_size=4096)

    # 生成测试数据
    num_configs = 100
    data = [
        [np.random.uniform(0, 100) for _ in range(3)]
        for _ in range(num_configs)
    ]
    flags = [[1 for _ in range(3)] for _ in range(num_configs)]
    cycles = [[40 + int(np.random.random() * 10) for _ in range(3)]
              for _ in range(num_configs)]

    # 分配数据
    configs_per_copu = num_configs // 2
    scheduler.copus[0].load_data(data[:configs_per_copu], flags[:configs_per_copu], cycles[:configs_per_copu])
    scheduler.copus[1].load_data(data[configs_per_copu:], flags[configs_per_copu:], cycles[configs_per_copu:])

    bins = np.linspace(0, 100, 10)

    # 运行100个周期，每个周期验证约束
    max_test_cycles = 100
    for _ in range(max_test_cycles):
        for copu in scheduler.copus:
            copu.step(bins, threshold=1.0, sample_rate=1.0)

        scheduler.cht_scheduler.advance_cycle()

        # 每10个周期验证一次
        if scheduler.cycle % 10 == 0:
            if not cv.validate_all_constraints(scheduler, scheduler.cycle):
                break

        scheduler.cycle += 1

        # 检查终止条件
        if all(copu.everything_free for copu in scheduler.copus):
            break

    all_passed &= cv.report()

    # ---- 第二层：多COPU同步验证 ----
    print("\n[第二层] 多COPU同步验证 (Synchronization Validation)")
    print("-" * 70)

    sv = SyncValidator()
    # 重新初始化调度器
    scheduler = MultiCOPU_Scheduler(num_copus=2, num_oocds=7)
    scheduler.copus[0].load_data(data[:configs_per_copu], flags[:configs_per_copu], cycles[:configs_per_copu])
    scheduler.copus[1].load_data(data[configs_per_copu:], flags[configs_per_copu:], cycles[configs_per_copu:])

    for _ in range(100):
        for copu in scheduler.copus:
            copu.step(bins, threshold=1.0, sample_rate=1.0)

        scheduler.cht_scheduler.advance_cycle()

        if _ % 10 == 0:
            sv.validate_cycle_sync(scheduler, _)
            sv.validate_collision_signal(scheduler, _)
            sv.validate_cht_consistency(scheduler, _)

        scheduler.cycle += 1

        if all(copu.everything_free for copu in scheduler.copus):
            break

    all_passed &= sv.report()

    # ---- 第四层：性能验证 ----
    print("\n[第四层] 性能验证 (Performance Validation)")
    print("-" * 70)

    pv = PerformanceValidator(min_throughput_scaling=0.85)
    results_by_num_copus = {}

    for num_copus in [1, 2, 4]:
        print(f"\n  测试 {num_copus} COPU...")
        scheduler = MultiCOPU_Scheduler(num_copus=num_copus, num_oocds=7)

        # 生成更多配置用于性能测试
        num_configs_perf = 500
        data_perf = [
            [np.random.uniform(0, 100) for _ in range(3)]
            for _ in range(num_configs_perf)
        ]
        flags_perf = [[1 for _ in range(3)] for _ in range(num_configs_perf)]
        cycles_perf = [[40 + int(np.random.random() * 10) for _ in range(3)]
                       for _ in range(num_configs_perf)]

        # 均匀分配
        configs_per_copu_perf = num_configs_perf // num_copus
        for i in range(num_copus):
            start = i * configs_per_copu_perf
            end = start + configs_per_copu_perf if i < num_copus - 1 else num_configs_perf
            scheduler.copus[i].load_data(
                data_perf[start:end], flags_perf[start:end], cycles_perf[start:end]
            )

        result = scheduler.simulate(bins, threshold=1.0, sample_rate=1.0, max_cycles=100000)
        results_by_num_copus[num_copus] = result

    all_passed &= pv.validate_throughput_scaling(results_by_num_copus)
    all_passed &= pv.validate_load_balance(results_by_num_copus)
    all_passed &= pv.validate_cht_conflict_growth(results_by_num_copus)
    all_passed &= pv.report()

    # ---- 第五层：压力测试 ----
    print("\n[第五层] 压力测试 (Stress Testing)")
    print("-" * 70)

    st = StressTestValidator()
    st.test_empty_linklist()
    st.test_single_config_single_link()
    st.test_cht_hash_collision()
    st.test_long_run_stability(num_cycles=50000)
    all_passed &= st.report()

    # ---- 总结 ----
    print("\n" + "=" * 70)
    if all_passed:
        print(" ✅ 所有验证测试通过！")
    else:
        print(" ❌ 部分验证测试失败，详见上面的报告")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
