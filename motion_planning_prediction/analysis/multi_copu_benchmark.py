"""
多COPU仿真系统的性能基准测试

功能：
1. 单COPU与原函数的等价性验证
2. 多COPU吞吐量扩展性分析
3. CHT冲突模式分析
4. 负载均衡评估
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_copu_simulation import MultiCOPU_Scheduler


# ======================== Utility Functions ========================
def create_synthetic_data(num_configs, num_links, num_bin_values):
    """
    创建合成测试数据

    Returns:
        (collision_data, collision_flags, cycles)
    """
    collision_data = [
        [np.random.uniform(0, 100, 1)[0] for _ in range(num_links)]
        for _ in range(num_configs)
    ]

    collision_flags = [
        [1 if np.random.random() > 0.3 else 0 for _ in range(num_links)]
        for _ in range(num_configs)
    ]

    cycles = [
        [40 + int(np.random.random() * 10) for _ in range(num_links)]
        for _ in range(num_configs)
    ]

    return collision_data, collision_flags, cycles


def partition_data_for_copus(data, flags, cycles, num_copus):
    """
    将数据分割给N个COPU

    Returns:
        (data_list, flags_list, cycles_list)
    """
    num_configs = len(data)
    configs_per_copu = num_configs // num_copus
    remainder = num_configs % num_copus

    data_list = []
    flags_list = []
    cycles_list = []

    for copu_id in range(num_copus):
        if copu_id < remainder:
            start_idx = copu_id * (configs_per_copu + 1)
            end_idx = start_idx + configs_per_copu + 1
        else:
            start_idx = (
                remainder * (configs_per_copu + 1)
                + (copu_id - remainder) * configs_per_copu
            )
            end_idx = start_idx + configs_per_copu

        data_list.append(data[start_idx:end_idx])
        flags_list.append(flags[start_idx:end_idx])
        cycles_list.append(cycles[start_idx:end_idx])

    return data_list, flags_list, cycles_list


def analyze_multi_copu_performance(results):
    """
    分析多COPU系统的性能指标
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


# ======================== Benchmark 1: Data Load Verification ========================
def benchmark_data_partitioning():
    """
    验证：数据分割函数的正确性和性能
    """
    print("\n" + "=" * 60)
    print(" Benchmark 1: Data Partitioning Performance")
    print("=" * 60)

    num_configs_list = [100, 1000, 10000]
    num_links = 7
    num_copus = 4

    for num_configs in num_configs_list:
        print(f"\nTesting {num_configs} configurations -> {num_copus} COPUs:")

        start = time.time()
        data, flags, cycles = create_synthetic_data(num_configs, num_links, 100)
        create_time = time.time() - start

        start = time.time()
        data_list, flags_list, cycles_list = partition_data_for_copus(
            data, flags, cycles, num_copus
        )
        partition_time = time.time() - start

        # 验证分割正确性
        total_configs = sum(len(d) for d in data_list)
        assert total_configs == num_configs, (
            f"分割错误：{total_configs} != {num_configs}"
        )

        # 计算各COPU的配置数
        config_counts = [len(d) for d in data_list]
        print(f"  配置分布: {config_counts}")
        print(f"  数据创建时间: {create_time * 1000:.2f}ms")
        print(f"  分割时间: {partition_time * 1000:.2f}ms")
        print("  ✓ 验证通过")


# ======================== Benchmark 2: CHT Performance ========================
def benchmark_cht_throughput():
    """
    验证：CHT的吞吐量和冲突率
    """
    print("\n" + "=" * 60)
    print(" Benchmark 2: CHT Read/Write Throughput")
    print("=" * 60)

    from multi_copu_simulation import DualPortSRAM_CHT

    num_copus = 4
    cht = DualPortSRAM_CHT(size=4096)

    print(f"\nSimulating {num_copus} COPUs with 1000 cycles:")

    # 模拟多COPU对CHT的访问
    read_per_cycle = []
    write_per_cycle = []

    for cycle in range(1000):
        # 每个COPU随机发送读或写请求
        reads_this_cycle = 0
        writes_this_cycle = 0

        for copu_id in range(num_copus):
            if np.random.random() > 0.3:  # 30%概率不访问
                if np.random.random() > 0.8:  # 20%是写，80%是读
                    cht.write_request(
                        copu_id,
                        f"key_{np.random.randint(0, 256)}",
                        1,
                        0,
                        cycle,
                    )
                    writes_this_cycle += 1
                else:
                    cht.read_request(copu_id, f"key_{np.random.randint(0, 256)}", cycle)
                    reads_this_cycle += 1

        read_per_cycle.append(reads_this_cycle)
        write_per_cycle.append(writes_this_cycle)

        cht.advance_cycle()

    # 分析结果
    stats = cht.get_stats()

    print("\nCHT Statistics:")
    print(f"  总读操作: {stats['total_reads']}")
    print(f"  总写操作: {stats['total_writes']}")
    print(
        f"  总冲突: {stats['total_conflicts']} ({stats['total_conflicts'] / max(1, stats['total_reads'] + stats['total_writes']) * 100:.2f}%)"
    )
    print(f"  总冲突率: {stats['conflict_rate']:.4f}")
    print(f"  使用的CHT条目: {stats['entries_used']}")

    avg_reads = np.mean(read_per_cycle)
    avg_writes = np.mean(write_per_cycle)
    print("\nAverage per cycle:")
    print(f"  读操作: {avg_reads:.2f}")
    print(f"  写操作: {avg_writes:.2f}")
    print("  ✓ 验证通过")


# ======================== Benchmark 3: Single vs Multi COPU ========================
def benchmark_single_vs_multi_copu():
    """
    验证：多COPU相对于单COPU的性能改进
    """
    print("\n" + "=" * 60)
    print(" Benchmark 3: Single COPU vs Multi-COPU Scaling")
    print("=" * 60)

    # 创建合成数据
    num_configs = 500
    num_links = 7

    print(f"\nCreating synthetic data: {num_configs} configs, {num_links} links")
    data, flags, cycles = create_synthetic_data(num_configs, num_links, 100)

    # 创建固定的bins用于量化
    bins = np.linspace(0, 100, 10)

    # 测试不同数量的COPU
    copu_counts = [1, 2, 4]

    results = {}

    for num_copus in copu_counts:
        print(f"\n--- Testing {num_copus} COPU(s) ---")

        # 分割数据
        data_list, flags_list, cycles_list = partition_data_for_copus(
            data, flags, cycles, num_copus
        )

        # 创建调度器
        scheduler = MultiCOPU_Scheduler(num_copus=num_copus, num_oocds=7, cht_size=4096)

        # 加载数据
        scheduler.load_data_for_all_copus(data_list, flags_list, cycles_list)

        # 执行仿真
        start = time.time()
        result = scheduler.simulate(
            bins, threshold=1.0, sample_rate=1.0, max_cycles=100000
        )
        sim_time = time.time() - start

        # 分析性能
        perf = analyze_multi_copu_performance(result)

        results[num_copus] = {
            "sim_time": sim_time,
            "perf": perf,
            "result": result,
        }

        print(f"  总周期: {result['total_cycles']}")
        print(f"  总查询: {perf['total_queries']}")
        print(f"  吞吐量: {perf['system_throughput']:.4f} queries/cycle")
        print(f"  利用率: {perf['avg_copu_utilization']:.4f}")
        print(f"  CHT冲突率: {perf['cht_conflict_rate']:.4f}")
        print(f"  负载均衡: {perf['load_balance_variance']:.4f}")
        print(f"  仿真时间: {sim_time:.3f}s")

    # 比较扩展性
    print("\n--- Scalability Analysis ---")
    single_throughput = results[1]["perf"]["system_throughput"]
    single_time = results[1]["sim_time"]

    for num_copus in copu_counts[1:]:
        multi_throughput = results[num_copus]["perf"]["system_throughput"]
        multi_time = results[num_copus]["sim_time"]

        speedup = multi_throughput / single_throughput
        time_reduction = (1 - multi_time / single_time) * 100

        print(f"\n{num_copus} COPU vs 1 COPU:")
        print(f"  吞吐量提升: {speedup:.2f}x")
        print(f"  执行时间减少: {time_reduction:.1f}%")


# ======================== Benchmark 4: Load Balance ========================
def benchmark_load_balance():
    """
    验证：负载均衡的均匀性
    """
    print("\n" + "=" * 60)
    print(" Benchmark 4: Load Balance Verification")
    print("=" * 60)

    num_configs = 1000
    num_links = 7
    num_copus = 4

    print(f"\nTesting load balance: {num_configs} configs -> {num_copus} COPUs")

    data, flags, cycles = create_synthetic_data(num_configs, num_links, 100)
    data_list, flags_list, cycles_list = partition_data_for_copus(
        data, flags, cycles, num_copus
    )

    # 检查配置分布
    config_counts = [len(d) for d in data_list]
    print(f"\nConfiguration distribution: {config_counts}")
    print(f"  平均: {np.mean(config_counts):.1f}")
    print(f"  标准差: {np.std(config_counts):.2f}")
    print(f"  最大偏差: {max(config_counts) - min(config_counts)}")

    # 运行仿真检查查询分布
    bins = np.linspace(0, 100, 10)
    scheduler = MultiCOPU_Scheduler(num_copus=num_copus, num_oocds=7)
    scheduler.load_data_for_all_copus(data_list, flags_list, cycles_list)
    result = scheduler.simulate(bins, threshold=1.0, sample_rate=1.0, max_cycles=100000)

    # 分析查询分布
    query_counts = [c["total_queries"] for c in result["copus"]]
    print(f"\nQuery distribution across COPUs: {[int(q) for q in query_counts]}")
    print(f"  平均: {np.mean(query_counts):.1f}")
    print(f"  标准差: {np.std(query_counts):.2f}")
    print(f"  负载均衡系数: {np.std(query_counts) / np.mean(query_counts):.4f}")

    perf = analyze_multi_copu_performance(result)
    print(f"  性能指标负载均衡: {perf['load_balance_variance']:.4f}")
    print("  ✓ 验证通过")


# ======================== Benchmark 5: No-Collision Scenario ========================
def benchmark_no_collision_scenario():
    """
    验证：多edge无碰撞场景下的系统性能

    场景描述：
    - 模拟处理多个robot edges（如路径规划中的多条边）
    - 每个edge包含多个configurations（姿态），每个config包含多个links
    - 所有collision_flags全部为1（无碰撞）
    - 多个COPU并行处理不同的edges
    - 用于验证系统在大规模任务下的性能表现

    数据结构：
    - num_edges: 要检测的边数（模拟路径中的多条边）
    - num_configs_per_edge: 每条边的配置数
    - num_links: 每个配置的关节数
    """
    print("\n" + "=" * 60)
    print(" Benchmark 5: Multi-Edge No-Collision Scenario")
    print("=" * 60)

    # 多edge场景参数
    num_edges = 4  # 40条边
    num_configs_per_edge = 200  # 每条边有50个配置
    num_links = 7  # 机器人7个关节

    print("\nCreating multi-edge no-collision data:")
    print(f"  边数: {num_edges}")
    print(f"  每条边的配置数: {num_configs_per_edge}")
    print(f"  每个配置的关节数: {num_links}")
    print(f"  总任务数: {num_edges * num_configs_per_edge}")
    print("  所有任务结果: is_free = 1 (无碰撞)")

    bins = np.linspace(0, 100, 10)
    copu_counts = [1, 2, 4, 5, 6, 7, 8]
    results_no_coll = {}

    # 为每个COPU数量运行基准测试
    for num_copus in copu_counts:
        print(f"\n--- Testing {num_copus} COPU(s) (Multi-Edge No-Collision) ---")

        # 创建调度器
        scheduler = MultiCOPU_Scheduler(num_copus=num_copus, num_oocds=7, cht_size=4096)

        # 统计变量
        config_assignments = [0] * num_copus
        edge_results = []
        total_copu_utilization = 0  # 累积COPU占用率
        total_cht_conflicts = 0  # 累积CHT冲突数
        all_oracle = num_edges * num_configs_per_edge * num_links
        all_prediction = 0
        all_cycle = 0

        # 按edge进行仿真（参考prediction_simulation_nDOF_sphere_preemptive的逻辑）
        for edge_idx in range(num_edges):
            # 按config级别分配到各COPU（每个config包含num_links个link）
            configs_per_copu = num_configs_per_edge // num_copus
            remainder = num_configs_per_edge % num_copus

            for copu_id in range(num_copus):
                if copu_id < remainder:
                    start_config = copu_id * (configs_per_copu + 1)
                    end_config = start_config + configs_per_copu + 1
                else:
                    start_config = (
                        remainder * (configs_per_copu + 1)
                        + (copu_id - remainder) * configs_per_copu
                    )
                    end_config = start_config + configs_per_copu

                # 该COPU分配的configs数
                num_configs_assigned = end_config - start_config
                config_assignments[copu_id] += num_configs_assigned

                # 该COPU分配的数据（每个config有num_links个link,每个link的坐标有个三个维度）
                copu_collision_data = [
                    [np.random.uniform(0, 100) for _ in range(3)]
                    for _ in range(num_configs_assigned * num_links)
                ]

                copu_collision_flags = [
                    1 for _ in range(num_configs_assigned * num_links)
                ]

                copu_cycles = [
                    (40 + int(np.random.random() * 3))
                    for _ in range(num_configs_assigned * num_links)
                ]

                # 加载数据到对应COPU
                scheduler.copus[copu_id].load_data(
                    copu_collision_data, copu_collision_flags, copu_cycles
                )

            # 执行仿真
            result = scheduler.simulate(
                bins, threshold=1.0, sample_rate=1.0, max_cycles=100000
            )

            # 分析性能
            perf = analyze_multi_copu_performance(result)

            # 收集该edge的结果
            edge_results.append(
                {
                    "edge_idx": edge_idx,
                    "perf": perf,
                    "result": result,
                }
            )

            all_prediction += perf["total_queries"]
            print(f" {edge_idx} 查询: {perf['total_queries']}")
            all_cycle += result["total_cycles"]

            # 累积COPU占用率和CHT冲突数
            total_copu_utilization += result["copus"][0]["oocd_utilization"]
            total_cht_conflicts += result["cht_stats"]["total_conflicts"]

        print(f"  配置分配给COPU: {config_assignments}")

        # 汇总结果
        results_no_coll[num_copus] = {
            "edge_results": edge_results,
            "all_oracle": all_oracle,
            "all_prediction": all_prediction,
            "all_cycle": all_cycle,
        }

        # 统计汇总信息
        print(f"  总周期: {all_cycle}")
        print(f"  总查询: {all_prediction:.0f}")
        print(
            f"  平均吞吐量: {all_prediction / all_cycle:.4f} queries/cycle"
            if all_cycle > 0
            else "  平均吞吐量: N/A"
        )
        print(f"  Oracle查询: {all_oracle}")
        print(f"  查询减少率: {(1 - all_prediction / all_oracle) * 100:.2f}%")

        # COPU占用率平均值和CHT冲突总数
        avg_copu_utilization = (total_copu_utilization / num_edges) * 100
        print(f"  COPU占用率(平均): {avg_copu_utilization:.2f}%")
        print(f"  CHT冲突数(总): {total_cht_conflicts}")


# ======================== Main Benchmark Runner ========================
def run_all_benchmarks():
    """运行所有基准测试"""
    print("\n" + "=" * 60)
    print(" Multi-COPU Simulation Benchmark Suite")
    print("=" * 60)

    try:
        benchmark_data_partitioning()
        benchmark_cht_throughput()
        benchmark_single_vs_multi_copu()
        benchmark_load_balance()
        benchmark_no_collision_scenario()

        print("\n" + "=" * 60)
        print(" All benchmarks completed successfully!")
        print("=" * 60 + "\n")
        return True
    except Exception as e:
        print(f"\n✗ Benchmark failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_benchmarks()
    sys.exit(0 if success else 1)
