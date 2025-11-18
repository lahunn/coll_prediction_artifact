"""
再现基准测试中的具体场景：4个edges，每个edge 200个configs，8个COPUs
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_copu_simulation import MultiCOPU_Scheduler, analyze_multi_copu_performance

# 参数（来自benchmark_no_collision_scenario）
num_edges = 2  # 简化为2个edges
num_configs_per_edge = 200
num_links = 7
num_copus = 8

bins = np.linspace(0, 100, 10)
all_oracle = num_edges * num_configs_per_edge * num_links
all_prediction = 0

print("=" * 80)
print(f"Benchmark Scenario: {num_edges} edges, {num_configs_per_edge} configs/edge, {num_copus} COPUs")
print("=" * 80)

for edge_idx in range(num_edges):
    print(f"\nEdge {edge_idx}:")
    
    # 为每个edge创建新的scheduler
    scheduler = MultiCOPU_Scheduler(num_copus=num_copus, num_oocds=7, cht_size=4096)
    
    configs_per_copu = num_configs_per_edge // num_copus
    remainder = num_configs_per_edge % num_copus
    
    total_linklist_items = 0
    
    for copu_id in range(num_copus):
        if copu_id < remainder:
            start_config = copu_id * (configs_per_copu + 1)
            end_config = start_config + configs_per_copu + 1
        else:
            start_config = remainder * (configs_per_copu + 1) + (copu_id - remainder) * configs_per_copu
            end_config = start_config + configs_per_copu
        
        num_configs_assigned = end_config - start_config
        
        # 按照基准测试中的方式生成数据
        copu_collision_data = [
            [np.random.uniform(0, 100) for _ in range(3)]
            for _ in range(num_configs_assigned * num_links)  # 注意：这里乘以了num_links
        ]
        
        copu_collision_flags = [
            1 for _ in range(num_configs_assigned * num_links)
        ]
        
        copu_cycles = [
            (40 + int(np.random.random() * 3))
            for _ in range(num_configs_assigned * num_links)
        ]
        
        # 加载数据
        scheduler.copus[copu_id].load_data(copu_collision_data, copu_collision_flags, copu_cycles)
        total_linklist_items += len(scheduler.copus[copu_id].linklist)
        
        print(f"  COPU[{copu_id}]: {len(scheduler.copus[copu_id].linklist)} linklist items")
    
    print(f"  Total linklist items for this edge: {total_linklist_items}")
    
    # 执行仿真
    result = scheduler.simulate(bins, threshold=1.0, sample_rate=1.0, max_cycles=100000)
    
    # 分析性能
    perf = analyze_multi_copu_performance(result)
    
    # 收集结果
    all_prediction += perf["total_queries"]
    
    print(f"  Total query_count for this edge: {perf['total_queries']}")
    print(f"  Expected (oracle): {num_configs_per_edge * num_links}")
    
    # 检查是否超过
    if perf["total_queries"] > total_linklist_items:
        print(f"  WARNING: query_count ({perf['total_queries']}) > linklist ({total_linklist_items})")
    else:
        print(f"  OK: query_count ({perf['total_queries']}) <= linklist ({total_linklist_items})")

print("\n" + "=" * 80)
print(f"Final Results:")
print(f"  all_oracle: {all_oracle}")
print(f"  all_prediction: {all_prediction}")
print(f"  Query reduction: {(1 - all_prediction / all_oracle) * 100:.2f}%")
