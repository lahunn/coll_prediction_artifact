"""
Performance analysis functions for multi-COPU simulations.
"""

import numpy as np


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