"""
common_simulation_utils.py

本模块包含多个仿真脚本中重复或相似的功能函数，便于统一调用和维护。
"""
import os
import numpy as np
from tqdm import tqdm

# =====================
# 参数与bins相关工具
# =====================
def get_bins(binnumber=16, minval=-1, maxval=1):
    """
    生成等间隔bins数组
    """
    intervalsize = (maxval - minval) / binnumber
    bins = np.zeros(binnumber)
    start = minval
    for i in range(binnumber):
        bins[i] = start
        start += intervalsize
    return bins

# =====================
# 打印统计信息
# =====================
def print_final_statistics(
    total_checks,
    fall_prediction,
    fall_oracle,
    fall_cycle=None,
    theoretical_min_cycles=None,
    total_pred_coll_cycles=None,
    total_pred_noncoll_cycles=None,
    total_oracle_coll_cycles=None,
    total_oracle_noncoll_cycles=None,
    extra_stats=None,
    title=None
):
    """
    统一打印仿真统计信息
    """
    if title:
        print("\n" + title)
    else:
        print("\n" + "=" * 50)
    print("Final Statistics:")
    print(f"  Total Actual Checks: {total_checks}")
    print(f"  Total Prediction Queries: {fall_prediction:.2f}")
    print(f"  Total Oracle Queries: {fall_oracle}")
    if total_checks > 0:
        print(f"  Query Reduction Rate: {(1 - fall_prediction / total_checks) * 100:.2f}%")
    if fall_oracle > 0:
        print(f"  Query Difference (Prediction - Oracle): {(fall_prediction - fall_oracle) / fall_oracle * 100:.2f}%")
    if fall_cycle is not None:
        print(f"\n  Total Cycles (Prediction): {fall_cycle}")
    if theoretical_min_cycles is not None:
        print(f"  Total Cycles (Oracle): {theoretical_min_cycles}")
        if fall_cycle:
            print(f"  Cycle Efficiency: {(theoretical_min_cycles / fall_cycle) * 100:.2f}%")
    if total_pred_coll_cycles is not None:
        print(f"\n  Prediction Coll Edge Cycles: {total_pred_coll_cycles}")
    if total_pred_noncoll_cycles is not None:
        print(f"  Prediction Non-Coll Edge Cycles: {total_pred_noncoll_cycles}")
    if total_oracle_coll_cycles is not None:
        print(f"  Oracle Coll Edge Cycles: {total_oracle_coll_cycles}")
    if total_oracle_noncoll_cycles is not None:
        print(f"  Oracle Non-Coll Edge Cycles: {total_oracle_noncoll_cycles}")
    if extra_stats:
        for k, v in extra_stats.items():
            print(f"  {k}: {v}")
    print("=" * 50)

# =====================
# Benchmark范围解析
# =====================
def parse_benchrange(benchmarks_arg):
    """
    解析基准测试范围参数，支持单个数字、范围字符串如"2-10"，或逗号分隔的列表
    """
    if isinstance(benchmarks_arg, int):
        return [benchmarks_arg]
    if isinstance(benchmarks_arg, str):
        if '-' in benchmarks_arg:
            start, end = map(int, benchmarks_arg.split('-'))
            return list(range(start, end + 1))
        elif ',' in benchmarks_arg:
            return [int(x) for x in benchmarks_arg.split(',')]
        else:
            return [int(benchmarks_arg)]
    raise ValueError(f"Unrecognized benchmarks_arg: {benchmarks_arg}")

# =====================
# 数据加载通用接口
# =====================
def load_pickle_data(filepath):
    """
    加载pkl文件，返回内容
    """
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# =====================
# 其他可复用工具函数可在此添加
# =====================
