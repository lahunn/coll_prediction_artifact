"""
common_simulation_utils.py

本模块包含多个仿真脚本中重复或相似的功能函数，便于统一调用和维护。
"""
import os
import numpy as np
import argparse
from tqdm import tqdm

# =====================
# 常量定义
# =====================
DEFAULT_QUANT_BITS = 4

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
# 统计初始化工具
# =====================
def initialize_statistics(extra_keys=None):
    """
    初始化标准仿真统计指标字典
    """
    stats = {
        "fall_prediction": 0.0,
        "fall_oracle": 0,
        "total_checks": 0,
        "fall_cycle": 0,
        "theoretical_min_cycles": 0,
        "total_pred_coll_cycles": 0,
        "total_pred_noncoll_cycles": 0,
        "total_oracle_coll_cycles": 0,
        "total_oracle_noncoll_cycles": 0,
    }
    if extra_keys:
        for key in extra_keys:
            stats[key] = 0.0
    return stats

# =====================
# 参数解析工具
# =====================
def create_common_parser(description):
    """
    创建包含标准位置参数的ArgumentParser
    标准参数顺序: threshold, sample_rate, qnoncoll_multiplier, data_folder, basename, benchmarks, robot_name, collision_model_type, num_oocds
    所有参数均有默认值，且作为可选位置参数。
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("threshold", type=float, nargs="?", default=0.5, help="Prediction threshold")
    parser.add_argument("sample_rate", type=float, nargs="?", default=0.1, help="Sampling rate")
    parser.add_argument("qnoncoll_multiplier", type=int, nargs="?", default=8, help="Queue length multiplier")
    parser.add_argument("data_folder", type=str, nargs="?", default="../../trace_files/scene_benchmarks/bit_collision_data", help="Data folder path")
    parser.add_argument("basename", type=str, nargs="?", default="iiwa_7", help="Basename (e.g. iiwa_7)")
    parser.add_argument("benchmarks", type=str, nargs="?", default="1-10", help="Benchmarks (e.g. '1-10', '5', '1,3')")
    parser.add_argument("robot_name", type=str, nargs="?", default="iiwa", help="Robot name (e.g. iiwa)")
    parser.add_argument("collision_model_type", type=str, nargs="?", default="link", choices=["link", "sphere"], help="Collision model type")
    parser.add_argument("num_oocds", type=int, nargs="?", default=8, help="Number of parallel OOCDs")
    return parser

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
# 仿真配置工具
# =====================
def setup_simulation(robot_name, quant_bits, collision_model_type, qnoncoll_multiplier):
    """
    统一设置仿真参数，包括bins计算、机器人参数获取等
    
    Returns:
        bins, num_elements, check_cost, qnoncoll_len, print_title
    """
    # 延迟导入以避免潜在的循环依赖
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    import simulation_utils as su
    from trace_generation.config.ana_parameters import get_robot_params

    # 获取机器人参数
    robot_params = get_robot_params(robot_name)
    
    # 计算bins
    bins = su.calculate_bins_from_workspace(robot_name, quant_bits)
    
    if collision_model_type == "sphere":
        num_elements = robot_params["sphere_num"]
        check_cost = robot_params["sphere_cost"]
        title_prefix = "Sphere"
    else:
        num_elements = robot_params["obb_num"]
        check_cost = robot_params["obb_cost"]
        title_prefix = "OBB"
        
    qnoncoll_len = num_elements * qnoncoll_multiplier
    print_title = f"=== {title_prefix} Collision Detection Prediction Simulation ==="
    
    return bins, num_elements, check_cost, qnoncoll_len, print_title

# =====================
# Oracle计算工具
# =====================
def calculate_oracle_metrics(edge_coll, num_elements, num_oocds, check_cost):
    """
    计算单条Edge的Oracle指标（理论最优值）
    
    Returns:
        actual_checks: 顺序检查所需的总次数
        oracle_queries: Oracle预测的查询数
        oracle_cycles: Oracle预测的周期数
        coll_found_oracle: 是否发现碰撞
    """
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    import simulation_utils as su
    
    actual_checks = 0
    # 累计实际查询总数（理想的顺序检查）
    for pose_coll in edge_coll:
        try:
            first_collision_index = pose_coll.index(0)
            actual_checks += first_collision_index + 1
        except ValueError:
            actual_checks += len(pose_coll)

    # Oracle 计算
    coll_found_oracle = any(
        link_coll == 0 for pose_coll in edge_coll for link_coll in pose_coll
    )
    
    if coll_found_oracle:
        oracle_queries = 1
    else:
        oracle_queries = num_elements * len(edge_coll)
        
    oracle_cycles = su.calculate_oracle_cycles(edge_coll, num_oocds, check_cost)

    return actual_checks, oracle_queries, oracle_cycles, coll_found_oracle

def aggregate_oracle_stats(edge_link_coll_data, num_elements, num_oocds, check_cost):
    """
    对一组 edges 累计理论 Oracle 指标
    """
    stats = {
        "total_checks": 0,
        "total_oracle_queries": 0,
        "total_oracle_cycles": 0,
        "total_oracle_coll_cycles": 0,
        "total_oracle_noncoll_cycles": 0,
        "total_coll_edges": 0,
        "total_noncoll_edges": 0
    }
    
    for edge_coll in edge_link_coll_data:
        if not edge_coll:
            continue
            
        actual_checks, oracle_queries, oracle_cycles, coll_found_oracle = calculate_oracle_metrics(
            edge_coll, num_elements, num_oocds, check_cost
        )
        
        stats["total_checks"] += actual_checks
        stats["total_oracle_queries"] += oracle_queries
        stats["total_oracle_cycles"] += oracle_cycles
        
        if coll_found_oracle:
            stats["total_oracle_coll_cycles"] += oracle_cycles
            stats["total_coll_edges"] += 1
        else:
            stats["total_oracle_noncoll_cycles"] += oracle_cycles
            stats["total_noncoll_edges"] += 1
            
    return stats

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
    if total_checks is not None:
        print(f"  Total Actual Checks: {total_checks}")
    print(f"  Total Prediction Queries: {fall_prediction:.2f}")
    print(f"  Total Oracle Queries: {fall_oracle}")
    if total_checks is not None and total_checks > 0:
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
