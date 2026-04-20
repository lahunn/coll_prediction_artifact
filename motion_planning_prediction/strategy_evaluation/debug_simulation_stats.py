
import sys
import os
import argparse
import pickle
from tqdm import tqdm
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import simulation_utils as su
from simulation_core.simulators import (
    simulate_parallel_collision_detection_sphere,
    simulate_parallel_collision_detection_link,
)

def main():
    basename = "iiwa_7"
    benchid = 1
    data_folder = "../../trace_files/scene_benchmarks/bit_collision_data/G1"
    num_oocds = 7
    check_cost = 60
    threshold = 1.0
    sample_rate = 0.125
    qnoncoll_multiplier = 8
    
    # 获取机器人参数
    from trace_generation.config.ana_parameters import get_robot_params
    from trace_generation.core.robot.environment import RobotEnv
    from trace_generation.core.collision.sphere_detector import SphereEnvGeometric
    
    robot_name = "iiwa"
    robot_params = get_robot_params(robot_name)
    
    # 获取 sphere-link 映射
    temp_env = RobotEnv(robot_name, OBB_GUI=False, enable_self_collision=False)
    temp_sphere_env = SphereEnvGeometric(robot_env=temp_env, robot_name=robot_name)
    temp_sphere_env._initialize_sphere_metadata()
    sphere_link_ids = temp_sphere_env.sphere_link_ids
    link_to_spheres = {}
    sphere_to_link = []
    for idx, link_id in enumerate(sphere_link_ids):
        lid = int(link_id)
        link_to_spheres.setdefault(lid, []).append(idx)
        sphere_to_link.append(lid)
    num_spheres_per_pose = len(sphere_link_ids)
    temp_sphere_env.close()
    temp_env.close()

    bins = su.calculate_bins_from_workspace(robot_name, 4)
    qnoncoll_len = qnoncoll_multiplier * num_oocds

    # 加载数据
    edge_link_data, edge_link_coll_data, edge_link_coords_data = su.load_data_with_link_coords(
        basename, benchid, data_folder
    )

    output_csv = "debug_edge_stats_g1_b1.csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Edge_Idx", "Strategy", "Queries", "Cycles", "Theoretical_Min_Cycles", "OOCD_Util"])

        strategies = ["sphere_coord", "link_coord"]
        
        for strategy in strategies:
            colldict = {}
            if strategy == "link_coord":
                iterator = zip(edge_link_coords_data, edge_link_coll_data)
            else:
                iterator = zip(edge_link_data, edge_link_coll_data)
            
            for i, (edge_coords, edge_coll) in enumerate(iterator):
                if not edge_coll: continue
                
                linklist, linklist_coll = su.csp_rearrange(edge_coords, edge_coll, groupsize=4)
                
                if strategy == "link_coord":
                    q, cd, cf, cy, util, dt = simulate_parallel_collision_detection_link(
                        linklist, linklist_coll, colldict, threshold, sample_rate, bins,
                        link_to_spheres, sphere_to_link, num_spheres_per_pose,
                        qnoncoll_len=qnoncoll_len, cycle_check=check_cost, num_oocds=num_oocds,
                        collect_deadtime=True
                    )
                else:
                    q, cd, cf, cy, util, dt = simulate_parallel_collision_detection_sphere(
                        linklist, linklist_coll, colldict, threshold, sample_rate, bins,
                        link_to_spheres, sphere_to_link, num_spheres_per_pose,
                        qnoncoll_len=qnoncoll_len * 4, cycle_check=check_cost, num_oocds=num_oocds,
                        collect_deadtime=True
                    )
                
                theo_min = (q * check_cost) / num_oocds
                writer.writerow([i, strategy, f"{q:.2f}", cy, f"{theo_min:.2f}", f"{util:.4f}"])

    print(f"Debug report saved to {output_csv}")

if __name__ == "__main__":
    main()
