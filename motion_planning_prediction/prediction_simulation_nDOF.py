import sys
import numpy as np
from tqdm import tqdm
import simulation_utils as su

# --- Simulation Settings ---
binnumber = 16
intervalsize = 2 / binnumber
bins = np.zeros(binnumber)
start = -1
for i in range(binnumber):
    bins[i] = start
    start += intervalsize

# --- Global Statistics ---
fall_prediction = 0
fall_oracle = 0

# --- Simulation Parameters from Command Line ---
threshold = float(sys.argv[1])
sample_rate = float(sys.argv[2])
qnoncoll_len = int(7 * int(sys.argv[3]))
planner_type = sys.argv[4]

# --- Benchmark Range ---
if planner_type == "MPNET":
    benchrange = [0, 13, 14, 15, 16, 17, 19, 1, 20, 21, 23, 24, 25, 27, 28, 29, 2, 30, 32, 34, 35, 36, 37, 39, 3, 44, 45, 46, 49, 4, 53, 55, 56, 57, 58, 59, 5, 63, 64, 65, 70, 71, 75, 7, 82, 83, 85, 87, 88, 8, 90, 91, 92, 93, 95, 96, 97, 98]
else:
    benchrange = range(2000, 2200)

# --- Main Simulation Loop ---
for benchid in tqdm(benchrange):
    all_prediction = 0
    all_oracle = 0
    colldict = {}

    edge_link_data, edge_link_coll_data = su.load_data(planner_type, benchid, "nDOF")

    if edge_link_data is None:
        continue

    for edge, edge_coll in zip(edge_link_data, edge_link_coll_data):
        if not edge_coll:
            continue

        # --- Oracle Calculation ---
        coll_found_oracle = any(link_coll == 0 for pose_coll in edge_coll for link_coll in pose_coll)
        if coll_found_oracle:
            all_oracle += 1
        else:
            all_oracle += (7 * len(edge_coll))

        # --- CSP Rearrangement ---
        linklist, linklist_coll = su.csp_rearrange(edge, edge_coll, groupsize=4)

        # --- Run Centralized Simulation ---
        edge_query_count, colldict, _ = su.simulate_parallel_collision_detection(
            linklist,
            linklist_coll,
            colldict,
            threshold,
            sample_rate,
            bins,
            qnoncoll_len=qnoncoll_len
        )

        all_prediction += edge_query_count

    fall_oracle += all_oracle
    fall_prediction += all_prediction
    print(all_prediction, all_oracle)

print(f"Final Prediction Queries: {fall_prediction}, Final Oracle Queries: {fall_oracle}")
