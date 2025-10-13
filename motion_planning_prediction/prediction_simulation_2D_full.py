import sys
import numpy as np
from tqdm import tqdm
import simulation_utils as su

# --- Simulation Settings ---
binnumber = 32
start_val = -1
if sys.argv[4] == "MPNET":
    intervalsize = 40 / binnumber
    start_val = -20
else:
    intervalsize = 2 / binnumber

bins = np.zeros(binnumber)
for i in range(binnumber):
    bins[i] = start_val
    start_val += intervalsize

# --- Global Statistics ---
fall_prediction = 0
fall_oracle = 0

# --- Simulation Parameters from Command Line ---
threshold = float(sys.argv[1])
sample_rate = float(sys.argv[2])
qnoncoll_len = int(1 * int(sys.argv[3]))
planner_type = sys.argv[4]

# --- Benchmark Range ---
benchrange = range(0, 201)
if planner_type == "GNN":
    benchrange = range(1, 201)

# --- Main Simulation Loop ---
for benchid in tqdm(benchrange):
    all_prediction = 0
    all_oracle = 0
    colldict = {}

    edge_link_data, edge_link_coll_data = su.load_data(planner_type, benchid, "2D")

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
            all_oracle += len(edge_coll)

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
