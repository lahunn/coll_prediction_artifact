import numpy as np
import random
import pickle
import sys


def reutrn_keyy(code):
    """Creates a hash key from a quantized code."""
    bitsize = len(code)
    keyy = ""
    for j in range(0, bitsize):
        if code[j] < 10:
            keyy = keyy + "0"
        keyy = keyy + str(code[j])
    return keyy


def csp_rearrange(edge, edgeyarr, groupsize=8):
    """
    Rearranges the poses on a path according to a hierarchical sampling strategy (CSP).
    """
    num_steps = len(edge)
    rearr = [edge[-1]]
    rearryarr = [edgeyarr[-1]]
    # Hierarchical sampling order
    for i in [0, 4, 2, 6, 1, 5, 3, 7]:
        for j in range(i, num_steps - 1, 8):
            rearr.append(edge[j])
            rearryarr.append(edgeyarr[j])

    group = []
    grouparr = []
    for pose, posecoll in zip(rearr, rearryarr):
        for link, linkcoll in zip(pose, posecoll):
            group.append(link)
            grouparr.append(linkcoll)
    return group, grouparr


def load_data(planner_type, benchid, dimension):
    """
    Loads motion trace data from a pickle file based on planner, benchmark ID, and dimension.
    """
    path_prefix = "../trace_files/motion_traces/"
    if dimension == "2D":
        if planner_type == "BIT":
            filename = f"{path_prefix}logfiles_BIT_2D/coord_motiom_{benchid}.pkl"
        elif planner_type == "GNN":
            filename = f"{path_prefix}logfiles_GNN_2D/coord_motiom_{benchid}.pkl"
        elif planner_type == "MPNET":
            filename = f"{path_prefix}logfiles_MPNET_2D/link_info_1_{benchid}.pkl"
        else:
            return None, None
    elif dimension == "nDOF":
        if planner_type == "BIT":
            filename = f"{path_prefix}logfiles_BIT_link/coord_motiom_{benchid}.pkl"
        elif planner_type == "GNN":
            filename = f"{path_prefix}logfiles_GNN_link/coord_gnn_motiom_{benchid}.pkl"
        elif planner_type == "MPNET":
            filename = f"{path_prefix}logfiles_MPNET_7D/coord_bench_3_{benchid}.pkl"
        else:
            return None, None
    else:
        return None, None

    try:
        with open(filename, "rb") as f:
            if planner_type == "MPNET":
                return pickle.load(f, encoding="latin1")
            else:
                return pickle.load(f)
    except FileNotFoundError:
        return None, None


def load_sphere_data(benchid, data_folder):
    """
    Loads sphere collision data from a pickle file.
    Format: (sphere_link_data, sphere_link_coll_data)
    """
    filename = f"{data_folder}/obstacles_{benchid}_sphere.pkl"
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            # 新格式: (sphere_link_data, sphere_link_coll_data)
            if isinstance(data, tuple) and len(data) == 2:
                return data
            # 兼容旧格式: (qarr, rarr, yarr)
            elif isinstance(data, tuple) and len(data) == 3:
                print(f"Warning: Old format detected in {filename}, converting...", file=sys.stderr)
                qarr_sphere, rarr_sphere, yarr_sphere = data
                return qarr_sphere, yarr_sphere  # 返回坐标和碰撞标签
            else:
                return None, None
    except FileNotFoundError:
        print(f"Warning: Sphere data file not found at {filename}", file=sys.stderr)
        return None, None


def update_collision_dict(colldict, hash_key, is_free, sample_rate):
    """
    Updates the collision history dictionary.
    """
    if hash_key in colldict:
        if is_free == 1 and random.random() <= sample_rate and colldict[hash_key][is_free] < 15:
            colldict[hash_key][is_free] += 1
        elif colldict[hash_key][is_free] < 15 and is_free == 0:
            colldict[hash_key][is_free] += 1
    else:
        colldict[hash_key] = [0, 0]
        if is_free == 1 and random.random() <= sample_rate and colldict[hash_key][is_free] < 15:
            colldict[hash_key][is_free] += 1
        elif colldict[hash_key][is_free] < 15 and is_free == 0:
            colldict[hash_key][is_free] += 1
    return colldict


def predict_collision(colldict, hash_key, threshold):
    """
    Predicts collision based on the history dictionary.
    """
    if hash_key in colldict:
        if colldict[hash_key][0] > colldict[hash_key][1] * threshold:
            return True  # Predict collision
        else:
            return False  # Predict free
    else:
        return False  # Predict free for unseen configurations


def simulate_parallel_collision_detection(
    linklist, linklist_coll, colldict, threshold, sample_rate, bins, qnoncoll_len=56, qcoll_len=8, cycle_check=40
):
    """
    Simulates the parallel collision detection process using OOCDs and prediction.
    """
    oocds = [[0, 0, 0, 0] for _ in range(7)]
    qcoll, qnoncoll = [], []
    cycle = 0
    first_two_running = 0
    first_two_checked = 0
    coll_found = 0
    links_remaining = len(linklist)
    everything_free = 0
    query_count = 0.0

    while not coll_found and not everything_free:
        # Process completed checks and schedule new ones
        for oocd_id in range(len(oocds)):
            oocd = oocds[oocd_id]
            if oocd[2] == 1 and oocd[3] <= cycle:
                query_count += 1
                if oocd[1] == 0:
                    coll_found = 1
                colldict = update_collision_dict(colldict, oocd[0], oocd[1], sample_rate)

            if oocd[3] <= cycle:
                if len(qcoll) > 0 and first_two_checked < cycle:
                    first_two_running += 1
                    if first_two_running == 1:
                        first_two_checked = cycle + cycle_check
                    oocds[oocd_id] = [qcoll[0][0], qcoll[0][1], 1, cycle + cycle_check]
                    del qcoll[0]
                elif len(qnoncoll) == qnoncoll_len or (
                    links_remaining == 0 and len(qnoncoll) > 0
                ) and first_two_checked < cycle:
                    oocds[oocd_id] = [qnoncoll[0][0], qnoncoll[0][1], 1, cycle + cycle_check]
                    del qnoncoll[0]
                else:
                    oocds[oocd_id] = [0, 0, 0, 0]

        # Predict and queue next link
        if len(linklist) > 0:
            link, linkcoll = linklist[0], linklist_coll[0]
            # This quantization part is script-specific, so we assume bins are passed or configured elsewhere
            # For now, let's create a placeholder for the key
            # In a real scenario, the binning logic would also be centralized or passed in.
            # For this fix, we'll assume a simple hash based on the link data itself.
            # NOTE: The binning logic is still in the main scripts, which is acceptable for now.
            # The key part is that the simulation loop itself is centralized.
            code_quant = np.digitize(link, bins, right=True)
            keyy = reutrn_keyy(code_quant)

            is_collision_predicted = predict_collision(colldict, keyy, threshold)

            if is_collision_predicted:
                if len(qcoll) < qcoll_len:
                    qcoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]
            else:
                if len(qnoncoll) < qnoncoll_len:
                    qnoncoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]

        links_remaining = len(linklist)
        if links_remaining == 0 and not any(oocd[3] > cycle for oocd in oocds) and not qnoncoll and not qcoll:
            everything_free = 1

        cycle += 1

    # Account for unfinished checks
    for oocd in oocds:
        if oocd[3] > cycle:
            query_count += (cycle_check - oocd[3] + cycle) / cycle_check

    return query_count, colldict, coll_found
