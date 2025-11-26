"""
Data loading utilities for motion traces and collision data.
"""

import pickle
import sys


def load_motion_trace_data(planner_type, benchid, dimension):
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


def load_data(basename, benchid, data_folder, collision_model_type="link"):
    """
    Loads collision data from a pickle file.

    Args:
        basename: Base name of the dataset (e.g., "iiwa_7")
        benchid: Benchmark number
        data_folder: Path to the data folder
        collision_model_type: Type of collision model ("link" or "sphere", default="link")

    Returns:
        (collision_data, collision_flags) tuple or (None, None)

    File naming convention:
        {basename}_{benchid:04d}_{collision_model_type}.pkl
    """
    filename = f"{data_folder}/{basename}_{benchid:04d}_{collision_model_type}.pkl"

    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) >= 2:
                return data[0], data[1]
    except FileNotFoundError:
        pass

    print(f"Warning: Collision data file not found at {filename}", file=sys.stderr)
    return None, None


def load_data_with_cycles(basename, benchid, data_folder, collision_model_type="link"):
    """
    Loads collision data with cycles from a pickle file.

    Args:
        basename: Base name of the dataset (e.g., "iiwa_7")
        benchid: Benchmark number
        data_folder: Path to the data folder
        collision_model_type: Type of collision model ("link" or "sphere", default="link")

    Returns:
        (collision_data, collision_flags, cycles) tuple or (None, None, None)

    File naming convention:
        - Sphere model: {basename}_{benchid:04d}_sphere_geometric_cycles.pkl
        - Link model: {basename}_{benchid:04d}_{collision_model_type}_cycles.pkl
    """
    if collision_model_type == "sphere":
        filename = f"{data_folder}/{basename}_{benchid:04d}_sphere_geometric_cycles.pkl"
    else:
        filename = (
            f"{data_folder}/{basename}_{benchid:04d}_{collision_model_type}_cycles.pkl"
        )

    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 3:
                return data[0], data[1], data[2]
    except FileNotFoundError:
        pass

    print(
        f"Warning: Collision data with cycles file not found at {filename}",
        file=sys.stderr,
    )
    return None, None, None
