import numpy as np
import math

# Constants
ADDR_CSR_CTRL = 0x0000
ADDR_STATUS = 0x0004
ADDR_COLLISION_MAP = 0x0008
ADDR_MOTION_STATUS = 0x000C

ADDR_SGU_CONFIG = 0x1000
ADDR_SGU_GEOM = 0x1500
ADDR_OCTREE_BASE = 0x2000
ADDR_SAS_RAM_BASE = 0x4000

# Fixed Point Conversions
def float_to_q2_14(val):
    limit = 2.0
    val = max(-limit, min(val, limit - (2**-14)))
    int_val = int(val * (2**14))
    return int_val & 0xFFFF

def float_to_q3_13(val):
    limit = 4.0
    val = max(-limit, min(val, limit - (2**-13)))
    int_val = int(val * (2**13))
    return int_val & 0xFFFF

def pack_sphere_geom(x, y, z, r2):
    qx = float_to_q2_14(x)
    qy = float_to_q2_14(y)
    qz = float_to_q2_14(z)
    qr2 = float_to_q2_14(r2)
    w0 = (qy << 16) | qx
    w1 = (qr2 << 16) | qz
    return w0, w1

def pack_link_params(rot_id, tx, ty, tz, active, is_last, axis_type, joint_idx, invert_axis):
    tx_f = float_to_q2_14(tx)
    ty_f = float_to_q2_14(ty)
    tz_f = float_to_q2_14(tz)
    raw_val = (
        ((rot_id & 0x1F) << 57) |
        ((tx_f & 0xFFFF) << 41) |
        ((ty_f & 0xFFFF) << 25) |
        ((tz_f & 0xFFFF) << 9) |
        (int(active) << 8) |
        (int(is_last) << 7) |
        ((axis_type & 0x3) << 5) |
        ((joint_idx & 0xF) << 1) |
        (int(invert_axis) << 0)
    )
    w0 = raw_val & 0xFFFFFFFF
    w1 = (raw_val >> 32) & 0xFFFFFFFF
    return w0, w1

def pack_llut_entry(count, start_row):
    """
    Packs count and start_row into a 10-bit LLUT entry.
    Format: {count[5:0], start_row[3:0]}
    """
    return ((count & 0x3F) << 4) | (start_row & 0xF)

def pack_sas_motion_config(motion_id, pose_count, start_pose, step_delta):
    def pack_joints(joints):
        val = 0
        for angle in reversed(joints):
            val = (val << 16) | (float_to_q3_13(angle) & 0xFFFF)
        return val
    p_start = pack_joints(start_pose)
    p_delta = pack_joints(step_delta)
    full_val = (motion_id & 0xFFFF) << 240
    full_val |= (pose_count & 0xFFFF) << 224
    full_val |= (p_delta & ((1<<112)-1)) << 112
    full_val |= (p_start & ((1<<112)-1))
    return full_val

# Octree Generation (Simplified version for driver)
WORLD_SIZE = 2.0
MAX_LEVEL = 6
GRID_RES = 64

def generate_octree_data(obstacles):
    grid = np.zeros((GRID_RES, GRID_RES, GRID_RES), dtype=bool)
    for center, half_ext in obstacles:
        # Simplified mapping
        ix_min = max(0, int((center[0] - half_ext[0] + 1.0) / 2.0 * 64))
        ix_max = min(63, int((center[0] + half_ext[0] + 1.0) / 2.0 * 64))
        iy_min = max(0, int((center[1] - half_ext[1] + 1.0) / 2.0 * 64))
        iy_max = min(63, int((center[1] + half_ext[1] + 1.0) / 2.0 * 64))
        iz_min = max(0, int((center[2] - half_ext[2] + 1.0) / 2.0 * 64))
        iz_max = min(63, int((center[2] + half_ext[2] + 1.0) / 2.0 * 64))
        grid[ix_min : ix_max + 1, iy_min : iy_max + 1, iz_min : iz_max + 1] = True

    queue = [((0, 64), (0, 64), (0, 64), 0)]
    addr_ptr = 1
    mem_image = []
    idx = 0
    while idx < len(queue):
        xr, yr, zr, level = queue[idx]
        idx += 1
        child_status = 0
        children_to_add = []
        mid_x, mid_y, mid_z = (xr[0] + xr[1]) // 2, (yr[0] + yr[1]) // 2, (zr[0] + zr[1]) // 2
        x_slices = [(xr[0], mid_x), (mid_x, xr[1])]
        y_slices = [(yr[0], mid_y), (mid_y, yr[1])]
        z_slices = [(zr[0], mid_z), (mid_z, zr[1])]
        for k in range(8):
            xi, yi, zi = k & 1, (k >> 1) & 1, (k >> 2) & 1
            block = grid[x_slices[xi][0]:x_slices[xi][1], y_slices[yi][0]:y_slices[yi][1], z_slices[zi][0]:z_slices[zi][1]]
            if not np.any(block): status = 0
            elif np.all(block) or level + 1 == MAX_LEVEL: status = 2
            else:
                status = 1
                children_to_add.append((x_slices[xi], y_slices[yi], z_slices[zi], level + 1))
            child_status |= status << (k * 2)
        base_idx = addr_ptr if children_to_add else 0
        addr_ptr += len(children_to_add)
        queue.extend(children_to_add)
        val = (child_status << 16) | (base_idx & 0x3FF)
        mem_image.append(val)
    return mem_image
