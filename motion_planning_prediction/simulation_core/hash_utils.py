"""
Hash and binning utilities for collision detection.
"""

import numpy as np
import json
import os


def calculate_bins(quant_min, quant_max, quant_bits):
    """
    计算量化分箱的边界

    Args:
        quant_min: 坐标的最小值
        quant_max: 坐标的最大值
        quant_bits: 量化位数（每个维度的比特数）

    Returns:
        bins
    """
    num_bins = 2**quant_bits
    bins = np.linspace(quant_min, quant_max, num_bins)
    # 为每个维度返回相同的bins（可以后续扩展为不同bins）
    return bins


def return_keyy(code, quant_bits):
    """
    将量化编码转换为二进制字符串，使用bit interleaving方式

    Args:
        code: 量化编码数组，例如 [3, 5, 2]（每个元素是量化值）
        quant_bits: 每个量化值的比特宽度（例如4表示每个值用4位表示）

    Returns:
        keyy: 二进制编码字符串，使用bit interleaving：对于每个bit位置，从每个维度取该bit位置的bit

    说明：
        对于3个维度，每个4bit，bit interleaving顺序为：
        bit0_dim0, bit0_dim1, bit0_dim2, bit1_dim0, bit1_dim1, bit1_dim2, ...
        例如：code=[3, 5, 2], quant_bits=4 -> "110101001000"
    """
    bitsize = len(code)
    keyy = ""

    for bit_pos in range(quant_bits):
        for dim in range(bitsize):
            # 从每个维度的当前bit位置提取bit
            bit = (int(code[dim]) >> bit_pos) & 1
            keyy += str(bit)

    return keyy


def compute_hash_keyy(link_coords, bins):
    """
    Args:
        link_coords: 单个link的坐标列表（7D: [x, y, z, qx, qy, qz, qw] 或 [x, y, z, radius]）
        bins: 包含三个分箱边界数组的列表 [bins_x, bins_y, bins_z]

    Returns:
        hash_key: 量化编码后的hash key字符串
    """
    # 对每个维度使用对应的bins进行量化
    code_quant = []
    for dim in range(3):
        coord = link_coords[dim]
        dim_bins = bins[dim]
        quant_val = np.digitize(coord, dim_bins, right=True)
        code_quant.append(quant_val)

    code_quant = np.array(code_quant)
    # 从第一个bins计算quant_bits（假设所有维度使用相同的量化位数）
    quant_bits = (len(bins[0]) - 1).bit_length()
    # 转换为hash key字符串
    keyy = return_keyy(code_quant, quant_bits)
    return keyy


def calculate_bins_from_workspace(robot_name, quant_bits):
    """
    根据机器人名称读取workspace信息，计算三个维度上的量化分箱边界

    Args:
        robot_name: 机器人名称，用于构造workspace文件路径
        quant_bits: 量化位数（每个维度的比特数）

    Returns:
        bins: 包含三个分箱边界数组的列表 [bins_x, bins_y, bins_z]
    """
    # 构造workspace文件路径（直接使用项目根目录路径）
    workspace_file = f"/home/lanh/project/robot_sim/coll_prediction_artifact/trace_generation/data/workspace_bounds/{robot_name}_workspace.json"

    # 检查文件是否存在
    if not os.path.exists(workspace_file):
        raise FileNotFoundError(f"Workspace file not found: {workspace_file}")

    # 读取workspace信息
    with open(workspace_file, "r") as f:
        workspace_data = json.load(f)

    # 提取各维度的范围
    x_min = workspace_data["x_start"]
    x_max = workspace_data["x_end"]
    y_min = workspace_data["y_start"]
    y_max = workspace_data["y_end"]
    z_min = workspace_data["z_start"]
    z_max = workspace_data["z_end"]

    # 计算每个维度的bins
    num_bins = 2**quant_bits
    bins_x = np.linspace(x_min, x_max, num_bins)
    bins_y = np.linspace(y_min, y_max, num_bins)
    bins_z = np.linspace(z_min, z_max, num_bins)

    return [bins_x, bins_y, bins_z]
