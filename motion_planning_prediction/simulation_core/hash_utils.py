"""
Hash and binning utilities for collision detection.
"""

import numpy as np


def calculate_bins(quant_min, quant_max, quant_bits):
    """
    计算量化分箱的边界

    Args:
        quant_min: 坐标的最小值
        quant_max: 坐标的最大值
        quant_bits: 量化位数（每个维度的比特数）

    Returns:
        bins: 分箱边界数组
    """
    num_bins = 2**quant_bits
    bins = np.linspace(quant_min, quant_max, num_bins)
    return bins


def return_keyy(code, quant_bits):
    """
    将量化编码转换为二进制字符串

    Args:
        code: 量化编码数组，例如 [3, 5, 2]（每个元素是量化值）
        quant_bits: 每个量化值的比特宽度（例如4表示每个值用4位表示）

    Returns:
        keyy: 二进制编码字符串，例如 "001101010010"（每个元素转为二进制后拼接）

    说明：
        假定 quant_bits=4，则每个量化值用4个比特表示
        例如：code=[3, 5, 2], quant_bits=4 -> "0011" + "0101" + "0010" = "001101010010"
        最终返回的二进制字符串长度为 len(code) * quant_bits
    """
    bitsize = len(code)
    keyy = ""

    for j in range(bitsize):
        # 将每个量化值转为二进制，用零补齐到quant_bits位
        binary_str = format(int(code[j]), f"0{quant_bits}b")
        keyy = keyy + binary_str

    return keyy


def compute_hash_keyy(link_coords, bins):
    """
    Args:
        link_coords: 单个link的坐标列表（7D: [x, y, z, qx, qy, qz, qw] 或 [x, y, z, radius]）
        bins: 分箱边界数组

    Returns:
        hash_key: 量化编码后的hash key字符串
    """
    # 只对坐标部分[0:3]进行量化
    code_quant = np.digitize(link_coords[0:3], bins, right=True)
    # 从bins计算quant_bits
    quant_bits = (len(bins) - 1).bit_length()
    # 转换为hash key字符串
    keyy = return_keyy(code_quant, quant_bits)
    return keyy
