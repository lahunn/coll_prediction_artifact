"""
Collision prediction functions.
"""

import random
import numpy as np
from .constants import MAX_COLLISION_COUNT
from .hash_utils import return_keyy


def update_collision_dict(colldict, hash_key, is_free, sample_rate):
    """
    Updates the collision history dictionary.
    """
    if hash_key in colldict:
        if (
            is_free == 1
            and random.random() <= sample_rate
            and colldict[hash_key][is_free] < MAX_COLLISION_COUNT
        ):
            colldict[hash_key][is_free] += 1
        elif colldict[hash_key][is_free] < MAX_COLLISION_COUNT and is_free == 0:
            colldict[hash_key][is_free] += 1
    else:
        colldict[hash_key] = [0, 0]
        if (
            is_free == 1
            and random.random() <= sample_rate
            and colldict[hash_key][is_free] < MAX_COLLISION_COUNT
        ):
            colldict[hash_key][is_free] += 1
        elif colldict[hash_key][is_free] < MAX_COLLISION_COUNT and is_free == 0:
            colldict[hash_key][is_free] += 1
    return colldict


def predict_collision(colldict, hash_key, threshold):
    """
    Predicts collision based on the history dictionary.
    """
    if hash_key in colldict:
        if colldict[hash_key][0] > colldict[hash_key][1] * threshold:
            return True
        else:
            return False
    else:
        return False


def calculate_accuracy(predictions, actuals):
    """
    计算预测准确率

    Args:
        predictions: 预测结果列表 (True/False)
        actuals: 实际结果列表 (0/1, 0表示碰撞)

    Returns:
        float: 准确率 (0-1)
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return 0.0

    correct = 0
    for pred, act in zip(predictions, actuals):
        pred_collision = pred
        actual_collision = act == 0
        if pred_collision == actual_collision:
            correct += 1

    return correct / len(predictions)


def enqueue_predictions(
    linklist,
    linklist_coll,
    qcoll,
    qnoncoll,
    colldict,
    threshold,
    bins,
    qcoll_len,
    qnoncoll_len,
    linklist_cycles=None,
    predictions=None,
    actuals=None,
):
    if len(linklist) > 0:
        link, linkcoll = linklist[0], linklist_coll[0]
        code_quant = np.digitize(link, bins, right=True)
        quant_bits = (len(bins) - 1).bit_length()
        keyy = return_keyy(code_quant, quant_bits)
        is_collision_predicted = predict_collision(colldict, keyy, threshold)
        if predictions is not None:
            predictions.append(is_collision_predicted)
        if actuals is not None:
            actuals.append(linkcoll)
        if is_collision_predicted:
            if len(qcoll) < qcoll_len:
                if linklist_cycles is not None:
                    link_cycle = linklist_cycles[0]
                    qcoll.append([keyy, linkcoll, link_cycle])
                    del linklist_cycles[0]
                else:
                    qcoll.append([keyy, linkcoll])
                del linklist[0]
                del linklist_coll[0]
        else:
            if len(qnoncoll) < qnoncoll_len:
                if linklist_cycles is not None:
                    link_cycle = linklist_cycles[0]
                    qnoncoll.append([keyy, linkcoll, link_cycle])
                    del linklist_cycles[0]
                else:
                    qnoncoll.append([keyy, linkcoll])
                del linklist[0]
                del linklist_coll[0]
