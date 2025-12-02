"""
Collision prediction functions.
"""

import random
from .constants import MAX_COLLISION_COUNT
from .hash_utils import compute_hash_keyy


def update_collision_dict(colldict, hash_key, is_free, sample_rate):
    """
    Updates the collision history dictionary.
    当计数器达到MAX_COLLISION_COUNT时，两个计数器同时除以2（右移1位）
    """
    # 检查是否需要更新计数器
    should_update = False
    if is_free == 1 and random.random() <= sample_rate:
        should_update = True
    elif is_free == 0:
        should_update = True

    if should_update:
        if hash_key not in colldict:
            colldict[hash_key] = [0, 0]
        else:
            # 检查是否达到计数器上限
            if colldict[hash_key][is_free] >= MAX_COLLISION_COUNT:
                # 饱和计数器：两个计数器同时除以2（右移1位）
                colldict[hash_key][0] = colldict[hash_key][0] // 2
                colldict[hash_key][1] = colldict[hash_key][1] // 2

        # 增加计数
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


def initialize_cht():
    """
    初始化碰撞历史表 (CHT)
    """
    return {}


def inherit_cht(colldict, decay_factor=1.0):
    """
    继承碰撞历史表，应用衰减因子

    Args:
        colldict: 原始碰撞历史表
        decay_factor: 衰减因子 (0-1)，1.0表示无衰减

    Returns:
        继承后的碰撞历史表
    """
    inherited = {}
    for key, counts in colldict.items():
        inherited[key] = [int(counts[0] * decay_factor), int(counts[1] * decay_factor)]
    return inherited


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
        keyy = compute_hash_keyy(link, bins)
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


def predict_next_config(
    linklist,
    linklist_coll,
    qcoll,
    qnoncoll,
    bins,
    threshold,
    cht_scheduler,
    qcoll_size,
    qnoncoll_size,
):
    """预测下一个配置并入队"""
    if len(linklist) > 0:
        link = linklist[0]
        linkcoll = linklist_coll[0]

        keyy = compute_hash_keyy(link, bins)

        # 查询CHT
        is_ready = False
        coll_count, noncoll_count = 0, 0

        is_ready, data = cht_scheduler.get_read_result(keyy)
        if is_ready:
            coll_count, noncoll_count = data

        if is_ready:
            is_collision_predicted = coll_count > noncoll_count * threshold

            # 入队
            if is_collision_predicted:
                if len(qcoll) < qcoll_size:
                    qcoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]
            else:
                if len(qnoncoll) < qnoncoll_size:
                    qnoncoll.append([keyy, linkcoll])
                    del linklist[0]
                    del linklist_coll[0]
