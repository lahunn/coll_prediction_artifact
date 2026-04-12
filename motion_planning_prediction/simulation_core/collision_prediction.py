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


def submit_cht_write(cht_scheduler, pred_id, hash_key, is_collision, sample_rate):
    """
    根据OOCD结果提交CHT写操作，参考update_collision_dict的逻辑

    Args:
        cht_scheduler: CHT访问调度器
        pred_id: prediction的ID
        hash_key: 配置的哈希键
        is_collision: OOCD结果 (0表示碰撞, 1表示无碰撞)
        sample_rate: 采样率，用于决定是否更新无碰撞计数
    """

    if is_collision == 0:
        # 碰撞：增加碰撞计数
        cht_scheduler.submit_write(pred_id, hash_key, 1, 0)
    else:
        # 无碰撞：按sample_rate概率增加无碰撞计数
        if random.random() <= sample_rate:
            cht_scheduler.submit_write(pred_id, hash_key, 0, 1)


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


def enqueue_predictions_by_link(
    linklist,
    linklist_coll,
    qcoll,
    qnoncoll,
    colldict,
    threshold,
    bins,
    qcoll_len,
    qnoncoll_len,
    link_to_spheres,
    sphere_to_link,
    num_spheres_per_pose,
    pose_cursor,
):
    """
    对属于同一link的所有sphere进行预测并入队。

    每次从linklist中取一个link的所有sphere，对每个sphere分别进行预测（使用各自的坐标），
    然后将它们分别入队到qcoll或qnoncoll。

    与enqueue_link_predictions的区别：
    - enqueue_link_predictions: 整个link作为一个任务入队（payload包含所有sphere）
    - enqueue_predictions_by_link: 每个sphere作为独立任务入队（但同一link的sphere一起预测）
    """
    if not linklist:
        return

    cursor_in_pose = pose_cursor[0]
    if cursor_in_pose >= num_spheres_per_pose:
        pose_cursor[0] = cursor_in_pose % num_spheres_per_pose
        cursor_in_pose = pose_cursor[0]

    link_id = sphere_to_link[cursor_in_pose]
    sphere_indices = link_to_spheres[link_id]
    count = len(sphere_indices)

    # 逐个处理该link的每个sphere，只有当队列有空间时才入队
    for i in range(count):
        if not linklist:  # 安全检查
            break

        coord = linklist[0]
        coll = linklist_coll[0]
        keyy = compute_hash_keyy(coord, bins)
        is_collision_predicted = predict_collision(colldict, keyy, threshold)

        if is_collision_predicted:
            if len(qcoll) < qcoll_len:
                qcoll.append([keyy, coll])
                del linklist[0]
                del linklist_coll[0]
                pose_cursor[0] = (pose_cursor[0] + 1) % num_spheres_per_pose
            else:
                # 队列满，停止处理，下次重试
                break
        else:
            if len(qnoncoll) < qnoncoll_len:
                qnoncoll.append([keyy, coll])
                del linklist[0]
                del linklist_coll[0]
                pose_cursor[0] = (pose_cursor[0] + 1) % num_spheres_per_pose
            else:
                # 队列满，停止处理，下次重试
                break


def enqueue_link_predictions(
    linklist,
    linklist_coll,
    qcoll,
    qnoncoll,
    colldict,
    threshold,
    bins,
    qcoll_len,
    qnoncoll_len,
    link_to_spheres,
    sphere_to_link,
    num_spheres_per_pose,
    pose_cursor,
):
    """Enqueue one link-level prediction task and remove that link's spheres.

    This keeps queue entries link-granular (single hash key), while the payload
    carries all sphere collision labels (and optional cycles) for that link so
    dispatch can fan out per sphere.
    """
    if not linklist:
        return

    cursor_in_pose = pose_cursor[0]
    if cursor_in_pose >= num_spheres_per_pose:
        pose_cursor[0] = cursor_in_pose % num_spheres_per_pose
        cursor_in_pose = pose_cursor[0]

    link_id = sphere_to_link[cursor_in_pose]
    sphere_indices = link_to_spheres[link_id]
    count = len(sphere_indices)

    if len(linklist) < count or len(linklist_coll) < count:
        return

    coords_slice = linklist[:count]
    colls_slice = linklist_coll[:count]

    keyy = compute_hash_keyy(coords_slice[0], bins)
    is_collision_predicted = predict_collision(colldict, keyy, threshold)

    task = [keyy, colls_slice]

    if is_collision_predicted:
        if len(qcoll) < qcoll_len:
            qcoll.append(task)
            del linklist[:count]
            del linklist_coll[:count]
            pose_cursor[0] = (pose_cursor[0] + count) % num_spheres_per_pose
    else:
        if len(qnoncoll) < qnoncoll_len:
            qnoncoll.append(task)
            del linklist[:count]
            del linklist_coll[:count]
            pose_cursor[0] = (pose_cursor[0] + count) % num_spheres_per_pose


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
    copu_id=0,
):
    """预测下一个配置并入队"""
    if len(linklist) > 0:
        link = linklist[0]
        linkcoll = linklist_coll[0]

        keyy = compute_hash_keyy(link, bins)

        # 查询CHT
        is_ready = False
        coll_count, noncoll_count = 0, 0

        is_ready, data = cht_scheduler.get_read_result(keyy, copu_id=copu_id)
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
