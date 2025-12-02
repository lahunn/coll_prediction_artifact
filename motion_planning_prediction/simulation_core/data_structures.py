"""
Data structures for collision detection simulation.
"""

from collections import deque


# Class for OOCD state
class OOCDState:
    def __init__(self, hash_key, result, busy, free_cycle):
        self.hash_key = hash_key
        self.result = result
        self.busy = busy
        self.free_cycle = free_cycle

    def reset(self):
        self.hash_key = 0
        self.result = 1
        self.busy = 0
        self.free_cycle = 0


class OOCDStatePreemptive:
    def __init__(self, hash_key, result, busy, free_cycle, task_type):
        self.hash_key = hash_key
        self.result = result
        self.busy = busy
        self.free_cycle = free_cycle
        self.task_type = task_type

    def reset(self):
        self.hash_key = 0
        self.result = 1
        self.busy = 0
        self.free_cycle = 0
        self.task_type = 0


class Prediction:
    def __init__(self, qcoll_len, qnoncoll_len):
        self.qcoll = deque(maxlen=qcoll_len)
        self.qnoncoll = deque(maxlen=qnoncoll_len)
        self.linklist = []
        self.linklist_coll = []
