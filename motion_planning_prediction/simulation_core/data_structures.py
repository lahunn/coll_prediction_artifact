"""
Data structures for collision detection simulation.
"""

from collections import deque, namedtuple

# Named tuple for OOCD state
OOCDState = namedtuple("OOCDState", ["hash_key", "result", "busy", "free_cycle"])
OOCDStatePreemptive = namedtuple(
    "OOCDStatePreemptive", ["hash_key", "result", "busy", "free_cycle", "task_type"]
)


class Prediction:
    def __init__(self, qcoll_len, qnoncoll_len):
        self.qcoll = deque(maxlen=qcoll_len)
        self.qnoncoll = deque(maxlen=qnoncoll_len)
        self.linklist = []
        self.linklist_coll = []
