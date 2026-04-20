"""
Constants for collision detection simulation.
"""
from trace_generation.config.ana_parameters import sphere_cost

NUM_OOCDS = 8
DEFAULT_NUM_DEDICATED_OOCDS = 128
MAX_COLLISION_COUNT = 15
DEFAULT_QNONCOLL_LEN = 64
DEFAULT_QCOLL_LEN = 8
# DEFAULT_CYCLE_CHECK = sphere_cost
DEFAULT_CYCLE_CHECK = 60  # 固定周期，简化调度
CHT_DEFAULT_SIZE = 4096
ONE_CYCLE_DELAY = 0
