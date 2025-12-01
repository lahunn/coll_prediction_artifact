"""
Simulation Core Package

This package contains the core simulation functionality for collision detection,
organized into modular components for better maintainability.
"""

from .constants import (
    NUM_OOCDS,
    MAX_COLLISION_COUNT,
    DEFAULT_QNONCOLL_LEN,
    DEFAULT_QCOLL_LEN,
    DEFAULT_CYCLE_CHECK,
)
from .data_structures import (
    OOCDState,
    OOCDStatePreemptive,
    Prediction,
)
from .hash_utils import (
    calculate_bins,
    return_keyy,
    compute_hash_keyy,
    calculate_bins_from_workspace,
)
from .collision_prediction import (
    update_collision_dict,
    predict_collision,
    calculate_accuracy,
    enqueue_predictions,
    initialize_cht,
    inherit_cht,
)
from .data_loader import (
    load_motion_trace_data,
    load_data,
    load_data_with_cycles,
)
from .data_preprocessing import (
    csp_rearrange,
    csp_rearrange_with_cycles,
    generate_recursive_reorder,
    recursive_binary_reorder,
    allocate_edge_data_to_copus,
)
from .oocd_processor import (
    process_oocds,
    process_oocd_states_preemptive,
    handle_preemption,
    process_oocd_states_dedicated,
)
from .simulators import (
    simulate_parallel_collision_detection,
    simulate_parallel_collision_detection_real_cycles,
    simulate_parallel_collision_detection_with_tracking,
    simulate_parallel_collision_detection_preemptive,
    simulate_parallel_collision_detection_dedicated,
    simulate_parallel_collision_detection_double_buffer,
)
from .oracle_utils import (
    calculate_oracle_cycles,
    calculate_oracle_cycles_for_edges,
)

__all__ = [
    # Constants
    'NUM_OOCDS',
    'MAX_COLLISION_COUNT',
    'DEFAULT_QNONCOLL_LEN',
    'DEFAULT_QCOLL_LEN',
    'DEFAULT_CYCLE_CHECK',
    
    # Data Structures
    'OOCDState',
    'OOCDStatePreemptive',
    'Prediction',
    
    # Hash Utils
    'calculate_bins',
    'return_keyy',
    'compute_hash_keyy',
    'calculate_bins_from_workspace',
    
    # Collision Prediction
    'update_collision_dict',
    'predict_collision',
    'calculate_accuracy',
    'enqueue_predictions',
    'initialize_cht',
    'inherit_cht',
    
    # Data Loader
    'load_motion_trace_data',
    'load_data',
    'load_data_with_cycles',
    
    # Data Preprocessing
    'csp_rearrange',
    'csp_rearrange_with_cycles',
    'generate_recursive_reorder',
    'recursive_binary_reorder',
    'allocate_edge_data_to_copus',
    
    # OOCD Processor
    'process_oocds',
    'process_oocd_states_preemptive',
    'handle_preemption',
    'process_oocd_states_dedicated',
    
    # Simulators
    'simulate_parallel_collision_detection',
    'simulate_parallel_collision_detection_real_cycles',
    'simulate_parallel_collision_detection_with_tracking',
    'simulate_parallel_collision_detection_preemptive',
    'simulate_parallel_collision_detection_dedicated',
    'simulate_parallel_collision_detection_double_buffer',
    
    # Oracle Utils
    'calculate_oracle_cycles',
    'calculate_oracle_cycles_for_edges',
]
