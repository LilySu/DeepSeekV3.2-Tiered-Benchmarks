"""
Reference MoE implementations for validation and testing.

These are pure-PyTorch implementations that produce numerically correct
results, used to validate the optimised Triton/DeepGEMM kernels.
"""

from .moe_block import ReferenceMoEBlock
from .moe_ops import (
    reference_expert_dispatch,
    reference_expert_gather,
    reference_swiglu_expert,
)

__all__ = [
    "ReferenceMoEBlock",
    "reference_expert_dispatch",
    "reference_expert_gather",
    "reference_swiglu_expert",
]
