"""
Unsloth MoE -- Mixture-of-Experts utilities for DeepSeek-V3.

This sub-package provides:
  - Grouped GEMM interface for efficient expert dispatch
  - Reference MoE implementations for testing and validation
  - Triton kernel auto-tuning infrastructure

Designed to work with DeepGEMM FP8 grouped GEMM on Hopper GPUs and
fall back to reference implementations on other hardware.
"""

from .grouped_gemm.interface import GroupedGEMMInterface
from .grouped_gemm.reference.moe_block import ReferenceMoEBlock
from .grouped_gemm.reference.moe_ops import (
    reference_expert_dispatch,
    reference_expert_gather,
)

__all__ = [
    "GroupedGEMMInterface",
    "ReferenceMoEBlock",
    "reference_expert_dispatch",
    "reference_expert_gather",
]
