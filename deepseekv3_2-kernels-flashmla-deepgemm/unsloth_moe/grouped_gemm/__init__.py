"""
Grouped GEMM sub-package for MoE expert dispatch.

Provides both high-performance Triton/DeepGEMM kernels and reference
implementations for validation.
"""

from .interface import GroupedGEMMInterface

__all__ = ["GroupedGEMMInterface"]
