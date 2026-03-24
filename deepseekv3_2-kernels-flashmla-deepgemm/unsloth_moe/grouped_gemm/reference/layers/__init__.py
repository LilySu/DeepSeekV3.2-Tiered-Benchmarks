"""
Reference MoE layer implementations for various architectures.

Provides reference implementations that can be used for:
  - Numerical validation of optimised kernels
  - Architecture comparison studies
  - Unit testing
"""

from .llama4_moe import Llama4MoELayer
from .qwen3_moe import Qwen3MoELayer

__all__ = ["Llama4MoELayer", "Qwen3MoELayer"]
