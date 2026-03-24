"""
Triton kernels for grouped GEMM MoE dispatch.

Sub-modules:
  - forward: Forward-pass grouped GEMM kernels
  - backward: Backward-pass grouped GEMM kernels
  - autotuning: Kernel auto-tuning infrastructure
  - tuning: Pre-computed tuning configs for DeepSeek-V3 shapes
"""

from .forward import grouped_gemm_forward
from .backward import grouped_gemm_backward
from .autotuning import AutoTuner, TuningConfig
from .tuning import DEEPSEEK_V3_TUNING_CONFIGS

__all__ = [
    "grouped_gemm_forward",
    "grouped_gemm_backward",
    "AutoTuner",
    "TuningConfig",
    "DEEPSEEK_V3_TUNING_CONFIGS",
]
