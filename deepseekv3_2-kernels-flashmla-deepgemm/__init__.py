"""
DeepSeek-V3 FlashMLA + DeepGEMM Kernel Package
================================================

High-performance kernel implementation for DeepSeek-V3 (671B) inference and
fine-tuning, combining:

  - **FlashMLA**: Flash-decoding kernel purpose-built for Multi-head Latent
    Attention (MLA).  Handles the compressed KV cache (kv_lora_rank=512) and
    partial RoPE (qk_rope=64) natively on Hopper GPUs.
  - **DeepGEMM**: FP8 grouped GEMM kernel optimised for Mixture-of-Experts
    dispatch (256 experts, top-8).  Uses Hopper TMA for high-throughput
    expert-parallel matrix multiplies.

Architecture reference: arXiv 2412.19437 (DeepSeek-V3 Technical Report).

This package is the **highest-performance** kernel combination for DeepSeek-V3
because FlashMLA and DeepGEMM together cover the two most compute-intensive
operations (MLA attention and MoE expert dispatch) with hardware-native
acceleration on H100/H200 GPUs.

Note: DeepSeek-V3 does **not** use Dynamic Sparse Attention (DSA). DSA stub
modules are provided for interface compatibility only.
"""

from .config import DEEPSEEK_V3_CONFIG
from .model import DeepSeekV3Model
from .cache import KVCache
from .mla_attention import MLAttention
from .rope_partial import YaRNRotaryEmbedding
from .fp8_utils import (
    quantize_fp8_block,
    dequantize_fp8_block,
    per_block_cast_to_fp8,
    FP8TensorWrapper,
)
from .moe_grouped_gemm import MoEGroupedGEMM
from .moe_router import MoERouter

__version__ = "0.1.0"
__all__ = [
    "DEEPSEEK_V3_CONFIG",
    "DeepSeekV3Model",
    "KVCache",
    "MLAttention",
    "YaRNRotaryEmbedding",
    "quantize_fp8_block",
    "dequantize_fp8_block",
    "per_block_cast_to_fp8",
    "FP8TensorWrapper",
    "MoEGroupedGEMM",
    "MoERouter",
]
