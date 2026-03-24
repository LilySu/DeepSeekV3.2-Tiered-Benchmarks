"""
Unsloth utility functions for DeepSeek-V3.

General-purpose utilities used across the Unsloth-optimised components:

  - Device detection and capability checks
  - Mixed-precision helpers
  - Memory estimation
  - Model patching utilities
  - Gradient accumulation helpers

Reference: Unsloth (https://github.com/unslothai/unsloth)
"""

from __future__ import annotations

import gc
import os
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Device / capability detection
# ---------------------------------------------------------------------------

def get_device_info() -> Dict[str, Any]:
    """Get information about the current CUDA device.

    Returns
    -------
    dict with keys: name, capability, memory_gb, is_hopper, supports_fp8
    """
    if not torch.cuda.is_available():
        return {
            "name": "cpu",
            "capability": (0, 0),
            "memory_gb": 0.0,
            "is_hopper": False,
            "supports_fp8": False,
            "supports_tma": False,
        }

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    cap = (props.major, props.minor)
    mem_gb = props.total_mem / (1024 ** 3)

    return {
        "name": props.name,
        "capability": cap,
        "memory_gb": mem_gb,
        "is_hopper": props.major >= 9,
        "supports_fp8": props.major >= 9,  # SM90+
        "supports_tma": props.major >= 9,  # Hopper TMA
    }


def is_hopper_gpu() -> bool:
    """Check if the current GPU is a Hopper (SM90+) device."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.major >= 9


def supports_fp8() -> bool:
    """Check if the current device supports FP8 (E4M3) natively."""
    return is_hopper_gpu() and hasattr(torch, "float8_e4m3fn")


def supports_flash_mla() -> bool:
    """Check if FlashMLA is available."""
    try:
        import flash_mla
        return True
    except ImportError:
        return False


def supports_deepgemm() -> bool:
    """Check if DeepGEMM is available."""
    try:
        import deep_gemm
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Memory utilities
# ---------------------------------------------------------------------------

def estimate_model_memory(
    hidden_size: int = 7168,
    num_layers: int = 61,
    vocab_size: int = 129280,
    num_experts: int = 256,
    expert_intermediate: int = 2048,
    shared_intermediate: int = 2048,
    first_k_dense: int = 3,
    dense_intermediate: int = 18432,
    dtype_bytes: int = 2,
) -> Dict[str, float]:
    """Estimate model memory usage in GB.

    Returns
    -------
    dict with keys:
      - embedding: float (GB)
      - attention: float (GB)
      - dense_ffn: float (GB)
      - moe_ffn: float (GB)
      - total: float (GB)
    """
    # Embedding
    emb = vocab_size * hidden_size * dtype_bytes

    # Per-layer attention params
    # q_a_proj: H * q_lora (7168 * 1536)
    # q_b_proj: q_lora * (heads * qk_head) (1536 * 128*192)
    # kv_a_proj: H * (kv_lora + rope) (7168 * 576)
    # kv_b_proj: kv_lora * (heads * (nope + v)) (512 * 128*256)
    # o_proj: (heads * v) * H (128*128 * 7168)
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_head_dim = 192
    heads = 128
    v_head = 128
    qk_nope = 128

    attn_per_layer = (
        hidden_size * q_lora_rank  # q_a_proj
        + q_lora_rank * heads * qk_head_dim  # q_b_proj
        + hidden_size * (kv_lora_rank + 64)  # kv_a_proj
        + kv_lora_rank * heads * (qk_nope + v_head)  # kv_b_proj
        + heads * v_head * hidden_size  # o_proj
    ) * dtype_bytes

    attn_total = attn_per_layer * num_layers

    # Dense FFN (first k layers)
    dense_per_layer = 3 * hidden_size * dense_intermediate * dtype_bytes
    dense_total = dense_per_layer * first_k_dense

    # MoE FFN
    expert_params = 3 * hidden_size * expert_intermediate  # gate, up, down
    shared_params = 3 * hidden_size * shared_intermediate
    moe_per_layer = (num_experts * expert_params + shared_params) * dtype_bytes
    moe_total = moe_per_layer * (num_layers - first_k_dense)

    total = emb + attn_total + dense_total + moe_total

    return {
        "embedding_gb": emb / (1024 ** 3),
        "attention_gb": attn_total / (1024 ** 3),
        "dense_ffn_gb": dense_total / (1024 ** 3),
        "moe_ffn_gb": moe_total / (1024 ** 3),
        "total_gb": total / (1024 ** 3),
    }


def estimate_kv_cache_memory(
    batch_size: int = 1,
    seq_len: int = 4096,
    num_layers: int = 61,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    dtype_bytes: int = 2,
) -> float:
    """Estimate KV cache memory in GB."""
    per_layer = batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
    return per_layer * num_layers / (1024 ** 3)


def clear_memory() -> None:
    """Aggressively clear GPU and CPU memory caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Mixed precision helpers
# ---------------------------------------------------------------------------

def get_optimal_dtype() -> torch.dtype:
    """Get the best dtype for the current hardware."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def upcast_for_computation(
    x: torch.Tensor, target_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Upcast tensor for numerically sensitive computations."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.to(target_dtype)
    return x


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

def freeze_model_except(
    model: nn.Module,
    unfrozen_names: Optional[List[str]] = None,
) -> int:
    """Freeze all parameters except those matching given names.

    Parameters
    ----------
    model : nn.Module
    unfrozen_names : list of str
        Substrings to match against parameter names. Matching parameters
        remain trainable.

    Returns
    -------
    num_trainable : int
        Number of trainable parameters after freezing.
    """
    if unfrozen_names is None:
        unfrozen_names = ["lora_"]

    for name, param in model.named_parameters():
        param.requires_grad = any(un in name for un in unfrozen_names)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by category.

    Returns
    -------
    dict with keys: total, trainable, frozen, embedding, attention, ffn
    """
    total = 0
    trainable = 0
    embedding = 0
    attention = 0
    ffn = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        if "embed" in name:
            embedding += n
        elif "attn" in name or "q_" in name or "kv_" in name or "o_proj" in name:
            attention += n
        elif "mlp" in name or "expert" in name or "gate_proj" in name:
            ffn += n

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "embedding": embedding,
        "attention": attention,
        "ffn": ffn,
    }


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------

class GradientAccumulator:
    """Helper for gradient accumulation with proper scaling.

    Parameters
    ----------
    accumulation_steps : int
    max_grad_norm : float
        For gradient clipping.
    """

    def __init__(
        self,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self._step = 0

    def should_step(self) -> bool:
        """Whether the optimizer should step on this iteration."""
        self._step += 1
        return self._step % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss by accumulation steps."""
        return loss / self.accumulation_steps

    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return the norm."""
        return torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            self.max_grad_norm,
        ).item()

    def reset(self) -> None:
        self._step = 0
