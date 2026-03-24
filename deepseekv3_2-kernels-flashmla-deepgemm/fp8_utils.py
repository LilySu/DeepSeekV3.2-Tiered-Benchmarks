"""
FP8 utilities for DeepGEMM integration.

DeepGEMM operates on E4M3 FP8 tensors with **per-block** (128x128) scale
factors.  This module provides:

  - Block-wise FP8 quantisation / dequantisation
  - Scale-factor computation (absmax per 128x128 tile)
  - A lightweight ``FP8TensorWrapper`` that bundles data + scale for dispatch

Reference: DeepGEMM, DeepSeek-V3 Technical Report arXiv 2412.19437 Sec 3.4.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# E4M3 range (IEEE 754): max representable value
FP8_E4M3_MAX: float = 448.0
FP8_E4M3_MIN: float = -448.0

# Default tile size for block quantisation (matches DeepGEMM)
BLOCK_SIZE: int = 128


# ---------------------------------------------------------------------------
# Core quantisation routines
# ---------------------------------------------------------------------------

def _compute_block_scales(
    x: torch.Tensor,
    block_m: int = BLOCK_SIZE,
    block_n: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Compute per-block absmax scale factors for a 2-D tensor.

    Parameters
    ----------
    x : Tensor [M, N]  (float16 / bfloat16 / float32)
    block_m, block_n : tile dimensions

    Returns
    -------
    scales : Tensor [ceil(M/block_m), ceil(N/block_n)]  float32
    """
    M, N = x.shape
    # Pad to multiple of block size
    pm = (block_m - M % block_m) % block_m
    pn = (block_n - N % block_n) % block_n
    if pm or pn:
        x = torch.nn.functional.pad(x, (0, pn, 0, pm))

    Mp, Np = x.shape
    # Reshape into blocks
    x_blocks = x.reshape(Mp // block_m, block_m, Np // block_n, block_n)
    x_blocks = x_blocks.permute(0, 2, 1, 3)  # [Bm, Bn, block_m, block_n]

    # Absmax per block
    absmax = x_blocks.abs().amax(dim=(-2, -1)).float()  # [Bm, Bn]
    # Scale = absmax / FP8_MAX  (avoid division by zero)
    scales = absmax / FP8_E4M3_MAX
    scales = scales.clamp(min=1e-12)
    return scales


def quantize_fp8_block(
    x: torch.Tensor,
    block_m: int = BLOCK_SIZE,
    block_n: int = BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise a 2-D tensor to FP8 E4M3 with per-block scaling.

    Parameters
    ----------
    x : Tensor [M, N]

    Returns
    -------
    x_fp8 : Tensor [M, N]  float8_e4m3fn  (or float16 fallback)
    scales : Tensor [ceil(M/block_m), ceil(N/block_n)]  float32
    """
    assert x.ndim == 2, f"Expected 2-D tensor, got {x.ndim}-D"
    M, N = x.shape

    scales = _compute_block_scales(x, block_m, block_n)

    # Pad input
    pm = (block_m - M % block_m) % block_m
    pn = (block_n - N % block_n) % block_n
    x_padded = x
    if pm or pn:
        x_padded = torch.nn.functional.pad(x, (0, pn, 0, pm))

    Mp, Np = x_padded.shape
    Bm, Bn = Mp // block_m, Np // block_n

    # Scale and clamp
    x_blocks = x_padded.reshape(Bm, block_m, Bn, block_n).permute(0, 2, 1, 3).float()
    x_scaled = x_blocks / scales[:, :, None, None]
    x_scaled = x_scaled.clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)

    # Reshape back
    x_quant = x_scaled.permute(0, 2, 1, 3).reshape(Mp, Np)
    # Remove padding
    x_quant = x_quant[:M, :N]

    # Cast to FP8 if available, else keep float16 with simulated quantisation
    if hasattr(torch, "float8_e4m3fn"):
        x_fp8 = x_quant.to(torch.float8_e4m3fn)
    else:
        # Fallback: round to FP8 precision by clamping mantissa bits
        x_fp8 = _simulate_fp8_e4m3(x_quant)

    return x_fp8, scales


def quantize_activations_fp8(
    x: torch.Tensor,
    channel_block: int = BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise activations to FP8 E4M3 with per-token scaling (1x128 blocks).

    Per the DeepSeek-V3 paper, activations use 1x128 block quantization
    (one scale per token per 128 channels), NOT 128x128 like weights.
    This preserves per-token dynamic range while still quantizing channels.

    Parameters
    ----------
    x : Tensor [T, D]  (tokens x channels)

    Returns
    -------
    x_fp8 : Tensor [T, D]  float8_e4m3fn
    scales : Tensor [T, ceil(D/128)]  float32
    """
    return quantize_fp8_block(x, block_m=1, block_n=channel_block)


def dequantize_fp8_block(
    x_fp8: torch.Tensor,
    scales: torch.Tensor,
    block_m: int = BLOCK_SIZE,
    block_n: int = BLOCK_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantise a block-quantised FP8 tensor back to higher precision.

    Parameters
    ----------
    x_fp8 : Tensor [M, N]
    scales : Tensor [Bm, Bn]

    Returns
    -------
    x : Tensor [M, N] in ``output_dtype``
    """
    M, N = x_fp8.shape
    x = x_fp8.float()

    pm = (block_m - M % block_m) % block_m
    pn = (block_n - N % block_n) % block_n
    if pm or pn:
        x = torch.nn.functional.pad(x, (0, pn, 0, pm))

    Mp, Np = x.shape
    Bm, Bn = Mp // block_m, Np // block_n

    x_blocks = x.reshape(Bm, block_m, Bn, block_n).permute(0, 2, 1, 3)
    x_deq = x_blocks * scales[:, :, None, None]
    x_deq = x_deq.permute(0, 2, 1, 3).reshape(Mp, Np)
    return x_deq[:M, :N].to(output_dtype)


def per_block_cast_to_fp8(
    x: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience: quantise any-dim tensor by treating last two dims as [M, N].

    If input has more than 2 dims, the leading dimensions are flattened into M.

    Returns
    -------
    x_fp8 : Tensor (same leading shape), last dim = N
    scales : Tensor [ceil(M/block), ceil(N/block)]
    """
    orig_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim == 1:
        x = x.unsqueeze(0)

    x_fp8, scales = quantize_fp8_block(x, block_size, block_size)

    if len(orig_shape) > 2:
        x_fp8 = x_fp8.reshape(orig_shape)
    elif len(orig_shape) == 1:
        x_fp8 = x_fp8.squeeze(0)

    return x_fp8, scales


# ---------------------------------------------------------------------------
# FP8 simulation fallback (for GPUs without native FP8 support)
# ---------------------------------------------------------------------------

def _simulate_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Simulate FP8 E4M3 quantisation via rounding.

    This is used when ``torch.float8_e4m3fn`` is not available (e.g., on
    pre-Hopper GPUs or CPU).
    """
    # Clamp to FP8 range
    x = x.clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    # Simulate 3-bit mantissa: round to nearest representable
    # For each value, compute the exponent and round mantissa
    sign = x.sign()
    abs_x = x.abs().clamp(min=1e-30)
    exp = torch.floor(torch.log2(abs_x))
    # Mantissa has 3 bits -> 8 levels
    mantissa = abs_x / (2.0 ** exp)  # [1, 2)
    mantissa_q = torch.round(mantissa * 8.0) / 8.0
    result = sign * mantissa_q * (2.0 ** exp)
    return result.to(x.dtype)


# ---------------------------------------------------------------------------
# FP8 tensor wrapper
# ---------------------------------------------------------------------------

class FP8TensorWrapper:
    """Lightweight container for an FP8 tensor + its block scale factors.

    This is the interface expected by :class:`MoEGroupedGEMM` for DeepGEMM
    dispatch.

    Parameters
    ----------
    data : Tensor  [M, N] in FP8 format
    scales : Tensor [Bm, Bn] float32 block scales
    block_size : int  tile size used during quantisation
    """

    __slots__ = ("data", "scales", "block_size", "shape", "dtype")

    def __init__(
        self,
        data: torch.Tensor,
        scales: torch.Tensor,
        block_size: int = BLOCK_SIZE,
    ) -> None:
        self.data = data
        self.scales = scales
        self.block_size = block_size
        self.shape = data.shape
        self.dtype = data.dtype

    @classmethod
    def from_float(
        cls,
        x: torch.Tensor,
        block_size: int = BLOCK_SIZE,
    ) -> "FP8TensorWrapper":
        """Create from a full-precision tensor."""
        if x.ndim > 2:
            M = 1
            for s in x.shape[:-1]:
                M *= s
            x_2d = x.reshape(M, x.shape[-1])
        else:
            x_2d = x
        data, scales = quantize_fp8_block(x_2d, block_size, block_size)
        return cls(data.reshape(x.shape), scales, block_size)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Dequantise back to full precision."""
        data_2d = self.data.reshape(-1, self.data.shape[-1])
        return dequantize_fp8_block(
            data_2d, self.scales, self.block_size, self.block_size, dtype
        ).reshape(self.data.shape)

    def to(self, device: torch.device) -> "FP8TensorWrapper":
        """Move to device."""
        return FP8TensorWrapper(
            self.data.to(device),
            self.scales.to(device),
            self.block_size,
        )

    def __repr__(self) -> str:
        return (
            f"FP8TensorWrapper(shape={list(self.shape)}, "
            f"block_size={self.block_size}, "
            f"scales_shape={list(self.scales.shape)})"
        )
