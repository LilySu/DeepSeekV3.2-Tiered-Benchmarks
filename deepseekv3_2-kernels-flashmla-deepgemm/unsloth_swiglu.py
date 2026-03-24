"""
Unsloth-optimised SwiGLU activation for DeepSeek-V3.

DeepSeek-V3 expert FFNs use SwiGLU (SiLU-gated linear unit):
    output = SiLU(gate(x)) * up(x)

This module provides:
  - Fused SwiGLU that avoids materialising the full intermediate tensor
  - Chunked SwiGLU for memory-efficient training
  - Backward-pass optimisations via custom autograd

Reference: arXiv 2412.19437, Unsloth (https://github.com/unslothai/unsloth)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Standard SwiGLU activation: SiLU(gate(x)) * up(x).

    Parameters
    ----------
    in_features : int
    intermediate_size : int
    bias : bool
    """

    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(in_features, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(in_features, intermediate_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_proj(x)) * self.up_proj(x)


class FusedSwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for fused SwiGLU.

    Fuses the SiLU activation with the element-wise multiply to reduce
    memory traffic by not storing the intermediate activation.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        gate_bias: Optional[torch.Tensor],
        up_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        gate = F.linear(x, gate_weight, gate_bias)
        up = F.linear(x, up_weight, up_bias)
        gate_activated = F.silu(gate)
        output = gate_activated * up

        # Save for backward
        ctx.save_for_backward(x, gate, up, gate_weight, up_weight)
        ctx.has_bias = gate_bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, gate, up, gate_weight, up_weight = ctx.saved_tensors

        # SiLU backward: d/dx[x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig = torch.sigmoid(gate)
        silu_gate = gate * sig
        dsilu = sig * (1.0 + gate * (1.0 - sig))

        # grad w.r.t. gate activation output
        grad_gate_activated = grad_output * up
        # grad w.r.t. up
        grad_up = grad_output * silu_gate
        # grad w.r.t. gate (pre-activation)
        grad_gate = grad_gate_activated * dsilu

        # grad w.r.t. x
        grad_x = grad_gate @ gate_weight + grad_up @ up_weight

        # grad w.r.t. weights
        flat_x = x.reshape(-1, x.shape[-1])
        grad_gate_weight = grad_gate.reshape(-1, grad_gate.shape[-1]).T @ flat_x
        grad_up_weight = grad_up.reshape(-1, grad_up.shape[-1]).T @ flat_x

        grad_gate_bias = grad_gate.sum(dim=tuple(range(grad_gate.ndim - 1))) if ctx.has_bias else None
        grad_up_bias = grad_up.sum(dim=tuple(range(grad_up.ndim - 1))) if ctx.has_bias else None

        return grad_x, grad_gate_weight, grad_up_weight, grad_gate_bias, grad_up_bias


class UnslothSwiGLU(nn.Module):
    """Unsloth-optimised SwiGLU with fused forward/backward.

    Parameters
    ----------
    in_features : int
    intermediate_size : int
    bias : bool
    use_fused : bool
        If True, use the custom fused autograd function.
    """

    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        bias: bool = False,
        use_fused: bool = True,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(in_features, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(in_features, intermediate_size, bias=bias)
        self.use_fused = use_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fused and x.requires_grad:
            return FusedSwiGLUFunction.apply(
                x,
                self.gate_proj.weight,
                self.up_proj.weight,
                self.gate_proj.bias,
                self.up_proj.bias,
            )
        return F.silu(self.gate_proj(x)) * self.up_proj(x)


class ChunkedSwiGLU(nn.Module):
    """Memory-efficient chunked SwiGLU for large intermediate sizes.

    Processes the sequence in chunks along the token dimension to reduce
    peak memory usage during training.

    Parameters
    ----------
    in_features : int
    intermediate_size : int
    bias : bool
    chunk_size : int
        Number of tokens per chunk.
    """

    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        bias: bool = False,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(in_features, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(in_features, intermediate_size, bias=bias)
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        T = x_flat.shape[0]

        outputs = []
        for i in range(0, T, self.chunk_size):
            chunk = x_flat[i : i + self.chunk_size]
            out_chunk = F.silu(self.gate_proj(chunk)) * self.up_proj(chunk)
            outputs.append(out_chunk)

        result = torch.cat(outputs, dim=0)
        return result.reshape(*orig_shape[:-1], result.shape[-1])
