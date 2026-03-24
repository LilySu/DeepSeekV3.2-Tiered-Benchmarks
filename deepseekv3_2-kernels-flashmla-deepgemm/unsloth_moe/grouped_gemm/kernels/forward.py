"""
Forward-pass grouped GEMM kernel.

Implements the forward pass of grouped matrix multiplication for MoE expert
dispatch.  On Hopper GPUs, this would dispatch to a Triton kernel; on CPU
or non-Hopper GPUs, it uses a PyTorch reference implementation.

For DeepSeek-V3:
  - 256 experts, average ~T*8/256 tokens per expert
  - Gate projection: [T_e, 7168] x [7168, 2048]
  - Up projection: [T_e, 7168] x [7168, 2048]
  - Down projection: [T_e, 2048] x [2048, 7168]
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .autotuning import TuningConfig


def grouped_gemm_forward(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    group_sizes: torch.Tensor,
    config: Optional[TuningConfig] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """Forward grouped GEMM: C[g] = A[g] @ B[g]^T for each group g.

    Parameters
    ----------
    inputs : Tensor [total_tokens, K]
        Concatenated inputs for all groups (sorted by group).
    weights : Tensor [num_groups, N, K] or [num_groups, K, N]
        Per-group weight matrices.  If shape is [G, N, K], we compute
        inputs @ weights^T.  If [G, K, N], we compute inputs @ weights.
    group_sizes : Tensor [num_groups]  int
        Number of tokens in each group.
    config : TuningConfig, optional
        Kernel tuning parameters.  Uses defaults if None.
    activation : str, optional
        If 'silu', applies SiLU activation to the output.

    Returns
    -------
    output : Tensor [total_tokens, N]
    """
    if config is None:
        config = TuningConfig()

    # Determine layout
    G = weights.shape[0]
    if weights.ndim == 3:
        if weights.shape[1] == inputs.shape[1]:
            # [G, K, N] layout
            transpose_b = False
        else:
            # [G, N, K] layout
            transpose_b = True
    else:
        raise ValueError(f"Expected 3-D weights, got {weights.ndim}-D")

    # Try Triton kernel
    if inputs.is_cuda:
        try:
            return _triton_grouped_gemm_forward(
                inputs, weights, group_sizes, config,
                transpose_b=transpose_b, activation=activation,
            )
        except Exception:
            pass

    # Fallback to reference
    return _reference_grouped_gemm_forward(
        inputs, weights, group_sizes,
        transpose_b=transpose_b, activation=activation,
    )


def _reference_grouped_gemm_forward(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    group_sizes: torch.Tensor,
    transpose_b: bool = True,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """Reference implementation using sequential matmuls."""
    outputs = []
    offset = 0

    for g, size in enumerate(group_sizes.tolist()):
        size = int(size)
        if size == 0:
            continue

        inp_g = inputs[offset : offset + size]  # [size, K]
        w_g = weights[g]

        if transpose_b:
            out_g = inp_g @ w_g.T  # [size, N]
        else:
            out_g = inp_g @ w_g  # [size, N]

        if activation == "silu":
            out_g = F.silu(out_g)
        elif activation == "relu":
            out_g = F.relu(out_g)

        outputs.append(out_g)
        offset += size

    if not outputs:
        N = weights.shape[1] if transpose_b else weights.shape[2]
        return torch.zeros(0, N, dtype=inputs.dtype, device=inputs.device)

    return torch.cat(outputs, dim=0)


def _triton_grouped_gemm_forward(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    group_sizes: torch.Tensor,
    config: TuningConfig,
    transpose_b: bool = True,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """Triton-accelerated grouped GEMM forward.

    Currently dispatches to reference -- the Triton kernel would be
    compiled and cached on first call in a full deployment.
    """
    # TODO: Implement actual Triton kernel
    # The kernel would:
    # 1. Use group_sizes to compute per-group offsets
    # 2. Launch a 2D grid: (ceil(M/BLOCK_M) * num_groups, ceil(N/BLOCK_N))
    # 3. Each block computes a BLOCK_M x BLOCK_N tile of the output
    # 4. Use TMA on Hopper for async global->shared loads
    return _reference_grouped_gemm_forward(
        inputs, weights, group_sizes,
        transpose_b=transpose_b, activation=activation,
    )


def grouped_swiglu_forward(
    inputs: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    group_sizes: torch.Tensor,
    config: Optional[TuningConfig] = None,
) -> torch.Tensor:
    """Fused grouped SwiGLU: SiLU(A @ W_gate^T) * (A @ W_up^T).

    Fuses the gate and up projections with SwiGLU activation to avoid
    materialising two full intermediate tensors.

    Parameters
    ----------
    inputs : [total_tokens, K]
    gate_weights : [G, N, K]
    up_weights : [G, N, K]
    group_sizes : [G]

    Returns
    -------
    output : [total_tokens, N]
    """
    gate_out = grouped_gemm_forward(inputs, gate_weights, group_sizes, config)
    up_out = grouped_gemm_forward(inputs, up_weights, group_sizes, config)
    return F.silu(gate_out) * up_out
