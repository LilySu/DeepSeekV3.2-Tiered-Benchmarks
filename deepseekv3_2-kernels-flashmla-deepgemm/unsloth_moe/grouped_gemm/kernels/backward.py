"""
Backward-pass grouped GEMM kernel.

Implements the backward pass for grouped matrix multiplication, computing
gradients with respect to both inputs and weights for each expert group.

For MoE training with DeepSeek-V3:
  - dL/dA[g] = dL/dC[g] @ B[g]   (input gradient)
  - dL/dB[g] = dL/dC[g]^T @ A[g]  (weight gradient)

These are themselves grouped GEMMs with the same group structure.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .autotuning import TuningConfig


def grouped_gemm_backward(
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    group_sizes: torch.Tensor,
    config: Optional[TuningConfig] = None,
    needs_input_grad: bool = True,
    needs_weight_grad: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward pass for grouped GEMM.

    Given forward: C[g] = A[g] @ B[g]^T

    Computes:
      - dA[g] = dC[g] @ B[g]  (if needs_input_grad)
      - dB[g] = dC[g]^T @ A[g]  (if needs_weight_grad)

    Parameters
    ----------
    grad_output : Tensor [total_tokens, N]
        Gradient of loss w.r.t. forward output.
    inputs : Tensor [total_tokens, K]
        Saved inputs from forward pass.
    weights : Tensor [num_groups, N, K]
        Weight matrices (same as forward).
    group_sizes : Tensor [num_groups]
    config : TuningConfig, optional
    needs_input_grad : bool
    needs_weight_grad : bool

    Returns
    -------
    grad_input : Tensor [total_tokens, K] or None
    grad_weight : Tensor [num_groups, N, K] or None
    """
    if config is None:
        config = TuningConfig()

    # Try Triton kernel
    if inputs.is_cuda:
        try:
            return _triton_grouped_gemm_backward(
                grad_output, inputs, weights, group_sizes, config,
                needs_input_grad, needs_weight_grad,
            )
        except Exception:
            pass

    # Fallback
    return _reference_grouped_gemm_backward(
        grad_output, inputs, weights, group_sizes,
        needs_input_grad, needs_weight_grad,
    )


def _reference_grouped_gemm_backward(
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    group_sizes: torch.Tensor,
    needs_input_grad: bool = True,
    needs_weight_grad: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Reference backward using sequential matmuls."""
    G = weights.shape[0]
    K = weights.shape[2]
    N = weights.shape[1]

    grad_input = torch.zeros_like(inputs) if needs_input_grad else None
    grad_weight = torch.zeros_like(weights) if needs_weight_grad else None

    offset = 0
    for g, size in enumerate(group_sizes.tolist()):
        size = int(size)
        if size == 0:
            continue

        grad_out_g = grad_output[offset : offset + size]  # [size, N]
        inp_g = inputs[offset : offset + size]  # [size, K]
        w_g = weights[g]  # [N, K]

        if needs_input_grad:
            # dA = dC @ B  ([size, N] @ [N, K] -> [size, K])
            grad_input[offset : offset + size] = grad_out_g @ w_g

        if needs_weight_grad:
            # dB = dC^T @ A  ([N, size] @ [size, K] -> [N, K])
            grad_weight[g] = grad_out_g.T @ inp_g

        offset += size

    return grad_input, grad_weight


def _triton_grouped_gemm_backward(
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    group_sizes: torch.Tensor,
    config: TuningConfig,
    needs_input_grad: bool = True,
    needs_weight_grad: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Triton-accelerated grouped GEMM backward.

    Currently dispatches to reference implementation.
    """
    # TODO: Implement Triton backward kernels
    return _reference_grouped_gemm_backward(
        grad_output, inputs, weights, group_sizes,
        needs_input_grad, needs_weight_grad,
    )


class GroupedGEMMFunction(torch.autograd.Function):
    """Autograd wrapper for grouped GEMM forward + backward.

    Usage::

        output = GroupedGEMMFunction.apply(inputs, weights, group_sizes)
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        group_sizes: torch.Tensor,
    ) -> torch.Tensor:
        from .forward import grouped_gemm_forward
        ctx.save_for_backward(inputs, weights, group_sizes)
        return grouped_gemm_forward(inputs, weights, group_sizes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, weights, group_sizes = ctx.saved_tensors
        grad_input, grad_weight = grouped_gemm_backward(
            grad_output, inputs, weights, group_sizes,
            needs_input_grad=ctx.needs_input_grad[0],
            needs_weight_grad=ctx.needs_input_grad[1],
        )
        return grad_input, grad_weight, None  # No gradient for group_sizes


class GroupedSwiGLUFunction(torch.autograd.Function):
    """Autograd wrapper for grouped SwiGLU (gate + up + activation).

    Forward:  out = SiLU(A @ W_gate^T) * (A @ W_up^T)
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        group_sizes: torch.Tensor,
    ) -> torch.Tensor:
        from .forward import grouped_gemm_forward

        gate_out = grouped_gemm_forward(inputs, gate_weights, group_sizes)
        up_out = grouped_gemm_forward(inputs, up_weights, group_sizes)
        gate_activated = F.silu(gate_out)
        output = gate_activated * up_out

        ctx.save_for_backward(
            inputs, gate_weights, up_weights, group_sizes,
            gate_out, up_out,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (inputs, gate_weights, up_weights, group_sizes,
         gate_out, up_out) = ctx.saved_tensors

        # SwiGLU backward
        sig = torch.sigmoid(gate_out)
        silu_gate = gate_out * sig
        dsilu = sig * (1.0 + gate_out * (1.0 - sig))

        grad_gate_activated = grad_output * up_out
        grad_up_out = grad_output * silu_gate
        grad_gate_out = grad_gate_activated * dsilu

        # Backward through grouped GEMMs
        grad_input_gate, grad_gate_w = grouped_gemm_backward(
            grad_gate_out, inputs, gate_weights, group_sizes,
        )
        grad_input_up, grad_up_w = grouped_gemm_backward(
            grad_up_out, inputs, up_weights, group_sizes,
        )

        grad_input = grad_input_gate + grad_input_up

        return grad_input, grad_gate_w, grad_up_w, None
