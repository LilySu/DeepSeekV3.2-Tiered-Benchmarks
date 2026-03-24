"""
Grouped GEMM interface for MoE expert dispatch.

Provides a unified API that dispatches to:
  1. DeepGEMM FP8 grouped GEMM (Hopper, highest performance)
  2. Triton grouped GEMM kernels (Ampere+)
  3. PyTorch reference implementation (CPU / any GPU)

The interface handles token sorting, group-size computation, and result
gathering -- the caller only needs to provide the router output.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedGEMMInterface:
    """Unified grouped GEMM dispatcher.

    Parameters
    ----------
    num_experts : int
        Total number of experts (256 for DeepSeek-V3).
    hidden_size : int
        Model hidden dimension.
    intermediate_size : int
        Expert FFN intermediate dimension.
    use_fp8 : bool
        Whether to attempt FP8 (DeepGEMM) dispatch.
    block_size : int
        FP8 quantisation tile size.
    """

    def __init__(
        self,
        num_experts: int = 256,
        hidden_size: int = 7168,
        intermediate_size: int = 2048,
        use_fp8: bool = True,
        block_size: int = 128,
    ) -> None:
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_fp8 = use_fp8
        self.block_size = block_size

        # Backend detection
        self._deepgemm = self._try_deepgemm()
        self._triton = self._try_triton()

    @staticmethod
    def _try_deepgemm():
        try:
            import deep_gemm
            return deep_gemm
        except ImportError:
            return None

    @staticmethod
    def _try_triton():
        try:
            import triton
            return triton
        except ImportError:
            return None

    @property
    def backend(self) -> str:
        """Currently active backend."""
        if self._deepgemm is not None and self.use_fp8:
            return "deepgemm_fp8"
        elif self._triton is not None:
            return "triton"
        return "pytorch"

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_weights: list,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens through experts via grouped GEMM.

        Parameters
        ----------
        hidden_states : [T, D]
            Flattened token representations.
        expert_weights : list of (gate_w, up_w, down_w) tuples
            Weight matrices for each expert.
        topk_weights : [T, top_k]
            Routing weights.
        topk_indices : [T, top_k]
            Expert assignments.

        Returns
        -------
        output : [T, D]
        """
        if self.backend == "deepgemm_fp8":
            return self._dispatch_deepgemm(
                hidden_states, expert_weights, topk_weights, topk_indices
            )
        elif self.backend == "triton":
            return self._dispatch_triton(
                hidden_states, expert_weights, topk_weights, topk_indices
            )
        return self._dispatch_pytorch(
            hidden_states, expert_weights, topk_weights, topk_indices
        )

    def _sort_by_expert(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sort tokens by expert assignment.

        Returns
        -------
        sorted_states : [T*K, D]
        sorted_weights : [T*K]
        sorted_expert_ids : [T*K]
        group_sizes : [num_experts]
        """
        T, K = topk_indices.shape
        D = hidden_states.shape[-1]

        # Expand hidden states for each top-k assignment
        token_ids = torch.arange(T, device=hidden_states.device)
        token_ids = token_ids.unsqueeze(1).expand(-1, K).reshape(-1)

        flat_indices = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)

        # Sort by expert ID
        sorted_order = flat_indices.argsort(stable=True)
        sorted_expert_ids = flat_indices[sorted_order]
        sorted_token_ids = token_ids[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        sorted_states = hidden_states[sorted_token_ids]

        group_sizes = torch.bincount(
            sorted_expert_ids, minlength=self.num_experts
        )

        return sorted_states, sorted_weights, sorted_expert_ids, group_sizes

    def _dispatch_pytorch(
        self,
        hidden_states: torch.Tensor,
        expert_weights: list,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reference PyTorch dispatch (no grouped GEMM)."""
        T, D = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        for k in range(topk_indices.shape[1]):
            for e in range(self.num_experts):
                mask = topk_indices[:, k] == e
                if not mask.any():
                    continue
                tokens = hidden_states[mask]
                gate_w, up_w, down_w = expert_weights[e]
                gate_out = F.silu(tokens @ gate_w.T)
                up_out = tokens @ up_w.T
                expert_out = (gate_out * up_out) @ down_w.T
                w = topk_weights[mask, k].unsqueeze(-1)
                output[mask] += w * expert_out

        return output

    def _dispatch_deepgemm(
        self,
        hidden_states: torch.Tensor,
        expert_weights: list,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """DeepGEMM FP8 grouped GEMM dispatch."""
        sorted_states, sorted_weights, sorted_expert_ids, group_sizes = (
            self._sort_by_expert(hidden_states, topk_weights, topk_indices)
        )

        T, K = topk_indices.shape
        D = hidden_states.shape[-1]

        # Stack expert weights
        gate_ws = torch.stack([ew[0] for ew in expert_weights])  # [E, I, D]
        up_ws = torch.stack([ew[1] for ew in expert_weights])
        down_ws = torch.stack([ew[2] for ew in expert_weights])

        # DeepGEMM grouped GEMM calls
        from ..fp8_utils import per_block_cast_to_fp8

        inp_fp8, inp_scales = per_block_cast_to_fp8(sorted_states)

        gate_out = self._deepgemm.m_grouped_gemm_fp8_fp8_bf16_nt(
            inp_fp8, inp_scales,
            gate_ws, torch.ones(1, device=hidden_states.device),
            group_sizes,
        )
        up_out = self._deepgemm.m_grouped_gemm_fp8_fp8_bf16_nt(
            inp_fp8, inp_scales,
            up_ws, torch.ones(1, device=hidden_states.device),
            group_sizes,
        )

        hidden = F.silu(gate_out) * up_out

        hidden_fp8, hidden_scales = per_block_cast_to_fp8(hidden)
        down_out = self._deepgemm.m_grouped_gemm_fp8_fp8_bf16_nt(
            hidden_fp8, hidden_scales,
            down_ws, torch.ones(1, device=hidden_states.device),
            group_sizes,
        )

        # Weighted scatter
        output = torch.zeros(T, D, dtype=hidden_states.dtype, device=hidden_states.device)
        token_ids = torch.arange(T, device=hidden_states.device)
        token_ids = token_ids.unsqueeze(1).expand(-1, K).reshape(-1)
        flat_indices = topk_indices.reshape(-1)
        sorted_order = flat_indices.argsort(stable=True)
        original_token_ids = token_ids[sorted_order]

        output.scatter_add_(
            0,
            original_token_ids.unsqueeze(-1).expand(-1, D),
            down_out * sorted_weights.unsqueeze(-1),
        )
        return output

    def _dispatch_triton(
        self,
        hidden_states: torch.Tensor,
        expert_weights: list,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Triton grouped GEMM dispatch (placeholder -- falls back to PyTorch)."""
        # TODO: Implement Triton grouped GEMM kernels
        return self._dispatch_pytorch(
            hidden_states, expert_weights, topk_weights, topk_indices
        )
