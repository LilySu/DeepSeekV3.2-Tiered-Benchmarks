"""
MoE dispatch using DeepGEMM FP8 grouped GEMM.

This module implements the Mixture-of-Experts feed-forward layer for
DeepSeek-V3, using DeepGEMM's FP8 grouped GEMM kernel on Hopper GPUs for
maximum throughput.

Architecture:
  - 256 routed experts + 1 shared expert
  - Each expert is a standard SwiGLU FFN
  - Routing via sigmoid top-8 with group restriction (n_group=8, topk_group=4)
  - Expert weights are stored in FP8 (E4M3) with 128x128 block scales
  - DeepGEMM dispatches all active experts in a single grouped GEMM call

The grouped GEMM avoids the overhead of 256 individual small matmuls by
batching them into one kernel launch that uses Hopper TMA for efficient
data movement.

Reference: arXiv 2412.19437, Section 3.3-3.4.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepSeekV3Config, DEEPSEEK_V3_CONFIG
from .moe_router import MoERouter
from .fp8_utils import (
    FP8TensorWrapper,
    quantize_fp8_block,
    dequantize_fp8_block,
    per_block_cast_to_fp8,
    BLOCK_SIZE,
)


# ---------------------------------------------------------------------------
# Expert FFN
# ---------------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """Single SwiGLU expert feed-forward network.

    gate_proj, up_proj: hidden_size -> expert_intermediate_size
    down_proj: expert_intermediate_size -> hidden_size
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Grouped GEMM dispatcher (eager fallback + DeepGEMM hook)
# ---------------------------------------------------------------------------

def _try_import_deepgemm():
    """Attempt to import the DeepGEMM library."""
    try:
        import deep_gemm
        return deep_gemm
    except ImportError:
        return None


class GroupedGEMMDispatcher:
    """Dispatch grouped GEMM via DeepGEMM (FP8) or PyTorch fallback.

    On Hopper GPUs with DeepGEMM installed, this uses the FP8 grouped GEMM
    kernel.  Otherwise it falls back to sequential expert matmuls.
    """

    def __init__(self, use_fp8: bool = True, block_size: int = BLOCK_SIZE) -> None:
        self.use_fp8 = use_fp8
        self.block_size = block_size
        self._deep_gemm = _try_import_deepgemm()

    @property
    def has_deepgemm(self) -> bool:
        return self._deep_gemm is not None

    def grouped_gemm_fp8(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        weight_scales: torch.Tensor,
        group_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """FP8 grouped GEMM via DeepGEMM.

        Parameters
        ----------
        inputs : [total_tokens, K]  bfloat16
        weights : [num_groups, K, N]  fp8
        weight_scales : per-block scales for weights
        group_sizes : [num_groups] int  -- tokens per group

        Returns
        -------
        output : [total_tokens, N]  bfloat16
        """
        if self._deep_gemm is not None:
            # Quantise inputs to FP8
            inp_fp8, inp_scales = per_block_cast_to_fp8(inputs, self.block_size)
            return self._deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
                inp_fp8, inp_scales,
                weights, weight_scales,
                group_sizes,
            )
        else:
            # Fallback: sequential matmul
            return self._sequential_gemm(inputs, weights, group_sizes)

    def _sequential_gemm(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        group_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback sequential GEMM for each group."""
        outputs = []
        offset = 0
        for g, size in enumerate(group_sizes.tolist()):
            if size == 0:
                continue
            inp_g = inputs[offset : offset + size]  # [size, K]
            w_g = weights[g].float()  # [K, N]
            out_g = inp_g.float() @ w_g
            outputs.append(out_g.to(inputs.dtype))
            offset += size
        if outputs:
            return torch.cat(outputs, dim=0)
        return torch.zeros(0, weights.shape[-1], dtype=inputs.dtype, device=inputs.device)


# ---------------------------------------------------------------------------
# Main MoE layer
# ---------------------------------------------------------------------------

class MoEGroupedGEMM(nn.Module):
    """DeepSeek-V3 Mixture-of-Experts layer with DeepGEMM dispatch.

    Parameters
    ----------
    config : DeepSeekV3Config
    layer_idx : int
        Layer index (first_k_dense_replace layers use dense FFN).
    """

    def __init__(
        self,
        config: DeepSeekV3Config = DEEPSEEK_V3_CONFIG,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.moe.num_experts
        self.top_k = config.moe.num_experts_per_tok
        self.is_dense = layer_idx < config.moe.first_k_dense_replace

        if self.is_dense:
            # Dense layers use a single FFN
            self.dense_ffn = ExpertFFN(
                config.hidden_size,
                config.intermediate_size,
            )
        else:
            # Router
            self.router = MoERouter(config)

            # Routed experts
            self.experts = nn.ModuleList([
                ExpertFFN(
                    config.hidden_size,
                    config.moe.expert_intermediate_size,
                )
                for _ in range(self.num_experts)
            ])

            # Shared expert(s)
            if config.moe.num_shared_experts > 0:
                self.shared_expert = ExpertFFN(
                    config.hidden_size,
                    config.moe.shared_expert_intermediate_size,
                )
            else:
                self.shared_expert = None

            # Grouped GEMM dispatcher
            self.dispatcher = GroupedGEMMDispatcher(
                use_fp8=config.moe.use_fp8_gemm,
                block_size=config.moe.fp8_block_size,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        hidden_states : [batch, seq_len, hidden_size]

        Returns
        -------
        output : [batch, seq_len, hidden_size]
        router_logits : [batch*seq_len, num_experts] or None for dense layers
        """
        bsz, seq_len, dim = hidden_states.shape

        if self.is_dense:
            return self.dense_ffn(hidden_states), None

        # Flatten for routing
        flat = hidden_states.reshape(-1, dim)  # [T, D]
        T = flat.shape[0]

        # Route
        topk_weights, topk_indices, router_logits = self.router(flat)
        # topk_weights: [T, top_k], topk_indices: [T, top_k]

        # Shared expert contribution
        if self.shared_expert is not None:
            shared_out = self.shared_expert(flat)  # [T, D]
        else:
            shared_out = torch.zeros_like(flat)

        # Expert dispatch — use DeepGEMM grouped GEMM when available
        if self.dispatcher.has_deepgemm and not self.training:
            routed_out = self._dispatch_experts_grouped_gemm(flat, topk_weights, topk_indices)
        else:
            routed_out = self._dispatch_experts(flat, topk_weights, topk_indices)

        # Combine
        output = shared_out + routed_out
        output = output.reshape(bsz, seq_len, dim)

        return output, router_logits

    def _dispatch_experts(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts and aggregate results.

        Uses DeepGEMM grouped GEMM when available, otherwise falls back
        to a token-level scatter/gather with individual expert forward passes.

        Parameters
        ----------
        flat : [T, D]
        topk_weights : [T, top_k]
        topk_indices : [T, top_k]

        Returns
        -------
        output : [T, D]
        """
        T, D = flat.shape

        # For each expert, gather tokens assigned to it
        output = torch.zeros_like(flat)

        # Build expert -> token mapping
        expert_mask = torch.zeros(
            self.num_experts, T, dtype=torch.bool, device=flat.device
        )
        for k in range(self.top_k):
            indices = topk_indices[:, k]  # [T]
            for e in range(self.num_experts):
                expert_mask[e] |= (indices == e)

        # Process each expert
        for e in range(self.num_experts):
            token_mask = expert_mask[e]  # [T]
            if not token_mask.any():
                continue

            token_indices = token_mask.nonzero(as_tuple=True)[0]
            expert_input = flat[token_indices]  # [num_tokens_e, D]

            # Expert forward
            expert_output = self.experts[e](expert_input)  # [num_tokens_e, D]

            # Weighted scatter back
            for k in range(self.top_k):
                mask_k = topk_indices[token_indices, k] == e
                if mask_k.any():
                    valid_tokens = token_indices[mask_k]
                    weights = topk_weights[valid_tokens, k].unsqueeze(-1)
                    output[valid_tokens] += weights * expert_output[mask_k]

        return output

    def _dispatch_experts_grouped_gemm(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch using DeepGEMM grouped GEMM (H100 optimised path).

        This performs all expert gate_proj matmuls in a single grouped GEMM,
        then all up_proj matmuls, then combines with SwiGLU, and finally
        all down_proj matmuls.

        Parameters
        ----------
        flat : [T, D]
        topk_weights : [T, top_k]
        topk_indices : [T, top_k]

        Returns
        -------
        output : [T, D]
        """
        T, D = flat.shape

        # Sort tokens by expert
        flat_indices = topk_indices.reshape(-1)  # [T*top_k]
        flat_weights = topk_weights.reshape(-1)  # [T*top_k]
        token_ids = torch.arange(T, device=flat.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        sorted_order = flat_indices.argsort()
        sorted_expert_ids = flat_indices[sorted_order]
        sorted_token_ids = token_ids[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        # Group sizes
        group_sizes = torch.bincount(sorted_expert_ids, minlength=self.num_experts)

        # Gather inputs in sorted order
        sorted_inputs = flat[sorted_token_ids]  # [T*top_k, D]

        # Stack expert weights for grouped GEMM (cached after first call)
        if not hasattr(self, '_stacked_gate_w'):
            self._stacked_gate_w = torch.stack([e.gate_proj.weight.T for e in self.experts])
            self._stacked_up_w = torch.stack([e.up_proj.weight.T for e in self.experts])
            self._stacked_down_w = torch.stack([e.down_proj.weight.T for e in self.experts])
            # Compute per-block FP8 scales from weight magnitudes (128x128 blocks)
            bs = self.dispatcher.block_size
            def _compute_block_scales(w):
                """Compute per-block max for FP8 scaling. w: [E, M, N]"""
                E, M, N = w.shape
                Mb = (M + bs - 1) // bs
                Nb = (N + bs - 1) // bs
                scales = torch.ones(E, Mb, Nb, device=w.device, dtype=torch.float32)
                for bi in range(Mb):
                    for bj in range(Nb):
                        block = w[:, bi*bs:min((bi+1)*bs, M), bj*bs:min((bj+1)*bs, N)]
                        scales[:, bi, bj] = block.abs().amax(dim=(-1, -2)).clamp(min=1e-12)
                return scales
            self._gate_scales = _compute_block_scales(self._stacked_gate_w)
            self._up_scales = _compute_block_scales(self._stacked_up_w)
            self._down_scales = _compute_block_scales(self._stacked_down_w)

        # Grouped GEMM: gate
        gate_out = self.dispatcher.grouped_gemm_fp8(
            sorted_inputs, self._stacked_gate_w,
            self._gate_scales,
            group_sizes,
        )

        # Grouped GEMM: up
        up_out = self.dispatcher.grouped_gemm_fp8(
            sorted_inputs, self._stacked_up_w,
            self._up_scales,
            group_sizes,
        )

        # SwiGLU
        hidden = F.silu(gate_out) * up_out

        # Grouped GEMM: down
        down_out = self.dispatcher.grouped_gemm_fp8(
            hidden, self._stacked_down_w,
            self._down_scales,
            group_sizes,
        )

        # Weighted scatter back
        output = torch.zeros(T, D, dtype=flat.dtype, device=flat.device)
        output.scatter_add_(
            0,
            sorted_token_ids.unsqueeze(-1).expand(-1, D),
            down_out * sorted_weights.unsqueeze(-1),
        )

        return output
