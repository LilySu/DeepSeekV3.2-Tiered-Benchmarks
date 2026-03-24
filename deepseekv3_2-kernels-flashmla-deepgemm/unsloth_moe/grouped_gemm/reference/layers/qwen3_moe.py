"""
Qwen3 MoE layer reference implementation.

Provided for comparison with DeepSeek-V3's MoE.  Key differences:

  - Qwen3: softmax routing, standard top-k, auxiliary loss
  - DeepSeek-V3: sigmoid routing, group-restricted top-k, no auxiliary loss
  - Qwen3: typically fewer experts (e.g., 64), lower top-k (e.g., 4)
  - DeepSeek-V3: 256 experts, top-8, with group structure

This reference is useful for:
  - Validating shared grouped GEMM infrastructure
  - Comparing routing strategies and expert counts
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3MoEExpert(nn.Module):
    """Single Qwen3 SwiGLU expert."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoERouter(nn.Module):
    """Qwen3 softmax top-k router with auxiliary loss."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 4,
        aux_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts
        self.aux_loss_weight = aux_loss_weight

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(hidden_states)
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)
        # Renormalise
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_indices, logits

    def compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Z-loss + load balance loss (Qwen3 style)."""
        T = router_logits.shape[0]
        probs = F.softmax(router_logits, dim=-1)

        # Load balance loss
        avg_probs = probs.mean(dim=0)
        counts = torch.bincount(
            topk_indices.reshape(-1), minlength=self.num_experts
        ).float()
        freq = counts / (T * self.top_k)
        lb_loss = self.num_experts * (avg_probs * freq).sum()

        # Z-loss (router logit regularisation)
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        return self.aux_loss_weight * (lb_loss + 0.001 * z_loss)


class Qwen3MoELayer(nn.Module):
    """Qwen3 MoE layer (reference implementation).

    Parameters
    ----------
    hidden_size : int
    intermediate_size : int
    num_experts : int
    top_k : int
    num_shared_experts : int
    shared_intermediate_size : int
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 2048,
        num_experts: int = 64,
        top_k: int = 4,
        num_shared_experts: int = 1,
        shared_intermediate_size: int = 4096,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Qwen3MoERouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            Qwen3MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

        # Shared expert
        self.has_shared = num_shared_experts > 0
        if self.has_shared:
            self.shared_expert = Qwen3MoEExpert(
                hidden_size, shared_intermediate_size
            )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : [B, S, D]

        Returns
        -------
        output : [B, S, D]
        router_logits : [B*S, num_experts]
        """
        bsz, seq_len, dim = hidden_states.shape
        flat = hidden_states.reshape(-1, dim)

        topk_weights, topk_indices, router_logits = self.router(flat)

        output = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = topk_indices[:, k] == e
                if not mask.any():
                    continue
                tokens = flat[mask]
                expert_out = self.experts[e](tokens)
                w = topk_weights[mask, k].unsqueeze(-1)
                output[mask] += w * expert_out

        # Shared expert
        if self.has_shared:
            shared_out = self.shared_expert(flat)
            output = output + shared_out

        return output.reshape(bsz, seq_len, dim), router_logits

    def compute_loss(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary routing loss."""
        return self.router.compute_aux_loss(router_logits, topk_indices)
