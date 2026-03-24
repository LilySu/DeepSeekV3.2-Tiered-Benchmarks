"""
Llama-4 MoE layer reference implementation.

Provided for comparison with DeepSeek-V3's MoE.  Key differences:

  - Llama-4: softmax routing, standard top-k (no group restriction)
  - DeepSeek-V3: sigmoid routing, group-restricted top-k (n_group=8, topk_group=4)
  - Llama-4: auxiliary load-balance loss
  - DeepSeek-V3: no auxiliary loss (noaux_tc)

This reference is useful for:
  - Validating shared grouped GEMM infrastructure
  - Comparing routing strategies
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Llama4MoEExpert(nn.Module):
    """Single Llama-4 SwiGLU expert."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Llama4MoERouter(nn.Module):
    """Llama-4 softmax top-k router."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(hidden_states)
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)
        # Renormalise
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_indices, logits


class Llama4MoELayer(nn.Module):
    """Llama-4 MoE layer (reference implementation).

    Parameters
    ----------
    hidden_size : int
    intermediate_size : int
    num_experts : int
    top_k : int
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_experts: int = 16,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Llama4MoERouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            Llama4MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

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
        T = flat.shape[0]

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

        return output.reshape(bsz, seq_len, dim), router_logits

    def load_balance_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Auxiliary load-balance loss (Llama-4 style)."""
        probs = F.softmax(router_logits, dim=-1)
        avg_probs = probs.mean(dim=0)  # [num_experts]
        # Expert usage frequency
        topk_indices = torch.topk(probs, k=self.top_k, dim=-1).indices
        counts = torch.bincount(
            topk_indices.reshape(-1), minlength=self.num_experts
        ).float()
        freq = counts / (router_logits.shape[0] * self.top_k)
        return self.num_experts * (avg_probs * freq).sum()
