"""
Reference MoE block implementation.

A complete, numerically-correct MoE layer using only PyTorch operations.
Used as the gold-standard reference for testing optimised kernels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceMoEBlock(nn.Module):
    """Reference MoE block with SwiGLU experts.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension (7168 for DeepSeek-V3).
    intermediate_size : int
        Expert FFN intermediate dimension (2048).
    num_experts : int
        Total number of routed experts (256).
    top_k : int
        Number of experts per token (8).
    n_group : int
        Number of expert groups for routing (8).
    topk_group : int
        Number of groups selected per token (4).
    num_shared_experts : int
        Number of shared (always-active) experts (1).
    shared_intermediate_size : int
        Shared expert intermediate size (2048).
    """

    def __init__(
        self,
        hidden_size: int = 7168,
        intermediate_size: int = 2048,
        num_experts: int = 256,
        top_k: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        num_shared_experts: int = 1,
        shared_intermediate_size: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group

        # Router gate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Routed experts: each has gate_proj, up_proj, down_proj
        self.expert_gate = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.expert_up = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.expert_down = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )

        # Shared expert
        self.has_shared = num_shared_experts > 0
        if self.has_shared:
            self.shared_gate_proj = nn.Linear(
                hidden_size, shared_intermediate_size, bias=False
            )
            self.shared_up_proj = nn.Linear(
                hidden_size, shared_intermediate_size, bias=False
            )
            self.shared_down_proj = nn.Linear(
                shared_intermediate_size, hidden_size, bias=False
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
        flat = hidden_states.reshape(-1, dim)  # [T, D]
        T = flat.shape[0]

        # Routing
        logits = self.gate(flat)  # [T, E]
        scores = torch.sigmoid(logits)

        # Group-restricted top-k
        topk_weights, topk_indices = self._group_topk(scores)

        # Normalise weights
        denom = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        topk_weights = topk_weights / denom

        # Routed expert computation
        output = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = topk_indices[:, k] == e
                if not mask.any():
                    continue
                tokens = flat[mask]  # [n, D]
                gate_out = tokens @ self.expert_gate[e].T  # [n, I]
                up_out = tokens @ self.expert_up[e].T  # [n, I]
                expert_out = F.silu(gate_out) * up_out  # [n, I]
                expert_out = expert_out @ self.expert_down[e].T  # [n, D]
                w = topk_weights[mask, k].unsqueeze(-1)
                output[mask] += w * expert_out

        # Shared expert
        if self.has_shared:
            shared = self.shared_down_proj(
                F.silu(self.shared_gate_proj(flat)) * self.shared_up_proj(flat)
            )
            output = output + shared

        return output.reshape(bsz, seq_len, dim), logits

    def _group_topk(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Group-restricted top-k routing."""
        T = scores.shape[0]
        experts_per_group = self.num_experts // self.n_group

        grouped = scores.view(T, self.n_group, experts_per_group)
        group_scores = grouped.sum(dim=-1)

        _, top_groups = torch.topk(group_scores, k=self.topk_group, dim=-1)

        group_mask = torch.zeros(T, self.n_group, device=scores.device, dtype=scores.dtype)
        group_mask.scatter_(1, top_groups, 1.0)
        expert_mask = group_mask.unsqueeze(-1).expand(T, self.n_group, experts_per_group)
        expert_mask = expert_mask.reshape(T, self.num_experts)

        masked_scores = scores * expert_mask
        topk_weights, topk_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return topk_weights, topk_indices
