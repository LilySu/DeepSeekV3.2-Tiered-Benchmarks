"""
MoE Router for DeepSeek-V3.

DeepSeek-V3 uses a **sigmoid** router with group-restricted top-k selection:

  1. Compute sigmoid scores for all 256 experts.
  2. Partition experts into ``n_group=8`` groups of 32 experts each.
  3. Select ``topk_group=4`` groups based on highest cumulative score.
  4. Within selected groups, pick ``top_k=8`` experts overall.
  5. Normalise selected weights (``norm_topk_prob=True``).

This avoids the auxiliary load-balancing loss entirely (``noaux_tc``),
relying instead on the group-topk structure for balanced routing.

Reference: arXiv 2412.19437, Section 3.3.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepSeekV3Config, DEEPSEEK_V3_CONFIG


class MoERouter(nn.Module):
    """Sigmoid router with group-restricted top-k for DeepSeek-V3.

    Parameters
    ----------
    config : DeepSeekV3Config
        Model configuration.
    """

    def __init__(self, config: DeepSeekV3Config = DEEPSEEK_V3_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 7168
        self.num_experts = config.moe.num_experts  # 256
        self.top_k = config.moe.num_experts_per_tok  # 8
        self.n_group = config.moe.n_group  # 8
        self.topk_group = config.moe.topk_group  # 4
        self.norm_topk_prob = config.moe.norm_topk_prob  # True
        self.scoring_func = config.moe.scoring_func  # "sigmoid"

        assert self.num_experts % self.n_group == 0, (
            f"num_experts ({self.num_experts}) must be divisible by "
            f"n_group ({self.n_group})"
        )
        self.experts_per_group = self.num_experts // self.n_group  # 32

        # Gate projection: hidden_size -> num_experts
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Parameters
        ----------
        hidden_states : Tensor [batch * seq_len, hidden_size]
            Flattened token representations.

        Returns
        -------
        topk_weights : Tensor [num_tokens, top_k]
            Normalised routing weights for selected experts.
        topk_indices : Tensor [num_tokens, top_k]  int64
            Expert indices for each token.
        router_logits : Tensor [num_tokens, num_experts]
            Raw logits (for monitoring / debugging).
        """
        # (1) Compute scores
        logits = self.gate(hidden_states)  # [T, E]
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)  # [T, E]
        elif self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            raise ValueError(f"Unknown scoring_func: {self.scoring_func}")

        # (2) Group-restricted top-k selection
        topk_weights, topk_indices = self._group_topk(scores)

        # (3) Normalise
        if self.norm_topk_prob:
            denom = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            topk_weights = topk_weights / denom

        return topk_weights, topk_indices, logits

    def _group_topk(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts with group-level restriction.

        Steps:
          1. Reshape scores to [T, n_group, experts_per_group].
          2. Sum scores per group -> [T, n_group].
          3. Select topk_group groups per token.
          4. Mask out experts in non-selected groups.
          5. Global top-k from remaining experts.

        Parameters
        ----------
        scores : [T, num_experts]

        Returns
        -------
        topk_weights : [T, top_k]
        topk_indices : [T, top_k]
        """
        T = scores.shape[0]

        # Reshape into groups
        grouped = scores.view(T, self.n_group, self.experts_per_group)

        # Group-level scores (sum within each group)
        group_scores = grouped.sum(dim=-1)  # [T, n_group]

        # Select top groups
        _, top_groups = torch.topk(
            group_scores, k=self.topk_group, dim=-1
        )  # [T, topk_group]

        # Build mask for selected groups
        group_mask = torch.zeros(
            T, self.n_group, device=scores.device, dtype=scores.dtype
        )
        group_mask.scatter_(1, top_groups, 1.0)  # [T, n_group]

        # Expand mask to expert level
        expert_mask = group_mask.unsqueeze(-1).expand(
            T, self.n_group, self.experts_per_group
        )
        expert_mask = expert_mask.reshape(T, self.num_experts)  # [T, E]

        # Zero out non-selected groups
        masked_scores = scores * expert_mask

        # Global top-k
        topk_weights, topk_indices = torch.topk(
            masked_scores, k=self.top_k, dim=-1
        )  # [T, top_k]

        return topk_weights, topk_indices

    def compute_load_balance_stats(
        self, topk_indices: torch.Tensor
    ) -> dict:
        """Compute load-balance statistics (for monitoring, not loss).

        Parameters
        ----------
        topk_indices : [T, top_k]

        Returns
        -------
        dict with keys:
          - expert_counts: [num_experts] token count per expert
          - max_load: max tokens assigned to any expert
          - min_load: min tokens assigned to any expert
          - load_balance_ratio: max / mean (lower is better, 1.0 = perfect)
        """
        flat = topk_indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_experts).float()
        mean_load = counts.mean()
        max_load = counts.max()
        min_load = counts.min()
        ratio = (max_load / mean_load).item() if mean_load > 0 else float("inf")
        return {
            "expert_counts": counts,
            "max_load": max_load.item(),
            "min_load": min_load.item(),
            "load_balance_ratio": ratio,
        }
