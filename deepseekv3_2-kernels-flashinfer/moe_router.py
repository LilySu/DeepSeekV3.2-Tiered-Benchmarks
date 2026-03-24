# MoE sigmoid routing for DeepSeek-V3 -- pure PyTorch.
#
# DeepSeek-V3 uses hierarchical group-based routing:
#   n_group=8, topk_group=4, topk_method="noaux_tc"
#
# Routing algorithm (arXiv 2412.19437, Section 2.2):
#   1. Compute sigmoid scores for all 256 experts
#   2. Add learned correction bias (e_score_correction_bias)
#   3. Group experts into 8 groups of 32 experts each
#   4. Score each group by sum of top-2 expert scores within it
#   5. Select top-4 groups
#   6. Within selected groups, apply flat top-8 selection
#   7. Normalize weights from ORIGINAL sigmoid scores (not bias-adjusted)
#   8. Scale by routed_scaling_factor=2.5
#
# The "noaux_tc" method means:
#   - No auxiliary load-balancing loss (saves 0.5-1% training FLOPS)
#   - Top-k with correction bias for implicit load balancing
#   - The bias is learned end-to-end as part of the routing parameters
#
# Key differences from GLM-5:
#   - GLM-5: n_group=1, topk_group=1 (flat routing, no group selection)
#   - DeepSeek-V3: n_group=8, topk_group=4 (hierarchical group routing)
#   - The group routing is the key innovation for 256-expert scalability

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopkRouter(nn.Module):
    """MoE routing layer: linear projection -> per-expert scores.

    Uses sigmoid activation (not softmax) -- a key DeepSeek-V3 design choice.
    Includes learned correction bias for implicit load balancing.
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.n_routed_experts = cfg["n_routed_experts"]
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.hidden_size))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )

    def forward(self, x):
        x = x.view(-1, self.hidden_size)
        return F.linear(x.float(), self.weight.float())


def sigmoid_topk_route(
    router_logits: torch.Tensor,     # [num_tokens, n_routed_experts]
    correction_bias: torch.Tensor,   # [n_routed_experts]
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid-based expert routing with hierarchical group selection.

    DeepSeek-V3 routing algorithm:
        1. sigmoid(logits) -> scores in [0, 1]
        2. scores + correction_bias -> biased scores for selection
        3. Group experts: 256 experts / 8 groups = 32 per group
        4. Score groups: sum of top-2 expert scores within each group
        5. Select top-4 groups
        6. Mask out experts in non-selected groups
        7. Select top-8 experts from remaining
        8. Normalize using ORIGINAL (unbiased) sigmoid scores
        9. Scale by routed_scaling_factor

    Args:
        router_logits:  [N, E] raw router output (E=256)
        correction_bias: [E] additive bias for load balancing
        top_k: number of experts to select per token (8)
        n_group: number of expert groups (8)
        topk_group: number of groups to select (4)
        norm_topk_prob: whether to normalize routing weights
        routed_scaling_factor: scale factor applied after normalization (2.5)

    Returns:
        topk_indices: [N, top_k] selected expert indices
        topk_weights: [N, top_k] routing weights (normalized and scaled)
    """
    scores = router_logits.sigmoid()
    scores_biased = scores + correction_bias.unsqueeze(0)

    if n_group > 1 and topk_group < n_group:
        # Hierarchical group-based routing (DeepSeek-V3 core routing algorithm)
        n_experts = scores_biased.shape[-1]
        experts_per_group = n_experts // n_group

        # Score each group by top-2 experts within it
        group_scores = (
            scores_biased.view(-1, n_group, experts_per_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        # Select top groups
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        # Expand to per-expert mask
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, n_group, experts_per_group)
            .reshape(-1, n_experts)
        )
        scores_biased = scores_biased.masked_fill(~score_mask.bool(), 0.0)

    # Select top-k experts
    topk_indices = torch.topk(scores_biased, k=top_k, dim=-1, sorted=False)[1]

    # Get weights from ORIGINAL scores (not bias-adjusted)
    topk_weights = scores.gather(1, topk_indices)

    # Normalize
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

    # Scale
    topk_weights = topk_weights * routed_scaling_factor

    return topk_indices, topk_weights
