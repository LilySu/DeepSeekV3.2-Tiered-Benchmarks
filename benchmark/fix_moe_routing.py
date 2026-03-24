#!/usr/bin/env python3
"""
MoE routing debug and fix utility for DeepSeek-V3.

STUB: Replaces fix_dsa_indexer.py from GLM5 -- DeepSeek-V3 does not use DSA.

This module provides utilities for debugging and fixing MoE routing issues
specific to DeepSeek-V3's grouped routing mechanism:

  - Expert load imbalance detection
  - Routing collapse detection (all tokens going to same expert)
  - Group selection bias analysis
  - Auxiliary-loss-free load balancing verification

DeepSeek-V3's routing uses n_group=8 groups with topk_group=4 group selection,
then top-2 experts per selected group for a total of 8 active experts per token.

Reference: arXiv:2412.19437, Section 2.1.2 (DeepSeekMoE Architecture)
"""

from __future__ import annotations

import sys
from typing import Dict, Any, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG


def diagnose_routing(
    gate_logits: "torch.Tensor",
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Diagnose routing quality from gate logits.

    Args:
        gate_logits: (n_tokens, n_experts) gating logits.
        n_group: Number of expert groups.
        topk_group: Number of groups to select.
        top_k: Total experts per token.

    Returns:
        Dictionary of diagnostic metrics.
    """
    tokens, n_experts = gate_logits.shape
    experts_per_group = n_experts // n_group

    probs = gate_logits.sigmoid()  # DeepSeek-V3 uses sigmoid, NOT softmax
    topk_vals, topk_idx = probs.topk(top_k, dim=-1)

    # Expert load counts
    expert_counts = torch.zeros(n_experts, device=gate_logits.device)
    for k in range(top_k):
        expert_counts.scatter_add_(0, topk_idx[:, k], torch.ones(tokens, device=gate_logits.device))

    expected_load = tokens * top_k / n_experts
    cv = expert_counts.float().std() / expert_counts.float().mean()
    max_load = expert_counts.max().item()
    min_load = expert_counts.min().item()
    empty_experts = (expert_counts == 0).sum().item()

    # Group-level analysis
    group_counts = expert_counts.reshape(n_group, experts_per_group).sum(dim=1)
    group_cv = group_counts.float().std() / group_counts.float().mean()

    # Routing entropy (higher = more diverse routing)
    avg_probs = probs.mean(dim=0)
    entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum().item()
    max_entropy = (torch.tensor(n_experts).float().log()).item()
    normalized_entropy = entropy / max_entropy

    # Routing collapse detection
    # If most tokens select the same top-K experts, routing has collapsed
    unique_experts_per_token = []
    for i in range(min(100, tokens)):
        unique_experts_per_token.append(topk_idx[i].unique().numel())
    avg_unique = sum(unique_experts_per_token) / len(unique_experts_per_token)

    diagnostics = {
        "n_tokens": tokens,
        "n_experts": n_experts,
        "n_group": n_group,
        "topk_group": topk_group,
        "top_k": top_k,
        "expert_load": {
            "expected": expected_load,
            "mean": expert_counts.float().mean().item(),
            "std": expert_counts.float().std().item(),
            "cv": cv.item(),
            "max": max_load,
            "min": min_load,
            "max_min_ratio": max_load / max(min_load, 1),
            "empty_experts": empty_experts,
        },
        "group_load": {
            "cv": group_cv.item(),
            "counts": group_counts.tolist(),
        },
        "routing_diversity": {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "avg_unique_experts_per_token": avg_unique,
        },
        "health": "ok",
    }

    # Health checks
    issues = []
    if cv.item() > 1.0:
        issues.append(f"High expert load imbalance (CV={cv.item():.3f} > 1.0)")
    if empty_experts > n_experts * 0.1:
        issues.append(f"Many empty experts ({empty_experts}/{n_experts})")
    if normalized_entropy < 0.5:
        issues.append(f"Low routing entropy ({normalized_entropy:.3f} < 0.5)")
    if max_load > expected_load * 5:
        issues.append(f"Extreme max load ({max_load} vs expected {expected_load:.1f})")
    if avg_unique < top_k * 0.9:
        issues.append(f"Routing near-collapse (avg unique experts = {avg_unique:.1f})")

    if issues:
        diagnostics["health"] = "warning"
        diagnostics["issues"] = issues

    return diagnostics


def fix_load_balance(
    gate_logits: "torch.Tensor",
    expert_counts: "torch.Tensor",
    bias_strength: float = 0.1,
) -> "torch.Tensor":
    """
    Apply auxiliary-loss-free load balancing bias.

    DeepSeek-V3 uses a bias term added to gating logits to encourage
    balanced routing without an auxiliary loss term. This avoids the
    performance degradation seen with auxiliary losses.

    The bias is updated based on observed expert loads:
      - Over-loaded experts get negative bias (reduced selection)
      - Under-loaded experts get positive bias (increased selection)
    """
    n_experts = gate_logits.shape[1]
    expected_load = expert_counts.float().mean()

    # Compute bias: negative for over-loaded, positive for under-loaded
    load_ratio = expert_counts.float() / expected_load.clamp(min=1)
    bias = -bias_strength * (load_ratio - 1.0)

    # Apply bias to logits
    adjusted_logits = gate_logits + bias.unsqueeze(0)

    return adjusted_logits


def print_diagnostics(diag: Dict[str, Any]) -> None:
    """Pretty-print routing diagnostics."""
    print(f"\n  MoE Routing Diagnostics")
    print(f"  {'='*50}")
    print(f"  Tokens: {diag['n_tokens']}, Experts: {diag['n_experts']}")
    print(f"  Groups: {diag['n_group']}, TopK_Group: {diag['topk_group']}, TopK: {diag['top_k']}")

    el = diag["expert_load"]
    print(f"\n  Expert Load:")
    print(f"    Expected: {el['expected']:.1f}")
    print(f"    Mean:     {el['mean']:.1f} (std={el['std']:.1f})")
    print(f"    Range:    [{el['min']:.0f}, {el['max']:.0f}] (ratio={el['max_min_ratio']:.2f})")
    print(f"    CV:       {el['cv']:.3f}")
    print(f"    Empty:    {el['empty_experts']}")

    gl = diag["group_load"]
    print(f"\n  Group Load:")
    print(f"    CV:     {gl['cv']:.3f}")
    print(f"    Counts: {gl['counts']}")

    rd = diag["routing_diversity"]
    print(f"\n  Routing Diversity:")
    print(f"    Entropy:            {rd['entropy']:.4f}")
    print(f"    Normalized:         {rd['normalized_entropy']:.4f}")
    print(f"    Avg unique/token:   {rd['avg_unique_experts_per_token']:.1f}")

    health = diag["health"]
    print(f"\n  Health: {health.upper()}")
    if "issues" in diag:
        for issue in diag["issues"]:
            print(f"    WARNING: {issue}")


def main():
    if not HAS_TORCH:
        print("PyTorch required for MoE routing debug")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEEPSEEK_V3_CONFIG

    print("=" * 60)
    print("  DeepSeek-V3 MoE Routing Debug & Fix")
    print("  (Replaces fix_dsa_indexer.py -- DSA not applicable)")
    print("=" * 60)

    # Test with random logits
    for n_tokens in [512, 2048, 8192]:
        print(f"\n  --- {n_tokens} tokens ---")
        logits = torch.randn(n_tokens, cfg["n_routed_experts"], device=device)

        diag = diagnose_routing(
            logits,
            n_group=cfg["n_group"],
            topk_group=cfg["topk_group"],
            top_k=cfg["num_experts_per_tok"],
        )
        print_diagnostics(diag)

    # Test load balancing fix
    print(f"\n  --- Load Balance Fix Test ---")
    n_tokens = 2048
    logits = torch.randn(n_tokens, cfg["n_routed_experts"], device=device)
    # Create imbalanced scenario: boost first few experts
    logits[:, :8] += 5.0

    diag_before = diagnose_routing(logits, cfg["n_group"], cfg["topk_group"], cfg["num_experts_per_tok"])
    print(f"  Before fix: CV={diag_before['expert_load']['cv']:.3f}")

    # Apply fix
    probs = F.softmax(logits, dim=-1)
    _, topk_idx = probs.topk(cfg["num_experts_per_tok"], dim=-1)
    counts = torch.zeros(cfg["n_routed_experts"], device=device)
    for k in range(cfg["num_experts_per_tok"]):
        counts.scatter_add_(0, topk_idx[:, k], torch.ones(n_tokens, device=device))

    adjusted = fix_load_balance(logits, counts, bias_strength=1.0)
    diag_after = diagnose_routing(adjusted, cfg["n_group"], cfg["topk_group"], cfg["num_experts_per_tok"])
    print(f"  After fix:  CV={diag_after['expert_load']['cv']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
