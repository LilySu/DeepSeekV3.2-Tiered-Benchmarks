#!/usr/bin/env python3
"""
STUB: fix_dsa_indexer.py -- DeepSeek-V3 does NOT use Dynamic Sparse Attention (DSA).

This file exists for parity with the GLM-5 project structure but redirects to
MoE routing debug utilities instead, since DeepSeek-V3's complexity lives in
its grouped expert routing mechanism rather than sparse attention.

See benchmark/fix_moe_routing.py for the actual implementation of:
  - Expert load imbalance detection
  - Group selection bias analysis
  - Routing collapse detection
  - Auxiliary-loss-free load balancing verification

DeepSeek-V3 routing: n_group=8, topk_group=4, top_k=8, 256 routed experts.
Reference: arXiv:2412.19437, Section 2.1.2

Usage:
    python -m benchmark.fix_dsa_indexer
    # This simply invokes fix_moe_routing.main()
"""

from __future__ import annotations

import sys
import os
from typing import Dict, Any, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG


# ---------------------------------------------------------------------------
# DSA stub -- explain why it does not apply
# ---------------------------------------------------------------------------

DSA_NOT_APPLICABLE_REASON = """
DeepSeek-V3 does NOT use Dynamic Sparse Attention (DSA).

DSA is a technique from GLM-5 / other architectures where attention patterns
are dynamically pruned based on importance scores. DeepSeek-V3 instead uses
standard causal attention with Multi-head Latent Attention (MLA) for KV
compression.

The complexity in DeepSeek-V3 that is analogous to DSA debugging lives in the
MoE routing layer:
  - 256 routed experts partitioned into 8 groups of 32
  - Per-token: select top-4 groups, then top-2 experts per group = 8 experts
  - Auxiliary-loss-free load balancing via bias terms
  - Sigmoid gating (not softmax) for expert weights

Common MoE routing issues this module can diagnose:
  1. Expert load imbalance (some experts overloaded, others idle)
  2. Group selection bias (certain groups always/never selected)
  3. Routing collapse (all tokens routed to same experts)
  4. Auxiliary bias divergence (bias terms growing unbounded)
"""


def explain_no_dsa():
    """Print explanation of why DSA does not apply to DeepSeek-V3."""
    print(DSA_NOT_APPLICABLE_REASON)


# ---------------------------------------------------------------------------
# MoE routing diagnostics (delegated to fix_moe_routing)
# ---------------------------------------------------------------------------

def diagnose_expert_load(
    gate_logits: "torch.Tensor",
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Compute per-expert load distribution from gate logits.

    This is the primary debug function for DeepSeek-V3's grouped routing.
    It replaces the DSA indexer debug from GLM-5.

    Args:
        gate_logits: (n_tokens, n_experts) raw gating logits before sigmoid
        n_group: Number of expert groups (8 for DeepSeek-V3)
        topk_group: Groups selected per token (4 for DeepSeek-V3)
        top_k: Total experts per token (8 for DeepSeek-V3)

    Returns:
        Dictionary with expert loads, group loads, and health indicators.
    """
    from benchmark.fix_moe_routing import diagnose_routing
    return diagnose_routing(gate_logits, n_group=n_group, topk_group=topk_group, top_k=top_k)


def diagnose_group_bias(
    gate_logits: "torch.Tensor",
    n_group: int = 8,
    topk_group: int = 4,
) -> Dict[str, Any]:
    """
    Analyze group selection bias across tokens.

    In DeepSeek-V3, the first stage of routing selects topk_group=4 out of
    n_group=8 groups. If certain groups are consistently favored, this creates
    an imbalance that the per-expert bias cannot fully correct.
    """
    tokens, n_experts = gate_logits.shape
    experts_per_group = n_experts // n_group

    probs = torch.sigmoid(gate_logits)
    group_probs = probs.view(tokens, n_group, experts_per_group)
    group_scores = group_probs.sum(dim=-1)  # (tokens, n_group)

    # Which groups are selected per token
    _, group_idx = group_scores.topk(topk_group, dim=-1)  # (tokens, topk_group)

    # Count group selection frequency
    group_freq = torch.zeros(n_group, device=gate_logits.device)
    for g in range(topk_group):
        for gi in range(n_group):
            group_freq[gi] += (group_idx[:, g] == gi).sum().float()

    expected_freq = tokens * topk_group / n_group
    group_cv = group_freq.std() / group_freq.mean()

    # Pairwise co-selection frequency
    co_selection = torch.zeros(n_group, n_group, device=gate_logits.device)
    for i in range(tokens):
        selected = group_idx[i]
        for a in range(topk_group):
            for b in range(a + 1, topk_group):
                co_selection[selected[a], selected[b]] += 1
                co_selection[selected[b], selected[a]] += 1

    return {
        "group_freq": group_freq.tolist(),
        "expected_freq": expected_freq,
        "group_cv": group_cv.item(),
        "co_selection_matrix": co_selection.tolist(),
        "most_selected_group": group_freq.argmax().item(),
        "least_selected_group": group_freq.argmin().item(),
        "max_min_ratio": (group_freq.max() / group_freq.min().clamp(min=1)).item(),
    }


def diagnose_bias_terms(
    bias: "torch.Tensor",
    n_group: int = 8,
) -> Dict[str, Any]:
    """
    Analyze the auxiliary-loss-free load balancing bias terms.

    DeepSeek-V3 uses additive bias terms on gating logits (updated during
    training) to encourage balanced routing without an auxiliary loss.
    If these biases grow too large, they dominate the gating signal.
    """
    n_experts = bias.shape[0]
    experts_per_group = n_experts // n_group

    group_biases = bias.view(n_group, experts_per_group)
    group_mean = group_biases.mean(dim=1)
    group_std = group_biases.std(dim=1)

    issues = []
    if bias.abs().max() > 10.0:
        issues.append(f"Bias magnitude too large: max={bias.abs().max():.2f} (should be < 10)")
    if bias.std() > 5.0:
        issues.append(f"Bias variance too high: std={bias.std():.2f}")
    if (group_mean.max() - group_mean.min()) > 3.0:
        issues.append(f"Group bias imbalance: range={group_mean.max() - group_mean.min():.2f}")

    return {
        "bias_stats": {
            "mean": bias.mean().item(),
            "std": bias.std().item(),
            "min": bias.min().item(),
            "max": bias.max().item(),
            "abs_mean": bias.abs().mean().item(),
        },
        "group_bias_mean": group_mean.tolist(),
        "group_bias_std": group_std.tolist(),
        "issues": issues,
        "health": "ok" if not issues else "warning",
    }


def print_full_diagnostics(
    gate_logits: "torch.Tensor",
    bias: Optional["torch.Tensor"] = None,
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
) -> None:
    """Run all diagnostics and print a full report."""
    print("=" * 60)
    print("  DeepSeek-V3 MoE Routing Full Diagnostic Report")
    print("  (Replaces DSA indexer debug -- DSA not applicable)")
    print("=" * 60)

    # Expert load
    load_diag = diagnose_expert_load(gate_logits, n_group, topk_group, top_k)
    from benchmark.fix_moe_routing import print_diagnostics
    print_diagnostics(load_diag)

    # Group bias
    print(f"\n  Group Selection Bias Analysis")
    print(f"  {'-'*40}")
    group_diag = diagnose_group_bias(gate_logits, n_group, topk_group)
    print(f"    Group frequencies: {group_diag['group_freq']}")
    print(f"    Expected per group: {group_diag['expected_freq']:.1f}")
    print(f"    Group CV: {group_diag['group_cv']:.3f}")
    print(f"    Most selected: group {group_diag['most_selected_group']}")
    print(f"    Least selected: group {group_diag['least_selected_group']}")
    print(f"    Max/min ratio: {group_diag['max_min_ratio']:.2f}")

    # Bias terms (if provided)
    if bias is not None:
        print(f"\n  Load Balancing Bias Analysis")
        print(f"  {'-'*40}")
        bias_diag = diagnose_bias_terms(bias, n_group)
        stats = bias_diag["bias_stats"]
        print(f"    Mean:     {stats['mean']:.4f}")
        print(f"    Std:      {stats['std']:.4f}")
        print(f"    Range:    [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    Abs mean: {stats['abs_mean']:.4f}")
        print(f"    Health:   {bias_diag['health'].upper()}")
        if bias_diag["issues"]:
            for issue in bias_diag["issues"]:
                print(f"    WARNING: {issue}")


def main():
    """Main entry point -- runs MoE routing debug (not DSA)."""
    explain_no_dsa()

    if not HAS_TORCH:
        print("PyTorch required for routing debug.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEEPSEEK_V3_CONFIG

    print("Running MoE routing diagnostics (replaces DSA indexer debug)...")
    print()

    for n_tokens in [512, 2048]:
        torch.manual_seed(42)
        logits = torch.randn(n_tokens, cfg["n_routed_experts"], device=device)
        bias = torch.randn(cfg["n_routed_experts"], device=device) * 0.5

        print_full_diagnostics(
            logits, bias,
            n_group=cfg["n_group"],
            topk_group=cfg["topk_group"],
            top_k=cfg["num_experts_per_tok"],
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
