"""H100 Category 8: Sparse Pattern Validation for DeepSeek-V3.

ADAPTED: DeepSeek-V3 does NOT use DSA sparse attention. Instead, this tests
MoE routing sparsity patterns: expert utilization, group balance, and routing
stability across tokens.

With n_group=8, topk_group=4, we expect 4 of 8 groups to be active per token.
With top_k=8 from 256 experts, only 3.1% of experts are used per token.

Requirements: CUDA GPU (any).
Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 2.2
"""

import sys
import torch
from .conftest import make_cfg, skip_no_cuda, jaccard


@skip_no_cuda
def h100_test_moe_expert_utilization():
    """Verify that all experts get used across a batch of tokens."""
    print("\n[H100-Sparse-1] MoE expert utilization")
    from importlib import import_module
    router = import_module("deepseekv3_2-kernels-flashinfer.moe_router")

    device = "cuda"
    torch.manual_seed(42)
    N, E = 1024, 256
    logits = torch.randn(N, E, device=device)
    bias = torch.zeros(E, device=device)

    indices, _ = router.sigmoid_topk_route(logits, bias, top_k=8, n_group=8, topk_group=4)

    used_experts = indices.unique()
    utilization = len(used_experts) / E

    ok = utilization > 0.5  # At least 50% of experts should be used with 1024 tokens
    print(f"  {len(used_experts)}/{E} experts used ({utilization*100:.1f}%)")
    print(f"  {'PASS' if ok else 'FAIL'} utilization > 50%")
    return ok


@skip_no_cuda
def h100_test_moe_group_balance():
    """Verify that group selection is roughly balanced across tokens."""
    print("\n[H100-Sparse-2] MoE group balance")
    from importlib import import_module
    router = import_module("deepseekv3_2-kernels-flashinfer.moe_router")

    device = "cuda"
    torch.manual_seed(42)
    N, E = 2048, 256
    n_group, experts_per_group = 8, 32
    logits = torch.randn(N, E, device=device)
    bias = torch.zeros(E, device=device)

    indices, _ = router.sigmoid_topk_route(logits, bias, top_k=8, n_group=n_group, topk_group=4)

    # Count which groups get selected
    group_counts = torch.zeros(n_group, dtype=torch.long, device=device)
    for g in range(n_group):
        mask = (indices >= g * experts_per_group) & (indices < (g + 1) * experts_per_group)
        group_counts[g] = mask.sum()

    total = group_counts.sum().item()
    expected_per_group = total / n_group
    max_deviation = (group_counts.float() - expected_per_group).abs().max().item() / expected_per_group

    ok = max_deviation < 0.5  # No group should deviate more than 50% from mean
    print(f"  Group counts: {group_counts.tolist()}")
    print(f"  Max deviation from mean: {max_deviation*100:.1f}%")
    print(f"  {'PASS' if ok else 'FAIL'} deviation < 50%")
    return ok


@skip_no_cuda
def h100_test_moe_routing_stability():
    """Similar tokens should route to similar experts (stability check)."""
    print("\n[H100-Sparse-3] MoE routing stability")
    from importlib import import_module
    router = import_module("deepseekv3_2-kernels-flashinfer.moe_router")

    device = "cuda"
    torch.manual_seed(42)
    E = 256
    base = torch.randn(1, E, device=device)
    noise = torch.randn(10, E, device=device) * 0.01  # Small perturbation
    logits = base + noise
    bias = torch.zeros(E, device=device)

    indices, _ = router.sigmoid_topk_route(logits, bias, top_k=8, n_group=8, topk_group=4)

    # All 10 tokens should route to very similar experts
    jaccard_scores = []
    for i in range(1, 10):
        j = jaccard(indices[0], indices[i])
        jaccard_scores.append(j)

    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
    ok = avg_jaccard > 0.5  # Similar inputs should have >50% expert overlap
    print(f"  Avg Jaccard similarity: {avg_jaccard:.3f}")
    print(f"  {'PASS' if ok else 'FAIL'} avg Jaccard > 0.5")
    return ok
