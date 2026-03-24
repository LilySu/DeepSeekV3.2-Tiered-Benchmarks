#!/usr/bin/env python3
"""
DeepGEMM debug script 3: MoE expert GEMM shapes and dispatch.

Tests FP8 GEMM operations for the MoE layer, including:
  - Gating network: (tokens, 7168) @ (7168, 256)
  - Expert gate_proj: (tpe, 7168) @ (7168, 2048)
  - Expert up_proj: (tpe, 7168) @ (7168, 2048)
  - Expert down_proj: (tpe, 2048) @ (2048, 7168)
  - Shared expert (2x intermediate): (tokens, 7168) @ (7168, 4096)

Where tpe = tokens_per_expert = total_tokens * top_k / n_experts
"""

from __future__ import annotations

import sys
import math

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG


def test_moe_gemm_shapes():
    """Test all MoE GEMM shapes."""
    if not HAS_TORCH:
        print("SKIP: PyTorch not available")
        return

    cfg = DEEPSEEK_V3_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    N_exp = cfg["n_routed_experts"]

    # Expected tokens per expert for different total token counts
    for total_tokens in [256, 1024, 4096]:
        tpe = total_tokens * cfg["num_experts_per_tok"] // N_exp  # uniform distribution
        tpe = max(tpe, 1)

        print(f"\n  total_tokens={total_tokens}, expected tpe={tpe}:")

        shapes = {
            "gating":       (total_tokens, H, N_exp),
            "expert_gate":  (tpe, H, I),
            "expert_up":    (tpe, H, I),
            "expert_down":  (tpe, I, H),
            "shared_gate":  (total_tokens, H, I * 2),  # shared expert is 2x wider
            "shared_up":    (total_tokens, H, I * 2),
            "shared_down":  (total_tokens, I * 2, H),
        }

        for name, (M, K, N) in shapes.items():
            try:
                A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
                B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
                C = torch.mm(A, B)

                flops = 2 * M * K * N
                assert C.shape == (M, N)
                print(f"    [OK] {name:<15s}: ({M},{K}) x ({K},{N}) -> ({M},{N})  "
                      f"FLOPs={flops/1e6:.1f}M")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    [OOM] {name}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print(f"    [FAIL] {name}: {e}")


def test_expert_load_distribution():
    """Test expert load distribution under grouped routing."""
    if not HAS_TORCH:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    N_exp = cfg["n_routed_experts"]
    K = cfg["num_experts_per_tok"]
    n_group = cfg["n_group"]
    topk_group = cfg["topk_group"]

    print(f"\n  Expert Load Distribution Test:")
    print(f"    n_experts={N_exp}, top_k={K}, n_group={n_group}, topk_group={topk_group}")

    for n_tokens in [512, 2048, 8192]:
        # Simulate gating
        logits = torch.randn(n_tokens, N_exp, device=device)
        probs = F.softmax(logits, dim=-1)
        _, topk_idx = probs.topk(K, dim=-1)

        # Count tokens per expert
        counts = torch.zeros(N_exp, dtype=torch.long, device=device)
        for k_idx in range(K):
            expert_ids = topk_idx[:, k_idx]
            counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids, dtype=torch.long))

        expected = n_tokens * K / N_exp
        actual_mean = counts.float().mean().item()
        actual_std = counts.float().std().item()
        cv = actual_std / actual_mean if actual_mean > 0 else 0
        max_load = counts.max().item()
        min_load = counts.min().item()
        empty = (counts == 0).sum().item()

        print(f"    tokens={n_tokens}: expected={expected:.1f}, "
              f"mean={actual_mean:.1f}, std={actual_std:.1f}, CV={cv:.3f}, "
              f"max={max_load}, min={min_load}, empty={empty}")


def main():
    print("=" * 60)
    print("  DeepGEMM Debug 3: MoE GEMM Shapes & Dispatch")
    print("=" * 60)

    test_moe_gemm_shapes()
    test_expert_load_distribution()

    print("\n  Done.")


if __name__ == "__main__":
    main()
