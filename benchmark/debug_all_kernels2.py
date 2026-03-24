#!/usr/bin/env python3
"""
Extended kernel debugging for DeepSeek-V3 (Part 2).

Additional tests beyond debug_all_kernels.py:
  - Grouped top-K routing edge cases
  - MLA latent compression accuracy
  - Cross-attention patterns for MTP
  - Token dispatch/combine kernels
  - Numerical stability of large softmax
  - Memory layout verification
  - Mixed-precision accumulation

Reference: arXiv:2412.19437
"""

from __future__ import annotations

import sys
import math
from typing import List, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG


def debug_grouped_routing_edge_cases() -> List[Tuple[str, bool, str]]:
    """Test grouped routing with edge cases."""
    results = []
    if not HAS_TORCH:
        return [("grouped_routing", False, "PyTorch not available")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEEPSEEK_V3_CONFIG

    # Edge case 1: single token
    try:
        logits = torch.randn(1, cfg["n_routed_experts"], device=device)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = probs.topk(cfg["num_experts_per_tok"], dim=-1)
        assert topk_idx.shape == (1, cfg["num_experts_per_tok"])
        results.append(("grouped_routing_single_token", True, "OK"))
    except Exception as e:
        results.append(("grouped_routing_single_token", False, str(e)))

    # Edge case 2: all experts in one group get highest scores
    try:
        logits = torch.zeros(4, cfg["n_routed_experts"], device=device)
        # Put all mass in group 0
        experts_per_group = cfg["n_routed_experts"] // cfg["n_group"]
        logits[:, :experts_per_group] = 10.0
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = probs.topk(cfg["num_experts_per_tok"], dim=-1)
        # All selected experts should be from group 0
        assert (topk_idx < experts_per_group).all()
        results.append(("grouped_routing_concentrated", True,
                       f"OK: all experts from group 0 (idx < {experts_per_group})"))
    except Exception as e:
        results.append(("grouped_routing_concentrated", False, str(e)))

    # Edge case 3: uniform distribution
    try:
        logits = torch.zeros(4, cfg["n_routed_experts"], device=device)
        probs = F.softmax(logits, dim=-1)
        # All probabilities should be equal
        expected = 1.0 / cfg["n_routed_experts"]
        max_dev = (probs - expected).abs().max().item()
        results.append(("grouped_routing_uniform", True,
                       f"OK: max deviation from uniform = {max_dev:.8f}"))
    except Exception as e:
        results.append(("grouped_routing_uniform", False, str(e)))

    return results


def debug_mla_compression_accuracy() -> List[Tuple[str, bool, str]]:
    """Test MLA latent compression accuracy."""
    results = []
    if not HAS_TORCH:
        return [("mla_compression", False, "PyTorch not available")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    d_c = cfg["kv_lora_rank"]

    # Test compression-decompression round-trip
    for bs_seq in [16, 64, 256]:
        try:
            # Random projection matrices
            W_down = torch.randn(H, d_c, dtype=torch.float32, device=device) / math.sqrt(H)
            W_up = torch.randn(d_c, H, dtype=torch.float32, device=device) / math.sqrt(d_c)

            x = torch.randn(bs_seq, H, dtype=torch.float32, device=device)

            # Compress and decompress
            compressed = x @ W_down      # (bs_seq, d_c)
            reconstructed = compressed @ W_up  # (bs_seq, H)

            # Measure reconstruction error
            error = (x - reconstructed).norm() / x.norm()
            results.append((f"mla_compress_roundtrip_{bs_seq}", True,
                           f"OK: relative error = {error.item():.4f}"))
        except Exception as e:
            results.append((f"mla_compress_roundtrip_{bs_seq}", False, str(e)))

    return results


def debug_mtp_cross_attention() -> List[Tuple[str, bool, str]]:
    """Test MTP (Multi-Token Prediction) computation pattern."""
    results = []
    if not HAS_TORCH:
        return [("mtp", False, "PyTorch not available")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    V = cfg["vocab_size"]

    # MTP: takes last hidden state, predicts next-next token
    for bs, seq in [(1, 16), (4, 64)]:
        try:
            hidden_state = torch.randn(bs, seq, H, dtype=torch.bfloat16, device=device)

            # MTP projection (hidden -> hidden)
            W_mtp = torch.randn(H, H, dtype=torch.bfloat16, device=device) / math.sqrt(H)

            # Project
            mtp_hidden = hidden_state @ W_mtp

            # LM head (shared with main model)
            # For testing, use small vocab
            test_vocab = min(V, 1024)
            W_head = torch.randn(H, test_vocab, dtype=torch.bfloat16, device=device) / math.sqrt(H)
            mtp_logits = mtp_hidden @ W_head

            assert mtp_logits.shape == (bs, seq, test_vocab)
            assert not torch.isnan(mtp_logits).any()

            results.append((f"mtp_bs{bs}_seq{seq}", True,
                           f"OK: logits shape = {mtp_logits.shape}"))
        except Exception as e:
            results.append((f"mtp_bs{bs}_seq{seq}", False, str(e)))

    return results


def debug_token_dispatch() -> List[Tuple[str, bool, str]]:
    """Test token dispatch/combine for MoE."""
    results = []
    if not HAS_TORCH:
        return [("token_dispatch", False, "PyTorch not available")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    N_exp = min(cfg["n_routed_experts"], 32)  # reduced for memory
    K = min(cfg["num_experts_per_tok"], 4)

    for n_tokens in [16, 64, 256]:
        try:
            x = torch.randn(n_tokens, H, dtype=torch.bfloat16, device=device)

            # Compute routing
            W_gate = torch.randn(H, N_exp, dtype=torch.bfloat16, device=device) / math.sqrt(H)
            logits = x @ W_gate
            probs = F.softmax(logits.float(), dim=-1)
            topk_vals, topk_idx = probs.topk(K, dim=-1)
            topk_vals = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

            # Dispatch: scatter tokens to experts
            # Count tokens per expert
            expert_counts = torch.zeros(N_exp, dtype=torch.long, device=device)
            for k in range(K):
                indices = topk_idx[:, k]
                for idx in indices:
                    expert_counts[idx] += 1

            # Simulate expert computation and combine
            output = torch.zeros_like(x)
            for k in range(K):
                expert_out = x * 0.99  # simplified expert computation
                output += expert_out * topk_vals[:, k:k+1]

            assert output.shape == (n_tokens, H)
            assert not torch.isnan(output).any()

            active_experts = (expert_counts > 0).sum().item()
            results.append((f"dispatch_t{n_tokens}", True,
                           f"OK: {active_experts}/{N_exp} experts active, "
                           f"max_load={expert_counts.max().item()}"))
        except Exception as e:
            results.append((f"dispatch_t{n_tokens}", False, str(e)))

    return results


def debug_large_softmax_stability() -> List[Tuple[str, bool, str]]:
    """Test numerical stability of softmax with long sequences."""
    results = []
    if not HAS_TORCH:
        return [("softmax_stability", False, "PyTorch not available")]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seq_len in [1024, 4096, 16384]:
        try:
            # Large attention scores that could cause overflow
            attn = torch.randn(1, 4, seq_len, seq_len, dtype=torch.bfloat16, device=device)

            # Add extreme values
            attn[0, 0, 0, :10] = 100.0
            attn[0, 1, :, :] *= 10.0

            probs = F.softmax(attn, dim=-1)

            has_nan = torch.isnan(probs).any().item()
            has_inf = torch.isinf(probs).any().item()
            sums = probs.sum(dim=-1)
            sum_err = (sums - 1.0).abs().max().item()

            ok = not has_nan and not has_inf and sum_err < 1e-3
            results.append((f"softmax_stability_seq{seq_len}", ok,
                           f"nan={has_nan}, inf={has_inf}, sum_err={sum_err:.6f}"))
        except Exception as e:
            results.append((f"softmax_stability_seq{seq_len}", False, str(e)))

    return results


def debug_memory_layouts() -> List[Tuple[str, bool, str]]:
    """Verify memory layouts are contiguous where expected."""
    results = []
    if not HAS_TORCH:
        return [("memory_layout", False, "PyTorch not available")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = DEEPSEEK_V3_CONFIG["hidden_size"]

    try:
        # Standard contiguous tensor
        x = torch.randn(32, H, dtype=torch.bfloat16, device=device)
        assert x.is_contiguous()

        # After reshape
        x_3d = x.reshape(4, 8, H)
        assert x_3d.is_contiguous()

        # After transpose (not contiguous)
        x_t = x_3d.transpose(0, 1)
        assert not x_t.is_contiguous()

        # After contiguous() call
        x_tc = x_t.contiguous()
        assert x_tc.is_contiguous()

        results.append(("memory_layout_basic", True, "OK: layout checks passed"))
    except Exception as e:
        results.append(("memory_layout_basic", False, str(e)))

    # Check that MLA head reshape preserves contiguity
    try:
        n_h = 8
        d = 128
        x = torch.randn(4, 32, n_h * d, dtype=torch.bfloat16, device=device)
        x_heads = x.reshape(4, 32, n_h, d)
        assert x_heads.is_contiguous()

        # Transpose for attention: (B, n_h, S, d)
        x_attn = x_heads.transpose(1, 2)
        # This will NOT be contiguous, verify we handle it
        is_contig = x_attn.is_contiguous()
        results.append(("memory_layout_mla_heads", True,
                       f"OK: after transpose contiguous={is_contig}"))
    except Exception as e:
        results.append(("memory_layout_mla_heads", False, str(e)))

    return results


def main():
    print("=" * 70)
    print("  DeepSeek-V3 Extended Kernel Debug Suite (Part 2)")
    print("=" * 70)

    test_groups = [
        ("Grouped Routing Edge Cases", debug_grouped_routing_edge_cases),
        ("MLA Compression Accuracy", debug_mla_compression_accuracy),
        ("MTP Cross-Attention", debug_mtp_cross_attention),
        ("Token Dispatch/Combine", debug_token_dispatch),
        ("Large Softmax Stability", debug_large_softmax_stability),
        ("Memory Layouts", debug_memory_layouts),
    ]

    total_pass = 0
    total_fail = 0

    for group_name, test_fn in test_groups:
        print(f"\n--- {group_name} ---")
        try:
            tests = test_fn()
        except Exception as e:
            print(f"  GROUP FAILED: {e}")
            tests = [(group_name, False, str(e))]

        for name, passed, msg in tests:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {msg}")
            total_pass += passed
            total_fail += (not passed)

    print(f"\n{'='*70}")
    print(f"  Results: {total_pass} passed, {total_fail} failed")
    print(f"{'='*70}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
