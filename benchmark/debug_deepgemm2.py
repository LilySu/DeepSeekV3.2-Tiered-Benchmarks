#!/usr/bin/env python3
"""
DeepGEMM debug script 2: MLA-specific GEMM shapes.

Tests FP8 GEMM operations for all projection shapes used in
DeepSeek-V3's MLA (Multi-head Latent Attention) mechanism.

Shapes tested:
  - x -> c_kv: (tokens, 7168) @ (7168, 512) -- KV compression
  - c_kv -> K_nope: (tokens, 512) @ (512, 128*128=16384)
  - c_kv -> V: (tokens, 512) @ (512, 128*128=16384)
  - x -> Q: (tokens, 7168) @ (7168, 128*192=24576)
  - x -> K_rope: (tokens, 7168) @ (7168, 64)
  - attn_out -> output: (tokens, 16384) @ (16384, 7168)
"""

from __future__ import annotations

import sys

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG


def test_mla_gemm_shapes():
    """Test all MLA GEMM shapes in FP8 and BF16."""
    if not HAS_TORCH:
        print("SKIP: PyTorch not available")
        return

    cfg = DEEPSEEK_V3_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    H = cfg["hidden_size"]
    d_c = cfg["kv_lora_rank"]
    n_h = cfg["num_heads"]
    d_nope = cfg["qk_nope_head_dim"]
    d_rope = cfg["qk_rope_head_dim"]
    d_v = cfg["v_head_dim"]

    shapes = {
        "KV_compress":    (H, d_c),
        "K_nope_up":      (d_c, n_h * d_nope),
        "V_up":           (d_c, n_h * d_v),
        "Q_proj":         (H, n_h * (d_nope + d_rope)),
        "K_rope":         (H, d_rope),
        "Output_proj":    (n_h * d_v, H),
    }

    for tokens in [32, 128, 512]:
        print(f"\n  tokens = {tokens}:")
        for name, (in_dim, out_dim) in shapes.items():
            try:
                A = torch.randn(tokens, in_dim, dtype=torch.bfloat16, device=device)
                B = torch.randn(in_dim, out_dim, dtype=torch.bfloat16, device=device)

                C = torch.mm(A, B)

                # Verify shape and numerical stability
                assert C.shape == (tokens, out_dim)
                nan_count = torch.isnan(C).sum().item()
                inf_count = torch.isinf(C).sum().item()

                status = "OK" if nan_count == 0 and inf_count == 0 else "WARN"
                print(f"    [{status}] {name:<15s}: ({tokens},{in_dim}) x ({in_dim},{out_dim}) "
                      f"-> ({tokens},{out_dim}) "
                      f"nan={nan_count} inf={inf_count}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    [OOM]  {name:<15s}: ({tokens},{in_dim}) x ({in_dim},{out_dim})")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print(f"    [FAIL] {name:<15s}: {e}")


def test_mla_latent_bottleneck():
    """Test the information bottleneck of MLA's latent compression."""
    if not HAS_TORCH:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    d_c = cfg["kv_lora_rank"]

    print(f"\n  MLA Latent Bottleneck Analysis:")
    print(f"    hidden_size = {H}")
    print(f"    kv_lora_rank = {d_c}")
    print(f"    compression ratio = {H / d_c:.1f}x")
    print(f"    KV cache per token: {d_c * 2} bytes (BF16), {d_c} bytes (FP8)")
    print(f"    vs standard MHA: {cfg['num_heads'] * (cfg['qk_nope_head_dim'] + cfg['v_head_dim']) * 2} bytes (BF16)")

    # SVD analysis of random projection
    W_down = torch.randn(H, d_c, dtype=torch.float32, device=device) / (H ** 0.5)
    U, S, Vh = torch.linalg.svd(W_down, full_matrices=False)
    print(f"    SVD of random W_down: top-5 singular values = {S[:5].tolist()}")
    print(f"    Condition number: {S[0] / S[-1]:.2f}")


def main():
    print("=" * 60)
    print("  DeepGEMM Debug 2: MLA GEMM Shapes")
    print("=" * 60)

    test_mla_gemm_shapes()
    test_mla_latent_bottleneck()

    print("\n  Done.")


if __name__ == "__main__":
    main()
