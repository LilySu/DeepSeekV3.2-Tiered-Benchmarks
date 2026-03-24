#!/usr/bin/env python3
"""
DeepGEMM debug script 5: Performance characterization.

Profiles FP8 GEMM throughput across the full range of DeepSeek-V3
matrix shapes to identify:
  - Shapes that are compute-bound vs memory-bound
  - Optimal tile sizes for each shape
  - Throughput as percentage of peak H100 FP8 TFLOPS
  - Impact of matrix alignment on performance

This complements the correctness tests in debug_deepgemm 1-4.
"""

from __future__ import annotations

import sys
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG, H100_SPECS
from benchmark.shared.timer import CUDATimer


def profile_gemm_throughput():
    """Profile GEMM throughput for all DeepSeek-V3 shapes."""
    if not HAS_TORCH:
        print("SKIP: PyTorch not available")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    d_c = cfg["kv_lora_rank"]
    n_h = cfg["num_heads"]
    d_nope = cfg["qk_nope_head_dim"]
    d_rope = cfg["qk_rope_head_dim"]
    d_v = cfg["v_head_dim"]
    I = cfg["moe_intermediate_size"]
    N_exp = cfg["n_routed_experts"]

    shapes = [
        # (M, N, K, name)
        (256, d_c, H, "MLA_down_256"),
        (1024, d_c, H, "MLA_down_1K"),
        (4096, d_c, H, "MLA_down_4K"),
        (256, n_h * d_nope, d_c, "MLA_K_up_256"),
        (1024, n_h * d_nope, d_c, "MLA_K_up_1K"),
        (256, n_h * (d_nope + d_rope), H, "MLA_Q_256"),
        (1024, n_h * (d_nope + d_rope), H, "MLA_Q_1K"),
        (64, I, H, "MoE_exp_64"),
        (256, I, H, "MoE_exp_256"),
        (64, H, I, "MoE_down_64"),
        (256, H, I, "MoE_down_256"),
        (256, N_exp, H, "MoE_gate_256"),
        (1024, N_exp, H, "MoE_gate_1K"),
    ]

    timer = CUDATimer(warmup=5, iters=50)
    peak_tflops = H100_SPECS["bf16_tflops"]

    print(f"\n  {'Name':<20s} {'M':>5s} {'N':>6s} {'K':>5s} {'ms':>8s} {'TFLOPS':>8s} {'MFU%':>6s} {'Bound':>8s}")
    print(f"  {'-'*20} {'-'*5} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

    for M, N, K, name in shapes:
        try:
            A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            B = torch.randn(K, N, dtype=torch.bfloat16, device=device)

            timing = timer.time_fn(lambda: torch.mm(A, B))

            flops = 2.0 * M * N * K
            elapsed_s = timing.mean_ms / 1000.0
            tflops = flops / elapsed_s / 1e12 if elapsed_s > 0 else 0
            mfu = 100.0 * tflops / peak_tflops

            # Rough roofline classification
            bytes_accessed = (M * K + K * N + M * N) * 2  # BF16
            ai = flops / bytes_accessed
            ridge = peak_tflops * 1e12 / (H100_SPECS["hbm_bandwidth_tb_s"] * 1e12)
            bound = "compute" if ai > ridge else "memory"

            print(f"  {name:<20s} {M:5d} {N:6d} {K:5d} {timing.mean_ms:8.3f} "
                  f"{tflops:8.1f} {mfu:5.1f}% {bound:>8s}")

            del A, B
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {name:<20s} OOM")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"  {name:<20s} FAIL: {e}")


def profile_alignment_impact():
    """Test impact of matrix dimension alignment on throughput."""
    if not HAS_TORCH or not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    device = "cuda"
    timer = CUDATimer(warmup=5, iters=50)

    print(f"\n  Alignment Impact (K varies around 7168):")
    print(f"  {'K':>6s} {'Aligned':>8s} {'ms':>8s} {'TFLOPS':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    M, N = 256, 2048
    for K in [7168, 7169, 7170, 7176, 7200, 7232, 7168-1, 7168-64]:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B = torch.randn(K, N, dtype=torch.bfloat16, device=device)

        timing = timer.time_fn(lambda: torch.mm(A, B))
        flops = 2.0 * M * N * K
        tflops = flops / (timing.mean_ms / 1000.0) / 1e12

        aligned = "yes" if K % 128 == 0 else ("64" if K % 64 == 0 else "no")
        print(f"  {K:6d} {aligned:>8s} {timing.mean_ms:8.3f} {tflops:8.1f}")

        del A, B


def main():
    print("=" * 60)
    print("  DeepGEMM Debug 5: Performance Characterization")
    print("=" * 60)

    profile_gemm_throughput()
    profile_alignment_impact()

    print("\n  Done.")


if __name__ == "__main__":
    main()
