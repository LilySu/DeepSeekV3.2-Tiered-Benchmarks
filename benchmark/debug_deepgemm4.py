#!/usr/bin/env python3
"""
DeepGEMM debug script 4: FP8 accumulation and mixed-precision.

Tests the numerical behavior of FP8 computation with different
accumulation strategies, replicating DeepSeek-V3's approach:
  - FP8 e4m3 inputs with FP32 accumulation
  - Block-wise scaling with 128x128 tiles
  - Mixed-precision output (FP8 compute, BF16 output)

This is critical because the DeepSeek-V3 paper (Section 3.3) notes
that accumulation precision directly affects training stability.
"""

from __future__ import annotations

import sys
import math

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_accumulation_precision():
    """Compare FP32 vs BF16 accumulation for FP8 GEMM."""
    if not HAS_TORCH:
        print("SKIP: PyTorch not available")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Typical MoE expert shape
    M, K, N = 64, 7168, 2048

    A = torch.randn(M, K, dtype=torch.float32, device=device)
    B = torch.randn(K, N, dtype=torch.float32, device=device)

    # FP32 reference
    C_fp32 = torch.mm(A, B)

    # BF16 accumulation
    C_bf16_acc = torch.mm(A.bfloat16(), B.bfloat16()).float()

    # Simulated FP8 with FP32 accumulation
    if hasattr(torch, "float8_e4m3fn"):
        A_fp8 = A.to(torch.float8_e4m3fn)
        B_fp8 = B.to(torch.float8_e4m3fn)
        # Cast back to float32 for accumulation
        C_fp8_fp32acc = torch.mm(A_fp8.float(), B_fp8.float())

        # FP8 with BF16 accumulation (less precise)
        C_fp8_bf16acc = torch.mm(A_fp8.bfloat16(), B_fp8.bfloat16()).float()

        err_bf16 = (C_fp32 - C_bf16_acc).abs().mean().item()
        err_fp8_fp32 = (C_fp32 - C_fp8_fp32acc).abs().mean().item()
        err_fp8_bf16 = (C_fp32 - C_fp8_bf16acc).abs().mean().item()

        print(f"\n  Accumulation precision ({M}x{K}x{N}):")
        print(f"    BF16 compute, BF16 acc: err={err_bf16:.6f}")
        print(f"    FP8 compute, FP32 acc:  err={err_fp8_fp32:.6f}")
        print(f"    FP8 compute, BF16 acc:  err={err_fp8_bf16:.6f}")
        print(f"    Ratio (FP8_FP32 / BF16): {err_fp8_fp32 / err_bf16:.2f}x")
    else:
        err_bf16 = (C_fp32 - C_bf16_acc).abs().mean().item()
        print(f"\n  BF16 acc error: {err_bf16:.6f}")
        print(f"  FP8 not available in this PyTorch build")


def test_block_accumulation():
    """Test block-wise accumulation strategy from DeepSeek-V3."""
    if not HAS_TORCH:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    M, K, N = 256, 7168, 2048
    block_k = 128  # accumulate in blocks of 128 along K dimension

    A = torch.randn(M, K, dtype=torch.float32, device=device)
    B = torch.randn(K, N, dtype=torch.float32, device=device)

    # Reference
    C_ref = torch.mm(A, B)

    # Block accumulation in FP32
    C_block_fp32 = torch.zeros(M, N, dtype=torch.float32, device=device)
    for k_start in range(0, K, block_k):
        k_end = min(k_start + block_k, K)
        A_block = A[:, k_start:k_end]
        B_block = B[k_start:k_end, :]
        C_block_fp32 += torch.mm(A_block, B_block)

    err_block = (C_ref - C_block_fp32).abs().max().item()
    print(f"\n  Block accumulation (block_k={block_k}):")
    print(f"    Max error vs single GEMM: {err_block:.10f}")
    print(f"    (Should be ~0 since both are FP32)")

    # Block accumulation with BF16 blocks, FP32 accumulator
    C_block_mixed = torch.zeros(M, N, dtype=torch.float32, device=device)
    for k_start in range(0, K, block_k):
        k_end = min(k_start + block_k, K)
        A_block = A[:, k_start:k_end].bfloat16()
        B_block = B[k_start:k_end, :].bfloat16()
        # Compute in BF16, accumulate in FP32
        partial = torch.mm(A_block, B_block).float()
        C_block_mixed += partial

    err_mixed = (C_ref - C_block_mixed).abs().mean().item()
    print(f"    BF16 blocks + FP32 acc: mean_err={err_mixed:.6f}")

    # Compare with pure BF16
    C_pure_bf16 = torch.mm(A.bfloat16(), B.bfloat16()).float()
    err_pure = (C_ref - C_pure_bf16).abs().mean().item()
    print(f"    Pure BF16:              mean_err={err_pure:.6f}")
    print(f"    Improvement from block acc: {err_pure / err_mixed:.2f}x")


def test_online_scaling():
    """Test online (dynamic) scaling for FP8 quantization."""
    if not HAS_TORCH:
        return
    if not hasattr(torch, "float8_e4m3fn"):
        print("\n  FP8 not available, skipping online scaling test")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp8_max = 448.0  # e4m3 max

    # Simulate activation distributions with varying magnitude
    print(f"\n  Online scaling test:")
    for scale_factor in [0.01, 1.0, 100.0, 10000.0]:
        x = torch.randn(256, 7168, dtype=torch.float32, device=device) * scale_factor

        # Compute per-tensor scale
        amax = x.abs().max()
        scale = amax / fp8_max

        # Quantize
        x_scaled = x / scale
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        x_dequant = x_fp8.float() * scale

        rel_err = (x - x_dequant).abs().mean() / x.abs().mean()
        print(f"    scale_factor={scale_factor:>8.1f}: amax={amax.item():.4f}, "
              f"quant_scale={scale.item():.6f}, rel_err={rel_err.item():.6f}")


def main():
    print("=" * 60)
    print("  DeepGEMM Debug 4: FP8 Accumulation & Mixed Precision")
    print("=" * 60)

    test_accumulation_precision()
    test_block_accumulation()
    test_online_scaling()

    print("\n  Done.")


if __name__ == "__main__":
    main()
