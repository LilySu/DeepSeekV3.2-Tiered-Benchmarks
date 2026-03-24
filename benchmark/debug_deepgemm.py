#!/usr/bin/env python3
"""
DeepGEMM debug script 1: Basic FP8 GEMM validation.

DeepGEMM is DeepSeek's open-source FP8 GEMM library optimized for
H100 GPUs. This script validates the basic FP8 GEMM operation using
PyTorch's native FP8 support as a reference.

Tests:
  - FP8 e4m3 GEMM correctness
  - Block-wise scaling accuracy
  - Comparison with BF16 reference
  - Small matrix edge cases

Reference:
  - DeepSeek-V3 (arXiv: 2412.19437), Section 3.3
  - DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
"""

from __future__ import annotations

import sys
import math

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_basic_fp8_gemm():
    """Test basic FP8 GEMM against BF16 reference."""
    if not HAS_TORCH:
        print("SKIP: PyTorch not available")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_shapes = [
        (128, 128, 128),
        (256, 512, 256),
        (512, 512, 7168),     # MLA projection shape
        (64, 2048, 7168),     # MoE expert shape
    ]

    for M, N, K in test_shapes:
        print(f"\n  Testing ({M}, {N}, {K}):")

        # BF16 reference
        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device)
        C_ref = torch.mm(A_bf16.float(), B_bf16.float())

        # FP8 simulation (quantize, compute, compare)
        if hasattr(torch, "float8_e4m3fn"):
            A_fp8 = A_bf16.to(torch.float8_e4m3fn)
            B_fp8 = B_bf16.to(torch.float8_e4m3fn)
            C_fp8 = torch.mm(A_fp8.to(torch.float32), B_fp8.to(torch.float32))

            # Error analysis
            abs_err = (C_ref - C_fp8).abs()
            max_err = abs_err.max().item()
            mean_err = abs_err.mean().item()
            rel_err = abs_err.mean().item() / C_ref.abs().mean().item()

            print(f"    FP8 vs FP32: max_err={max_err:.6f}, mean_err={mean_err:.6f}, rel_err={rel_err:.6f}")
        else:
            print(f"    FP8 dtype not available, skipping FP8 test")

        # BF16 error (as baseline)
        C_bf16 = torch.mm(A_bf16, B_bf16)
        bf16_err = (C_ref - C_bf16.float()).abs().mean().item()
        print(f"    BF16 vs FP32: mean_err={bf16_err:.6f}")


def test_block_scaling():
    """Test block-wise scaling for FP8 quantization."""
    if not HAS_TORCH:
        print("SKIP: PyTorch not available")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    M, K = 512, 7168  # MLA down-projection
    block_h, block_w = 128, 128

    x = torch.randn(M, K, dtype=torch.float32, device=device)

    # Per-tensor scaling
    scale_tensor = x.abs().max() / 448.0
    x_quant_tensor = torch.round(x / scale_tensor).clamp(-448, 448) * scale_tensor
    err_tensor = (x - x_quant_tensor).abs().mean().item()

    # Per-block scaling
    nh, nw = M // block_h, K // block_w
    blocks = x.reshape(nh, block_h, nw, block_w)
    block_max = blocks.abs().permute(0, 2, 1, 3).reshape(nh, nw, -1).amax(dim=-1)
    scales = (block_max / 448.0).clamp(min=1e-12)
    scales_exp = scales[:, :, None, None].expand(nh, nw, block_h, block_w).permute(0, 2, 1, 3).reshape(M, K)
    x_quant_block = torch.round(x / scales_exp).clamp(-448, 448) * scales_exp
    err_block = (x - x_quant_block).abs().mean().item()

    print(f"\n  Block scaling test ({M}x{K}, blocks={block_h}x{block_w}):")
    print(f"    Per-tensor error: {err_tensor:.6f}")
    print(f"    Per-block error:  {err_block:.6f}")
    print(f"    Improvement:      {err_tensor / err_block:.2f}x")


def main():
    print("=" * 60)
    print("  DeepGEMM Debug 1: Basic FP8 GEMM Validation")
    print("=" * 60)

    test_basic_fp8_gemm()
    test_block_scaling()

    print("\n  Done.")


if __name__ == "__main__":
    main()
