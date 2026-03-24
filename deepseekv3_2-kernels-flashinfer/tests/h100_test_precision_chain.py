"""H100 Category 9: Precision Boundary Stress for DeepSeek-V3.

DeepSeek-V3 has ~244 FP8<->BF16 crossings per forward pass (4 per layer x 61 layers).
Chain multiple roundtrips to measure accumulated drift.

Requirements: CUDA GPU (any).
Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 3.3 -- FP8 Training
"""

import sys
import torch
from .conftest import assert_close, cosine_sim, skip_no_cuda


@skip_no_cuda
def h100_test_precision_chain_roundtrips():
    """Chain 61 FP8 roundtrips (simulating one per layer) and measure accumulated error."""
    print("\n[H100-Prec-1] Chained FP8 roundtrips (61 iterations)")
    from importlib import import_module
    fp8 = import_module("deepseekv3_2-kernels-flashinfer.fp8_utils")

    device = "cuda"
    torch.manual_seed(42)
    x_orig = torch.randn(4, 512, device=device, dtype=torch.float32)
    x = x_orig.clone()

    num_layers = 61  # DeepSeek-V3 layer count
    for i in range(num_layers):
        x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
        x = fp8.dequantize_fp8(x_fp8, scales, block_size=128).float()

    abs_err = (x - x_orig).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    cos = cosine_sim(x, x_orig)

    mask = x_orig.abs() > 0.1
    rel_err = (abs_err[mask] / x_orig[mask].abs()).max().item() if mask.any() else 0

    print(f"  After {num_layers} roundtrips:")
    print(f"    max_abs_err:  {max_abs:.4f}")
    print(f"    mean_abs_err: {mean_abs:.6f}")
    print(f"    max_rel_err:  {rel_err:.4f}")
    print(f"    cosine_sim:   {cos:.6f}")

    ok = cos > 0.90
    print(f"  {'PASS' if ok else 'FAIL'} cosine_sim {cos:.4f} > 0.90 threshold")
    return ok


@skip_no_cuda
def h100_test_precision_full_pipeline():
    """Full pipeline: quantize activations, do GEMM in FP8, dequantize, measure drift."""
    print("\n[H100-Prec-2] Full FP8 pipeline precision")
    from importlib import import_module
    fp8 = import_module("deepseekv3_2-kernels-flashinfer.fp8_utils")

    device = "cuda"
    torch.manual_seed(42)

    # Simulate DeepSeek-V3 MoE expert computation in FP8
    hidden = torch.randn(32, 7168, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2048, 7168, device=device, dtype=torch.bfloat16)

    # Reference: BF16 GEMM
    ref_out = hidden.float() @ weight.float().t()

    # FP8 path: quantize input, dequantize, then GEMM
    h_fp8, h_scales = fp8.quantize_activations_deepgemm(hidden.float(), block_size=128)
    h_deq = fp8.dequantize_fp8(h_fp8, h_scales, block_size=128)
    fp8_out = h_deq.float() @ weight.float().t()

    cos = cosine_sim(ref_out, fp8_out)
    max_rel = ((ref_out - fp8_out).abs() / (ref_out.abs() + 1e-6)).max().item()

    ok = cos > 0.99
    print(f"  FP8 pipeline: cos_sim={cos:.6f}, max_rel_err={max_rel:.4f}")
    print(f"  {'PASS' if ok else 'FAIL'} cosine_sim > 0.99")
    return ok
