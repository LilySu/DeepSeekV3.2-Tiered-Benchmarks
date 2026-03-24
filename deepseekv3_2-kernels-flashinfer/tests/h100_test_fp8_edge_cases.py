"""H100 Category 4: FP8 Numeric Edge Cases for DeepSeek-V3.

E4M3 max=448, min_subnormal~0.001953. DeepSeek-V3 uses 128x128 block-wise
FP8 quantization for activations and KV cache. Tests exercise pathological
input distributions that could cause silent overflow or precision loss.

Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 3.3 -- FP8 Training
"""

import sys
import torch
from .conftest import assert_close, skip_no_cuda


@skip_no_cuda
def h100_test_fp8_overflow_detection():
    """Values > 448 should be handled by per-block scaling, not silently overflow."""
    print("\n[H100-FP8-1] Overflow detection (outliers > 448)")
    from importlib import import_module
    fp8 = import_module("deepseekv3_2-kernels-flashinfer.fp8_utils")

    device = "cuda"
    x = torch.randn(4, 512, device=device) * 0.1
    x[0, 0] = 1000.0
    x[1, 100] = -500.0
    x[2, 255] = 448.1

    x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
    x_rt = fp8.dequantize_fp8(x_fp8, scales, block_size=128)

    ok = True
    if not torch.isfinite(x_rt).all():
        print("  FAIL non-finite values after FP8 roundtrip with outliers")
        ok = False

    outlier_err = abs(x_rt[0, 0].item() - 1000.0) / 1000.0
    if outlier_err > 0.1:
        print(f"  FAIL outlier error {outlier_err:.3f} > 0.10 for x=1000")
        ok = False
    else:
        print(f"  PASS outlier x=1000 roundtrip error={outlier_err:.3f}")

    neg_err = abs(x_rt[1, 100].item() - (-500.0)) / 500.0
    if neg_err > 0.1:
        print(f"  FAIL negative outlier error {neg_err:.3f}")
        ok = False
    else:
        print(f"  PASS negative outlier x=-500 roundtrip error={neg_err:.3f}")

    return ok


@skip_no_cuda
def h100_test_fp8_zero_handling():
    """All-zero blocks should not produce NaN/Inf scales."""
    print("\n[H100-FP8-2] Zero block handling")
    from importlib import import_module
    fp8 = import_module("deepseekv3_2-kernels-flashinfer.fp8_utils")

    device = "cuda"
    x = torch.zeros(2, 256, device=device)
    x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
    x_rt = fp8.dequantize_fp8(x_fp8, scales, block_size=128)

    ok = torch.isfinite(scales).all() and torch.isfinite(x_rt).all()
    ok = ok and x_rt.abs().max() < 0.01
    print(f"  {'PASS' if ok else 'FAIL'} zero blocks produce finite scales")
    return ok


@skip_no_cuda
def h100_test_fp8_subnormal_precision():
    """Very small values near FP8 subnormal range should not become exactly zero."""
    print("\n[H100-FP8-3] Subnormal precision")
    from importlib import import_module
    fp8 = import_module("deepseekv3_2-kernels-flashinfer.fp8_utils")

    device = "cuda"
    x = torch.full((2, 128), 0.01, device=device)  # Small but non-zero
    x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
    x_rt = fp8.dequantize_fp8(x_fp8, scales, block_size=128)

    ok = x_rt.abs().mean().item() > 0.001  # Should preserve small values
    print(f"  {'PASS' if ok else 'FAIL'} subnormal mean={x_rt.abs().mean().item():.6f}")
    return ok


@skip_no_cuda
def h100_test_fp8_flashinfer_kv_scale_correctness():
    """FlashInfer KV cache FP8 scale must be consistent across quantize/dequantize."""
    print("\n[H100-FP8-4] FlashInfer KV scale correctness")
    from importlib import import_module
    fp8 = import_module("deepseekv3_2-kernels-flashinfer.fp8_utils")

    device = "cuda"
    torch.manual_seed(42)
    ckv = torch.randn(4, 64, 512, dtype=torch.bfloat16, device=device) * 2.0
    kpe = torch.randn(4, 64, 64, dtype=torch.bfloat16, device=device) * 2.0

    kv_fp8, scale = fp8.quantize_kv_flashinfer(ckv, kpe)
    ckv_rt, kpe_rt = fp8.dequantize_kv_flashinfer(kv_fp8, scale, head_dim_ckv=512)

    metrics = fp8.compute_fp8_error(ckv.float(), ckv_rt.float())
    ok = metrics["cosine_sim"] > 0.99
    print(f"  {'PASS' if ok else 'FAIL'} KV roundtrip cos_sim={metrics['cosine_sim']:.6f}")
    return ok
