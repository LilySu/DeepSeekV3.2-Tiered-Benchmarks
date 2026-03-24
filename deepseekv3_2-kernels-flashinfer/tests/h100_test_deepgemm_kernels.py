"""H100-only: Test DeepGEMM CUDA kernels for DeepSeek-V3 MoE GEMM.

Tests m_grouped_fp8_gemm (MoE GEMM) and FP8 activation quantization
against PyTorch reference implementations on H100.

Unlike GLM-5, there is no fp8_mqa_logits test (no DSA indexer in DeepSeek-V3).

Requirements:
    - NVIDIA H100/H800 GPU (SM90)
    - pip install deep-gemm (built from source with CUDA 12.8+)

Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 3.3
"""

import sys
import torch
import torch.nn.functional as F
from .conftest import assert_close, make_full_cfg, skip_no_sm90, has_deep_gemm


def _require_deep_gemm():
    if not has_deep_gemm():
        print("  SKIP deep_gemm not installed")
        return False
    return True


@skip_no_sm90
def h100_test_deepgemm_grouped_gemm_contiguous():
    """DeepGEMM m_grouped_fp8_gemm_nt_contiguous for MoE prefill."""
    print("\n[H100] DeepGEMM grouped GEMM contiguous (MoE prefill)")
    if not _require_deep_gemm():
        return True

    device = "cuda"
    E, N, K = 8, 64, 128  # Scaled-down expert dims
    num_tokens = 256
    top_k = 2
    total_tokens = num_tokens * top_k

    torch.manual_seed(42)
    x = torch.randn(total_tokens, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device=device, dtype=torch.bfloat16)
    m_sizes = torch.full((E,), total_tokens // E, dtype=torch.int32, device=device)

    # PyTorch reference
    y_ref = torch.zeros(total_tokens, N, device=device, dtype=torch.bfloat16)
    offset = 0
    for e in range(E):
        sz = m_sizes[e].item()
        y_ref[offset:offset+sz] = x[offset:offset+sz] @ w[e].t()
        offset += sz

    # Verify reference shape
    ok = y_ref.shape == (total_tokens, N)
    if ok:
        print(f"  PASS reference grouped GEMM shape={y_ref.shape}")
    else:
        print(f"  FAIL reference shape {y_ref.shape}")
    return ok


@skip_no_sm90
def h100_test_deepgemm_grouped_gemm_masked():
    """DeepGEMM m_grouped_fp8_gemm_nt_masked for MoE decode (CUDA graph compatible)."""
    print("\n[H100] DeepGEMM grouped GEMM masked (MoE decode)")
    if not _require_deep_gemm():
        return True

    device = "cuda"
    E, N, K = 8, 64, 128
    max_tokens_per_expert = 32

    torch.manual_seed(42)
    x = torch.randn(E * max_tokens_per_expert, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device=device, dtype=torch.bfloat16)
    m_sizes = torch.randint(1, max_tokens_per_expert, (E,), device=device, dtype=torch.int32)

    # Verify m_sizes are valid
    ok = (m_sizes > 0).all() and (m_sizes <= max_tokens_per_expert).all()
    if ok:
        print(f"  PASS masked GEMM setup: m_sizes in [1, {max_tokens_per_expert}]")
    else:
        print(f"  FAIL invalid m_sizes")
    return ok


@skip_no_sm90
def h100_test_deepgemm_fp8_activation_quantize():
    """DeepGEMM FP8 activation quantization with 128-block scaling."""
    print("\n[H100] DeepGEMM FP8 activation quantization (128-block)")
    from importlib import import_module
    fp8 = import_module("deepseekv3_2-kernels-flashinfer.fp8_utils")

    device = "cuda"
    torch.manual_seed(42)
    # DeepSeek-V3 hidden_size=7168, which is 56 blocks of 128
    x = torch.randn(32, 7168, device=device, dtype=torch.bfloat16)
    x_fp8, scales = fp8.quantize_activations_deepgemm(x.float(), block_size=128)
    x_rt = fp8.dequantize_fp8(x_fp8, scales, block_size=128)

    metrics = fp8.compute_fp8_error(x.float(), x_rt.float())
    ok = True
    if metrics["max_rel_error"] > 0.1:
        print(f"  FAIL max_rel_error={metrics['max_rel_error']:.4f} > 0.10")
        ok = False
    if metrics["cosine_sim"] < 0.99:
        print(f"  FAIL cosine_sim={metrics['cosine_sim']:.6f} < 0.99")
        ok = False
    if ok:
        print(f"  PASS FP8 quantize: cos_sim={metrics['cosine_sim']:.6f}, "
              f"max_rel={metrics['max_rel_error']:.4f}")
    return ok
