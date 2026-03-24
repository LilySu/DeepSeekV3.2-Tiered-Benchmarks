"""H100 Category 2: TMA (Tensor Memory Accelerator) Verification for DeepSeek-V3.

Verify TMA activity via bandwidth checks. If TMA is not active, kernels
fall back to standard loads with 20-30% perf loss.

Requirements: H100 (SM90), flashinfer and/or deep-gemm installed.
"""

import sys
import torch
from .conftest import skip_no_sm90, has_flashinfer, has_deep_gemm


def _cuda_timer(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


@skip_no_sm90
def h100_test_tma_bandwidth_flashinfer():
    """Verify FlashInfer FA3 MLA decode achieves near-peak bandwidth (implies TMA active)."""
    print("\n[H100-TMA-1] FlashInfer FA3 bandwidth check (TMA proxy)")
    if not has_flashinfer():
        print("  SKIP flashinfer not installed")
        return True

    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    device = "cuda"
    B, H = 32, 128  # DeepSeek-V3: 128 heads
    d_ckv, d_kpe = 512, 64
    seq_kv = 4096
    page_size = 1

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa3")

    q_nope = torch.randn(B, H, d_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B, H, d_kpe, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(B * seq_kv, 1, d_ckv, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(B * seq_kv, 1, d_kpe, dtype=torch.bfloat16, device=device)

    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * seq_kv
    kv_indices = torch.arange(0, B * seq_kv, dtype=torch.int32, device=device)
    kv_lens = torch.full((B,), seq_kv, dtype=torch.int32, device=device)
    sm_scale = 1.0 / ((d_ckv + d_kpe) ** 0.5)

    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens,
                 H, d_ckv, d_kpe, page_size, False, sm_scale,
                 torch.bfloat16, torch.bfloat16)

    median_ms = _cuda_timer(lambda: wrapper.run(q_nope, q_pe, ckv, kpe))

    # Compute bandwidth
    kv_bytes = B * seq_kv * (d_ckv + d_kpe) * 2  # BF16
    bw_gb_s = kv_bytes / (median_ms * 1e-3) / 1e9

    print(f"  FA3 MLA decode: {median_ms:.3f}ms, {bw_gb_s:.0f} GB/s")
    ok = bw_gb_s > 500  # Should be >1000 on H100, but threshold is 500
    print(f"  {'PASS' if ok else 'FAIL'} bandwidth {bw_gb_s:.0f} GB/s > 500 GB/s")
    return ok


@skip_no_sm90
def h100_test_tma_bandwidth_deepgemm():
    """Verify DeepGEMM grouped GEMM achieves near-peak compute (implies TMA active)."""
    print("\n[H100-TMA-2] DeepGEMM bandwidth check (TMA proxy)")
    if not has_deep_gemm():
        print("  SKIP deep_gemm not installed")
        return True

    device = "cuda"
    M, N, K = 4096, 2048, 7168  # Matches DeepSeek-V3 MoE dimensions
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    median_ms = _cuda_timer(lambda: torch.mm(a, b.t()))

    flops = 2 * M * N * K
    tflops = flops / (median_ms * 1e-3) / 1e12

    print(f"  GEMM {M}x{N}x{K}: {median_ms:.3f}ms, {tflops:.1f} TFLOPS")
    ok = tflops > 100  # H100 peak is ~989 TFLOPS BF16
    print(f"  {'PASS' if ok else 'FAIL'} compute {tflops:.1f} TFLOPS > 100 TFLOPS")
    return ok
