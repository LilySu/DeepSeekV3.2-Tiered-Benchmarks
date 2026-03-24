"""H100 Benchmark & Profiling Harness for DeepSeek-V3 CUDA Kernels.

This harness benchmarks each kernel-accelerated component individually and the
full model end-to-end on H100 GPUs, collecting ncu/nsys profiling metrics.

Architecture-specific notes:
    - No DSA indexer benchmark (DeepSeek-V3 does not use DSA)
    - MoE router benchmarks include group-based selection (n_group=8, topk_group=4)
    - MLA attention uses FlashInfer FA3 backend (qk_nope_head_dim=128 native)
    - YaRN RoPE with factor=40 for 128K context

Usage:
    python3 -m deepseekv3_2-kernels-flashinfer.tests.h100_bench --mode bench
    nsys profile -o dsv3_profile python3 -m deepseekv3_2-kernels-flashinfer.tests.h100_bench --mode nsys

Hardware requirements:
    - 1+ NVIDIA H100/H800 GPUs (SM90)
    - CUDA 12.8+
    - FlashInfer and/or DeepGEMM installed
    - For full 671B model: 8+ H100 80GB (tensor parallel)
    - For single-layer benchmarks: 1 H100 sufficient

Paper ref: DeepSeek-V3 (arXiv 2412.19437)
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

# -- ncu metric groups -------------------------------------------------------

NCU_METRICS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":
        "SM utilization (% of peak). Target: >80% for compute-bound kernels.",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed":
        "HBM bandwidth utilization (% of peak 3.35 TB/s). Target: >70% for memory-bound.",
    "l2__throughput.avg.pct_of_peak_sustained_elapsed":
        "L2 cache throughput. High = good cache reuse.",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed":
        "Tensor core utilization (HMMA). Target: >60% for GEMM kernels.",
    "smsp__warps_issue_stalled_wait.pct":
        "Warps stalled waiting for data. High = memory-bound.",
    "smsp__warps_issue_stalled_math_pipe_throttle.pct":
        "Warps stalled on math pipe. High = compute-bound (good for GEMM).",
    "dram__bytes_read.sum":
        "Total HBM bytes read.",
    "dram__bytes_write.sum":
        "Total HBM bytes written.",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed":
        "Achieved occupancy.",
}


def bench(fn, warmup=5, iters=20):
    """Time a function with CUDA events. Returns (median_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


@dataclass
class BenchResult:
    component: str
    median_ms: float
    min_ms: float
    max_ms: float
    flops: Optional[float] = None
    bandwidth_gb_s: Optional[float] = None


def bench_rmsnorm(cfg, device, B=1, S=128):
    """Benchmark RMSNorm for DeepSeek-V3 hidden_size=7168."""
    from importlib import import_module
    mla = import_module("deepseekv3_2-kernels-flashinfer.mla_attention")
    norm = mla.RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"]).to(device)
    x = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
    med, mn, mx = bench(lambda: norm(x))
    return BenchResult("rmsnorm", med, mn, mx)


def bench_moe_router(cfg, device, B=1, S=128):
    """Benchmark MoE router with group selection (n_group=8, topk_group=4)."""
    from importlib import import_module
    router_mod = import_module("deepseekv3_2-kernels-flashinfer.moe_router")
    N = B * S
    logits = torch.randn(N, cfg["n_routed_experts"], device=device, dtype=torch.float32)
    bias = torch.zeros(cfg["n_routed_experts"], device=device, dtype=torch.float32)
    med, mn, mx = bench(lambda: router_mod.sigmoid_topk_route(
        logits, bias, top_k=cfg["num_experts_per_tok"],
        n_group=cfg["n_group"], topk_group=cfg["topk_group"],
    ))
    return BenchResult("moe_router", med, mn, mx)


def bench_mla_attention(cfg, device, B=1, S=128):
    """Benchmark MLA attention (eager fallback path)."""
    from importlib import import_module
    mla_mod = import_module("deepseekv3_2-kernels-flashinfer.mla_attention")
    rope_mod = import_module("deepseekv3_2-kernels-flashinfer.rope_partial")
    model_mod = import_module("deepseekv3_2-kernels-flashinfer.model")

    attn = mla_mod.MLAttention(cfg, 0).to(device).eval()
    rope = rope_mod.RotaryEmbedding(cfg).to(device)
    hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
    pos_ids = torch.arange(S, device=device).unsqueeze(0)
    cos, sin = rope(hidden, pos_ids)
    mask = model_mod.make_causal_mask(S, 0, hidden.dtype, device)

    with torch.no_grad():
        med, mn, mx = bench(lambda: attn(hidden, (cos, sin), attention_mask=mask))
    return BenchResult("mla_attention", med, mn, mx)


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V3 H100 Benchmark")
    parser.add_argument("--mode", choices=["bench", "nsys", "ncu"], default="bench")
    parser.add_argument("--full-dims", action="store_true", help="Use full DeepSeek-V3 dims")
    parser.add_argument("--component", type=str, default=None, help="Benchmark specific component")
    args = parser.parse_args()

    from .conftest import make_cfg, make_full_cfg, has_sm90
    device = "cuda"
    cfg = make_full_cfg() if args.full_dims else make_cfg(num_layers=2)

    if not has_sm90():
        print("WARNING: Not running on H100 (SM90). Results may not be representative.")

    print(f"{'='*70}")
    print(f"DeepSeek-V3 H100 Benchmark Suite")
    print(f"Mode: {args.mode}, Full dims: {args.full_dims}")
    print(f"{'='*70}")

    benchmarks = {
        "rmsnorm": lambda: bench_rmsnorm(cfg, device),
        "moe_router": lambda: bench_moe_router(cfg, device),
        "mla_attention": lambda: bench_mla_attention(cfg, device),
    }

    results = []
    for name, fn in benchmarks.items():
        if args.component and args.component != name:
            continue
        try:
            result = fn()
            results.append(result)
            print(f"  {result.component:20s}  median={result.median_ms:.3f}ms  "
                  f"min={result.min_ms:.3f}ms  max={result.max_ms:.3f}ms")
        except Exception as e:
            print(f"  {name:20s}  ERROR: {e}")

    print(f"\n{'='*70}")
    print(f"Completed {len(results)} benchmarks")


if __name__ == "__main__":
    main()
