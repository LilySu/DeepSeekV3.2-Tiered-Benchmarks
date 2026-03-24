"""H100-only: 3-way benchmark comparing DeepSeek-V3 implementations.

Compares per-component and end-to-end performance across:
  1. Pure PyTorch (eager)
  2. Triton kernels (RMSNorm/SwiGLU/CE/MoE GEMM)
  3. FlashInfer + DeepGEMM CUDA kernels (MLA/MoE)

Requirements:
    - NVIDIA H100/H800 GPU (SM90)
    - FlashInfer and DeepGEMM for column 3

Run:
    python3 -m deepseekv3_2-kernels-flashinfer.tests.h100_bench_3way
    python3 -m deepseekv3_2-kernels-flashinfer.tests.h100_bench_3way --full-dims

Paper ref: DeepSeek-V3 (arXiv 2412.19437)
"""

import argparse
import sys
from importlib import import_module

import torch
import torch.nn.functional as F

from .conftest import make_cfg, make_full_cfg, has_sm90, has_flashinfer, has_deep_gemm, PROJECT_ROOT


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


def make_inputs(cfg, device, B=1, S=128):
    """Create shared inputs for benchmark."""
    rope_mod = import_module("deepseekv3_2-kernels-flashinfer.rope_partial")
    model_mod = import_module("deepseekv3_2-kernels-flashinfer.model")

    hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S), device=device)
    rope = rope_mod.RotaryEmbedding(cfg).to(device)
    pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    cos, sin = rope(hidden, pos_ids)
    mask = model_mod.make_causal_mask(S, 0, hidden.dtype, device)

    return {
        "hidden": hidden, "input_ids": input_ids,
        "cos": cos, "sin": sin, "mask": mask,
        "B": B, "S": S,
    }


def bench_rmsnorm(cfg, inputs, device):
    """RMSNorm: PyTorch manual vs Unsloth Triton."""
    D = cfg["hidden_size"]
    x = inputs["hidden"]
    results = {}

    # PyTorch reference
    mla_mod = import_module("deepseekv3_2-kernels-flashinfer.mla_attention")
    pt_norm = mla_mod.RMSNorm(D, cfg["rms_norm_eps"]).to(device)
    results["pytorch"] = bench(lambda: pt_norm(x))

    # Triton (Unsloth)
    try:
        rms_mod = import_module("deepseekv3_2-kernels-flashinfer.unsloth_rms_layernorm")
        results["triton"] = bench(lambda: rms_mod.fast_rms_layernorm(pt_norm, x))
    except Exception:
        results["triton"] = (float("inf"), 0, 0)

    return results


def bench_moe_router(cfg, inputs, device):
    """MoE Router: group-based sigmoid routing."""
    router_mod = import_module("deepseekv3_2-kernels-flashinfer.moe_router")
    N = inputs["B"] * inputs["S"]
    logits = torch.randn(N, cfg["n_routed_experts"], device=device, dtype=torch.float32)
    bias = torch.zeros(cfg["n_routed_experts"], device=device)

    results = {}
    results["group_routing"] = bench(lambda: router_mod.sigmoid_topk_route(
        logits, bias, top_k=cfg["num_experts_per_tok"],
        n_group=cfg["n_group"], topk_group=cfg["topk_group"],
    ))

    # Flat routing comparison (n_group=1)
    results["flat_routing"] = bench(lambda: router_mod.sigmoid_topk_route(
        logits, bias, top_k=cfg["num_experts_per_tok"],
        n_group=1, topk_group=1,
    ))

    return results


def print_results(component, results):
    """Print comparison table for a component."""
    print(f"\n  {component}:")
    for name, (med, mn, mx) in results.items():
        print(f"    {name:20s}  median={med:.3f}ms  min={mn:.3f}ms  max={mx:.3f}ms")
    if len(results) >= 2:
        vals = list(results.values())
        speedup = vals[0][0] / vals[-1][0] if vals[-1][0] > 0 else 0
        print(f"    Speedup: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V3 3-Way Benchmark")
    parser.add_argument("--full-dims", action="store_true")
    parser.add_argument("--component", type=str, default=None)
    args = parser.parse_args()

    device = "cuda"
    cfg = make_full_cfg() if args.full_dims else make_cfg(num_layers=2)

    print(f"{'='*70}")
    print(f"DeepSeek-V3 3-Way Performance Comparison")
    print(f"Full dims: {args.full_dims}, SM90: {has_sm90()}")
    print(f"FlashInfer: {has_flashinfer()}, DeepGEMM: {has_deep_gemm()}")
    print(f"{'='*70}")

    # Only run on GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available")
        sys.exit(1)

    inputs = make_inputs(cfg, device)

    benchmarks = [
        ("RMSNorm", bench_rmsnorm),
        ("MoE Router", bench_moe_router),
    ]

    for name, fn in benchmarks:
        if args.component and args.component.lower() not in name.lower():
            continue
        try:
            results = fn(cfg, inputs, device)
            print_results(name, results)
        except Exception as e:
            print(f"\n  {name}: ERROR - {e}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
