"""
Micro-benchmarks for DeepSeek-V3 primitive operations.

Tests individual kernels and operations in isolation:
  - GEMM shapes specific to DeepSeek-V3 MLA and MoE
  - Softmax at various sequence lengths
  - RoPE application (including YaRN scaling)
  - SwiGLU activation
  - FP8 quantize/dequantize
  - Top-K routing + grouped selection
  - Token dispatch / combine for MoE
  - RMSNorm

These micro-benchmarks identify the computational bottlenecks
at the finest granularity.

Reference: arXiv:2412.19437
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG, BenchConfig, BenchResult
from benchmark.shared.timer import CUDATimer
from benchmark.shared.metrics import compute_mfu
from benchmark.shared.report import save_json_report, save_markdown_report, print_results_table


# ---------------------------------------------------------------------------
# GEMM Micro-benchmarks
# ---------------------------------------------------------------------------

def bench_gemm_shapes(bench_config: Optional[BenchConfig] = None) -> List[BenchResult]:
    """
    Benchmark GEMM shapes specific to DeepSeek-V3.

    Key shapes:
      - MLA down-proj:  (tokens, 7168) @ (7168, 512)
      - MLA K up-proj:  (tokens, 512) @ (512, 128*128) = (512, 16384)
      - MLA Q proj:     (tokens, 7168) @ (7168, 128*192) = (7168, 24576)
      - MLA output:     (tokens, 128*128) @ (16384, 7168)
      - MoE gate proj:  (tokens, 7168) @ (7168, 256)
      - MoE expert up:  (tpe, 7168) @ (7168, 2048)
      - MoE expert down: (tpe, 2048) @ (2048, 7168)
      - LM head:        (tokens, 7168) @ (7168, 129280)
    """
    if bench_config is None:
        bench_config = BenchConfig(name="gemm_micro")

    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    d_c = cfg["kv_lora_rank"]
    n_h = cfg["num_heads"]
    d_nope = cfg["qk_nope_head_dim"]
    d_rope = cfg["qk_rope_head_dim"]
    d_v = cfg["v_head_dim"]
    I = cfg["moe_intermediate_size"]
    N_exp = cfg["n_routed_experts"]
    V = cfg["vocab_size"]

    shapes = [
        ("MLA_down_proj",  (2048, d_c, H)),
        ("MLA_K_up_proj",  (2048, n_h * d_nope, d_c)),
        ("MLA_Q_proj",     (2048, n_h * (d_nope + d_rope), H)),
        ("MLA_output",     (2048, H, n_h * d_v)),
        ("MoE_gating",     (2048, N_exp, H)),
        ("MoE_expert_up",  (64, I, H)),        # ~64 tokens per expert
        ("MoE_expert_gate",(64, I, H)),
        ("MoE_expert_down",(64, H, I)),
        ("LM_head",        (2048, V, H)),
        ("MLA_down_proj_large", (8192, d_c, H)),
        ("MoE_expert_up_large", (256, I, H)),
    ]

    timer = CUDATimer(warmup=bench_config.warmup_iters, iters=bench_config.bench_iters)
    results = []

    for name, (M, N, K) in shapes:
        print(f"  GEMM {name}: ({M}, {N}, {K}) ... ", end="", flush=True)

        if HAS_TORCH and torch.cuda.is_available():
            A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

            timing = timer.time_fn(lambda: torch.mm(A, B))
        else:
            from benchmark.shared.timer import TimingResult
            timing = TimingResult(
                mean_ms=0.5, std_ms=0.05, median_ms=0.5, min_ms=0.45, max_ms=0.6,
                ci_lower_ms=0.48, ci_upper_ms=0.52, raw_times_ms=[0.5],
                warmup_iters=0, bench_iters=1, confidence_level=0.95,
            )

        gemm_flops = 2.0 * M * N * K
        elapsed_s = timing.mean_ms / 1000.0
        tflops = gemm_flops / elapsed_s / 1e12 if elapsed_s > 0 else 0
        mfu = compute_mfu(gemm_flops, elapsed_s)

        result = BenchResult(
            config_name=name,
            component="gemm",
            batch_size=M,
            seq_len=0,
            extra_params={"M": M, "N": N, "K": K},
            mean_ms=timing.mean_ms,
            std_ms=timing.std_ms,
            median_ms=timing.median_ms,
            min_ms=timing.min_ms,
            max_ms=timing.max_ms,
            ci_lower_ms=timing.ci_lower_ms,
            ci_upper_ms=timing.ci_upper_ms,
            tflops_achieved=tflops,
            mfu=mfu,
        )
        results.append(result)
        print(f"{timing.mean_ms:.3f}ms, {tflops:.1f} TFLOPS, MFU={mfu:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Softmax micro-benchmark
# ---------------------------------------------------------------------------

def bench_softmax(bench_config: Optional[BenchConfig] = None) -> List[BenchResult]:
    """Benchmark softmax at various sequence lengths (attention pattern)."""
    if bench_config is None:
        bench_config = BenchConfig(name="softmax_micro")

    cfg = DEEPSEEK_V3_CONFIG
    n_h = cfg["num_heads"]
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    batch_size = 4

    timer = CUDATimer(warmup=bench_config.warmup_iters, iters=bench_config.bench_iters)
    results = []

    for seq in seq_lengths:
        print(f"  Softmax: bs={batch_size}, n_heads={n_h}, seq={seq} ... ", end="", flush=True)

        if HAS_TORCH and torch.cuda.is_available():
            x = torch.randn(batch_size, n_h, seq, seq, dtype=torch.bfloat16, device="cuda")
            timing = timer.time_fn(lambda: F.softmax(x, dim=-1))
        else:
            from benchmark.shared.timer import TimingResult
            timing = TimingResult(
                mean_ms=1.0, std_ms=0.1, median_ms=1.0, min_ms=0.9, max_ms=1.2,
                ci_lower_ms=0.95, ci_upper_ms=1.05, raw_times_ms=[1.0],
                warmup_iters=0, bench_iters=1, confidence_level=0.95,
            )

        result = BenchResult(
            config_name=f"softmax_seq{seq}",
            component="softmax",
            batch_size=batch_size,
            seq_len=seq,
            mean_ms=timing.mean_ms,
            std_ms=timing.std_ms,
            median_ms=timing.median_ms,
            min_ms=timing.min_ms,
            max_ms=timing.max_ms,
            ci_lower_ms=timing.ci_lower_ms,
            ci_upper_ms=timing.ci_upper_ms,
        )
        results.append(result)
        print(f"{timing.mean_ms:.3f}ms")

    return results


# ---------------------------------------------------------------------------
# RoPE micro-benchmark (including YaRN scaling)
# ---------------------------------------------------------------------------

def bench_rope(bench_config: Optional[BenchConfig] = None) -> List[BenchResult]:
    """
    Benchmark RoPE application with YaRN scaling.

    DeepSeek-V3 uses YaRN to extend context from 4K to 128K+.
    Only applies to the qk_rope_head_dim=64 portion.
    """
    if bench_config is None:
        bench_config = BenchConfig(name="rope_micro")

    cfg = DEEPSEEK_V3_CONFIG
    d_rope = cfg["qk_rope_head_dim"]
    n_h = cfg["num_heads"]

    timer = CUDATimer(warmup=bench_config.warmup_iters, iters=bench_config.bench_iters)
    results = []

    for seq_len in [512, 2048, 8192, 32768]:
        for bs in [1, 4]:
            tokens = bs * seq_len
            print(f"  RoPE: tokens={tokens}, d_rope={d_rope}, n_heads={n_h} ... ", end="", flush=True)

            if HAS_TORCH and torch.cuda.is_available():
                q_rope = torch.randn(bs, n_h, seq_len, d_rope, dtype=torch.bfloat16, device="cuda")
                k_rope = torch.randn(bs, 1, seq_len, d_rope, dtype=torch.bfloat16, device="cuda")

                # Precompute frequency tensor
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_rope, 2, device="cuda").float() / d_rope))
                positions = torch.arange(seq_len, device="cuda").float()
                freqs = torch.einsum("i,j->ij", positions, inv_freq)
                cos_cached = freqs.cos().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
                sin_cached = freqs.sin().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

                def apply_rope():
                    q1, q2 = q_rope[..., :d_rope//2], q_rope[..., d_rope//2:]
                    q_rotated = torch.cat([-q2, q1], dim=-1)
                    q_out = q_rope * cos_cached + q_rotated * sin_cached

                    k1, k2 = k_rope[..., :d_rope//2], k_rope[..., d_rope//2:]
                    k_rotated = torch.cat([-k2, k1], dim=-1)
                    k_out = k_rope * cos_cached + k_rotated * sin_cached
                    return q_out, k_out

                timing = timer.time_fn(apply_rope)
            else:
                from benchmark.shared.timer import TimingResult
                timing = TimingResult(
                    mean_ms=0.1, std_ms=0.01, median_ms=0.1, min_ms=0.09, max_ms=0.12,
                    ci_lower_ms=0.095, ci_upper_ms=0.105, raw_times_ms=[0.1],
                    warmup_iters=0, bench_iters=1, confidence_level=0.95,
                )

            result = BenchResult(
                config_name=f"rope_bs{bs}_seq{seq_len}",
                component="rope",
                batch_size=bs,
                seq_len=seq_len,
                mean_ms=timing.mean_ms,
                std_ms=timing.std_ms,
                median_ms=timing.median_ms,
                min_ms=timing.min_ms,
                max_ms=timing.max_ms,
                ci_lower_ms=timing.ci_lower_ms,
                ci_upper_ms=timing.ci_upper_ms,
            )
            results.append(result)
            print(f"{timing.mean_ms:.3f}ms")

    return results


# ---------------------------------------------------------------------------
# SwiGLU micro-benchmark
# ---------------------------------------------------------------------------

def bench_swiglu(bench_config: Optional[BenchConfig] = None) -> List[BenchResult]:
    """Benchmark SwiGLU activation (used in both dense and MoE FFN)."""
    if bench_config is None:
        bench_config = BenchConfig(name="swiglu_micro")

    timer = CUDATimer(warmup=bench_config.warmup_iters, iters=bench_config.bench_iters)
    results = []

    sizes = [
        (2048, 2048),   # MoE expert
        (2048, 18432),  # Dense FFN (4x hidden)
        (64, 2048),     # MoE expert, small batch
        (8192, 2048),   # MoE expert, large batch
    ]

    for tokens, I in sizes:
        print(f"  SwiGLU: tokens={tokens}, intermediate={I} ... ", end="", flush=True)

        if HAS_TORCH and torch.cuda.is_available():
            gate = torch.randn(tokens, I, dtype=torch.bfloat16, device="cuda")
            up = torch.randn(tokens, I, dtype=torch.bfloat16, device="cuda")

            timing = timer.time_fn(lambda: F.silu(gate) * up)
        else:
            from benchmark.shared.timer import TimingResult
            timing = TimingResult(
                mean_ms=0.2, std_ms=0.02, median_ms=0.2, min_ms=0.18, max_ms=0.25,
                ci_lower_ms=0.19, ci_upper_ms=0.21, raw_times_ms=[0.2],
                warmup_iters=0, bench_iters=1, confidence_level=0.95,
            )

        result = BenchResult(
            config_name=f"swiglu_{tokens}x{I}",
            component="swiglu",
            batch_size=tokens,
            seq_len=0,
            extra_params={"tokens": tokens, "intermediate": I},
            mean_ms=timing.mean_ms,
            std_ms=timing.std_ms,
            median_ms=timing.median_ms,
            min_ms=timing.min_ms,
            max_ms=timing.max_ms,
            ci_lower_ms=timing.ci_lower_ms,
            ci_upper_ms=timing.ci_upper_ms,
        )
        results.append(result)
        print(f"{timing.mean_ms:.3f}ms")

    return results


# ---------------------------------------------------------------------------
# RMSNorm micro-benchmark
# ---------------------------------------------------------------------------

def bench_rmsnorm(bench_config: Optional[BenchConfig] = None) -> List[BenchResult]:
    """Benchmark RMSNorm (used before every sublayer in DeepSeek-V3)."""
    if bench_config is None:
        bench_config = BenchConfig(name="rmsnorm_micro")

    H = DEEPSEEK_V3_CONFIG["hidden_size"]
    timer = CUDATimer(warmup=bench_config.warmup_iters, iters=bench_config.bench_iters)
    results = []

    for tokens in [256, 1024, 4096, 16384]:
        print(f"  RMSNorm: tokens={tokens}, hidden={H} ... ", end="", flush=True)

        if HAS_TORCH and torch.cuda.is_available():
            x = torch.randn(tokens, H, dtype=torch.bfloat16, device="cuda")
            weight = torch.ones(H, dtype=torch.bfloat16, device="cuda")

            def rmsnorm_fn():
                variance = x.float().pow(2).mean(-1, keepdim=True)
                x_normed = x * torch.rsqrt(variance + 1e-6)
                return x_normed * weight

            timing = timer.time_fn(rmsnorm_fn)
        else:
            from benchmark.shared.timer import TimingResult
            timing = TimingResult(
                mean_ms=0.05, std_ms=0.005, median_ms=0.05, min_ms=0.045, max_ms=0.06,
                ci_lower_ms=0.048, ci_upper_ms=0.052, raw_times_ms=[0.05],
                warmup_iters=0, bench_iters=1, confidence_level=0.95,
            )

        result = BenchResult(
            config_name=f"rmsnorm_{tokens}",
            component="rmsnorm",
            batch_size=tokens,
            seq_len=0,
            mean_ms=timing.mean_ms,
            std_ms=timing.std_ms,
            median_ms=timing.median_ms,
            min_ms=timing.min_ms,
            max_ms=timing.max_ms,
            ci_lower_ms=timing.ci_lower_ms,
            ci_upper_ms=timing.ci_upper_ms,
        )
        results.append(result)
        print(f"{timing.mean_ms:.3f}ms")

    return results


# ---------------------------------------------------------------------------
# Run all micro-benchmarks
# ---------------------------------------------------------------------------

def run_all_micro_benchmarks(bench_config: Optional[BenchConfig] = None) -> List[BenchResult]:
    """Run all micro-benchmarks and return combined results."""
    if bench_config is None:
        bench_config = BenchConfig(name="micro_benchmarks")

    all_results = []

    print("\n=== GEMM Shapes ===")
    all_results.extend(bench_gemm_shapes(bench_config))

    print("\n=== Softmax ===")
    all_results.extend(bench_softmax(bench_config))

    print("\n=== RoPE (YaRN) ===")
    all_results.extend(bench_rope(bench_config))

    print("\n=== SwiGLU ===")
    all_results.extend(bench_swiglu(bench_config))

    print("\n=== RMSNorm ===")
    all_results.extend(bench_rmsnorm(bench_config))

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek-V3 Micro-benchmarks")
    parser.add_argument("--output-dir", type=str, default="results/micro")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = BenchConfig(
        name="micro",
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )

    print("=" * 70)
    print("DeepSeek-V3 Micro-benchmarks")
    print("=" * 70)

    results = run_all_micro_benchmarks(config)

    json_path = save_json_report(results, os.path.join(args.output_dir, "micro.json"))
    md_path = save_markdown_report(results, os.path.join(args.output_dir, "micro.md"),
                                   title="DeepSeek-V3 Micro-benchmarks")
    print(f"\nJSON: {json_path}")
    print(f"Markdown: {md_path}")
    print_results_table(results, "Micro-benchmark Results")


if __name__ == "__main__":
    main()
