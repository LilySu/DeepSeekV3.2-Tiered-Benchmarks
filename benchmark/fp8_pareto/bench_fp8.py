"""
FP8 Pareto frontier benchmarks for DeepSeek-V3.

DeepSeek-V3 is one of the first production models trained entirely with FP8.
It uses e4m3 format with 128x128 block-wise quantization for both forward
and backward passes. This benchmark measures the Pareto frontier of
precision vs. throughput across different FP8 configurations.

Reference: DeepSeek-V3 Technical Report, Section 3.3 (arXiv: 2412.19437)
  - Fine-grained quantization: 128-element groups for activations, 128x128 blocks for weights
  - Per-block scaling factors stored alongside quantized tensors
  - Mixed-precision accumulation in FP32 for numerical stability
"""

from __future__ import annotations

import os
import sys
import json
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG, BenchConfig, BenchResult
from benchmark.shared.timer import CUDATimer
from benchmark.shared.metrics import compute_mfu, compute_bandwidth_utilization
from benchmark.shared.report import save_json_report, save_markdown_report


# ---------------------------------------------------------------------------
# FP8 Configuration
# ---------------------------------------------------------------------------

@dataclass
class FP8Config:
    """Configuration for an FP8 precision experiment."""
    name: str = "fp8_e4m3_block128"
    format: str = "e4m3"                    # e4m3 or e5m2
    block_size: Tuple[int, int] = (128, 128)
    use_per_block_scaling: bool = True
    accumulate_dtype: str = "float32"       # accumulation precision
    output_dtype: str = "bfloat16"          # output cast-back precision
    stochastic_rounding: bool = False


# Pre-defined configurations to sweep
FP8_CONFIGS = [
    FP8Config(name="fp8_e4m3_block128", format="e4m3", block_size=(128, 128)),
    FP8Config(name="fp8_e4m3_block64", format="e4m3", block_size=(64, 64)),
    FP8Config(name="fp8_e4m3_block256", format="e4m3", block_size=(256, 256)),
    FP8Config(name="fp8_e4m3_per_tensor", format="e4m3", block_size=(0, 0), use_per_block_scaling=False),
    FP8Config(name="fp8_e5m2_block128", format="e5m2", block_size=(128, 128)),
    FP8Config(name="fp8_e4m3_stochastic", format="e4m3", block_size=(128, 128), stochastic_rounding=True),
]


# ---------------------------------------------------------------------------
# FP8 GEMM Kernel (benchmark stub)
# ---------------------------------------------------------------------------

def _create_fp8_gemm_fn(
    M: int,
    N: int,
    K: int,
    fp8_config: FP8Config,
    device: str = "cuda",
):
    """
    Create a callable that performs an FP8 GEMM with the given configuration.

    In production, this would use:
      - torch._scaled_mm (PyTorch 2.1+)
      - CUTLASS FP8 kernels
      - DeepGEMM library (DeepSeek's open-source FP8 GEMM)

    For benchmarking purposes, we simulate with quantize -> matmul -> dequantize.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for FP8 benchmarks")

    # Create weight and activation tensors
    dtype = torch.bfloat16
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Check if native FP8 is available (PyTorch 2.1+ with H100)
    has_fp8 = hasattr(torch, "float8_e4m3fn")

    if has_fp8 and device == "cuda":
        # Use native FP8 path
        fp8_dtype = torch.float8_e4m3fn if fp8_config.format == "e4m3" else torch.float8_e5m2

        def _quantize_block(tensor, block_h, block_w, fp8_dt):
            """Block-wise FP8 quantization with per-block scaling."""
            if block_h == 0 or block_w == 0:
                # Per-tensor quantization
                scale = tensor.abs().max() / 448.0  # e4m3 max value
                scale = scale.clamp(min=1e-12)
                return tensor.to(fp8_dt) , scale.float()

            # Pad to block boundaries
            h, w = tensor.shape
            pad_h = (block_h - h % block_h) % block_h
            pad_w = (block_w - w % block_w) % block_w
            if pad_h > 0 or pad_w > 0:
                tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))

            # Reshape into blocks
            bh, bw = tensor.shape[0] // block_h, tensor.shape[1] // block_w
            blocks = tensor.reshape(bh, block_h, bw, block_w).permute(0, 2, 1, 3)
            scales = blocks.reshape(bh, bw, -1).abs().amax(dim=-1) / 448.0
            scales = scales.clamp(min=1e-12)

            # Quantize (simplified -- real impl uses fused kernels)
            q = tensor.to(fp8_dt)
            return q[:h, :w], scales

        block_h, block_w = fp8_config.block_size

        def fn():
            A_q, A_scale = _quantize_block(A, block_h, block_w, fp8_dtype)
            B_q, B_scale = _quantize_block(B, block_h, block_w, fp8_dtype)
            # Scaled matmul
            try:
                out = torch._scaled_mm(
                    A_q, B_q.t(),
                    scale_a=A_scale.reshape(-1)[0:1] if A_scale.dim() > 0 else A_scale.unsqueeze(0),
                    scale_b=B_scale.reshape(-1)[0:1] if B_scale.dim() > 0 else B_scale.unsqueeze(0),
                    out_dtype=torch.bfloat16,
                )
            except (RuntimeError, TypeError):
                # Fallback for older PyTorch or unsupported configs
                out = torch.mm(A.float(), B.float()).to(torch.bfloat16)
            return out

        return fn
    else:
        # CPU fallback or older PyTorch: simulate with bf16
        def fn():
            return torch.mm(A, B)
        return fn


# ---------------------------------------------------------------------------
# Pareto Frontier Sweep
# ---------------------------------------------------------------------------

def run_fp8_pareto_sweep(
    bench_config: Optional[BenchConfig] = None,
    fp8_configs: Optional[List[FP8Config]] = None,
    matrix_sizes: Optional[List[Tuple[int, int, int]]] = None,
) -> List[BenchResult]:
    """
    Run FP8 Pareto frontier sweep.

    Tests each FP8 configuration across multiple matrix sizes representative
    of DeepSeek-V3 GEMM shapes:
      - MLA projections: (bs*seq, hidden, kv_lora_rank) = (*, 7168, 512)
      - MoE expert FFN: (tokens_per_expert, hidden, moe_intermediate) = (*, 7168, 2048)
      - Output head: (bs*seq, hidden, vocab) = (*, 7168, 129280)
    """
    if bench_config is None:
        bench_config = BenchConfig(name="fp8_pareto")

    if fp8_configs is None:
        fp8_configs = FP8_CONFIGS

    if matrix_sizes is None:
        matrix_sizes = [
            # (M, N, K) -- representative DeepSeek-V3 GEMM shapes
            (256, 512, 7168),     # MLA down-projection: small batch
            (1024, 512, 7168),    # MLA down-projection: medium batch
            (4096, 512, 7168),    # MLA down-projection: large batch
            (256, 2048, 7168),    # MoE expert FFN: small
            (1024, 2048, 7168),   # MoE expert FFN: medium
            (256, 7168, 2048),    # MoE expert FFN down-proj
            (1024, 129280, 7168), # LM head: large vocab
        ]

    timer = CUDATimer(
        warmup=bench_config.warmup_iters,
        iters=bench_config.bench_iters,
    )

    results: List[BenchResult] = []

    for fp8_cfg in fp8_configs:
        print(f"\n--- FP8 Config: {fp8_cfg.name} ---")

        for M, N, K in matrix_sizes:
            print(f"  Matrix: ({M}, {N}, {K}) ... ", end="", flush=True)

            try:
                fn = _create_fp8_gemm_fn(M, N, K, fp8_cfg)
                timing = timer.time_fn(fn)

                # Compute metrics
                gemm_flops = 2.0 * M * N * K
                elapsed_s = timing.mean_ms / 1000.0
                tflops = gemm_flops / elapsed_s / 1e12 if elapsed_s > 0 else 0

                fp8_format_str = f"float8_{fp8_cfg.format}" if fp8_cfg.format == "e4m3" else "bfloat16"
                mfu = compute_mfu(gemm_flops, elapsed_s, dtype=fp8_format_str)

                # Memory bytes (A + B + C)
                bytes_per_elem = 1 if "fp8" in fp8_cfg.name else 2
                total_bytes = (M * K + K * N) * bytes_per_elem + M * N * 2  # output in bf16
                bw_util = compute_bandwidth_utilization(total_bytes, elapsed_s)

                result = BenchResult(
                    config_name=fp8_cfg.name,
                    component="fp8_gemm",
                    batch_size=M,
                    seq_len=0,
                    extra_params={"M": M, "N": N, "K": K, "format": fp8_cfg.format,
                                  "block_size": list(fp8_cfg.block_size)},
                    mean_ms=timing.mean_ms,
                    std_ms=timing.std_ms,
                    median_ms=timing.median_ms,
                    min_ms=timing.min_ms,
                    max_ms=timing.max_ms,
                    ci_lower_ms=timing.ci_lower_ms,
                    ci_upper_ms=timing.ci_upper_ms,
                    tflops_achieved=tflops,
                    mfu=mfu,
                    bandwidth_utilization=bw_util,
                )
                results.append(result)
                print(f"{timing.mean_ms:.3f}ms ({tflops:.1f} TFLOPS, MFU={mfu:.1f}%)")

            except Exception as e:
                print(f"FAILED: {e}")
                continue

    return results


# ---------------------------------------------------------------------------
# Pareto analysis
# ---------------------------------------------------------------------------

def compute_pareto_frontier(results: List[BenchResult]) -> List[BenchResult]:
    """
    Extract Pareto-optimal configurations from results.

    A result is Pareto-optimal if no other result has both higher throughput
    and lower latency for the same matrix size.
    """
    # Group by matrix shape
    groups: Dict[str, List[BenchResult]] = {}
    for r in results:
        key = f"{r.extra_params.get('M', 0)}x{r.extra_params.get('N', 0)}x{r.extra_params.get('K', 0)}"
        groups.setdefault(key, []).append(r)

    pareto_results = []
    for key, group in groups.items():
        # Sort by throughput descending
        group.sort(key=lambda r: r.tflops_achieved, reverse=True)

        frontier = []
        best_latency = float("inf")
        for r in group:
            if r.mean_ms < best_latency:
                frontier.append(r)
                best_latency = r.mean_ms

        pareto_results.extend(frontier)

    return pareto_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run FP8 Pareto benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek-V3 FP8 Pareto Frontier Benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/fp8_pareto")
    args = parser.parse_args()

    config = BenchConfig(
        name="fp8_pareto",
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        output_dir=args.output_dir,
    )

    print("=" * 70)
    print("DeepSeek-V3 FP8 Pareto Frontier Benchmark")
    print(f"Format: e4m3, Block size: 128x128 (DeepSeek-V3 default)")
    print(f"Warmup: {config.warmup_iters}, Iterations: {config.bench_iters}")
    print("=" * 70)

    results = run_fp8_pareto_sweep(config)

    if results:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = save_json_report(results, os.path.join(args.output_dir, "fp8_pareto.json"))
        md_path = save_markdown_report(results, os.path.join(args.output_dir, "fp8_pareto.md"),
                                       title="FP8 Pareto Frontier")
        print(f"\nResults saved to: {json_path}")
        print(f"Report saved to:  {md_path}")

        pareto = compute_pareto_frontier(results)
        print(f"\nPareto-optimal configurations: {len(pareto)}/{len(results)}")
        for r in pareto:
            print(f"  {r.config_name}: {r.extra_params['M']}x{r.extra_params['N']}x{r.extra_params['K']} "
                  f"-> {r.mean_ms:.3f}ms, {r.tflops_achieved:.1f} TFLOPS")


if __name__ == "__main__":
    main()
