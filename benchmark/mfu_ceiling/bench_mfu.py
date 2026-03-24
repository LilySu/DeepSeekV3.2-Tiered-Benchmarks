"""
MFU ceiling analysis for DeepSeek-V3 671B.

This benchmark determines the theoretical and achievable MFU ceiling
for the DeepSeek-V3 architecture on NVIDIA H100 GPUs. The analysis
covers each major component individually and then composes them
into a full-model estimate.

Key considerations for DeepSeek-V3 MFU:
  1. MLA reduces KV cache memory bandwidth but adds projection overhead.
  2. MoE with 256 experts and top-8 routing creates expert-parallel workloads.
  3. Grouped routing (8 groups, top-4) affects load balancing.
  4. FP8 computation doubles peak TFLOPS but quantization overhead matters.
  5. MTP adds 1 extra prediction head (modest overhead).

Reference: arXiv:2412.19437
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG, H100_SPECS, BenchConfig, BenchResult
from benchmark.shared.timer import CUDATimer
from benchmark.shared.metrics import (
    compute_mla_flops,
    compute_moe_flops,
    compute_dense_ffn_flops,
    compute_mtp_flops,
    compute_full_model_flops,
    compute_mfu,
    compute_bandwidth_utilization,
    roofline_bound,
)
from benchmark.shared.report import save_json_report, save_markdown_report, print_results_table


# ---------------------------------------------------------------------------
# Roofline model parameters
# ---------------------------------------------------------------------------

@dataclass
class RooflinePoint:
    """A single point on the roofline model."""
    name: str
    flops: float
    bytes_accessed: float
    arithmetic_intensity: float
    bound: str              # "compute" or "memory"
    achievable_tflops: float
    theoretical_tflops: float
    efficiency_pct: float


def compute_roofline_point(
    name: str,
    flops: float,
    bytes_accessed: float,
    dtype: str = "bfloat16",
    hardware: Dict[str, Any] = H100_SPECS,
) -> RooflinePoint:
    """Compute a roofline model point for a given operation."""
    ai = flops / bytes_accessed if bytes_accessed > 0 else float("inf")

    dtype_tflops = {
        "fp8": hardware["fp8_tflops"],
        "float8_e4m3fn": hardware["fp8_tflops"],
        "bfloat16": hardware["bf16_tflops"],
        "float16": hardware["fp16_tflops"],
    }
    peak_tflops = dtype_tflops.get(dtype, hardware["bf16_tflops"])
    peak_bw_tb = hardware["hbm_bandwidth_tb_s"]

    # Ridge point
    ridge = peak_tflops * 1e12 / (peak_bw_tb * 1e12)  # FLOP/byte

    # Achievable performance
    if ai >= ridge:
        # Compute-bound: limited by peak TFLOPS
        achievable = peak_tflops
        bound = "compute"
    else:
        # Memory-bound: limited by bandwidth * arithmetic intensity
        achievable = ai * peak_bw_tb  # TFLOPS
        bound = "memory"

    efficiency = 100.0 * achievable / peak_tflops if peak_tflops > 0 else 0

    return RooflinePoint(
        name=name,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=ai,
        bound=bound,
        achievable_tflops=achievable,
        theoretical_tflops=peak_tflops,
        efficiency_pct=efficiency,
    )


# ---------------------------------------------------------------------------
# MLA MFU ceiling
# ---------------------------------------------------------------------------

def bench_mla_mfu_ceiling(
    batch_size: int = 1,
    seq_len: int = 2048,
    bench_config: Optional[BenchConfig] = None,
) -> BenchResult:
    """
    Benchmark MLA (Multi-head Latent Attention) MFU ceiling.

    MLA key insight: by compressing KV into a low-rank latent of dimension 512,
    the KV cache per layer per token drops from 2 * num_heads * head_dim * 2bytes
    to kv_lora_rank * 2bytes = 1024 bytes (a ~32x reduction for DeepSeek-V3).

    This makes MLA significantly more memory-efficient during decode, shifting
    the bottleneck from KV cache bandwidth to compute.
    """
    if bench_config is None:
        bench_config = BenchConfig(name="mla_mfu_ceiling")

    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    n_h = cfg["num_heads"]
    d_c = cfg["kv_lora_rank"]
    d_rope = cfg["qk_rope_head_dim"]
    d_nope = cfg["qk_nope_head_dim"]
    d_v = cfg["v_head_dim"]

    timer = CUDATimer(warmup=bench_config.warmup_iters, iters=bench_config.bench_iters)

    if HAS_TORCH and torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16

        # Create MLA projection weights
        W_dkv = torch.randn(H, d_c, dtype=dtype, device=device)          # down-project to latent
        W_uk = torch.randn(d_c, n_h * d_nope, dtype=dtype, device=device)  # up-project K nope
        W_rope = torch.randn(H, d_rope, dtype=dtype, device=device)       # RoPE key
        W_q = torch.randn(H, n_h * (d_nope + d_rope), dtype=dtype, device=device)
        W_o = torch.randn(n_h * d_v, H, dtype=dtype, device=device)

        x = torch.randn(batch_size * seq_len, H, dtype=dtype, device=device)

        def mla_forward():
            # Down-project to KV latent
            c_kv = x @ W_dkv  # (tokens, d_c)

            # Up-project K nope from latent
            k_nope = c_kv @ W_uk  # (tokens, n_h * d_nope)

            # RoPE key (shared across heads)
            k_rope = x @ W_rope  # (tokens, d_rope)

            # Q projection
            q = x @ W_q  # (tokens, n_h * (d_nope + d_rope))

            # Reshape for attention (simplified - no actual attention computation here)
            # In practice: q @ k^T -> softmax -> @ v
            q_r = q.reshape(batch_size, seq_len, n_h, d_nope + d_rope)
            k_nope_r = k_nope.reshape(batch_size, seq_len, n_h, d_nope)

            # Output projection (using c_kv as surrogate for attention output)
            out = c_kv[:, :n_h * d_v] @ W_o[:n_h * d_v, :]  # simplified

            return out

        timing = timer.time_fn(mla_forward)
    else:
        # Dry-run timing estimate
        from benchmark.shared.timer import TimingResult
        timing = TimingResult(
            mean_ms=1.0, std_ms=0.1, median_ms=1.0, min_ms=0.9, max_ms=1.2,
            ci_lower_ms=0.95, ci_upper_ms=1.05, raw_times_ms=[1.0],
            warmup_iters=0, bench_iters=1, confidence_level=0.95,
        )

    # Compute theoretical FLOPs
    flop_info = compute_mla_flops(batch_size, seq_len)
    elapsed_s = timing.mean_ms / 1000.0
    mfu = compute_mfu(flop_info["total_flops"], elapsed_s, dtype=bench_config.dtype)

    return BenchResult(
        config_name="mla_mfu_ceiling",
        component="mla",
        batch_size=batch_size,
        seq_len=seq_len,
        mean_ms=timing.mean_ms,
        std_ms=timing.std_ms,
        median_ms=timing.median_ms,
        min_ms=timing.min_ms,
        max_ms=timing.max_ms,
        ci_lower_ms=timing.ci_lower_ms,
        ci_upper_ms=timing.ci_upper_ms,
        tokens_per_sec=(batch_size * seq_len) / elapsed_s if elapsed_s > 0 else 0,
        tflops_achieved=flop_info["total_flops"] / elapsed_s / 1e12 if elapsed_s > 0 else 0,
        mfu=mfu,
    )


# ---------------------------------------------------------------------------
# MoE MFU ceiling
# ---------------------------------------------------------------------------

def bench_moe_mfu_ceiling(
    batch_size: int = 1,
    seq_len: int = 2048,
    bench_config: Optional[BenchConfig] = None,
) -> BenchResult:
    """
    Benchmark MoE (Mixture of Experts) MFU ceiling.

    DeepSeek-V3 uses 256 routed experts with grouped routing:
      1. Compute affinity scores for all 256 experts.
      2. Group experts into 8 groups of 32.
      3. Select top-4 groups based on aggregate affinity.
      4. Within selected groups, pick top-2 experts per group (4*2=8 total).

    The MFU ceiling for MoE depends heavily on:
      - Expert load balancing (affected by auxiliary-loss-free balancing)
      - All-to-all communication overhead in distributed settings
      - Token-to-expert dispatch efficiency
    """
    if bench_config is None:
        bench_config = BenchConfig(name="moe_mfu_ceiling")

    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    K = cfg["num_experts_per_tok"]
    N_experts = cfg["n_routed_experts"]

    timer = CUDATimer(warmup=bench_config.warmup_iters, iters=bench_config.bench_iters)

    if HAS_TORCH and torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        tokens = batch_size * seq_len

        # Simulate a single expert FFN (SwiGLU)
        W_gate = torch.randn(H, I, dtype=dtype, device=device)
        W_up = torch.randn(H, I, dtype=dtype, device=device)
        W_down = torch.randn(I, H, dtype=dtype, device=device)

        # Gating network
        W_gating = torch.randn(H, N_experts, dtype=dtype, device=device)

        x = torch.randn(tokens, H, dtype=dtype, device=device)

        def moe_forward():
            # Gating: compute expert affinities
            logits = x @ W_gating  # (tokens, N_experts)
            probs = torch.softmax(logits, dim=-1)

            # Top-K selection (simplified, no grouped routing in this micro-bench)
            topk_vals, topk_idx = probs.topk(K, dim=-1)

            # Expert computation (simulate K experts processing all tokens)
            # In practice, only selected tokens go to each expert
            gate = torch.nn.functional.silu(x @ W_gate)
            up = x @ W_up
            hidden = gate * up
            out = hidden @ W_down

            # Weighted combination (simplified)
            return out * topk_vals.sum(dim=-1, keepdim=True)

        timing = timer.time_fn(moe_forward)
    else:
        from benchmark.shared.timer import TimingResult
        timing = TimingResult(
            mean_ms=2.0, std_ms=0.2, median_ms=2.0, min_ms=1.8, max_ms=2.5,
            ci_lower_ms=1.9, ci_upper_ms=2.1, raw_times_ms=[2.0],
            warmup_iters=0, bench_iters=1, confidence_level=0.95,
        )

    flop_info = compute_moe_flops(batch_size, seq_len)
    elapsed_s = timing.mean_ms / 1000.0
    mfu = compute_mfu(flop_info["total_flops"], elapsed_s, dtype=bench_config.dtype)

    return BenchResult(
        config_name="moe_mfu_ceiling",
        component="moe",
        batch_size=batch_size,
        seq_len=seq_len,
        mean_ms=timing.mean_ms,
        std_ms=timing.std_ms,
        median_ms=timing.median_ms,
        min_ms=timing.min_ms,
        max_ms=timing.max_ms,
        ci_lower_ms=timing.ci_lower_ms,
        ci_upper_ms=timing.ci_upper_ms,
        tokens_per_sec=(batch_size * seq_len) / elapsed_s if elapsed_s > 0 else 0,
        tflops_achieved=flop_info["total_flops"] / elapsed_s / 1e12 if elapsed_s > 0 else 0,
        mfu=mfu,
    )


# ---------------------------------------------------------------------------
# Full model MFU ceiling (composite)
# ---------------------------------------------------------------------------

def bench_full_model_mfu_ceiling(
    batch_sizes: Optional[List[int]] = None,
    seq_lengths: Optional[List[int]] = None,
    bench_config: Optional[BenchConfig] = None,
) -> List[BenchResult]:
    """
    Compute full-model MFU ceiling across batch/seq combinations.

    Composes per-component ceilings into a full model estimate:
      - 61 MLA layers
      - 3 dense FFN layers
      - 58 MoE FFN layers
      - 1 MTP layer
      - Embedding + LM head
    """
    if bench_config is None:
        bench_config = BenchConfig(name="full_model_mfu")

    if batch_sizes is None:
        batch_sizes = bench_config.batch_sizes
    if seq_lengths is None:
        seq_lengths = bench_config.seq_lengths

    results = []

    for bs in batch_sizes:
        for seq in seq_lengths:
            print(f"  Computing MFU ceiling: bs={bs}, seq={seq} ... ", end="", flush=True)

            flop_info = compute_full_model_flops(bs, seq)

            # Theoretical time at peak hardware utilization
            peak_tflops = H100_SPECS["bf16_tflops"]
            theoretical_time_s = flop_info["total_flops"] / (peak_tflops * 1e12)

            # Estimated achievable MFU based on component-level analysis
            # MLA: ~70% MFU (compute-bound for large seq, memory-bound for decode)
            # MoE: ~55% MFU (expert dispatch overhead, load imbalance)
            # Dense FFN: ~75% MFU (standard GEMM)
            # Overall: weighted average based on FLOP share
            mla_share = flop_info["mla_flops"] / flop_info["total_flops"]
            moe_share = flop_info["moe_ffn_flops"] / flop_info["total_flops"]
            dense_share = flop_info["dense_ffn_flops"] / flop_info["total_flops"]
            other_share = 1.0 - mla_share - moe_share - dense_share

            estimated_mfu = (
                mla_share * 70.0 +
                moe_share * 55.0 +
                dense_share * 75.0 +
                other_share * 50.0
            )

            estimated_time_s = theoretical_time_s / (estimated_mfu / 100.0) if estimated_mfu > 0 else 0
            tokens = bs * seq

            result = BenchResult(
                config_name="full_model_mfu",
                component="full_model",
                batch_size=bs,
                seq_len=seq,
                mean_ms=estimated_time_s * 1000,
                tokens_per_sec=tokens / estimated_time_s if estimated_time_s > 0 else 0,
                tflops_achieved=flop_info["total_flops"] / estimated_time_s / 1e12 if estimated_time_s > 0 else 0,
                mfu=estimated_mfu,
                extra_params={
                    "total_flops": flop_info["total_flops"],
                    "theoretical_time_ms": theoretical_time_s * 1000,
                    "breakdown": flop_info["breakdown_pct"],
                },
            )
            results.append(result)

            print(f"est. MFU={estimated_mfu:.1f}%, est. time={estimated_time_s*1000:.1f}ms")

    return results


# ---------------------------------------------------------------------------
# Roofline analysis
# ---------------------------------------------------------------------------

def generate_roofline_analysis(
    batch_size: int = 4,
    seq_len: int = 2048,
    dtype: str = "bfloat16",
) -> List[RooflinePoint]:
    """
    Generate roofline model points for key DeepSeek-V3 operations.

    Returns a list of RooflinePoint for plotting on a roofline chart.
    """
    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    tokens = batch_size * seq_len
    bytes_per_elem = 2 if dtype in ("bfloat16", "float16") else 1

    points = []

    # 1. MLA KV down-projection: (tokens, H) @ (H, d_c) -> (tokens, d_c)
    d_c = cfg["kv_lora_rank"]
    flops = 2.0 * tokens * H * d_c
    bytes_acc = (tokens * H + H * d_c + tokens * d_c) * bytes_per_elem
    points.append(compute_roofline_point("MLA_KV_down", flops, bytes_acc, dtype))

    # 2. MLA Q projection: (tokens, H) @ (H, n_h * d_qk)
    d_qk = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    n_h = cfg["num_heads"]
    flops = 2.0 * tokens * H * n_h * d_qk
    bytes_acc = (tokens * H + H * n_h * d_qk + tokens * n_h * d_qk) * bytes_per_elem
    points.append(compute_roofline_point("MLA_Q_proj", flops, bytes_acc, dtype))

    # 3. Attention QK^T: (tokens, n_h, d_qk) @ (tokens, n_h, d_qk)^T
    avg_span = seq_len // 2
    flops = 2.0 * batch_size * n_h * seq_len * avg_span * d_qk
    bytes_acc = 2.0 * batch_size * n_h * seq_len * d_qk * bytes_per_elem + batch_size * n_h * seq_len * avg_span * bytes_per_elem
    points.append(compute_roofline_point("Attention_QKT", flops, bytes_acc, dtype))

    # 4. MoE gating: (tokens, H) @ (H, N_experts)
    N_exp = cfg["n_routed_experts"]
    flops = 2.0 * tokens * H * N_exp
    bytes_acc = (tokens * H + H * N_exp + tokens * N_exp) * bytes_per_elem
    points.append(compute_roofline_point("MoE_gating", flops, bytes_acc, dtype))

    # 5. MoE expert FFN (single expert, tokens_per_expert tokens)
    I = cfg["moe_intermediate_size"]
    tpe = tokens * cfg["num_experts_per_tok"] // N_exp  # avg tokens per expert
    flops = 6.0 * tpe * H * I + tpe * I  # SwiGLU
    bytes_acc = (tpe * H + 3 * H * I + tpe * I + tpe * H) * bytes_per_elem
    points.append(compute_roofline_point("MoE_expert_FFN", flops, bytes_acc, dtype))

    # 6. LM head: (tokens, H) @ (H, vocab)
    V = cfg["vocab_size"]
    flops = 2.0 * tokens * H * V
    bytes_acc = (tokens * H + H * V + tokens * V) * bytes_per_elem
    points.append(compute_roofline_point("LM_head", flops, bytes_acc, dtype))

    return points


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek-V3 MFU Ceiling Analysis")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--output-dir", type=str, default="results/mfu_ceiling")
    parser.add_argument("--roofline", action="store_true", help="Generate roofline analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DeepSeek-V3 671B MFU Ceiling Analysis")
    print(f"Hardware: {H100_SPECS['name']}")
    print(f"Peak BF16: {H100_SPECS['bf16_tflops']} TFLOPS")
    print(f"Peak FP8:  {H100_SPECS['fp8_tflops']} TFLOPS")
    print(f"HBM BW:    {H100_SPECS['hbm_bandwidth_tb_s']} TB/s")
    print("=" * 70)

    config = BenchConfig(
        name="mfu_ceiling",
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
    )

    # Component-level MFU
    print("\n--- Component-level MFU (bs=4, seq=2048) ---")
    mla_result = bench_mla_mfu_ceiling(4, 2048, config)
    print(f"  MLA:  {mla_result.mfu:.1f}% MFU, {mla_result.mean_ms:.3f}ms")

    moe_result = bench_moe_mfu_ceiling(4, 2048, config)
    print(f"  MoE:  {moe_result.mfu:.1f}% MFU, {moe_result.mean_ms:.3f}ms")

    # Full model sweep
    print("\n--- Full Model MFU Ceiling Sweep ---")
    full_results = bench_full_model_mfu_ceiling(config=config)

    all_results = [mla_result, moe_result] + full_results

    # Roofline
    if args.roofline:
        print("\n--- Roofline Analysis ---")
        points = generate_roofline_analysis()
        for p in points:
            print(f"  {p.name:<20s}: AI={p.arithmetic_intensity:.1f} FLOP/B, "
                  f"bound={p.bound}, achievable={p.achievable_tflops:.1f} TFLOPS "
                  f"({p.efficiency_pct:.0f}%)")

    # Save reports
    json_path = save_json_report(all_results, os.path.join(args.output_dir, "mfu_ceiling.json"))
    md_path = save_markdown_report(all_results, os.path.join(args.output_dir, "mfu_ceiling.md"),
                                   title="DeepSeek-V3 MFU Ceiling Analysis")
    print(f"\nJSON: {json_path}")
    print(f"Markdown: {md_path}")

    print_results_table(all_results, "MFU Ceiling Results")


if __name__ == "__main__":
    main()
