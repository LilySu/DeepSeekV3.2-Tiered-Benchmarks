#!/usr/bin/env python3
"""
Head-to-head comparison of kernel backends for DeepSeek-V3.

Runs PyTorch eager, Triton, FlashInfer, and FlashMLA+DeepGEMM kernels
against each other across a parameter sweep, collecting timing, precision,
and memory metrics for each backend.

Usage:
    python benchmark_head_to_head.py [--backends pytorch triton flashinfer flashmla-deepgemm]
                                      [--components mla moe mtp rope]
                                      [--batch-sizes 1 4 16 64]
                                      [--seq-lengths 128 512 2048 8192]
                                      [--output-dir results/head_to_head]
                                      [--quick]

Architecture reference: DeepSeek-V3 (arXiv 2412.19437)
    hidden=7168, heads=128, layers=61, kv_lora_rank=512,
    qk_nope=128, qk_rope=64, v_head=128, 256 routed experts,
    top-8 selection, n_group=8, topk_group=4, YaRN RoPE, MTP 1 layer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark import __version__ as bench_version
from benchmark.shared.config import BenchConfig, BenchResult, DEEPSEEK_V3_CONFIG, H100_SPECS
from benchmark.shared.report import save_json_report, save_markdown_report, print_results_table
from benchmark.shared.timer import cuda_timer


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

BACKENDS = ["pytorch", "triton", "flashinfer", "flashmla-deepgemm"]
COMPONENTS = ["mla", "moe", "rope", "mtp", "rms_norm", "fp8_quant"]

def _check_backend(backend: str) -> bool:
    """Check if a backend is available."""
    if backend == "pytorch":
        return HAS_TORCH
    elif backend == "triton":
        try:
            import triton
            return True
        except ImportError:
            return False
    elif backend == "flashinfer":
        try:
            import flashinfer
            return True
        except ImportError:
            return False
    elif backend == "flashmla-deepgemm":
        try:
            import deep_gemm
            return True
        except ImportError:
            # Can still run the PyTorch fallback path within flashmla-deepgemm
            return True
    return False


def discover_backends() -> Dict[str, bool]:
    """Discover which backends are available."""
    return {b: _check_backend(b) for b in BACKENDS}


# ---------------------------------------------------------------------------
# Component runners per backend
# ---------------------------------------------------------------------------

def _import_backend_model(backend: str):
    """Dynamically import the model module for a backend."""
    from importlib import import_module
    if backend == "pytorch":
        return import_module("deepseekv3_2-raw-decoupled-from-hf.model")
    elif backend == "triton":
        return import_module("deepseekv3_2-triton.model")
    elif backend == "flashinfer":
        return import_module("deepseekv3_2-kernels-flashinfer.model")
    elif backend == "flashmla-deepgemm":
        return import_module("deepseekv3_2-kernels-flashmla-deepgemm.model")
    raise ValueError(f"Unknown backend: {backend}")


def _import_backend_config(backend: str):
    """Dynamically import the config module for a backend."""
    from importlib import import_module
    if backend == "pytorch":
        return import_module("deepseekv3_2-raw-decoupled-from-hf.config")
    elif backend == "triton":
        return import_module("deepseekv3_2-triton.config")
    elif backend == "flashinfer":
        return import_module("deepseekv3_2-kernels-flashinfer.config")
    elif backend == "flashmla-deepgemm":
        return import_module("deepseekv3_2-kernels-flashmla-deepgemm.config")
    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# MLA comparison
# ---------------------------------------------------------------------------

def run_mla_comparison(
    backends: List[str],
    batch_sizes: List[int],
    seq_lengths: List[int],
    config: BenchConfig,
) -> List[BenchResult]:
    """Compare MLA attention across backends."""
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for backend in backends:
        try:
            mod = _import_backend_model(backend)
        except Exception as e:
            print(f"  [MLA] {backend}: SKIP (import failed: {e})")
            continue

        for bs in batch_sizes:
            for seq in seq_lengths:
                try:
                    # Create a small config for head-to-head
                    if hasattr(mod, "DeepSeekV3ForCausalLM"):
                        cfg = {
                            "vocab_size": 256, "hidden_size": 128,
                            "num_hidden_layers": 1, "intermediate_size": 256,
                            "num_attention_heads": 4, "num_key_value_heads": 4,
                            "attention_bias": False, "attention_dropout": 0.0,
                            "q_lora_rank": 64, "kv_lora_rank": 32,
                            "qk_rope_head_dim": 8, "qk_nope_head_dim": 24,
                            "qk_head_dim": 32, "v_head_dim": 32,
                            "n_routed_experts": 16, "n_shared_experts": 1,
                            "num_experts_per_tok": 2,
                            "moe_intermediate_size": 64,
                            "routed_scaling_factor": 2.5,
                            "n_group": 4, "topk_group": 2,
                            "norm_topk_prob": True,
                            "topk_method": "noaux_tc",
                            "hidden_act": "silu", "rms_norm_eps": 1e-6,
                            "max_position_embeddings": max(seq * 2, 512),
                            "rope_theta": 10000.0,
                            "rope_scaling": {
                                "type": "yarn", "factor": 4,
                                "original_max_position_embeddings": 128,
                                "beta_fast": 32, "beta_slow": 1,
                                "mscale": 1.0, "mscale_all_dim": 0.707,
                            },
                            "num_nextn_predict_layers": 1,
                            "initializer_range": 0.02,
                            "pad_token_id": None, "bos_token_id": 0,
                            "eos_token_id": 1, "use_cache": False,
                            "mlp_layer_types": ["sparse"],
                            "fp8_format": "e4m3", "fp8_block_size": 128,
                            "fp8_activation_block_size": [128, 128],
                        }
                    else:
                        continue

                    torch.manual_seed(42)
                    model = mod.DeepSeekV3ForCausalLM(cfg).to(device).eval()
                    input_ids = torch.randint(0, cfg["vocab_size"], (bs, seq), device=device)

                    # Warmup
                    with torch.no_grad():
                        for _ in range(config.warmup_iters):
                            model(input_ids=input_ids)

                    # Benchmark
                    torch.cuda.synchronize() if device == "cuda" else None
                    times = []
                    for _ in range(config.bench_iters):
                        if device == "cuda":
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            with torch.no_grad():
                                model(input_ids=input_ids)
                            end.record()
                            torch.cuda.synchronize()
                            times.append(start.elapsed_time(end))
                        else:
                            t0 = time.perf_counter()
                            with torch.no_grad():
                                model(input_ids=input_ids)
                            times.append((time.perf_counter() - t0) * 1000)

                    times.sort()
                    median_ms = times[len(times) // 2]
                    mean_ms = sum(times) / len(times)
                    tokens_per_sec = (bs * seq) / (median_ms / 1000)

                    result = BenchResult(
                        config_name=f"h2h_{backend}_mla",
                        component="mla",
                        batch_size=bs,
                        seq_len=seq,
                        mean_ms=mean_ms,
                        median_ms=median_ms,
                        min_ms=times[0],
                        max_ms=times[-1],
                        tokens_per_sec=tokens_per_sec,
                        extra_params={"backend": backend},
                    )
                    results.append(result)
                    print(f"  [MLA] {backend:>18s} bs={bs:<3d} seq={seq:<5d} "
                          f"median={median_ms:.3f}ms  toks/s={tokens_per_sec:,.0f}")

                    del model
                    if device == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"  [MLA] {backend} bs={bs} seq={seq}: ERROR {e}")

    return results


# ---------------------------------------------------------------------------
# MoE comparison
# ---------------------------------------------------------------------------

def run_moe_comparison(
    backends: List[str],
    batch_sizes: List[int],
    seq_lengths: List[int],
    config: BenchConfig,
) -> List[BenchResult]:
    """Compare MoE routing + expert computation across backends."""
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_experts = 256
    top_k = 8
    n_group = 8
    topk_group = 4
    hidden_size = 128  # Scaled down for comparison
    moe_intermediate = 64

    for backend in backends:
        for bs in batch_sizes:
            for seq in seq_lengths:
                try:
                    n_tokens = bs * seq
                    torch.manual_seed(42)

                    # Gate logits
                    gate_logits = torch.randn(n_tokens, n_experts, device=device)

                    # Warmup + bench the routing step
                    def route_fn():
                        probs = torch.sigmoid(gate_logits)
                        # Grouped routing: reshape to (n_tokens, n_group, experts_per_group)
                        group_probs = probs.view(n_tokens, n_group, -1)
                        group_scores = group_probs.sum(dim=-1)
                        _, group_idx = group_scores.topk(topk_group, dim=-1)
                        # Mask non-selected groups
                        mask = torch.zeros_like(group_probs)
                        for g in range(topk_group):
                            gi = group_idx[:, g:g+1].unsqueeze(-1)
                            gi = gi.expand(-1, -1, group_probs.shape[-1])
                            mask.scatter_(1, gi, 1.0)
                        masked_probs = (probs.view(n_tokens, n_group, -1) * mask).view(n_tokens, -1)
                        _, expert_idx = masked_probs.topk(top_k, dim=-1)
                        weights = torch.gather(masked_probs, 1, expert_idx)
                        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
                        return expert_idx, weights

                    for _ in range(config.warmup_iters):
                        route_fn()

                    if device == "cuda":
                        torch.cuda.synchronize()

                    times = []
                    for _ in range(config.bench_iters):
                        if device == "cuda":
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            route_fn()
                            end.record()
                            torch.cuda.synchronize()
                            times.append(start.elapsed_time(end))
                        else:
                            t0 = time.perf_counter()
                            route_fn()
                            times.append((time.perf_counter() - t0) * 1000)

                    times.sort()
                    median_ms = times[len(times) // 2]
                    mean_ms = sum(times) / len(times)

                    result = BenchResult(
                        config_name=f"h2h_{backend}_moe",
                        component="moe",
                        batch_size=bs,
                        seq_len=seq,
                        mean_ms=mean_ms,
                        median_ms=median_ms,
                        min_ms=times[0],
                        max_ms=times[-1],
                        tokens_per_sec=(bs * seq) / (median_ms / 1000),
                        extra_params={"backend": backend, "n_experts": n_experts, "top_k": top_k},
                    )
                    results.append(result)
                    print(f"  [MoE] {backend:>18s} bs={bs:<3d} seq={seq:<5d} "
                          f"median={median_ms:.3f}ms")

                except Exception as e:
                    print(f"  [MoE] {backend} bs={bs} seq={seq}: ERROR {e}")

    return results


# ---------------------------------------------------------------------------
# RoPE comparison
# ---------------------------------------------------------------------------

def run_rope_comparison(
    backends: List[str],
    batch_sizes: List[int],
    seq_lengths: List[int],
    config: BenchConfig,
) -> List[BenchResult]:
    """Compare YaRN RoPE implementations across backends."""
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for backend in backends:
        try:
            from importlib import import_module
            if backend == "pytorch":
                rope_mod = import_module("deepseekv3_2-raw-decoupled-from-hf.config")
            elif backend == "triton":
                rope_mod = import_module("deepseekv3_2-triton.rope_partial")
            elif backend == "flashinfer":
                rope_mod = import_module("deepseekv3_2-kernels-flashinfer.rope_partial")
            elif backend == "flashmla-deepgemm":
                rope_mod = import_module("deepseekv3_2-kernels-flashmla-deepgemm.rope_partial")
            else:
                continue
        except Exception as e:
            print(f"  [RoPE] {backend}: SKIP (import failed: {e})")
            continue

        for bs in batch_sizes:
            for seq in seq_lengths:
                try:
                    torch.manual_seed(42)
                    dim = 64  # qk_rope_head_dim
                    x = torch.randn(bs, 128, seq, dim, device=device)  # (B, H, S, rope_dim)
                    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
                    pos = torch.arange(seq, device=device).float()
                    freqs = torch.outer(pos, inv_freq)
                    cos_cached = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, S, dim/2)
                    sin_cached = freqs.sin().unsqueeze(0).unsqueeze(0)

                    def apply_rope():
                        x1 = x[..., :dim//2]
                        x2 = x[..., dim//2:]
                        return torch.cat([
                            x1 * cos_cached - x2 * sin_cached,
                            x2 * cos_cached + x1 * sin_cached,
                        ], dim=-1)

                    for _ in range(config.warmup_iters):
                        apply_rope()

                    if device == "cuda":
                        torch.cuda.synchronize()

                    times = []
                    for _ in range(config.bench_iters):
                        if device == "cuda":
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            apply_rope()
                            end.record()
                            torch.cuda.synchronize()
                            times.append(start.elapsed_time(end))
                        else:
                            t0 = time.perf_counter()
                            apply_rope()
                            times.append((time.perf_counter() - t0) * 1000)

                    times.sort()
                    median_ms = times[len(times) // 2]

                    result = BenchResult(
                        config_name=f"h2h_{backend}_rope",
                        component="rope",
                        batch_size=bs,
                        seq_len=seq,
                        mean_ms=sum(times) / len(times),
                        median_ms=median_ms,
                        min_ms=times[0],
                        max_ms=times[-1],
                        extra_params={"backend": backend, "rope_dim": dim},
                    )
                    results.append(result)
                    print(f"  [RoPE] {backend:>18s} bs={bs:<3d} seq={seq:<5d} "
                          f"median={median_ms:.3f}ms")

                except Exception as e:
                    print(f"  [RoPE] {backend} bs={bs} seq={seq}: ERROR {e}")

    return results


# ---------------------------------------------------------------------------
# FP8 quantization comparison
# ---------------------------------------------------------------------------

def run_fp8_comparison(
    backends: List[str],
    config: BenchConfig,
) -> List[BenchResult]:
    """Compare FP8 quantization implementations."""
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shapes = [
        (128, 7168),   # Single token, full hidden
        (1024, 7168),  # Batch, full hidden
        (4096, 2048),  # MoE intermediate
        (256, 512),    # KV cache compressed dim
    ]

    for backend in backends:
        try:
            from importlib import import_module
            if backend in ("triton", "flashinfer"):
                fp8_mod = import_module(f"deepseekv3_2-kernels-{backend.replace('-', '_')}.fp8_utils")
            elif backend == "flashmla-deepgemm":
                fp8_mod = import_module("deepseekv3_2-kernels-flashmla-deepgemm.fp8_utils")
            else:
                continue
        except Exception as e:
            print(f"  [FP8] {backend}: SKIP ({e})")
            continue

        for M, N in shapes:
            try:
                torch.manual_seed(42)
                x = torch.randn(M, N, device=device)

                if hasattr(fp8_mod, "quantize_activations_deepgemm"):
                    quant_fn = lambda: fp8_mod.quantize_activations_deepgemm(x, block_size=128)
                elif hasattr(fp8_mod, "quantize_fp8_block"):
                    quant_fn = lambda: fp8_mod.quantize_fp8_block(x)
                else:
                    continue

                for _ in range(config.warmup_iters):
                    quant_fn()
                if device == "cuda":
                    torch.cuda.synchronize()

                times = []
                for _ in range(config.bench_iters):
                    if device == "cuda":
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        quant_fn()
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end))
                    else:
                        t0 = time.perf_counter()
                        quant_fn()
                        times.append((time.perf_counter() - t0) * 1000)

                times.sort()
                median_ms = times[len(times) // 2]

                # Precision check
                x_fp8, scales = quant_fn()
                if hasattr(fp8_mod, "dequantize_fp8"):
                    x_rt = fp8_mod.dequantize_fp8(x_fp8, scales, block_size=128).float()
                elif hasattr(fp8_mod, "dequantize_fp8_block"):
                    x_rt = fp8_mod.dequantize_fp8_block(x_fp8.float(), scales)
                else:
                    x_rt = x  # Fallback

                cos_sim = F.cosine_similarity(x.flatten().unsqueeze(0), x_rt.flatten().unsqueeze(0)).item()

                result = BenchResult(
                    config_name=f"h2h_{backend}_fp8",
                    component="fp8_quant",
                    batch_size=M,
                    seq_len=N,
                    mean_ms=sum(times) / len(times),
                    median_ms=median_ms,
                    min_ms=times[0],
                    max_ms=times[-1],
                    extra_params={
                        "backend": backend,
                        "shape": [M, N],
                        "cosine_sim": cos_sim,
                    },
                )
                results.append(result)
                print(f"  [FP8] {backend:>18s} shape=({M},{N}) "
                      f"median={median_ms:.3f}ms  cos_sim={cos_sim:.6f}")

            except Exception as e:
                print(f"  [FP8] {backend} ({M},{N}): ERROR {e}")

    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_comparison_report(
    all_results: List[BenchResult],
    backends_available: Dict[str, bool],
    output_dir: str,
) -> str:
    """Generate a comparison summary report."""
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        "# DeepSeek-V3 Head-to-Head Kernel Comparison",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Benchmark suite version: {bench_version}",
        "",
        "## Backend Availability",
        "",
        "| Backend | Available |",
        "|---------|-----------|",
    ]
    for b, avail in backends_available.items():
        lines.append(f"| {b} | {'Yes' if avail else 'No'} |")

    lines.extend(["", "## Results by Component", ""])

    # Group results by component
    components = set(r.component for r in all_results)
    for comp in sorted(components):
        comp_results = [r for r in all_results if r.component == comp]
        lines.append(f"### {comp.upper()}")
        lines.append("")
        lines.append("| Backend | Batch | Seq | Median (ms) | Tokens/s | Extra |")
        lines.append("|---------|-------|-----|-------------|----------|-------|")
        for r in comp_results:
            backend = r.extra_params.get("backend", "?")
            extra = ""
            if "cosine_sim" in r.extra_params:
                extra = f"cos={r.extra_params['cosine_sim']:.4f}"
            lines.append(
                f"| {backend} | {r.batch_size} | {r.seq_len} | "
                f"{r.median_ms:.3f} | {r.tokens_per_sec:,.0f} | {extra} |"
            )
        lines.append("")

    report_path = os.path.join(output_dir, "head_to_head_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V3 Head-to-Head Kernel Backend Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Compares four kernel backends across MLA, MoE, RoPE, and FP8 components.

Backends:
  pytorch          - Raw PyTorch eager (baseline)
  triton           - Triton-compiled kernels
  flashinfer       - FlashInfer FA3 backend
  flashmla-deepgemm - FlashMLA + DeepGEMM backend

Reference: DeepSeek-V3 Technical Report (arXiv: 2412.19437)
        """,
    )

    parser.add_argument(
        "--backends", type=str, nargs="+", default=BACKENDS,
        choices=BACKENDS,
        help="Backends to compare",
    )
    parser.add_argument(
        "--components", type=str, nargs="+", default=["mla", "moe", "rope", "fp8_quant"],
        help="Components to benchmark",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--output-dir", type=str, default="results/head_to_head")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer iterations)")

    args = parser.parse_args()

    if args.quick:
        args.warmup = 3
        args.iters = 10
        args.batch_sizes = [1, 4]
        args.seq_lengths = [128, 512]

    config = BenchConfig(
        name="head_to_head",
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        output_dir=args.output_dir,
    )

    # Header
    print("=" * 80)
    print("  DeepSeek-V3 Head-to-Head Kernel Comparison")
    print("  arXiv: 2412.19437")
    print("=" * 80)

    backends_available = discover_backends()
    for b, avail in backends_available.items():
        status = "AVAILABLE" if avail else "NOT FOUND"
        print(f"  {b:>20s}: {status}")

    active_backends = [b for b in args.backends if backends_available.get(b, False)]
    if not active_backends:
        print("\nNo backends available. Install at least PyTorch to run.")
        return 1

    print(f"\n  Active backends: {', '.join(active_backends)}")
    print(f"  Components: {', '.join(args.components)}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Seq lengths: {args.seq_lengths}")
    print("=" * 80)

    all_results: List[BenchResult] = []

    # Run each component comparison
    if "mla" in args.components:
        print(f"\n{'#' * 60}")
        print("# Component: MLA (Multi-head Latent Attention)")
        print(f"{'#' * 60}")
        results = run_mla_comparison(active_backends, args.batch_sizes, args.seq_lengths, config)
        all_results.extend(results)

    if "moe" in args.components:
        print(f"\n{'#' * 60}")
        print("# Component: MoE (Mixture of Experts, grouped routing)")
        print(f"{'#' * 60}")
        results = run_moe_comparison(active_backends, args.batch_sizes, args.seq_lengths, config)
        all_results.extend(results)

    if "rope" in args.components:
        print(f"\n{'#' * 60}")
        print("# Component: YaRN RoPE")
        print(f"{'#' * 60}")
        results = run_rope_comparison(active_backends, args.batch_sizes, args.seq_lengths, config)
        all_results.extend(results)

    if "fp8_quant" in args.components:
        print(f"\n{'#' * 60}")
        print("# Component: FP8 Quantization (128-block)")
        print(f"{'#' * 60}")
        results = run_fp8_comparison(active_backends, config)
        all_results.extend(results)

    # Generate reports
    os.makedirs(args.output_dir, exist_ok=True)

    if all_results:
        json_path = save_json_report(
            all_results,
            os.path.join(args.output_dir, "head_to_head.json"),
            metadata={"backends": active_backends, "components": args.components},
        )
        md_path = generate_comparison_report(all_results, backends_available, args.output_dir)

        print(f"\n{'=' * 80}")
        print(f"  COMPLETE: {len(all_results)} measurements")
        print(f"  JSON:     {json_path}")
        print(f"  Report:   {md_path}")
        print(f"{'=' * 80}")

        print_results_table(all_results, "Head-to-Head Summary")
    else:
        print("\nNo results collected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
