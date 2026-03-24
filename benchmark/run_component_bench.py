#!/usr/bin/env python3
"""
Quick component-level benchmark runner for DeepSeek-V3.

Runs only the component-level benchmarks (MLA, MoE, Dense FFN)
with configurable batch sizes and sequence lengths.

This is the recommended entry point for quick profiling of
individual DeepSeek-V3 architectural components.

Usage:
    python -m benchmark.run_component_bench --batch-sizes 1 4 --seq-lengths 512 2048
"""

from __future__ import annotations

import argparse
import os
import sys

from benchmark.shared.config import BenchConfig, DEEPSEEK_V3_CONFIG
from benchmark.shared.report import save_json_report, save_markdown_report, print_results_table
from benchmark.triple_report.bench_component import run_all_component_benchmarks


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V3 Component Benchmark Runner")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/component")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 mode")
    args = parser.parse_args()

    config = BenchConfig(
        name="component_bench",
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        use_fp8=args.fp8,
        dtype="float8_e4m3fn" if args.fp8 else "bfloat16",
    )

    print("=" * 70)
    print("DeepSeek-V3 Component Benchmark")
    print("=" * 70)
    print(f"  hidden_size:     {DEEPSEEK_V3_CONFIG['hidden_size']}")
    print(f"  num_heads:       {DEEPSEEK_V3_CONFIG['num_heads']}")
    print(f"  kv_lora_rank:    {DEEPSEEK_V3_CONFIG['kv_lora_rank']}")
    print(f"  n_experts:       {DEEPSEEK_V3_CONFIG['n_routed_experts']}")
    print(f"  top_k:           {DEEPSEEK_V3_CONFIG['num_experts_per_tok']}")
    print(f"  n_group:         {DEEPSEEK_V3_CONFIG['n_group']}")
    print(f"  topk_group:      {DEEPSEEK_V3_CONFIG['topk_group']}")
    print(f"  Precision:       {config.dtype}")
    print(f"  Batch sizes:     {args.batch_sizes}")
    print(f"  Seq lengths:     {args.seq_lengths}")
    print("=" * 70)

    results = run_all_component_benchmarks(
        bench_config=config,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
    )

    if results:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = save_json_report(results, os.path.join(args.output_dir, "component.json"))
        md_path = save_markdown_report(
            results,
            os.path.join(args.output_dir, "component.md"),
            title="DeepSeek-V3 Component Benchmarks",
        )
        print(f"\nJSON:     {json_path}")
        print(f"Markdown: {md_path}")
        print_results_table(results, "Component Results")
    else:
        print("\nNo results produced. Check PyTorch/CUDA availability.")


if __name__ == "__main__":
    main()
