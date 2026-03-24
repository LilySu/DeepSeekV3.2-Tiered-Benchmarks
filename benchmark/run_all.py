#!/usr/bin/env python3
"""
Master benchmark runner for DeepSeek-V3.

Orchestrates all benchmark sub-packages and produces a unified report.

Usage:
    python -m benchmark.run_all [--suite {micro,component,e2e,moe,fp8,mfu,all}]
                                [--output-dir results/]
                                [--quick]  # reduced iterations for CI

Reference: DeepSeek-V3 Technical Report (arXiv: 2412.19437)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
from typing import List, Optional

from benchmark.shared.config import BenchConfig, BenchResult, DEEPSEEK_V3_CONFIG, H100_SPECS
from benchmark.shared.report import save_json_report, save_markdown_report, print_results_table


def run_micro_benchmarks(config: BenchConfig) -> List[BenchResult]:
    """Run micro-level benchmarks (individual kernels)."""
    from benchmark.triple_report.bench_micro import run_all_micro_benchmarks
    return run_all_micro_benchmarks(config)


def run_component_benchmarks(config: BenchConfig) -> List[BenchResult]:
    """Run component-level benchmarks (MLA, MoE, FFN layers)."""
    from benchmark.triple_report.bench_component import run_all_component_benchmarks
    return run_all_component_benchmarks(config)


def run_e2e_benchmarks(config: BenchConfig) -> List[BenchResult]:
    """Run end-to-end benchmarks (full model forward pass)."""
    from benchmark.triple_report.bench_e2e import run_e2e_sweep
    return run_e2e_sweep(config)


def run_moe_sweep(config: BenchConfig) -> List[BenchResult]:
    """Run MoE parameter sweep benchmarks."""
    from benchmark.moe_sweep.bench_moe import run_moe_sweep
    return run_moe_sweep(bench_config=config)


def run_fp8_benchmarks(config: BenchConfig) -> List[BenchResult]:
    """Run FP8 Pareto frontier benchmarks."""
    from benchmark.fp8_pareto.bench_fp8 import run_fp8_pareto_sweep
    return run_fp8_pareto_sweep(config)


def run_mfu_analysis(config: BenchConfig) -> List[BenchResult]:
    """Run MFU ceiling analysis."""
    from benchmark.mfu_ceiling.bench_mfu import bench_full_model_mfu_ceiling
    return bench_full_model_mfu_ceiling(config=config)


SUITE_MAP = {
    "micro": run_micro_benchmarks,
    "component": run_component_benchmarks,
    "e2e": run_e2e_benchmarks,
    "moe": run_moe_sweep,
    "fp8": run_fp8_benchmarks,
    "mfu": run_mfu_analysis,
}


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V3 Master Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark suites:
  micro      - Individual kernel/op benchmarks (GEMM, softmax, RoPE, etc.)
  component  - Full layer benchmarks (MLA, MoE FFN, Dense FFN)
  e2e        - End-to-end model forward pass
  moe        - MoE parameter sweep (experts, top-K, groups)
  fp8        - FP8 precision-performance Pareto frontier
  mfu        - MFU ceiling analysis and roofline model
  all        - Run all suites
        """,
    )

    parser.add_argument(
        "--suite", type=str, nargs="+",
        default=["all"],
        choices=list(SUITE_MAP.keys()) + ["all"],
        help="Which benchmark suite(s) to run",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer iterations)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 2048])

    args = parser.parse_args()

    if args.quick:
        args.warmup = 3
        args.iters = 10

    config = BenchConfig(
        name="master_run",
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        output_dir=args.output_dir,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Header
    print("=" * 80)
    print("  DeepSeek-V3 671B Benchmark Suite")
    print("  Reference: arXiv:2412.19437")
    print("=" * 80)
    print(f"  Hardware target: {H100_SPECS['name']}")
    print(f"  Model: {DEEPSEEK_V3_CONFIG['num_layers']} layers "
          f"({DEEPSEEK_V3_CONFIG['num_dense_layers']} dense + {DEEPSEEK_V3_CONFIG['num_moe_layers']} MoE)")
    print(f"  Hidden: {DEEPSEEK_V3_CONFIG['hidden_size']}, Heads: {DEEPSEEK_V3_CONFIG['num_heads']}")
    print(f"  Experts: {DEEPSEEK_V3_CONFIG['n_routed_experts']}, Top-K: {DEEPSEEK_V3_CONFIG['num_experts_per_tok']}")
    print(f"  Groups: {DEEPSEEK_V3_CONFIG['n_group']}, TopK Groups: {DEEPSEEK_V3_CONFIG['topk_group']}")
    print(f"  Warmup: {config.warmup_iters}, Iterations: {config.bench_iters}")
    print("=" * 80)

    suites = list(SUITE_MAP.keys()) if "all" in args.suite else args.suite
    all_results: List[BenchResult] = []
    suite_timings = {}

    for suite_name in suites:
        print(f"\n{'#' * 70}")
        print(f"# Suite: {suite_name.upper()}")
        print(f"{'#' * 70}")

        t0 = time.time()
        try:
            results = SUITE_MAP[suite_name](config)
            elapsed = time.time() - t0
            suite_timings[suite_name] = elapsed

            # Save per-suite results
            if results:
                suite_dir = os.path.join(args.output_dir, suite_name)
                os.makedirs(suite_dir, exist_ok=True)
                save_json_report(results, os.path.join(suite_dir, f"{suite_name}.json"))
                save_markdown_report(
                    results,
                    os.path.join(suite_dir, f"{suite_name}.md"),
                    title=f"DeepSeek-V3 {suite_name.upper()} Benchmarks",
                )
                all_results.extend(results)

            print(f"\n  Suite {suite_name}: {len(results)} results in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            suite_timings[suite_name] = elapsed
            print(f"\n  Suite {suite_name} FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

    # Combined report
    if all_results:
        combined_json = save_json_report(
            all_results,
            os.path.join(args.output_dir, "all_benchmarks.json"),
            metadata={"suites_run": suites, "suite_timings_s": suite_timings},
        )
        combined_md = save_markdown_report(
            all_results,
            os.path.join(args.output_dir, "all_benchmarks.md"),
            title="DeepSeek-V3 Complete Benchmark Report",
            metadata={"suites_run": ", ".join(suites)},
        )

        print(f"\n{'=' * 80}")
        print(f"  COMPLETE: {len(all_results)} total results across {len(suites)} suites")
        print(f"  JSON:     {combined_json}")
        print(f"  Markdown: {combined_md}")
        print(f"{'=' * 80}")

        print_results_table(all_results, "All Results Summary")


if __name__ == "__main__":
    main()
