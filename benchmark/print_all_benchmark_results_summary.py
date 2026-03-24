#!/usr/bin/env python3
"""
Print all DeepSeek-V3 benchmark results in a comprehensive summary table.

Scans the results directory for all JSON benchmark files and produces
a formatted console output with tables grouped by component.

Usage:
    python -m benchmark.print_all_benchmark_results_summary
    python -m benchmark.print_all_benchmark_results_summary --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from benchmark.extract_summary import load_results, find_json_files, extract_summary, print_summary


def print_detailed_table(results: List[Dict[str, Any]], title: str = "All Results") -> None:
    """Print a detailed results table to stdout."""
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"  DeepSeek-V3 671B | Reference: arXiv:2412.19437")
    print(f"{'='*110}")

    # Group by component
    components: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        comp = r.get("component", "unknown")
        components.setdefault(comp, []).append(r)

    for comp_name in sorted(components.keys()):
        comp_results = components[comp_name]
        print(f"\n  --- {comp_name.upper()} ({len(comp_results)} results) ---")
        print(f"  {'Config':<35s} {'BS':>4s} {'Seq':>5s} {'Mean ms':>9s} "
              f"{'Std ms':>7s} {'CI95 low':>8s} {'CI95 hi':>8s} "
              f"{'MFU%':>6s} {'TFLOPS':>7s} {'Tok/s':>12s} {'Mem MB':>8s}")
        print(f"  {'-'*35} {'-'*4} {'-'*5} {'-'*9} {'-'*7} {'-'*8} {'-'*8} "
              f"{'-'*6} {'-'*7} {'-'*12} {'-'*8}")

        for r in sorted(comp_results, key=lambda x: (x.get("batch_size", 0), x.get("seq_len", 0))):
            config = r.get("config_name", "")[:35]
            bs = r.get("batch_size", 0)
            seq = r.get("seq_len", 0)
            mean = r.get("mean_ms", 0)
            std = r.get("std_ms", 0)
            ci_lo = r.get("ci_lower_ms", 0)
            ci_hi = r.get("ci_upper_ms", 0)
            mfu = r.get("mfu", 0)
            tflops = r.get("tflops_achieved", 0)
            tps = r.get("tokens_per_sec", 0)
            mem = r.get("peak_memory_mb", 0)

            print(f"  {config:<35s} {bs:4d} {seq:5d} {mean:9.3f} "
                  f"{std:7.3f} {ci_lo:8.3f} {ci_hi:8.3f} "
                  f"{mfu:6.1f} {tflops:7.1f} {tps:12,.0f} {mem:8.1f}")

    print(f"\n{'='*110}")


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print a comparison table showing component performance side-by-side."""
    print(f"\n{'='*80}")
    print(f"  Component Performance Comparison")
    print(f"{'='*80}")

    # Find common (bs, seq) pairs
    configs = set()
    for r in results:
        bs = r.get("batch_size", 0)
        seq = r.get("seq_len", 0)
        if bs > 0 and seq > 0:
            configs.add((bs, seq))

    components = sorted(set(r.get("component", "unknown") for r in results))

    print(f"\n  {'BS x Seq':<12s}", end="")
    for comp in components:
        print(f"  {comp:>12s}", end="")
    print("  (mean ms)")
    print(f"  {'-'*12}", end="")
    for _ in components:
        print(f"  {'-'*12}", end="")
    print()

    for bs, seq in sorted(configs):
        print(f"  {bs:3d} x {seq:<5d} ", end="")
        for comp in components:
            matching = [r for r in results
                       if r.get("component") == comp
                       and r.get("batch_size") == bs
                       and r.get("seq_len") == seq]
            if matching:
                mean = matching[0].get("mean_ms", 0)
                print(f"  {mean:12.3f}", end="")
            else:
                print(f"  {'---':>12s}", end="")
        print()

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Print all DeepSeek-V3 benchmark results")
    parser.add_argument("--results-dir", type=str, default="results/",
                        help="Directory with benchmark results")
    parser.add_argument("--comparison", action="store_true",
                        help="Print component comparison table")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print only summary statistics")
    args = parser.parse_args()

    # Load results
    all_results = []
    if os.path.isdir(args.results_dir):
        json_files = find_json_files(args.results_dir, recursive=True)
        if not json_files:
            print(f"No JSON result files found in {args.results_dir}")
            sys.exit(1)

        print(f"Found {len(json_files)} result file(s) in {args.results_dir}")
        for jf in json_files:
            try:
                results = load_results(jf)
                all_results.extend(results)
                print(f"  Loaded {len(results)} results from {jf}")
            except Exception as e:
                print(f"  Warning: {jf}: {e}", file=sys.stderr)
    else:
        print(f"Results directory not found: {args.results_dir}")
        print("Run benchmarks first: python -m benchmark.run_all")
        sys.exit(1)

    if not all_results:
        print("No results loaded.")
        sys.exit(1)

    # Print summary
    summary = extract_summary(all_results)
    print_summary(summary)

    if not args.summary_only:
        print_detailed_table(all_results)

    if args.comparison:
        print_comparison_table(all_results)


if __name__ == "__main__":
    main()
