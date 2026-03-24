#!/usr/bin/env python3
"""
Extract summary statistics from DeepSeek-V3 benchmark results.

Reads JSON result files and produces condensed summaries suitable for
inclusion in papers, presentations, or CI dashboards.

Usage:
    python -m benchmark.extract_summary results/all_benchmarks.json
    python -m benchmark.extract_summary results/ --recursive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_results(path: str) -> List[Dict[str, Any]]:
    """Load benchmark results from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "results" in data:
        return data["results"]
    else:
        return [data]


def find_json_files(directory: str, recursive: bool = True) -> List[str]:
    """Find all JSON result files in a directory."""
    files = []
    pattern = "**/*.json" if recursive else "*.json"
    for p in Path(directory).glob(pattern):
        if p.is_file() and not p.name.startswith("."):
            files.append(str(p))
    return sorted(files)


def extract_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract summary statistics from benchmark results.

    Returns a dict with:
      - per_component: stats grouped by component name
      - overall: aggregate statistics
      - best: best result for each metric
    """
    if not results:
        return {"error": "No results to summarize"}

    # Group by component
    components: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        comp = r.get("component", "unknown")
        components.setdefault(comp, []).append(r)

    per_component = {}
    for comp, comp_results in components.items():
        means = [r.get("mean_ms", 0) for r in comp_results if r.get("mean_ms", 0) > 0]
        mfus = [r.get("mfu", 0) for r in comp_results if r.get("mfu", 0) > 0]
        tps = [r.get("tokens_per_sec", 0) for r in comp_results if r.get("tokens_per_sec", 0) > 0]
        mems = [r.get("peak_memory_mb", 0) for r in comp_results if r.get("peak_memory_mb", 0) > 0]

        per_component[comp] = {
            "count": len(comp_results),
            "latency_ms": {
                "min": min(means) if means else 0,
                "max": max(means) if means else 0,
                "mean": sum(means) / len(means) if means else 0,
            },
            "mfu_pct": {
                "min": min(mfus) if mfus else 0,
                "max": max(mfus) if mfus else 0,
                "mean": sum(mfus) / len(mfus) if mfus else 0,
            },
            "throughput_tok_s": {
                "min": min(tps) if tps else 0,
                "max": max(tps) if tps else 0,
            },
            "peak_memory_mb": max(mems) if mems else 0,
        }

    # Overall
    all_mfus = [r.get("mfu", 0) for r in results if r.get("mfu", 0) > 0]
    all_tps = [r.get("tokens_per_sec", 0) for r in results if r.get("tokens_per_sec", 0) > 0]

    overall = {
        "total_benchmarks": len(results),
        "components": list(components.keys()),
        "best_mfu": max(all_mfus) if all_mfus else 0,
        "mean_mfu": sum(all_mfus) / len(all_mfus) if all_mfus else 0,
        "best_throughput": max(all_tps) if all_tps else 0,
    }

    # Best results
    best = {}
    if all_mfus:
        best_mfu_r = max(results, key=lambda r: r.get("mfu", 0))
        best["mfu"] = {
            "value": best_mfu_r.get("mfu", 0),
            "config": best_mfu_r.get("config_name", ""),
            "component": best_mfu_r.get("component", ""),
        }
    if all_tps:
        best_tp_r = max(results, key=lambda r: r.get("tokens_per_sec", 0))
        best["throughput"] = {
            "value": best_tp_r.get("tokens_per_sec", 0),
            "config": best_tp_r.get("config_name", ""),
            "component": best_tp_r.get("component", ""),
        }

    return {
        "per_component": per_component,
        "overall": overall,
        "best": best,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print a benchmark summary."""
    overall = summary.get("overall", {})
    print(f"\n{'='*60}")
    print(f"  DeepSeek-V3 Benchmark Summary")
    print(f"{'='*60}")
    print(f"  Total benchmarks: {overall.get('total_benchmarks', 0)}")
    print(f"  Components: {', '.join(overall.get('components', []))}")
    print(f"  Best MFU: {overall.get('best_mfu', 0):.1f}%")
    print(f"  Mean MFU: {overall.get('mean_mfu', 0):.1f}%")
    print(f"  Best throughput: {overall.get('best_throughput', 0):,.0f} tok/s")

    print(f"\n  {'Component':<15s} {'Count':>5s} {'MFU%':>8s} {'Latency ms':>12s} {'Peak MB':>8s}")
    print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*12} {'-'*8}")
    for comp, stats in summary.get("per_component", {}).items():
        mfu = stats.get("mfu_pct", {})
        lat = stats.get("latency_ms", {})
        print(f"  {comp:<15s} {stats['count']:5d} {mfu.get('mean', 0):7.1f}% "
              f"{lat.get('mean', 0):11.3f}  {stats.get('peak_memory_mb', 0):7.0f}")

    best = summary.get("best", {})
    if best:
        print(f"\n  Best results:")
        for metric, info in best.items():
            print(f"    {metric}: {info['value']:.2f} ({info['config']}, {info['component']})")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract DeepSeek-V3 benchmark summary")
    parser.add_argument("path", type=str, help="JSON file or directory to summarize")
    parser.add_argument("--recursive", action="store_true", help="Search directory recursively")
    parser.add_argument("--output", type=str, help="Save summary to JSON file")
    args = parser.parse_args()

    # Collect results
    all_results = []
    if os.path.isfile(args.path):
        all_results = load_results(args.path)
    elif os.path.isdir(args.path):
        for json_file in find_json_files(args.path, args.recursive):
            try:
                all_results.extend(load_results(json_file))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: could not load {json_file}: {e}", file=sys.stderr)
    else:
        print(f"Error: {args.path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(all_results)} results from {args.path}")

    summary = extract_summary(all_results)
    print_summary(summary)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
