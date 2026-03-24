"""
Report generation utilities for DeepSeek-V3 benchmarks.

Produces JSON and Markdown reports from BenchResult collections.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from benchmark.shared.config import BenchResult, DEEPSEEK_V3_CONFIG, H100_SPECS


# ---------------------------------------------------------------------------
# JSON Report
# ---------------------------------------------------------------------------

def save_json_report(
    results: List[BenchResult],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save benchmark results as a JSON report.

    Args:
        results: List of BenchResult objects.
        output_path: Path to write the JSON file.
        metadata: Optional metadata dict to include.

    Returns:
        Absolute path to the written file.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = {
        "metadata": {
            "model": "DeepSeek-V3 671B",
            "paper": "arXiv:2412.19437",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config_summary": {
                "hidden_size": DEEPSEEK_V3_CONFIG["hidden_size"],
                "num_layers": DEEPSEEK_V3_CONFIG["num_layers"],
                "num_heads": DEEPSEEK_V3_CONFIG["num_heads"],
                "kv_lora_rank": DEEPSEEK_V3_CONFIG["kv_lora_rank"],
                "n_routed_experts": DEEPSEEK_V3_CONFIG["n_routed_experts"],
                "num_experts_per_tok": DEEPSEEK_V3_CONFIG["num_experts_per_tok"],
                "n_group": DEEPSEEK_V3_CONFIG["n_group"],
                "topk_group": DEEPSEEK_V3_CONFIG["topk_group"],
            },
            "hardware": H100_SPECS["name"],
            **(metadata or {}),
        },
        "results": [r.to_dict() for r in results],
        "summary": _compute_summary(results),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return output_path


# ---------------------------------------------------------------------------
# Markdown Report
# ---------------------------------------------------------------------------

def save_markdown_report(
    results: List[BenchResult],
    output_path: str,
    title: str = "DeepSeek-V3 Benchmark Report",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save benchmark results as a Markdown report with tables.

    Args:
        results: List of BenchResult objects.
        output_path: Path to write the Markdown file.
        title: Report title.
        metadata: Optional metadata dict.

    Returns:
        Absolute path to the written file.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Model:** DeepSeek-V3 671B (arXiv: 2412.19437)")
    lines.append(f"**Hardware:** {H100_SPECS['name']}")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if metadata:
        lines.append("## Metadata")
        lines.append("")
        for k, v in metadata.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    # Group results by component
    components: Dict[str, List[BenchResult]] = {}
    for r in results:
        comp = r.component or "unknown"
        components.setdefault(comp, []).append(r)

    for comp_name, comp_results in sorted(components.items()):
        lines.append(f"## {comp_name.upper()} Benchmarks")
        lines.append("")

        # Table header
        lines.append("| Batch | SeqLen | Mean (ms) | Std (ms) | CI 95% (ms) | MFU (%) | Toks/s | Peak Mem (MB) |")
        lines.append("|------:|-------:|----------:|---------:|------------:|--------:|-------:|--------------:|")

        for r in sorted(comp_results, key=lambda x: (x.batch_size, x.seq_len)):
            ci = f"{r.ci_lower_ms:.3f}-{r.ci_upper_ms:.3f}"
            lines.append(
                f"| {r.batch_size:5d} | {r.seq_len:6d} | {r.mean_ms:9.3f} | {r.std_ms:8.3f} | "
                f"{ci:>11s} | {r.mfu:7.1f} | {r.tokens_per_sec:12,.0f} | {r.peak_memory_mb:13.1f} |"
            )

        lines.append("")

    # Summary
    summary = _compute_summary(results)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total benchmarks:** {summary['total_benchmarks']}")
    lines.append(f"- **Best MFU:** {summary['best_mfu']:.1f}% ({summary['best_mfu_config']})")
    lines.append(f"- **Best throughput:** {summary['best_throughput']:,.0f} tok/s ({summary['best_throughput_config']})")
    lines.append(f"- **Peak memory:** {summary['peak_memory_mb']:.1f} MB")
    lines.append("")

    # Model config reference
    lines.append("## Model Configuration Reference")
    lines.append("")
    lines.append("```")
    lines.append(f"hidden_size:       {DEEPSEEK_V3_CONFIG['hidden_size']}")
    lines.append(f"num_layers:        {DEEPSEEK_V3_CONFIG['num_layers']} (3 dense + 58 MoE)")
    lines.append(f"num_heads:         {DEEPSEEK_V3_CONFIG['num_heads']}")
    lines.append(f"kv_lora_rank:      {DEEPSEEK_V3_CONFIG['kv_lora_rank']}")
    lines.append(f"qk_rope_head_dim:  {DEEPSEEK_V3_CONFIG['qk_rope_head_dim']}")
    lines.append(f"qk_nope_head_dim:  {DEEPSEEK_V3_CONFIG['qk_nope_head_dim']}")
    lines.append(f"v_head_dim:        {DEEPSEEK_V3_CONFIG['v_head_dim']}")
    lines.append(f"n_routed_experts:  {DEEPSEEK_V3_CONFIG['n_routed_experts']}")
    lines.append(f"experts_per_tok:   {DEEPSEEK_V3_CONFIG['num_experts_per_tok']}")
    lines.append(f"n_group:           {DEEPSEEK_V3_CONFIG['n_group']}")
    lines.append(f"topk_group:        {DEEPSEEK_V3_CONFIG['topk_group']}")
    lines.append(f"moe_intermediate:  {DEEPSEEK_V3_CONFIG['moe_intermediate_size']}")
    lines.append(f"vocab_size:        {DEEPSEEK_V3_CONFIG['vocab_size']}")
    lines.append(f"mtp_layers:        {DEEPSEEK_V3_CONFIG['mtp_layers']}")
    lines.append("```")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_results_table(results: List[BenchResult], title: str = "Results") -> None:
    """Print results as a formatted table to stdout."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"{'Component':>8s} | {'BS':>4s} | {'Seq':>5s} | {'Mean ms':>9s} | {'MFU%':>6s} | {'Toks/s':>12s} | {'Mem MB':>8s}")
    print(f"{'-'*8}-+-{'-'*4}-+-{'-'*5}-+-{'-'*9}-+-{'-'*6}-+-{'-'*12}-+-{'-'*8}")

    for r in sorted(results, key=lambda x: (x.component, x.batch_size, x.seq_len)):
        print(
            f"{r.component:>8s} | {r.batch_size:4d} | {r.seq_len:5d} | "
            f"{r.mean_ms:9.3f} | {r.mfu:6.1f} | {r.tokens_per_sec:12,.0f} | {r.peak_memory_mb:8.1f}"
        )
    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_summary(results: List[BenchResult]) -> Dict[str, Any]:
    """Compute summary statistics across all results."""
    if not results:
        return {
            "total_benchmarks": 0,
            "best_mfu": 0.0,
            "best_mfu_config": "N/A",
            "best_throughput": 0.0,
            "best_throughput_config": "N/A",
            "peak_memory_mb": 0.0,
        }

    best_mfu_result = max(results, key=lambda r: r.mfu)
    best_tp_result = max(results, key=lambda r: r.tokens_per_sec)

    return {
        "total_benchmarks": len(results),
        "best_mfu": best_mfu_result.mfu,
        "best_mfu_config": f"bs={best_mfu_result.batch_size},seq={best_mfu_result.seq_len},{best_mfu_result.component}",
        "best_throughput": best_tp_result.tokens_per_sec,
        "best_throughput_config": f"bs={best_tp_result.batch_size},seq={best_tp_result.seq_len},{best_tp_result.component}",
        "peak_memory_mb": max(r.peak_memory_mb for r in results),
    }
