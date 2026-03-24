"""
Shared utilities for DeepSeek-V3 benchmarks.

Provides configuration dataclasses, timing helpers, FLOP/bandwidth metrics,
and report generation for all benchmark sub-packages.
"""

from benchmark.shared.config import DEEPSEEK_V3_CONFIG, H100_SPECS, BenchConfig, BenchResult
from benchmark.shared.timer import CUDATimer
from benchmark.shared.metrics import compute_mla_flops, compute_moe_flops, compute_mfu, compute_bandwidth_utilization
from benchmark.shared.report import save_json_report, save_markdown_report

__all__ = [
    "DEEPSEEK_V3_CONFIG",
    "H100_SPECS",
    "BenchConfig",
    "BenchResult",
    "CUDATimer",
    "compute_mla_flops",
    "compute_moe_flops",
    "compute_mfu",
    "compute_bandwidth_utilization",
    "save_json_report",
    "save_markdown_report",
]
