#!/usr/bin/env python3
"""
Debug import chain for DeepSeek-V3 benchmark suite.

Verifies that all benchmark modules can be imported correctly
and reports the dependency tree and any import errors.

Usage:
    python -m benchmark.debug_imports
"""

from __future__ import annotations

import importlib
import sys
import traceback
from typing import List, Tuple


MODULES_TO_CHECK = [
    # Core shared
    "benchmark",
    "benchmark.shared",
    "benchmark.shared.config",
    "benchmark.shared.metrics",
    "benchmark.shared.report",
    "benchmark.shared.timer",

    # Sub-packages
    "benchmark.fp8_pareto",
    "benchmark.fp8_pareto.bench_fp8",
    "benchmark.fp8_pareto.precision_experiment",
    "benchmark.mfu_ceiling",
    "benchmark.mfu_ceiling.bench_mfu",
    "benchmark.moe_sweep",
    "benchmark.moe_sweep.bench_moe",
    "benchmark.triple_report",
    "benchmark.triple_report.bench_micro",
    "benchmark.triple_report.bench_component",
    "benchmark.triple_report.bench_e2e",

    # Top-level scripts
    "benchmark.run_all",
    "benchmark.run_component_bench",
    "benchmark.extract_summary",
    "benchmark.generate_research_report_from_benchmark_results",
    "benchmark.print_all_benchmark_results_summary",
]

OPTIONAL_DEPS = [
    ("torch", "PyTorch (required for GPU benchmarks)"),
    ("torch.cuda", "CUDA support"),
    ("flash_attn", "Flash Attention"),
    ("triton", "OpenAI Triton"),
    ("numpy", "NumPy"),
    ("matplotlib", "Matplotlib (for visualization)"),
]


def check_imports() -> List[Tuple[str, bool, str]]:
    """Check all benchmark module imports."""
    results = []
    for module_name in MODULES_TO_CHECK:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "")
            msg = f"OK" + (f" (v{version})" if version else "")
            results.append((module_name, True, msg))
        except Exception as e:
            results.append((module_name, False, f"{type(e).__name__}: {e}"))
    return results


def check_optional_deps() -> List[Tuple[str, bool, str]]:
    """Check optional dependencies."""
    results = []
    for module_name, description in OPTIONAL_DEPS:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "available")
            results.append((module_name, True, f"{description}: {version}"))
        except ImportError:
            results.append((module_name, False, f"{description}: NOT INSTALLED"))
        except Exception as e:
            results.append((module_name, False, f"{description}: {e}"))
    return results


def check_config_values():
    """Verify config values are accessible and correct."""
    try:
        from benchmark.shared.config import DEEPSEEK_V3_CONFIG, H100_SPECS

        checks = [
            ("hidden_size", DEEPSEEK_V3_CONFIG["hidden_size"], 7168),
            ("num_heads", DEEPSEEK_V3_CONFIG["num_heads"], 128),
            ("kv_lora_rank", DEEPSEEK_V3_CONFIG["kv_lora_rank"], 512),
            ("qk_rope_head_dim", DEEPSEEK_V3_CONFIG["qk_rope_head_dim"], 64),
            ("qk_nope_head_dim", DEEPSEEK_V3_CONFIG["qk_nope_head_dim"], 128),
            ("v_head_dim", DEEPSEEK_V3_CONFIG["v_head_dim"], 128),
            ("n_routed_experts", DEEPSEEK_V3_CONFIG["n_routed_experts"], 256),
            ("num_experts_per_tok", DEEPSEEK_V3_CONFIG["num_experts_per_tok"], 8),
            ("n_group", DEEPSEEK_V3_CONFIG["n_group"], 8),
            ("topk_group", DEEPSEEK_V3_CONFIG["topk_group"], 4),
            ("num_layers", DEEPSEEK_V3_CONFIG["num_layers"], 61),
            ("vocab_size", DEEPSEEK_V3_CONFIG["vocab_size"], 129280),
            ("moe_intermediate_size", DEEPSEEK_V3_CONFIG["moe_intermediate_size"], 2048),
            ("mtp_layers", DEEPSEEK_V3_CONFIG["mtp_layers"], 1),
        ]

        results = []
        for name, actual, expected in checks:
            ok = actual == expected
            results.append((f"config.{name}", ok,
                           f"expected={expected}, got={actual}"))
        return results
    except Exception as e:
        return [("config_check", False, str(e))]


def main():
    print("=" * 70)
    print("  DeepSeek-V3 Import Debug")
    print("=" * 70)
    print(f"  Python: {sys.version}")
    print(f"  Path: {sys.path[:3]}...")

    # Module imports
    print("\n--- Benchmark Module Imports ---")
    import_results = check_imports()
    pass_count = 0
    for name, ok, msg in import_results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        pass_count += ok

    # Optional dependencies
    print("\n--- Optional Dependencies ---")
    dep_results = check_optional_deps()
    for name, ok, msg in dep_results:
        status = "OK  " if ok else "MISS"
        print(f"  [{status}] {msg}")

    # Config verification
    print("\n--- Config Verification ---")
    config_results = check_config_values()
    for name, ok, msg in config_results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")

    total = len(import_results)
    print(f"\n{'='*70}")
    print(f"  Imports: {pass_count}/{total} successful")
    print(f"{'='*70}")

    return 0 if pass_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
