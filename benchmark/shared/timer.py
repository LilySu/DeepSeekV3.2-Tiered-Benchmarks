"""
CUDA timer with bootstrap confidence intervals for DeepSeek-V3 benchmarks.

Provides precise GPU-side timing using CUDA events, with statistical
analysis via bootstrap resampling for robust confidence intervals.
"""

from __future__ import annotations

import time
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any

try:
    import torch
    import torch.cuda
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


@dataclass
class TimingResult:
    """Result of a timed kernel execution."""
    mean_ms: float
    std_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    ci_lower_ms: float
    ci_upper_ms: float
    raw_times_ms: List[float]
    warmup_iters: int
    bench_iters: int
    confidence_level: float


class CUDATimer:
    """
    High-precision CUDA timer using CUDA events.

    Usage:
        timer = CUDATimer(warmup=10, iters=100)
        result = timer.time_fn(lambda: my_kernel(x, y))
        print(f"Mean: {result.mean_ms:.3f} ms, 95% CI: [{result.ci_lower_ms:.3f}, {result.ci_upper_ms:.3f}]")
    """

    def __init__(
        self,
        warmup: int = 10,
        iters: int = 100,
        num_bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        device: Optional[str] = None,
    ):
        self.warmup = warmup
        self.iters = iters
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level
        self.device = device or ("cuda" if HAS_CUDA else "cpu")

    def time_fn(
        self,
        fn: Callable,
        *args,
        **kwargs,
    ) -> TimingResult:
        """
        Time a callable using CUDA events (or CPU fallback).

        Args:
            fn: Callable to time. Called as fn(*args, **kwargs).

        Returns:
            TimingResult with statistics and bootstrap CI.
        """
        if HAS_CUDA and self.device.startswith("cuda"):
            return self._time_cuda(fn, *args, **kwargs)
        else:
            return self._time_cpu(fn, *args, **kwargs)

    def _time_cuda(self, fn: Callable, *args, **kwargs) -> TimingResult:
        """Time using CUDA events for GPU-side precision."""
        import torch

        # Warmup
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        # Benchmark
        times_ms = []
        for _ in range(self.iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            fn(*args, **kwargs)
            end_event.record()

            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)  # milliseconds
            times_ms.append(elapsed)

        return self._build_result(times_ms)

    def _time_cpu(self, fn: Callable, *args, **kwargs) -> TimingResult:
        """Fallback CPU timing using time.perf_counter."""
        # Warmup
        for _ in range(self.warmup):
            fn(*args, **kwargs)

        # Benchmark
        times_ms = []
        for _ in range(self.iters):
            start = time.perf_counter()
            fn(*args, **kwargs)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

        return self._build_result(times_ms)

    def _build_result(self, times_ms: List[float]) -> TimingResult:
        """Compute statistics and bootstrap CI from raw timings."""
        n = len(times_ms)
        mean = sum(times_ms) / n
        sorted_times = sorted(times_ms)
        median = sorted_times[n // 2] if n % 2 == 1 else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        min_val = sorted_times[0]
        max_val = sorted_times[-1]
        variance = sum((t - mean) ** 2 for t in times_ms) / max(n - 1, 1)
        std = math.sqrt(variance)

        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(times_ms)

        return TimingResult(
            mean_ms=mean,
            std_ms=std,
            median_ms=median,
            min_ms=min_val,
            max_ms=max_val,
            ci_lower_ms=ci_lower,
            ci_upper_ms=ci_upper,
            raw_times_ms=times_ms,
            warmup_iters=self.warmup,
            bench_iters=self.iters,
            confidence_level=self.confidence_level,
        )

    def _bootstrap_ci(self, times_ms: List[float]) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for the mean.

        Uses the percentile method with `num_bootstrap_samples` resamples.
        """
        n = len(times_ms)
        if n < 2:
            mean = sum(times_ms) / max(n, 1)
            return (mean, mean)

        rng = random.Random(42)  # deterministic for reproducibility
        bootstrap_means = []

        for _ in range(self.num_bootstrap_samples):
            sample = [times_ms[rng.randint(0, n - 1)] for _ in range(n)]
            bootstrap_means.append(sum(sample) / n)

        bootstrap_means.sort()
        alpha = 1.0 - self.confidence_level
        lower_idx = int(math.floor(alpha / 2 * self.num_bootstrap_samples))
        upper_idx = int(math.ceil((1 - alpha / 2) * self.num_bootstrap_samples)) - 1

        lower_idx = max(0, min(lower_idx, len(bootstrap_means) - 1))
        upper_idx = max(0, min(upper_idx, len(bootstrap_means) - 1))

        return (bootstrap_means[lower_idx], bootstrap_means[upper_idx])


class MultiTimer:
    """
    Time multiple named regions within a single pass.

    Usage:
        mt = MultiTimer()
        with mt.region("mla_proj"):
            mla_projection(x)
        with mt.region("mla_attn"):
            mla_attention(q, k, v)
        mt.report()
    """

    def __init__(self):
        self._regions: Dict[str, List[float]] = {}
        self._current_region: Optional[str] = None
        self._start_time: float = 0.0

    class _RegionContext:
        def __init__(self, parent: "MultiTimer", name: str):
            self.parent = parent
            self.name = name

        def __enter__(self):
            if HAS_CUDA:
                import torch
                torch.cuda.synchronize()
            self.parent._start_time = time.perf_counter()
            return self

        def __exit__(self, *exc):
            if HAS_CUDA:
                import torch
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - self.parent._start_time) * 1000.0
            self.parent._regions.setdefault(self.name, []).append(elapsed_ms)

    def region(self, name: str) -> _RegionContext:
        return self._RegionContext(self, name)

    def report(self) -> Dict[str, Dict[str, float]]:
        """Return summary statistics for each timed region."""
        result = {}
        for name, times in self._regions.items():
            n = len(times)
            mean = sum(times) / n
            sorted_t = sorted(times)
            result[name] = {
                "mean_ms": mean,
                "min_ms": sorted_t[0],
                "max_ms": sorted_t[-1],
                "count": n,
                "total_ms": sum(times),
            }
        return result

    def print_report(self, title: str = "Region Timing") -> None:
        """Pretty-print the region timing report."""
        report = self.report()
        print(f"\n--- {title} ---")
        total_all = sum(v["total_ms"] for v in report.values())
        for name, stats in sorted(report.items(), key=lambda x: -x[1]["total_ms"]):
            pct = 100 * stats["total_ms"] / total_all if total_all > 0 else 0
            print(
                f"  {name:<20s}: mean={stats['mean_ms']:.3f}ms  "
                f"min={stats['min_ms']:.3f}ms  max={stats['max_ms']:.3f}ms  "
                f"n={stats['count']}  total={stats['total_ms']:.1f}ms ({pct:.1f}%)"
            )
        print(f"  {'TOTAL':<20s}: {total_all:.1f}ms\n")
