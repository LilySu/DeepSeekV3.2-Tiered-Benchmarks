"""
Auto-tuning infrastructure for grouped GEMM Triton kernels.

Provides a systematic way to search for optimal kernel configurations
(block sizes, number of warps, pipeline stages) for given matrix shapes.

For DeepSeek-V3, the key shapes are:
  - Gate/Up projection: [T_expert, 7168] x [7168, 2048]
  - Down projection: [T_expert, 2048] x [2048, 7168]

where T_expert varies based on routing decisions (average ~= T*8/256).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class TuningConfig:
    """Single tuning configuration for a Triton kernel.

    Parameters
    ----------
    block_m : int
        Tile size along the M (token) dimension.
    block_n : int
        Tile size along the N (output) dimension.
    block_k : int
        Tile size along the K (reduction) dimension.
    num_warps : int
        Number of warps per block.
    num_stages : int
        Number of software pipelining stages.
    """

    block_m: int = 64
    block_n: int = 64
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 3

    # Performance metrics (filled by auto-tuner)
    tflops: float = 0.0
    time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TuningConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProblemShape:
    """Matrix multiply problem shape for tuning."""

    M: int  # tokens per expert (variable)
    K: int  # input dimension
    N: int  # output dimension
    num_groups: int  # number of active experts

    @property
    def flops(self) -> int:
        """Total FLOPs for the grouped GEMM."""
        return 2 * self.M * self.K * self.N * self.num_groups


class AutoTuner:
    """Auto-tuning engine for grouped GEMM kernels.

    Parameters
    ----------
    cache_dir : str
        Directory to cache tuning results.
    warmup_iters : int
        Number of warmup iterations before measurement.
    measure_iters : int
        Number of iterations to measure.
    """

    def __init__(
        self,
        cache_dir: str = "/tmp/unsloth_moe_tuning",
        warmup_iters: int = 10,
        measure_iters: int = 50,
    ) -> None:
        self.cache_dir = cache_dir
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self._cache: Dict[str, TuningConfig] = {}

        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached tuning results."""
        cache_file = os.path.join(self.cache_dir, "tuning_cache.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
            for key, config_dict in data.items():
                self._cache[key] = TuningConfig.from_dict(config_dict)

    def _save_cache(self) -> None:
        """Save tuning results to disk."""
        cache_file = os.path.join(self.cache_dir, "tuning_cache.json")
        data = {k: v.to_dict() for k, v in self._cache.items()}
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _make_key(self, shape: ProblemShape) -> str:
        """Create a cache key for a problem shape."""
        return f"M{shape.M}_K{shape.K}_N{shape.N}_G{shape.num_groups}"

    def get_search_space(self) -> List[TuningConfig]:
        """Generate the search space of kernel configurations."""
        configs = []
        for bm in [32, 64, 128]:
            for bn in [32, 64, 128]:
                for bk in [16, 32, 64]:
                    for nw in [2, 4, 8]:
                        for ns in [2, 3, 4]:
                            configs.append(TuningConfig(
                                block_m=bm, block_n=bn, block_k=bk,
                                num_warps=nw, num_stages=ns,
                            ))
        return configs

    def tune(
        self,
        shape: ProblemShape,
        kernel_fn: Optional[Any] = None,
        search_space: Optional[List[TuningConfig]] = None,
    ) -> TuningConfig:
        """Find the best kernel configuration for a given shape.

        Parameters
        ----------
        shape : ProblemShape
        kernel_fn : callable, optional
            The Triton kernel to tune. If None, returns cached or default.
        search_space : list of TuningConfig, optional

        Returns
        -------
        best_config : TuningConfig
        """
        key = self._make_key(shape)
        if key in self._cache:
            return self._cache[key]

        if kernel_fn is None or not torch.cuda.is_available():
            # Return a reasonable default
            default = TuningConfig(
                block_m=64, block_n=64, block_k=32,
                num_warps=4, num_stages=3,
            )
            self._cache[key] = default
            return default

        if search_space is None:
            search_space = self.get_search_space()

        best_config = None
        best_time = float("inf")

        for config in search_space:
            try:
                elapsed = self._benchmark_config(shape, kernel_fn, config)
                if elapsed < best_time:
                    best_time = elapsed
                    best_config = config
                    best_config.time_ms = elapsed
                    best_config.tflops = (
                        shape.flops / (elapsed * 1e-3) / 1e12
                    )
            except Exception:
                continue

        if best_config is None:
            best_config = TuningConfig()

        self._cache[key] = best_config
        self._save_cache()
        return best_config

    def _benchmark_config(
        self,
        shape: ProblemShape,
        kernel_fn: Any,
        config: TuningConfig,
    ) -> float:
        """Benchmark a single configuration.

        Returns
        -------
        time_ms : float
            Average kernel execution time in milliseconds.
        """
        device = torch.device("cuda")

        # Create test tensors
        a = torch.randn(shape.M, shape.K, dtype=torch.bfloat16, device=device)
        b = torch.randn(shape.num_groups, shape.K, shape.N, dtype=torch.bfloat16, device=device)
        group_sizes = torch.full(
            (shape.num_groups,), shape.M // max(shape.num_groups, 1),
            dtype=torch.int32, device=device,
        )

        # Warmup
        for _ in range(self.warmup_iters):
            kernel_fn(a, b, group_sizes, config)

        torch.cuda.synchronize()

        # Measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(self.measure_iters):
            kernel_fn(a, b, group_sizes, config)
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / self.measure_iters
        return elapsed

    def get_cached_config(self, shape: ProblemShape) -> Optional[TuningConfig]:
        """Look up a cached configuration without running tuning."""
        key = self._make_key(shape)
        return self._cache.get(key)

    def clear_cache(self) -> None:
        """Clear all cached tuning results."""
        self._cache.clear()
        cache_file = os.path.join(self.cache_dir, "tuning_cache.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
