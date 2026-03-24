"""
H100 benchmark suite for DeepSeek-V3 FlashMLA + DeepGEMM.

Measures:
  - Attention throughput (tokens/sec) with FlashMLA
  - MoE expert dispatch throughput with DeepGEMM FP8
  - End-to-end layer throughput
  - Memory bandwidth utilisation

Requires: H100 GPU, FlashMLA, DeepGEMM.
"""

from __future__ import annotations

import sys
import os
import time
from typing import Dict

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config, DEEPSEEK_V3_CONFIG

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)


def _benchmark_fn(fn, warmup=10, iters=50):
    """Benchmark a function with CUDA timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


@requires_hopper
class TestH100AttentionBench:
    """Benchmark MLA attention on H100."""

    @pytest.mark.parametrize("seq_len", [1, 128, 1024, 4096])
    def test_eager_attention_throughput(self, seq_len):
        """Benchmark eager attention at various sequence lengths."""
        from mla_attention import eager_attention_forward

        B, H, D = 1, 128, 192
        D_v = 128
        device = torch.device("cuda")

        q = torch.randn(B, H, seq_len, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, H, seq_len, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, H, seq_len, D_v, device=device, dtype=torch.bfloat16)
        scale = D ** -0.5

        ms = _benchmark_fn(lambda: eager_attention_forward(q, k, v, scale, causal=True))
        tokens_per_sec = B * seq_len / (ms * 1e-3)
        print(f"\nEager attn seq_len={seq_len}: {ms:.3f}ms, {tokens_per_sec:.0f} tok/s")

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_mla_layer_throughput(self, batch_size):
        """Benchmark full MLA layer."""
        from mla_attention import MLAttention
        config = DEEPSEEK_V3_CONFIG

        device = torch.device("cuda")
        attn = MLAttention(config, layer_idx=0).to(device).eval().to(torch.bfloat16)

        S = 128
        x = torch.randn(batch_size, S, config.hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            ms = _benchmark_fn(lambda: attn(x))

        tokens_per_sec = batch_size * S / (ms * 1e-3)
        print(f"\nMLA B={batch_size} S={S}: {ms:.3f}ms, {tokens_per_sec:.0f} tok/s")


@requires_hopper
class TestH100MoEBench:
    """Benchmark MoE dispatch on H100."""

    def test_router_throughput(self):
        """Benchmark routing for 256 experts."""
        from moe_router import MoERouter
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")

        router = MoERouter(config).to(device).eval().to(torch.bfloat16)
        T = 4096
        x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            ms = _benchmark_fn(lambda: router(x))

        print(f"\nRouter T={T}: {ms:.3f}ms, {T / (ms * 1e-3):.0f} tok/s")

    def test_expert_ffn_throughput(self):
        """Benchmark single expert FFN."""
        from moe_grouped_gemm import ExpertFFN
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")

        expert = ExpertFFN(
            config.hidden_size, config.moe.expert_intermediate_size
        ).to(device).eval().to(torch.bfloat16)

        T = 64  # typical tokens per expert
        x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            ms = _benchmark_fn(lambda: expert(x))

        print(f"\nExpert FFN T={T}: {ms:.3f}ms")


@requires_hopper
class TestH100MemoryBench:
    """Benchmark memory usage on H100."""

    def test_kv_cache_memory(self):
        """Measure actual KV cache memory usage."""
        from cache import KVCache
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")

        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        cache = KVCache(
            config, max_batch_size=1, max_seq_len=4096,
            device=device, dtype=torch.bfloat16,
        )
        cache.allocate_all()

        after = torch.cuda.memory_allocated()
        used_mb = (after - before) / 1024**2

        print(f"\nKV cache memory (B=1, S=4096, all layers): {used_mb:.1f} MB")
        print(f"Reported: {cache.memory_bytes / 1024**2:.1f} MB")
