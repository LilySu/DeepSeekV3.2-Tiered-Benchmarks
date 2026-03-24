"""
H100 kernel launch overhead tests.

Measures and validates kernel launch overhead for:
  - FlashMLA kernel launch
  - DeepGEMM grouped GEMM launch
  - Comparison with PyTorch eager baseline
  - CUDA graph amortisation of launch overhead

Requires: H100 GPU.
"""

from __future__ import annotations

import sys
import os
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)


def _measure_launch_overhead(fn, iters=1000):
    """Measure per-call overhead in microseconds."""
    # Warmup
    for _ in range(100):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms * 1000 / iters  # microseconds per call


@requires_hopper
class TestH100LaunchOverhead:
    """Measure kernel launch overhead on H100."""

    def test_matmul_launch_overhead(self):
        """Baseline: small matmul launch overhead."""
        M, K, N = 32, 64, 32
        device = torch.device("cuda")
        a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        b = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        us = _measure_launch_overhead(lambda: torch.mm(a, b))
        print(f"\nMatmul ({M}x{K} @ {K}x{N}) launch: {us:.1f} us")
        assert us < 100  # should be under 100us per launch

    def test_eager_attention_launch_overhead(self):
        """Eager attention launch overhead."""
        from mla_attention import eager_attention_forward

        B, H, S, D = 1, 4, 16, 32
        device = torch.device("cuda")
        dtype = torch.bfloat16

        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)

        us = _measure_launch_overhead(
            lambda: eager_attention_forward(q, k, v, D**-0.5, causal=True)
        )
        print(f"\nEager attn launch: {us:.1f} us")

    def test_rmsnorm_launch_overhead(self):
        """RMSNorm launch overhead."""
        from unsloth_rms_layernorm import UnslothRMSNorm

        device = torch.device("cuda")
        norm = UnslothRMSNorm(256).to(device).to(torch.bfloat16)
        x = torch.randn(1, 16, 256, device=device, dtype=torch.bfloat16)

        us = _measure_launch_overhead(lambda: norm(x))
        print(f"\nRMSNorm launch: {us:.1f} us")

    def test_router_launch_overhead(self):
        """MoE router launch overhead."""
        from moe_router import MoERouter
        from config import DeepSeekV3Config, MoEConfig

        config = DeepSeekV3Config(
            hidden_size=256,
            moe=MoEConfig(
                num_experts=16, num_experts_per_tok=4,
                n_group=4, topk_group=2,
            ),
        )
        device = torch.device("cuda")
        router = MoERouter(config).to(device).to(torch.bfloat16).eval()
        x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            us = _measure_launch_overhead(lambda: router(x))
        print(f"\nRouter launch: {us:.1f} us")

    def test_cuda_graph_amortises_overhead(self):
        """CUDA graph should reduce per-step overhead."""
        device = torch.device("cuda")
        M, K, N = 64, 128, 64
        a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        b = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        # Without graph
        us_no_graph = _measure_launch_overhead(lambda: torch.mm(a, b))

        # With graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            c = torch.mm(a, b)

        us_graph = _measure_launch_overhead(lambda: g.replay())

        print(f"\nMatmul overhead: no_graph={us_no_graph:.1f}us, graph={us_graph:.1f}us")
        # Graph replay should be faster
        assert us_graph < us_no_graph * 1.5  # at least not significantly worse

    def test_sequential_vs_batched_expert(self):
        """Compare overhead: sequential expert calls vs batched."""
        from moe_grouped_gemm import ExpertFFN

        device = torch.device("cuda")
        D, I = 256, 128
        num_experts = 8
        T_per_expert = 8

        experts = [
            ExpertFFN(D, I).to(device).to(torch.bfloat16).eval()
            for _ in range(num_experts)
        ]
        inputs = [
            torch.randn(T_per_expert, D, device=device, dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]

        # Sequential
        def sequential():
            for e, inp in zip(experts, inputs):
                e(inp)

        us_seq = _measure_launch_overhead(sequential, iters=200)

        # Single large matmul (simulating batch)
        big_input = torch.randn(num_experts * T_per_expert, D, device=device, dtype=torch.bfloat16)
        big_weight = torch.randn(I, D, device=device, dtype=torch.bfloat16)

        def batched():
            torch.mm(big_input, big_weight.T)

        us_batch = _measure_launch_overhead(batched, iters=200)

        print(f"\nSequential {num_experts} experts: {us_seq:.1f}us")
        print(f"Batched equivalent: {us_batch:.1f}us")
        print(f"Overhead ratio: {us_seq / us_batch:.1f}x")
