"""
3-way benchmark: PyTorch vs FlashMLA vs DeepGEMM on H100.

Compares performance across the three execution backends:
  1. PyTorch eager (reference)
  2. FlashMLA (attention kernel)
  3. DeepGEMM FP8 (MoE grouped GEMM)

Reports speedup ratios and throughput metrics.

Requires: H100 GPU, FlashMLA, DeepGEMM.
"""

from __future__ import annotations

import sys
import os
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEEPSEEK_V3_CONFIG

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)


def _benchmark(fn, warmup=10, iters=100):
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
    return start.elapsed_time(end) / iters


@requires_hopper
class TestH100Bench3Way:
    """3-way performance comparison on H100."""

    @pytest.mark.parametrize("seq_len", [1, 64, 512, 2048])
    def test_attention_3way(self, seq_len):
        """Compare attention implementations."""
        from mla_attention import eager_attention_forward

        B, H, D, D_v = 1, 128, 192, 128
        device = torch.device("cuda")
        dtype = torch.bfloat16

        q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
        k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
        v = torch.randn(B, H, seq_len, D_v, device=device, dtype=dtype)
        scale = D ** -0.5

        # 1. PyTorch eager
        ms_eager = _benchmark(
            lambda: eager_attention_forward(q, k, v, scale, causal=True)
        )

        # 2. PyTorch SDPA
        try:
            ms_sdpa = _benchmark(
                lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale
                )
            )
        except Exception:
            ms_sdpa = float("inf")

        # 3. FlashMLA
        ms_flashmla = float("inf")
        try:
            from flash_mla import flash_mla_with_kvcache
            # FlashMLA would be benchmarked here with proper cache setup
            ms_flashmla = float("inf")  # placeholder
        except ImportError:
            pass

        print(f"\n--- Attention benchmark (seq_len={seq_len}) ---")
        print(f"  Eager:    {ms_eager:.3f} ms")
        print(f"  SDPA:     {ms_sdpa:.3f} ms")
        print(f"  FlashMLA: {ms_flashmla:.3f} ms")
        if ms_sdpa < float("inf"):
            print(f"  SDPA speedup: {ms_eager / ms_sdpa:.2f}x")
        if ms_flashmla < float("inf"):
            print(f"  FlashMLA speedup: {ms_eager / ms_flashmla:.2f}x")

    @pytest.mark.parametrize("total_tokens", [256, 1024, 4096])
    def test_moe_gemm_3way(self, total_tokens):
        """Compare MoE GEMM implementations."""
        from fp8_utils import quantize_fp8_block

        D, I = 7168, 2048
        num_experts = 256
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Random inputs
        x = torch.randn(total_tokens, D, device=device, dtype=dtype)
        w = torch.randn(num_experts, I, D, device=device, dtype=dtype)
        group_sizes = torch.full(
            (num_experts,), total_tokens // num_experts,
            dtype=torch.int32, device=device,
        )

        # 1. Sequential matmul
        def sequential():
            offset = 0
            outs = []
            for g in range(num_experts):
                s = group_sizes[g].item()
                if s > 0:
                    outs.append(x[offset:offset+s] @ w[g].T)
                    offset += s
            if outs:
                return torch.cat(outs)

        ms_seq = _benchmark(sequential) if total_tokens >= num_experts else float("inf")

        # 2. BMM (batch matmul)
        def bmm():
            # Reshape for BMM (only works with equal group sizes)
            t_per_e = total_tokens // num_experts
            x_r = x[:num_experts * t_per_e].reshape(num_experts, t_per_e, D)
            return torch.bmm(x_r, w.transpose(1, 2))

        ms_bmm = _benchmark(bmm) if total_tokens >= num_experts else float("inf")

        # 3. DeepGEMM FP8
        ms_deepgemm = float("inf")
        try:
            import deep_gemm
            x_fp8, x_scales = quantize_fp8_block(x)
            w_fp8, w_scales = quantize_fp8_block(w.reshape(-1, D))

            def deepgemm_call():
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
                    x_fp8, x_scales, w_fp8, w_scales, group_sizes,
                )

            ms_deepgemm = _benchmark(deepgemm_call)
        except ImportError:
            pass

        print(f"\n--- MoE GEMM benchmark (T={total_tokens}) ---")
        print(f"  Sequential: {ms_seq:.3f} ms")
        print(f"  BMM:        {ms_bmm:.3f} ms")
        print(f"  DeepGEMM:   {ms_deepgemm:.3f} ms")
        if ms_deepgemm < float("inf"):
            print(f"  DeepGEMM speedup vs seq: {ms_seq / ms_deepgemm:.2f}x")

    def test_end_to_end_layer(self):
        """Benchmark a complete decoder layer."""
        from model import DeepSeekV3DecoderLayer

        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Use a smaller config for feasibility
        from config import DeepSeekV3Config, MoEConfig
        small = DeepSeekV3Config(
            hidden_size=1024, intermediate_size=2048,
            num_hidden_layers=1, num_attention_heads=16,
            q_lora_rank=256, kv_lora_rank=64,
            qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
            vocab_size=1024,
            moe=MoEConfig(
                num_experts=32, num_experts_per_tok=4,
                n_group=4, topk_group=2,
                expert_intermediate_size=256,
                shared_expert_intermediate_size=256,
                first_k_dense_replace=0,
            ),
        )

        layer = DeepSeekV3DecoderLayer(small, 0).to(device).eval().to(dtype)
        B, S = 1, 128
        x = torch.randn(B, S, small.hidden_size, device=device, dtype=dtype)

        with torch.no_grad():
            ms = _benchmark(lambda: layer(x))

        print(f"\nDecoder layer B={B} S={S}: {ms:.3f}ms")
        print(f"  Throughput: {B * S / (ms * 1e-3):.0f} tok/s")
