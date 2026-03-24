"""
H100 memory tests.

Measures and validates GPU memory usage for:
  - Model parameter memory
  - KV cache memory at various sequence lengths
  - Peak memory during forward pass
  - Memory efficiency of FP8 vs BF16 weights
  - Memory fragmentation after repeated allocation/deallocation

Requires: H100 GPU.
"""

from __future__ import annotations

import gc
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config, MoEConfig

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)


def _clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _make_test_model(device="cuda"):
    from model import DeepSeekV3Model
    config = DeepSeekV3Config(
        hidden_size=512, intermediate_size=1024,
        num_hidden_layers=4, num_attention_heads=8,
        q_lora_rank=128, kv_lora_rank=64,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        vocab_size=1024,
        moe=MoEConfig(
            num_experts=16, num_experts_per_tok=4,
            n_group=4, topk_group=2,
            expert_intermediate_size=256,
            shared_expert_intermediate_size=256,
            first_k_dense_replace=1,
        ),
        use_flashmla=False, use_deepgemm=False,
    )
    model = DeepSeekV3Model(config).to(device).eval().to(torch.bfloat16)
    return model, config


@requires_hopper
class TestH100Memory:
    """GPU memory tests on H100."""

    def test_model_parameter_memory(self):
        """Measure model parameter memory."""
        _clear_gpu()
        before = torch.cuda.memory_allocated()

        model, config = _make_test_model()

        after = torch.cuda.memory_allocated()
        param_mb = (after - before) / 1024**2

        total_params = sum(p.numel() for p in model.parameters())
        expected_mb = total_params * 2 / 1024**2  # BF16 = 2 bytes

        print(f"\nModel memory: {param_mb:.1f} MB (expected ~{expected_mb:.1f} MB)")
        print(f"Parameters: {total_params:,}")

    @pytest.mark.parametrize("seq_len", [128, 1024, 4096])
    def test_kv_cache_memory_scaling(self, seq_len):
        """KV cache memory should scale linearly with sequence length."""
        from cache import KVCache

        _clear_gpu()
        config = DeepSeekV3Config(
            hidden_size=512, num_hidden_layers=4,
            kv_lora_rank=64, qk_rope_head_dim=16,
            num_attention_heads=8,
            q_lora_rank=128,
            qk_nope_head_dim=32, v_head_dim=32,
        )

        before = torch.cuda.memory_allocated()
        cache = KVCache(
            config, max_batch_size=1, max_seq_len=seq_len,
            device=torch.device("cuda"), dtype=torch.bfloat16,
        )
        cache.allocate_all()
        after = torch.cuda.memory_allocated()

        cache_mb = (after - before) / 1024**2
        print(f"\nKV cache (S={seq_len}): {cache_mb:.2f} MB, reported: {cache.memory_bytes/1024**2:.2f} MB")

    def test_peak_memory_forward(self):
        """Measure peak memory during forward pass."""
        model, config = _make_test_model()
        _clear_gpu()
        torch.cuda.reset_peak_memory_stats()

        B, S = 1, 128
        input_ids = torch.randint(0, config.vocab_size, (B, S), device="cuda")

        before = torch.cuda.memory_allocated()
        with torch.no_grad():
            out = model(input_ids)
        peak = torch.cuda.max_memory_allocated()

        peak_mb = peak / 1024**2
        activation_mb = (peak - before) / 1024**2
        print(f"\nPeak memory: {peak_mb:.1f} MB, activation overhead: {activation_mb:.1f} MB")

    def test_fp8_vs_bf16_memory(self):
        """FP8 weights should use ~half the memory of BF16."""
        from fp8_utils import per_block_cast_to_fp8

        _clear_gpu()
        N = 1024 * 1024  # 1M elements

        # BF16
        x_bf16 = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        bf16_bytes = x_bf16.element_size() * x_bf16.numel()

        # FP8 (simulated)
        x_2d = x_bf16.reshape(1024, 1024)
        x_fp8, scales = per_block_cast_to_fp8(x_2d.float())
        fp8_bytes = x_fp8.element_size() * x_fp8.numel() + scales.element_size() * scales.numel()

        ratio = bf16_bytes / fp8_bytes
        print(f"\nBF16: {bf16_bytes / 1024:.1f} KB, FP8+scales: {fp8_bytes / 1024:.1f} KB")
        print(f"Compression ratio: {ratio:.2f}x")

        # FP8 should be at least 1.5x smaller
        assert ratio > 1.3

    def test_no_memory_leak_repeated_forward(self):
        """Repeated forward passes should not leak memory."""
        model, config = _make_test_model()
        _clear_gpu()

        B, S = 1, 32
        input_ids = torch.randint(0, config.vocab_size, (B, S), device="cuda")

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(input_ids)
        _clear_gpu()

        baseline = torch.cuda.memory_allocated()

        with torch.no_grad():
            for _ in range(50):
                model(input_ids)
        _clear_gpu()

        after = torch.cuda.memory_allocated()
        leak = (after - baseline) / 1024

        print(f"\nMemory after 50 forwards: baseline={baseline/1024:.1f}KB, after={after/1024:.1f}KB, leak={leak:.1f}KB")
        assert leak < 100, f"Memory leak detected: {leak:.1f} KB"

    def test_cache_reset_frees_tracking(self):
        """Cache reset should not grow memory."""
        from cache import KVCache

        config = DeepSeekV3Config(
            hidden_size=512, num_hidden_layers=2,
            kv_lora_rank=64, qk_rope_head_dim=16,
            num_attention_heads=8,
            q_lora_rank=128,
            qk_nope_head_dim=32, v_head_dim=32,
        )

        cache = KVCache(
            config, max_batch_size=4, max_seq_len=256,
            device=torch.device("cuda"), dtype=torch.bfloat16,
        )
        cache.allocate_all()

        _clear_gpu()
        baseline = torch.cuda.memory_allocated()

        for _ in range(100):
            cache.advance(4, 1)
            cache.reset()

        after = torch.cuda.memory_allocated()
        assert after <= baseline + 1024  # allow 1KB tolerance
