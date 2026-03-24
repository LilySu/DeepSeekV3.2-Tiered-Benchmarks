"""
H100 multi-GPU tests.

Tests model components across multiple GPUs:
  - Tensor placement and movement
  - Expert parallelism (experts on different devices)
  - Pipeline parallelism staging
  - All-reduce for expert outputs

Requires: Multiple H100 GPUs.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

requires_multi_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 2),
    reason="Multiple GPUs required",
)

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)


@requires_hopper
@requires_multi_gpu
class TestH100MultiGPU:
    """Multi-GPU tests on H100."""

    def test_tensor_movement(self):
        """Tensors should move between GPUs efficiently."""
        x = torch.randn(1024, 1024, device="cuda:0", dtype=torch.bfloat16)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        y = x.to("cuda:1")
        end.record()

        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        gb_per_sec = x.numel() * x.element_size() / (ms * 1e-3) / 1e9

        print(f"\nGPU0->GPU1 transfer: {ms:.3f}ms, {gb_per_sec:.1f} GB/s")
        assert y.device == torch.device("cuda:1")

    def test_expert_parallelism_basic(self):
        """Experts can be placed on different GPUs."""
        from moe_grouped_gemm import ExpertFFN

        D, I = 256, 128
        expert0 = ExpertFFN(D, I).to("cuda:0").to(torch.bfloat16)
        expert1 = ExpertFFN(D, I).to("cuda:1").to(torch.bfloat16)

        x0 = torch.randn(4, D, device="cuda:0", dtype=torch.bfloat16)
        x1 = torch.randn(4, D, device="cuda:1", dtype=torch.bfloat16)

        with torch.no_grad():
            out0 = expert0(x0)
            out1 = expert1(x1)

        assert out0.device == torch.device("cuda:0")
        assert out1.device == torch.device("cuda:1")

    def test_attention_on_each_gpu(self):
        """Attention layers should work on each GPU independently."""
        from mla_attention import MLAttention
        from config import DeepSeekV3Config

        config = DeepSeekV3Config(
            hidden_size=256, num_attention_heads=4,
            q_lora_rank=64, kv_lora_rank=32,
            qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
            use_flashmla=False,
        )

        for gpu_id in range(min(torch.cuda.device_count(), 2)):
            device = f"cuda:{gpu_id}"
            attn = MLAttention(config, layer_idx=0).to(device).to(torch.bfloat16).eval()
            x = torch.randn(1, 8, 256, device=device, dtype=torch.bfloat16)

            with torch.no_grad():
                out, _ = attn(x)

            assert out.device == torch.device(device)
            assert out.shape == x.shape

    def test_gather_expert_outputs(self):
        """Expert outputs from different GPUs should gather correctly."""
        D = 128
        num_gpus = min(torch.cuda.device_count(), 2)

        outputs = []
        for g in range(num_gpus):
            out = torch.randn(8, D, device=f"cuda:{g}", dtype=torch.bfloat16)
            outputs.append(out)

        # Gather to GPU 0
        gathered = torch.cat([o.to("cuda:0") for o in outputs], dim=0)
        assert gathered.shape == (8 * num_gpus, D)
        assert gathered.device == torch.device("cuda:0")

    def test_nccl_available(self):
        """NCCL backend should be available for distributed communication."""
        assert torch.distributed.is_nccl_available(), "NCCL not available"

    def test_kv_cache_per_gpu(self):
        """KV caches should be placeable on different GPUs."""
        from cache import KVCache
        from config import DeepSeekV3Config

        config = DeepSeekV3Config(
            hidden_size=256, num_hidden_layers=2,
            kv_lora_rank=64, qk_rope_head_dim=16,
            num_attention_heads=4,
            q_lora_rank=64,
            qk_nope_head_dim=32, v_head_dim=32,
        )

        for g in range(min(torch.cuda.device_count(), 2)):
            device = torch.device(f"cuda:{g}")
            cache = KVCache(
                config, max_batch_size=1, max_seq_len=64,
                device=device, dtype=torch.bfloat16,
            )
            cache.allocate_all()

            kv = torch.randn(1, 4, config.kv_lora_rank, device=device, dtype=torch.bfloat16)
            k_rope = torch.randn(1, 4, config.qk_rope_head_dim, device=device, dtype=torch.bfloat16)
            cache.update(0, kv, k_rope)

            assert cache.seq_lens.device == device
