"""
H100 thermal stress tests.

Sustained workload tests to verify kernel stability under thermal load:
  - Sustained attention computation
  - Sustained MoE dispatch
  - Mixed workload (attention + MoE alternating)
  - Output consistency under thermal throttling

Requires: H100 GPU.
"""

from __future__ import annotations

import sys
import os
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config, MoEConfig

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)


def _make_stress_model():
    from model import DeepSeekV3Model
    config = DeepSeekV3Config(
        hidden_size=512, intermediate_size=1024,
        num_hidden_layers=4, num_attention_heads=8,
        q_lora_rank=128, kv_lora_rank=64,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        vocab_size=1024,
        moe=MoEConfig(
            num_experts=32, num_experts_per_tok=4,
            n_group=4, topk_group=2,
            expert_intermediate_size=256,
            shared_expert_intermediate_size=256,
            first_k_dense_replace=1,
        ),
        use_flashmla=False, use_deepgemm=False,
    )
    return DeepSeekV3Model(config).cuda().eval().to(torch.bfloat16), config


@requires_hopper
class TestH100Thermal:
    """Thermal stress tests on H100."""

    def test_sustained_attention(self):
        """Run attention for 30 seconds, verify output stability."""
        from mla_attention import MLAttention

        config = DeepSeekV3Config(
            hidden_size=512, num_attention_heads=8,
            q_lora_rank=128, kv_lora_rank=64,
            qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
            use_flashmla=False,
        )
        attn = MLAttention(config, layer_idx=0).cuda().eval().to(torch.bfloat16)

        B, S = 4, 128
        x = torch.randn(B, S, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        # Get reference output
        with torch.no_grad():
            ref, _ = attn(x)
            ref = ref.clone()

        # Sustained run
        start = time.time()
        iterations = 0
        max_diff = 0.0

        with torch.no_grad():
            while time.time() - start < 10:  # 10 seconds
                out, _ = attn(x)
                diff = (out - ref).abs().max().item()
                max_diff = max(max_diff, diff)
                iterations += 1

        duration = time.time() - start
        print(f"\nSustained attention: {iterations} iters in {duration:.1f}s")
        print(f"Max output deviation: {max_diff:.8f}")
        assert max_diff < 1e-4, f"Output drifted by {max_diff}"

    def test_sustained_moe(self):
        """Run MoE dispatch for 30 seconds."""
        from moe_grouped_gemm import MoEGroupedGEMM

        config = DeepSeekV3Config(
            hidden_size=512,
            moe=MoEConfig(
                num_experts=32, num_experts_per_tok=4,
                n_group=4, topk_group=2,
                expert_intermediate_size=256,
                shared_expert_intermediate_size=256,
                first_k_dense_replace=0,
            ),
        )
        moe = MoEGroupedGEMM(config, layer_idx=0).cuda().eval().to(torch.bfloat16)

        B, S = 2, 64
        x = torch.randn(B, S, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        start = time.time()
        iterations = 0

        with torch.no_grad():
            while time.time() - start < 10:
                out, _ = moe(x)
                iterations += 1

        duration = time.time() - start
        print(f"\nSustained MoE: {iterations} iters in {duration:.1f}s")
        print(f"Throughput: {iterations * B * S / duration:.0f} tok/s")

    def test_sustained_full_model(self):
        """Run full model forward for sustained period."""
        model, config = _make_stress_model()
        B, S = 1, 64
        input_ids = torch.randint(0, config.vocab_size, (B, S), device="cuda")

        start = time.time()
        iterations = 0

        with torch.no_grad():
            while time.time() - start < 15:
                model(input_ids)
                iterations += 1

        duration = time.time() - start
        print(f"\nSustained model forward: {iterations} iters in {duration:.1f}s")
        print(f"Throughput: {iterations * B * S / duration:.0f} tok/s")

    def test_output_consistency_under_load(self):
        """Output should remain consistent even under sustained GPU load."""
        model, config = _make_stress_model()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")

        # Get baseline
        with torch.no_grad():
            baseline = model(input_ids)["logits"].clone()

        # Run under load for a while
        dummy = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(100):
            torch.mm(dummy, dummy.T)  # GPU stress

        # Check output
        with torch.no_grad():
            after_load = model(input_ids)["logits"]

        torch.testing.assert_close(baseline, after_load)

    def test_gpu_temperature_accessible(self):
        """GPU temperature should be queryable (for monitoring)."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                temps = [int(t.strip()) for t in result.stdout.strip().split("\n")]
                for i, t in enumerate(temps):
                    print(f"\nGPU {i} temperature: {t}C")
                    assert t < 90, f"GPU {i} temperature too high: {t}C"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("nvidia-smi not available")
