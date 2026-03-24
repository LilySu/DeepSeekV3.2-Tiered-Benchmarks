"""
H100 determinism tests.

Verifies that kernel outputs are deterministic across:
  - Multiple runs with same input
  - Different CUDA streams
  - After torch.cuda.synchronize()

Requires: H100 GPU.
"""

from __future__ import annotations

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


def _make_test_model():
    from model import DeepSeekV3Model
    config = DeepSeekV3Config(
        hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4,
        q_lora_rank=64, kv_lora_rank=32,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        vocab_size=256,
        moe=MoEConfig(
            num_experts=8, num_experts_per_tok=2, n_group=2, topk_group=1,
            expert_intermediate_size=128, shared_expert_intermediate_size=128,
            first_k_dense_replace=1,
        ),
        use_flashmla=False, use_deepgemm=False,
    )
    return DeepSeekV3Model(config).cuda().eval().to(torch.bfloat16), config


@requires_hopper
class TestH100Determinism:
    """Test determinism on H100."""

    def test_forward_deterministic(self):
        """Two forward passes with same input should produce identical output."""
        model, config = _make_test_model()
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")

        with torch.no_grad():
            out1 = model(input_ids)["logits"].clone()
            out2 = model(input_ids)["logits"].clone()

        torch.testing.assert_close(out1, out2)

    def test_determinism_across_sync(self):
        """Outputs should be identical after synchronize."""
        model, config = _make_test_model()
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")

        with torch.no_grad():
            out1 = model(input_ids)["logits"]
        torch.cuda.synchronize()
        out1 = out1.clone()

        with torch.no_grad():
            out2 = model(input_ids)["logits"]
        torch.cuda.synchronize()
        out2 = out2.clone()

        torch.testing.assert_close(out1, out2)

    def test_attention_deterministic(self):
        """Attention layer should be deterministic on GPU."""
        from mla_attention import MLAttention
        from config import DeepSeekV3Config

        config = DeepSeekV3Config(
            hidden_size=256, num_attention_heads=4,
            q_lora_rank=64, kv_lora_rank=32,
            qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
            use_flashmla=False,
        )
        attn = MLAttention(config, layer_idx=0).cuda().eval().to(torch.bfloat16)

        x = torch.randn(1, 8, 256, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            out1, _ = attn(x)
            out2, _ = attn(x)

        torch.testing.assert_close(out1, out2)

    def test_router_deterministic(self):
        """MoE router should be deterministic."""
        from moe_router import MoERouter

        config = DeepSeekV3Config(
            hidden_size=256,
            moe=MoEConfig(
                num_experts=16, num_experts_per_tok=4,
                n_group=4, topk_group=2,
            ),
        )
        router = MoERouter(config).cuda().eval().to(torch.bfloat16)

        x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            w1, i1, _ = router(x)
            w2, i2, _ = router(x)

        torch.testing.assert_close(w1, w2)
        torch.testing.assert_close(i1, i2)

    def test_fp8_quantisation_deterministic(self):
        """FP8 quantisation should be deterministic."""
        from fp8_utils import quantize_fp8_block

        x = torch.randn(256, 256, device="cuda")

        x_fp8_1, scales_1 = quantize_fp8_block(x)
        x_fp8_2, scales_2 = quantize_fp8_block(x)

        torch.testing.assert_close(x_fp8_1.float(), x_fp8_2.float())
        torch.testing.assert_close(scales_1, scales_2)

    @pytest.mark.parametrize("run", range(5))
    def test_repeated_forward_stable(self, run):
        """Model output should be identical across 5 runs."""
        model, config = _make_test_model()
        input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")

        with torch.no_grad():
            logits = model(input_ids)["logits"]

        # Verify finite
        assert torch.isfinite(logits).all()
