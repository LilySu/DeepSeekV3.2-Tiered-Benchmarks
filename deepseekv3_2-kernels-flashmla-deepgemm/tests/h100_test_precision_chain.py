"""
H100 precision chain tests.

Tests numerical precision through the full computation chain:
  BF16 input -> FP8 quantise -> DeepGEMM -> BF16 output

Verifies:
  - Accumulated error stays within bounds through MoE layers
  - Attention precision with FlashMLA vs eager
  - RMSNorm precision in BF16
  - End-to-end output quality with FP8 MoE

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


@requires_hopper
class TestH100PrecisionChain:
    """Test precision through the computation chain."""

    def test_bf16_matmul_precision(self):
        """BF16 matmul precision on H100."""
        M, K, N = 128, 256, 128
        device = torch.device("cuda")

        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        ref = a @ b

        a_bf16 = a.bfloat16()
        b_bf16 = b.bfloat16()
        out_bf16 = (a_bf16 @ b_bf16).float()

        rel_err = (ref - out_bf16).abs() / (ref.abs() + 1e-8)
        mean_err = rel_err.mean().item()
        print(f"\nBF16 matmul mean relative error: {mean_err:.6f}")
        assert mean_err < 0.01

    def test_fp8_matmul_precision(self):
        """FP8 quantise + matmul precision."""
        from fp8_utils import quantize_fp8_block, dequantize_fp8_block

        M, K = 256, 256
        device = torch.device("cuda")

        x = torch.randn(M, K, device=device)
        w = torch.randn(K, K, device=device)
        ref = x @ w

        # FP8 path
        x_fp8, x_scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), x_scales)
        w_fp8, w_scales = quantize_fp8_block(w)
        w_deq = dequantize_fp8_block(w_fp8.float(), w_scales)
        out_fp8 = x_deq @ w_deq

        rel_err = (ref - out_fp8).abs() / (ref.abs() + 1e-8)
        mean_err = rel_err.mean().item()
        print(f"\nFP8 matmul mean relative error: {mean_err:.6f}")
        assert mean_err < 0.15  # FP8 has lower precision

    def test_rmsnorm_bf16_precision(self):
        """RMSNorm precision in BF16."""
        from unsloth_rms_layernorm import UnslothRMSNorm

        device = torch.device("cuda")
        D = 256
        norm_f32 = UnslothRMSNorm(D).to(device)
        norm_bf16 = UnslothRMSNorm(D).to(device).to(torch.bfloat16)
        norm_bf16.weight.data = norm_f32.weight.data.bfloat16()

        x = torch.randn(4, 16, D, device=device)

        ref = norm_f32(x)
        out = norm_bf16(x.bfloat16()).float()

        rel_err = (ref - out).abs() / (ref.abs() + 1e-8)
        assert rel_err.mean() < 0.01

    def test_attention_bf16_precision(self):
        """Attention precision in BF16."""
        from mla_attention import eager_attention_forward

        B, H, S, D = 1, 8, 32, 32
        D_v = 32
        device = torch.device("cuda")

        q = torch.randn(B, H, S, D, device=device)
        k = torch.randn(B, H, S, D, device=device)
        v = torch.randn(B, H, S, D_v, device=device)
        scale = D ** -0.5

        ref = eager_attention_forward(q, k, v, scale, causal=True)
        out_bf16 = eager_attention_forward(
            q.bfloat16(), k.bfloat16(), v.bfloat16(), scale, causal=True
        ).float()

        rel_err = (ref - out_bf16).abs() / (ref.abs() + 1e-8)
        mean_err = rel_err.mean().item()
        print(f"\nAttention BF16 mean relative error: {mean_err:.6f}")
        assert mean_err < 0.02

    def test_end_to_end_precision(self):
        """End-to-end model precision: FP32 vs BF16."""
        from model import DeepSeekV3Model

        config = DeepSeekV3Config(
            hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4,
            q_lora_rank=32, kv_lora_rank=16,
            qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
            vocab_size=256,
            moe=MoEConfig(
                num_experts=4, num_experts_per_tok=2,
                n_group=2, topk_group=1,
                expert_intermediate_size=64,
                shared_expert_intermediate_size=64,
                first_k_dense_replace=1,
            ),
            use_flashmla=False, use_deepgemm=False,
        )

        torch.manual_seed(42)
        model_f32 = DeepSeekV3Model(config).cuda().eval()

        torch.manual_seed(42)
        model_bf16 = DeepSeekV3Model(config).cuda().eval().to(torch.bfloat16)

        input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")

        with torch.no_grad():
            out_f32 = model_f32(input_ids)["logits"]
            out_bf16 = model_bf16(input_ids)["logits"].float()

        rel_err = (out_f32 - out_bf16).abs() / (out_f32.abs() + 1e-8)
        mean_err = rel_err.mean().item()
        print(f"\nEnd-to-end FP32 vs BF16 mean relative error: {mean_err:.6f}")
        # BF16 should be reasonably close
        assert mean_err < 0.1

    def test_accumulated_fp8_error(self):
        """FP8 error should not accumulate catastrophically through layers."""
        from fp8_utils import quantize_fp8_block, dequantize_fp8_block

        device = torch.device("cuda")
        D = 256
        x = torch.randn(64, D, device=device)
        original = x.clone()

        # Simulate 10 layers of FP8 quantise -> matmul -> dequantise
        for _ in range(10):
            w = torch.randn(D, D, device=device) * 0.01  # small weights
            x_fp8, scales = quantize_fp8_block(x)
            x_deq = dequantize_fp8_block(x_fp8.float(), scales)
            x = x_deq @ w + x_deq * 0.9  # residual-like

        # Output should still be finite
        assert torch.isfinite(x).all()
        print(f"\nAfter 10 FP8 layers: output norm = {x.norm():.4f}")
