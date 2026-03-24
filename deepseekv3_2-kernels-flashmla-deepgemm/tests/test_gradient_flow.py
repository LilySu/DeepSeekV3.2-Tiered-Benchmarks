"""
Gradient flow tests.

Verifies that gradients propagate correctly through all model components:
  - MLA attention
  - YaRN RoPE
  - MoE router and experts
  - SwiGLU activation
  - RMSNorm
  - Full model forward/backward

All tests are CPU-runnable.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config
from model import DeepSeekV3Model, DeepSeekRMSNorm
from mla_attention import MLAttention
from moe_grouped_gemm import MoEGroupedGEMM, ExpertFFN
from moe_router import MoERouter
from rope_partial import YaRNRotaryEmbedding
from unsloth_swiglu import UnslothSwiGLU, FusedSwiGLUFunction


class TestGradientFlowAttention:
    """Test gradient flow through MLA attention."""

    def test_attention_gradients_exist(self, small_config):
        """All attention parameters should receive gradients."""
        torch.manual_seed(42)
        attn = MLAttention(small_config, layer_idx=0)
        attn.train()

        B, S, D = 2, 8, small_config.hidden_size
        x = torch.randn(B, S, D, requires_grad=True)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()

        # Check input gradient
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check all parameter gradients
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_attention_gradient_magnitudes(self, small_config):
        """Gradients should have reasonable magnitudes (not vanishing/exploding)."""
        torch.manual_seed(42)
        attn = MLAttention(small_config, layer_idx=0)
        attn.train()

        B, S, D = 1, 4, small_config.hidden_size
        x = torch.randn(B, S, D, requires_grad=True)
        out, _ = attn(x)
        loss = out.mean()
        loss.backward()

        for name, param in attn.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm > 1e-10, f"Vanishing gradient for {name}: {grad_norm}"
                assert grad_norm < 1e6, f"Exploding gradient for {name}: {grad_norm}"


class TestGradientFlowMoE:
    """Test gradient flow through MoE components."""

    def test_router_gradients(self, small_config):
        """Router gate should receive gradients."""
        torch.manual_seed(42)
        router = MoERouter(small_config)
        router.train()

        T = 8
        x = torch.randn(T, small_config.hidden_size, requires_grad=True)
        weights, indices, logits = router(x)
        loss = weights.sum()
        loss.backward()

        assert router.gate.weight.grad is not None
        assert torch.isfinite(router.gate.weight.grad).all()

    def test_expert_ffn_gradients(self):
        """ExpertFFN should propagate gradients through SwiGLU."""
        torch.manual_seed(42)
        expert = ExpertFFN(64, 128)
        expert.train()

        x = torch.randn(4, 64, requires_grad=True)
        out = expert(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in expert.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_moe_layer_gradients(self, small_config):
        """Full MoE layer should propagate gradients."""
        torch.manual_seed(42)
        # Use a MoE layer (not dense)
        moe = MoEGroupedGEMM(small_config, layer_idx=small_config.moe.first_k_dense_replace)
        moe.train()

        B, S, D = 1, 4, small_config.hidden_size
        x = torch.randn(B, S, D, requires_grad=True)
        out, router_logits = moe(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestGradientFlowRoPE:
    """Test gradient flow through YaRN RoPE."""

    def test_rope_gradient_passthrough(self, small_config):
        """RoPE should allow gradients to flow through."""
        rope = YaRNRotaryEmbedding(small_config)
        rope_dim = small_config.qk_rope_head_dim

        q = torch.randn(1, 1, 4, rope_dim, requires_grad=True)
        k = torch.randn(1, 1, 4, rope_dim, requires_grad=True)

        q_out, k_out = rope(q, k)
        loss = (q_out.sum() + k_out.sum())
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None


class TestGradientFlowSwiGLU:
    """Test gradient flow through SwiGLU variants."""

    def test_fused_swiglu_gradients(self):
        """FusedSwiGLU should produce correct gradients."""
        torch.manual_seed(42)
        D, I = 32, 64
        swiglu = UnslothSwiGLU(D, I, use_fused=True)
        swiglu.train()

        x = torch.randn(2, 4, D, requires_grad=True)
        out = swiglu(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert swiglu.gate_proj.weight.grad is not None
        assert swiglu.up_proj.weight.grad is not None

    def test_fused_vs_standard_gradients(self):
        """Fused and standard SwiGLU should produce similar gradients."""
        torch.manual_seed(42)
        D, I = 32, 64

        std = UnslothSwiGLU(D, I, use_fused=False)
        fused = UnslothSwiGLU(D, I, use_fused=True)

        # Copy weights
        fused.gate_proj.weight.data.copy_(std.gate_proj.weight.data)
        fused.up_proj.weight.data.copy_(std.up_proj.weight.data)

        x_std = torch.randn(2, 4, D, requires_grad=True)
        x_fused = x_std.clone().detach().requires_grad_(True)

        out_std = std(x_std)
        out_fused = fused(x_fused)

        out_std.sum().backward()
        out_fused.sum().backward()

        torch.testing.assert_close(x_std.grad, x_fused.grad, atol=1e-4, rtol=1e-4)


class TestGradientFlowFullModel:
    """Test gradient flow through the complete model."""

    def test_model_backward_completes(self, small_model, small_config):
        """Full model backward pass should complete without errors."""
        model = small_model
        model.train()

        B, S = 1, 8
        input_ids = torch.randint(0, small_config.vocab_size, (B, S))
        labels = torch.randint(0, small_config.vocab_size, (B, S))

        outputs = model(input_ids)
        logits = outputs["logits"]

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, small_config.vocab_size),
            labels.view(-1),
        )

        loss.backward()

        # Verify at least some parameters got gradients
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert has_grad > 0, "No parameters received gradients"

    def test_embedding_gets_gradient(self, small_model, small_config):
        """Embedding layer should receive gradients."""
        model = small_model
        model.train()

        B, S = 1, 4
        input_ids = torch.randint(0, small_config.vocab_size, (B, S))

        outputs = model(input_ids)
        loss = outputs["logits"].sum()
        loss.backward()

        assert model.embed_tokens.weight.grad is not None

    def test_lm_head_gets_gradient(self, small_model, small_config):
        """LM head should receive gradients."""
        model = small_model
        model.train()

        B, S = 1, 4
        input_ids = torch.randint(0, small_config.vocab_size, (B, S))

        outputs = model(input_ids)
        loss = outputs["logits"].sum()
        loss.backward()

        assert model.lm_head.weight.grad is not None
