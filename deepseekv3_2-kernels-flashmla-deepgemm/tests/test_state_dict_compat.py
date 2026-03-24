"""
State dict HuggingFace compatibility tests.

Verifies that:
  - Model state dict keys match expected HuggingFace naming conventions
  - Config round-trips through dict serialisation
  - Parameter shapes match the architecture specification
  - State dict can be loaded with strict=True

CPU-runnable.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config, DEEPSEEK_V3_CONFIG
from model import DeepSeekV3Model


class TestConfigSerialization:
    """Test config to_dict / from_dict round-trip."""

    def test_config_roundtrip(self):
        """Config should survive dict serialisation."""
        config = DEEPSEEK_V3_CONFIG
        d = config.to_dict()
        config2 = DeepSeekV3Config.from_dict(d)

        assert config2.hidden_size == config.hidden_size
        assert config2.num_hidden_layers == config.num_hidden_layers
        assert config2.q_lora_rank == config.q_lora_rank
        assert config2.kv_lora_rank == config.kv_lora_rank
        assert config2.qk_nope_head_dim == config.qk_nope_head_dim
        assert config2.qk_rope_head_dim == config.qk_rope_head_dim
        assert config2.v_head_dim == config.v_head_dim
        assert config2.moe.num_experts == config.moe.num_experts
        assert config2.moe.num_experts_per_tok == config.moe.num_experts_per_tok
        assert config2.moe.n_group == config.moe.n_group
        assert config2.moe.topk_group == config.moe.topk_group

    def test_config_derived_fields(self):
        """Derived fields should be computed correctly after from_dict."""
        d = DEEPSEEK_V3_CONFIG.to_dict()
        config = DeepSeekV3Config.from_dict(d)
        assert config.qk_head_dim == 192
        assert config.kv_compressed_dim == 576

    def test_full_config_values(self):
        """Full config should have the DeepSeek-V3 671B values."""
        c = DEEPSEEK_V3_CONFIG
        assert c.hidden_size == 7168
        assert c.num_attention_heads == 128
        assert c.num_hidden_layers == 61
        assert c.q_lora_rank == 1536
        assert c.kv_lora_rank == 512
        assert c.qk_nope_head_dim == 128
        assert c.qk_rope_head_dim == 64
        assert c.v_head_dim == 128
        assert c.vocab_size == 129280
        assert c.rms_norm_eps == 1e-6
        assert c.moe.num_experts == 256
        assert c.moe.num_experts_per_tok == 8
        assert c.moe.n_group == 8
        assert c.moe.topk_group == 4
        assert c.moe.routing_type == "sigmoid"
        assert c.moe.aux_loss_type == "noaux_tc"
        assert c.moe.first_k_dense_replace == 3
        assert c.mtp.num_mtp_layers == 1
        assert c.rope_scaling.factor == 40.0
        assert c.rope_scaling.original_max_position_embeddings == 4096
        assert c.rope_scaling.beta_fast == 32.0
        assert c.rope_scaling.beta_slow == 1.0


class TestStateDictKeys:
    """Test that state dict keys follow expected naming."""

    def test_small_model_keys(self, small_model, small_config):
        """Small model state dict should have expected key patterns."""
        sd = small_model.state_dict()

        # Check embedding
        assert "embed_tokens.weight" in sd

        # Check layer structure
        for i in range(small_config.num_hidden_layers):
            prefix = f"layers.{i}"
            assert f"{prefix}.input_layernorm.weight" in sd
            assert f"{prefix}.post_attention_layernorm.weight" in sd

            # Attention keys
            assert f"{prefix}.self_attn.q_a_proj.weight" in sd
            assert f"{prefix}.self_attn.q_b_proj.weight" in sd
            assert f"{prefix}.self_attn.kv_a_proj.weight" in sd
            assert f"{prefix}.self_attn.kv_b_proj.weight" in sd
            assert f"{prefix}.self_attn.o_proj.weight" in sd

        # Check final norm and lm_head
        assert "norm.weight" in sd
        assert "lm_head.weight" in sd

    def test_moe_layer_keys(self, small_model, small_config):
        """MoE layers should have router and expert keys."""
        sd = small_model.state_dict()

        # MoE layer (after first_k_dense_replace)
        moe_idx = small_config.moe.first_k_dense_replace
        if moe_idx < small_config.num_hidden_layers:
            prefix = f"layers.{moe_idx}.mlp"
            assert f"{prefix}.router.gate.weight" in sd

    def test_dense_layer_keys(self, small_model, small_config):
        """Dense layers should have standard FFN keys."""
        sd = small_model.state_dict()
        prefix = "layers.0.mlp"  # First layer is always dense
        # Dense FFN or MoE dense_ffn
        has_ffn = any(k.startswith(prefix) for k in sd.keys())
        assert has_ffn


class TestParameterShapes:
    """Test that parameter shapes match architecture specification."""

    def test_attention_shapes(self, small_model, small_config):
        """Attention parameter shapes should match config."""
        layer = small_model.layers[0]
        attn = layer.self_attn

        c = small_config
        H = c.num_attention_heads

        assert attn.q_a_proj.weight.shape == (c.q_lora_rank, c.hidden_size)
        assert attn.q_b_proj.weight.shape == (H * c.qk_head_dim, c.q_lora_rank)
        assert attn.kv_a_proj.weight.shape == (
            c.kv_lora_rank + c.qk_rope_head_dim, c.hidden_size
        )
        assert attn.kv_b_proj.weight.shape == (
            H * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank
        )
        assert attn.o_proj.weight.shape == (c.hidden_size, H * c.v_head_dim)

    def test_embedding_shape(self, small_model, small_config):
        """Embedding should be [vocab_size, hidden_size]."""
        assert small_model.embed_tokens.weight.shape == (
            small_config.vocab_size, small_config.hidden_size
        )

    def test_lm_head_shape(self, small_model, small_config):
        """LM head should be [vocab_size, hidden_size]."""
        assert small_model.lm_head.weight.shape == (
            small_config.vocab_size, small_config.hidden_size
        )


class TestStateDictLoading:
    """Test state dict save/load cycle."""

    def test_load_strict(self, small_model, small_config):
        """State dict should load with strict=True."""
        sd = small_model.state_dict()
        model2 = DeepSeekV3Model(small_config)
        model2.load_state_dict(sd, strict=True)

    def test_loaded_model_produces_same_output(self, small_model, small_config):
        """Loaded model should produce identical outputs."""
        sd = small_model.state_dict()
        model2 = DeepSeekV3Model(small_config)
        model2.load_state_dict(sd, strict=True)
        model2.eval()

        torch.manual_seed(123)
        input_ids = torch.randint(0, small_config.vocab_size, (1, 4))

        with torch.no_grad():
            out1 = small_model(input_ids)["logits"]
            out2 = model2(input_ids)["logits"]

        torch.testing.assert_close(out1, out2)
