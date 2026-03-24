"""
Edge case tests.

Covers unusual but valid inputs:
  - Empty batches (batch_size=0)
  - Single-token sequences (seq_len=1)
  - Very long sequences
  - Single-batch inputs
  - All-same input tokens
  - Extreme values

CPU-runnable.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config
from model import DeepSeekV3Model
from cache import KVCache
from moe_router import MoERouter
from mla_attention import MLAttention, eager_attention_forward
from rope_partial import YaRNRotaryEmbedding
from fp8_utils import quantize_fp8_block, dequantize_fp8_block
from dsa_indexer import DSAIndexer
from dsa_sparse_attention import DSASparseAttention


class TestSingleTokenSequence:
    """Test with seq_len=1 (decode step)."""

    def test_model_forward_seq1(self, small_model, small_config):
        """Model should handle seq_len=1."""
        B = 1
        input_ids = torch.randint(0, small_config.vocab_size, (B, 1))
        with torch.no_grad():
            out = small_model(input_ids)
        assert out["logits"].shape == (B, 1, small_config.vocab_size)

    def test_attention_seq1(self, small_config):
        """Attention should work with seq_len=1."""
        torch.manual_seed(42)
        attn = MLAttention(small_config, layer_idx=0)
        B, S, D = 1, 1, small_config.hidden_size
        x = torch.randn(B, S, D)
        with torch.no_grad():
            out, _ = attn(x)
        assert out.shape == (B, S, D)

    def test_rope_seq1(self, small_config):
        """RoPE should work with seq_len=1."""
        rope = YaRNRotaryEmbedding(small_config)
        dim = small_config.qk_rope_head_dim
        q = torch.randn(1, 1, 1, dim)
        k = torch.randn(1, 1, 1, dim)
        q_out, k_out = rope(q, k, seq_len=1)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


class TestSingleBatch:
    """Test with batch_size=1."""

    def test_model_batch1(self, small_model, small_config):
        """Model should handle batch_size=1."""
        input_ids = torch.randint(0, small_config.vocab_size, (1, 8))
        with torch.no_grad():
            out = small_model(input_ids)
        assert out["logits"].shape == (1, 8, small_config.vocab_size)

    def test_router_single_token(self, small_config):
        """Router should work with a single token."""
        router = MoERouter(small_config)
        x = torch.randn(1, small_config.hidden_size)
        weights, indices, logits = router(x)
        assert weights.shape == (1, small_config.moe.num_experts_per_tok)


class TestLongSequence:
    """Test with longer sequences."""

    def test_model_longer_seq(self, small_model, small_config):
        """Model should handle moderately long sequences."""
        B = 1
        S = 64  # longer than typical test
        input_ids = torch.randint(0, small_config.vocab_size, (B, S))
        with torch.no_grad():
            out = small_model(input_ids)
        assert out["logits"].shape == (B, S, small_config.vocab_size)

    def test_rope_longer_seq(self, small_config):
        """RoPE should extend to longer sequences."""
        rope = YaRNRotaryEmbedding(small_config)
        dim = small_config.qk_rope_head_dim
        S = 128
        q = torch.randn(1, 1, S, dim)
        k = torch.randn(1, 1, S, dim)
        q_out, k_out = rope(q, k, seq_len=S)
        assert q_out.shape == q.shape


class TestAllSameTokens:
    """Test with all-identical input tokens."""

    def test_same_token_input(self, small_model, small_config):
        """Model should handle all-same tokens (e.g., padding)."""
        B, S = 1, 8
        input_ids = torch.zeros(B, S, dtype=torch.long)
        with torch.no_grad():
            out = small_model(input_ids)
        assert torch.isfinite(out["logits"]).all()

    def test_same_hidden_states(self, small_config):
        """Router should handle identical hidden states."""
        router = MoERouter(small_config)
        T = 4
        x = torch.ones(T, small_config.hidden_size)
        weights, indices, _ = router(x)
        # All tokens should get the same routing
        for t in range(1, T):
            torch.testing.assert_close(weights[0], weights[t])
            torch.testing.assert_close(indices[0], indices[t])


class TestExtremeValues:
    """Test with extreme input values."""

    def test_large_input_values(self, small_config):
        """Model components should handle large inputs gracefully."""
        torch.manual_seed(42)
        attn = MLAttention(small_config, layer_idx=0)
        B, S, D = 1, 4, small_config.hidden_size
        x = torch.randn(B, S, D) * 100  # large values

        with torch.no_grad():
            out, _ = attn(x)
        # Should still be finite (RMSNorm helps)
        assert torch.isfinite(out).all()

    def test_near_zero_input(self, small_config):
        """Near-zero inputs should produce finite outputs."""
        torch.manual_seed(42)
        attn = MLAttention(small_config, layer_idx=0)
        B, S, D = 1, 4, small_config.hidden_size
        x = torch.randn(B, S, D) * 1e-8

        with torch.no_grad():
            out, _ = attn(x)
        assert torch.isfinite(out).all()


class TestDSAStubs:
    """Test DSA stub modules (not applicable to DeepSeek-V3)."""

    def test_dsa_indexer_is_noop(self):
        """DSA indexer should be a no-op."""
        indexer = DSAIndexer()
        assert indexer.is_applicable() is False
        assert indexer.build_index() is None
        assert indexer.get_sparse_mask() is None
        assert indexer.get_block_indices() is None

    def test_dsa_sparse_attention_delegates(self):
        """DSA sparse attention should delegate to eager attention."""
        dsa = DSASparseAttention()
        assert dsa.is_applicable() is False

        B, H, S, D = 1, 2, 4, 8
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        out = dsa(q, k, v, causal=True)
        assert out.shape == (B, H, S, D)

    def test_dsa_repr(self):
        """DSA stubs should have informative repr."""
        indexer = DSAIndexer()
        assert "stub" in repr(indexer).lower()
        dsa = DSASparseAttention()
        assert "stub" in repr(dsa).lower()


class TestEagerAttentionEdgeCases:
    """Test eager attention with edge cases."""

    def test_single_head(self):
        """Single attention head should work."""
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        out = eager_attention_forward(q, k, v, scale=8**-0.5, causal=True)
        assert out.shape == (1, 1, 4, 8)

    def test_different_qkv_seq_lens(self):
        """Q and KV can have different sequence lengths (decode)."""
        q = torch.randn(1, 2, 1, 8)  # query seq_len = 1
        k = torch.randn(1, 2, 10, 8)  # kv seq_len = 10
        v = torch.randn(1, 2, 10, 8)
        out = eager_attention_forward(q, k, v, scale=8**-0.5, causal=True)
        assert out.shape == (1, 2, 1, 8)


class TestMTPOutput:
    """Test Multi-Token Prediction output."""

    def test_mtp_logits_produced(self, small_model, small_config):
        """Model should produce MTP logits when requested."""
        B, S = 1, 4
        input_ids = torch.randint(0, small_config.vocab_size, (B, S))
        with torch.no_grad():
            out = small_model(input_ids, output_mtp_logits=True)

        assert out["mtp_logits"] is not None
        assert len(out["mtp_logits"]) == small_config.mtp.num_mtp_layers
        for mtp in out["mtp_logits"]:
            assert mtp.shape == (B, S, small_config.vocab_size)

    def test_mtp_logits_not_produced_by_default(self, small_model, small_config):
        """MTP logits should not be produced unless requested."""
        B, S = 1, 4
        input_ids = torch.randint(0, small_config.vocab_size, (B, S))
        with torch.no_grad():
            out = small_model(input_ids)
        assert out["mtp_logits"] is None
