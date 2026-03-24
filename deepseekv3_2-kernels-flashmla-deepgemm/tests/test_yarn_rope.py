"""
YaRN RoPE correctness tests -- DeepSeek-V3 specific.

Verifies the YaRN (Yet another RoPE extensioN) implementation:
  - Frequency interpolation between original and extended ranges
  - Beta_fast and beta_slow correction boundaries
  - Magnitude scaling (mscale)
  - Partial RoPE (only qk_rope_head_dim=64 affected)
  - Position-dependent encoding
  - Cache extension

CPU-runnable.

Reference: arXiv 2412.19437, Section 3.2.1.
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config, YaRNRoPEConfig
from rope_partial import (
    YaRNRotaryEmbedding,
    apply_partial_rope,
    _yarn_find_correction_dim,
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
    _yarn_get_mscale,
)


class TestYaRNHelpers:
    """Test YaRN helper functions."""

    def test_correction_dim_positive(self):
        """Correction dimension should be non-negative for valid inputs."""
        dim = _yarn_find_correction_dim(1.0, 64, 10000.0, 4096)
        assert dim >= 0

    def test_correction_range_ordered(self):
        """Low boundary should be <= high boundary."""
        low, high = _yarn_find_correction_range(1.0, 32.0, 64, 10000.0, 4096)
        assert low <= high

    def test_correction_range_within_dim(self):
        """Boundaries should be within [0, dim-1]."""
        dim = 64
        low, high = _yarn_find_correction_range(1.0, 32.0, dim, 10000.0, 4096)
        assert 0 <= low <= dim - 1
        assert 0 <= high <= dim - 1

    def test_linear_ramp_mask_bounds(self):
        """Linear ramp mask should be in [0, 1]."""
        mask = _yarn_linear_ramp_mask(5, 20, 32)
        assert (mask >= 0).all()
        assert (mask <= 1).all()

    def test_linear_ramp_mask_shape(self):
        """Mask shape should match dimension."""
        mask = _yarn_linear_ramp_mask(0, 10, 32)
        assert mask.shape == (32,)

    def test_mscale_identity_for_small_factor(self):
        """mscale should be 1.0 for factor <= 1."""
        assert _yarn_get_mscale(0.5, 1.0) == 1.0
        assert _yarn_get_mscale(1.0, 1.0) == 1.0

    def test_mscale_increases_with_factor(self):
        """mscale should increase with scaling factor."""
        s1 = _yarn_get_mscale(10.0, 1.0)
        s2 = _yarn_get_mscale(40.0, 1.0)
        assert s2 > s1

    def test_mscale_for_deepseek_v3(self):
        """DeepSeek-V3 mscale with factor=40."""
        s = _yarn_get_mscale(40.0, 1.0)
        # 0.1 * 1.0 * ln(40) + 1 = 0.1 * 3.689 + 1 = 1.3689
        expected = 0.1 * 1.0 * math.log(40.0) + 1.0
        assert abs(s - expected) < 1e-6


class TestYaRNFrequencies:
    """Test YaRN frequency computation."""

    def test_inv_freq_shape(self, small_config):
        """Inverse frequencies should have dim // 2 elements."""
        rope = YaRNRotaryEmbedding(small_config)
        expected_len = small_config.qk_rope_head_dim // 2
        assert rope.inv_freq.shape == (expected_len,)

    def test_inv_freq_positive(self, small_config):
        """All inverse frequencies should be positive."""
        rope = YaRNRotaryEmbedding(small_config)
        assert (rope.inv_freq > 0).all()

    def test_inv_freq_interpolated(self):
        """YaRN frequencies should be between original and fully interpolated."""
        config = DeepSeekV3Config(
            qk_rope_head_dim=64,
            rope_scaling=YaRNRoPEConfig(factor=40.0, original_max_position_embeddings=4096),
        )
        rope = YaRNRotaryEmbedding(config)

        # Original (no interpolation) frequencies
        dim = 64
        base = 10000.0
        orig_freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        # Fully interpolated
        interp_freqs = orig_freqs / 40.0

        # YaRN frequencies should be between the two
        for i in range(len(rope.inv_freq)):
            assert rope.inv_freq[i] >= interp_freqs[i] - 1e-6
            assert rope.inv_freq[i] <= orig_freqs[i] + 1e-6

    def test_attention_scale_positive(self, small_config):
        """Attention scale should be positive."""
        rope = YaRNRotaryEmbedding(small_config)
        assert rope.get_attention_scale().item() > 0


class TestRoPEApplication:
    """Test RoPE application to Q and K."""

    def test_output_shape_matches_input(self, small_config):
        """Output shapes should match input shapes."""
        rope = YaRNRotaryEmbedding(small_config)
        dim = small_config.qk_rope_head_dim
        B, H, S = 2, 4, 8

        q = torch.randn(B, H, S, dim)
        k = torch.randn(B, 1, S, dim)

        q_out, k_out = rope(q, k)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_position_dependent(self, small_config):
        """Different positions should produce different encodings."""
        rope = YaRNRotaryEmbedding(small_config)
        dim = small_config.qk_rope_head_dim

        # Same vector at different positions
        v = torch.randn(1, 1, 1, dim)
        q1 = v.clone()
        k1 = v.clone()
        q2 = v.clone()
        k2 = v.clone()

        pos1 = torch.tensor([[0]])
        pos2 = torch.tensor([[100]])

        q1_out, _ = rope(q1, k1, position_ids=pos1)
        q2_out, _ = rope(q2, k2, position_ids=pos2)

        assert not torch.allclose(q1_out, q2_out, atol=1e-6)

    def test_relative_distance_property(self, small_config):
        """Dot product should depend on relative position, not absolute."""
        rope = YaRNRotaryEmbedding(small_config)
        dim = small_config.qk_rope_head_dim

        q = torch.randn(1, 1, 1, dim)
        k = torch.randn(1, 1, 1, dim)

        # Position 0 & 5 (distance 5)
        pos_q_a = torch.tensor([[0]])
        pos_k_a = torch.tensor([[5]])
        q_a, _ = rope(q.clone(), k.clone(), position_ids=pos_q_a)
        _, k_a = rope(q.clone(), k.clone(), position_ids=pos_k_a)
        dot_a = (q_a * k_a).sum()

        # Position 10 & 15 (same distance 5)
        pos_q_b = torch.tensor([[10]])
        pos_k_b = torch.tensor([[15]])
        q_b, _ = rope(q.clone(), k.clone(), position_ids=pos_q_b)
        _, k_b = rope(q.clone(), k.clone(), position_ids=pos_k_b)
        dot_b = (q_b * k_b).sum()

        # Dot products should be approximately equal (same relative distance)
        torch.testing.assert_close(dot_a, dot_b, atol=0.1, rtol=0.1)


class TestPartialRoPE:
    """Test partial RoPE application."""

    def test_nope_dimensions_unchanged(self, small_config):
        """Non-RoPE dimensions should not be modified."""
        rope = YaRNRotaryEmbedding(small_config)
        nope = small_config.qk_nope_head_dim
        rope_dim = small_config.qk_rope_head_dim
        total = nope + rope_dim

        q = torch.randn(1, 4, 8, total)
        k = torch.randn(1, 1, 8, total)

        q_out, k_out = apply_partial_rope(q, k, rope, nope, rope_dim)

        torch.testing.assert_close(q[..., :nope], q_out[..., :nope])
        torch.testing.assert_close(k[..., :nope], k_out[..., :nope])

    def test_rope_dimensions_modified(self, small_config):
        """RoPE dimensions should be modified (unless position=0)."""
        rope = YaRNRotaryEmbedding(small_config)
        nope = small_config.qk_nope_head_dim
        rope_dim = small_config.qk_rope_head_dim
        total = nope + rope_dim

        q = torch.randn(1, 4, 8, total)
        k = torch.randn(1, 1, 8, total)

        q_out, k_out = apply_partial_rope(q, k, rope, nope, rope_dim)

        # RoPE dimensions should change (for positions > 0)
        rope_slice_q = q[..., nope:nope + rope_dim]
        rope_slice_q_out = q_out[..., nope:nope + rope_dim]
        # At least some positions should differ
        assert not torch.allclose(rope_slice_q[:, :, 1:], rope_slice_q_out[:, :, 1:], atol=1e-6)


class TestCacheExtension:
    """Test cos/sin cache extension."""

    def test_cache_extends_on_demand(self, small_config):
        """Cache should extend when longer sequences are requested."""
        rope = YaRNRotaryEmbedding(small_config)
        dim = small_config.qk_rope_head_dim

        # Short sequence
        q1 = torch.randn(1, 1, 4, dim)
        k1 = torch.randn(1, 1, 4, dim)
        rope(q1, k1, seq_len=4)
        assert rope._cached_seq_len >= 4

        # Longer sequence
        q2 = torch.randn(1, 1, 32, dim)
        k2 = torch.randn(1, 1, 32, dim)
        rope(q2, k2, seq_len=32)
        assert rope._cached_seq_len >= 32

    def test_cache_reuse(self, small_config):
        """Repeated calls with same length should reuse cache."""
        rope = YaRNRotaryEmbedding(small_config)
        dim = small_config.qk_rope_head_dim

        q = torch.randn(1, 1, 8, dim)
        k = torch.randn(1, 1, 8, dim)

        rope(q, k, seq_len=8)
        cached_len_1 = rope._cached_seq_len

        rope(q, k, seq_len=8)
        cached_len_2 = rope._cached_seq_len

        assert cached_len_1 == cached_len_2
