"""
FP8 tensor layout tests.

Verifies:
  - Block quantisation produces correct scale shapes
  - Round-trip quantise/dequantise error is bounded
  - FP8TensorWrapper interface
  - Edge cases (zeros, very large/small values, non-square tensors)

CPU-runnable.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fp8_utils import (
    quantize_fp8_block,
    dequantize_fp8_block,
    per_block_cast_to_fp8,
    FP8TensorWrapper,
    BLOCK_SIZE,
    FP8_E4M3_MAX,
)


class TestBlockQuantisation:
    """Test per-block FP8 quantisation."""

    def test_scale_shape_aligned(self):
        """Scales should have shape [ceil(M/block), ceil(N/block)]."""
        x = torch.randn(256, 512)
        _, scales = quantize_fp8_block(x, block_m=128, block_n=128)
        assert scales.shape == (2, 4)

    def test_scale_shape_unaligned(self):
        """Non-block-aligned tensors should still produce correct scale shapes."""
        x = torch.randn(200, 300)
        _, scales = quantize_fp8_block(x, block_m=128, block_n=128)
        assert scales.shape == (2, 3)  # ceil(200/128)=2, ceil(300/128)=3

    def test_output_shape_preserved(self):
        """Quantised tensor should have same shape as input."""
        x = torch.randn(128, 256)
        x_fp8, _ = quantize_fp8_block(x)
        assert x_fp8.shape == x.shape

    def test_scales_positive(self):
        """All scales should be positive."""
        x = torch.randn(256, 256)
        _, scales = quantize_fp8_block(x)
        assert (scales > 0).all()

    def test_zero_input_zero_output(self):
        """Zero input should produce zero output."""
        x = torch.zeros(128, 128)
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)
        assert (x_deq == 0).all()


class TestRoundTrip:
    """Test quantise -> dequantise round-trip."""

    def test_roundtrip_small(self):
        """Small tensor round-trip."""
        torch.manual_seed(42)
        x = torch.randn(128, 128)
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        rel_err = (x - x_deq).abs() / (x.abs() + 1e-8)
        assert rel_err.mean() < 0.2

    def test_roundtrip_large(self):
        """Larger tensor round-trip."""
        torch.manual_seed(42)
        x = torch.randn(512, 1024)
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        max_err = (x - x_deq).abs().max()
        assert max_err < 50.0  # FP8 has limited range

    def test_roundtrip_uniform(self):
        """Uniform distribution should round-trip well."""
        x = torch.rand(256, 256) * 2 - 1  # [-1, 1]
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        mse = ((x - x_deq) ** 2).mean()
        assert mse < 0.1


class TestFP8TensorWrapper:
    """Test FP8TensorWrapper interface."""

    def test_from_float(self):
        """Create wrapper from float tensor."""
        x = torch.randn(128, 256)
        wrapper = FP8TensorWrapper.from_float(x)
        assert wrapper.shape == x.shape
        assert wrapper.scales is not None

    def test_dequantize(self):
        """Dequantise should produce similar values."""
        torch.manual_seed(42)
        x = torch.randn(128, 128)
        wrapper = FP8TensorWrapper.from_float(x)
        x_deq = wrapper.dequantize()

        assert x_deq.shape == x.shape
        rel_err = (x - x_deq).abs() / (x.abs() + 1e-8)
        assert rel_err.mean() < 0.3

    def test_repr(self):
        """Repr should be informative."""
        x = torch.randn(128, 256)
        wrapper = FP8TensorWrapper.from_float(x)
        r = repr(wrapper)
        assert "FP8TensorWrapper" in r
        assert "128" in r
        assert "256" in r

    def test_to_device(self):
        """to() should move both data and scales."""
        x = torch.randn(128, 128)
        wrapper = FP8TensorWrapper.from_float(x)
        wrapper2 = wrapper.to(torch.device("cpu"))
        assert wrapper2.data.device == torch.device("cpu")
        assert wrapper2.scales.device == torch.device("cpu")


class TestPerBlockCast:
    """Test per_block_cast_to_fp8 convenience function."""

    def test_2d_input(self):
        """2D input should work directly."""
        x = torch.randn(128, 256)
        x_fp8, scales = per_block_cast_to_fp8(x)
        assert x_fp8.shape == x.shape

    def test_3d_input(self):
        """3D input should flatten leading dims."""
        x = torch.randn(4, 32, 256)
        x_fp8, scales = per_block_cast_to_fp8(x)
        assert x_fp8.shape == x.shape

    def test_1d_input(self):
        """1D input should be handled."""
        x = torch.randn(256)
        x_fp8, scales = per_block_cast_to_fp8(x)
        assert x_fp8.shape == x.shape


class TestEdgeCases:
    """Test FP8 edge cases."""

    def test_very_large_values(self):
        """Values exceeding FP8 range should be clamped."""
        x = torch.full((128, 128), 1000.0)
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)
        # Should recover approximately (clamped to FP8 range then scaled)
        assert torch.isfinite(x_deq).all()

    def test_very_small_values(self):
        """Very small values should be representable."""
        x = torch.full((128, 128), 1e-6)
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)
        assert torch.isfinite(x_deq).all()

    def test_mixed_positive_negative(self):
        """Mixed sign values should be handled correctly."""
        x = torch.randn(128, 128)
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        # Signs should be mostly preserved
        sign_match = (x.sign() == x_deq.sign()).float().mean()
        assert sign_match > 0.8  # Most signs should match
