"""
H100 FP8 edge case tests.

Tests FP8 quantisation/dequantisation edge cases on H100 with native FP8:
  - Subnormal values
  - NaN / Inf handling
  - Very large matrices
  - Mixed-sign blocks
  - Sparse matrices
  - Block boundary effects

Requires: H100 GPU.
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
    FP8_E4M3_MAX,
    BLOCK_SIZE,
)

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)


@requires_hopper
class TestH100FP8EdgeCases:
    """FP8 edge cases on H100."""

    def test_all_zeros(self):
        """All-zero tensor should quantise cleanly."""
        x = torch.zeros(256, 256, device="cuda")
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)
        assert (x_deq == 0).all()

    def test_all_ones(self):
        """All-one tensor should have small round-trip error."""
        x = torch.ones(256, 256, device="cuda")
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)
        assert ((x - x_deq).abs() < 0.1).all()

    def test_very_large_values(self):
        """Large values should be clamped to FP8 range."""
        x = torch.full((128, 128), 1e6, device="cuda")
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)
        assert torch.isfinite(x_deq).all()

    def test_very_small_values(self):
        """Very small values should survive quantisation."""
        x = torch.full((128, 128), 1e-7, device="cuda")
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)
        assert torch.isfinite(x_deq).all()

    def test_mixed_magnitude_blocks(self):
        """Blocks with mixed magnitudes should use appropriate scales."""
        x = torch.randn(256, 256, device="cuda")
        # Make first block very large, second block very small
        x[:128, :128] *= 1000
        x[128:, 128:] *= 0.001

        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        # Different blocks should have different scales
        assert scales[0, 0] > scales[1, 1]

    def test_sparse_tensor(self):
        """Sparse tensor (mostly zeros) should quantise correctly."""
        x = torch.zeros(256, 256, device="cuda")
        # Sprinkle a few non-zero values
        x[10, 20] = 1.0
        x[100, 150] = -0.5
        x[200, 50] = 2.0

        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        # Non-zero values should be approximately preserved
        assert abs(x_deq[10, 20] - 1.0) < 0.2
        assert abs(x_deq[200, 50] - 2.0) < 0.5

    def test_non_square_large(self):
        """Large non-square matrix should work."""
        x = torch.randn(1024, 7168, device="cuda")
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        assert x_deq.shape == x.shape
        assert torch.isfinite(x_deq).all()

    def test_block_boundary_alignment(self):
        """Values at block boundaries should be handled correctly."""
        # Create tensor where block boundaries contain important values
        x = torch.zeros(256, 256, device="cuda")
        # Set values at the exact block boundaries
        for i in range(0, 256, BLOCK_SIZE):
            for j in range(0, 256, BLOCK_SIZE):
                x[i, j] = 1.0
                if i + BLOCK_SIZE - 1 < 256 and j + BLOCK_SIZE - 1 < 256:
                    x[i + BLOCK_SIZE - 1, j + BLOCK_SIZE - 1] = -1.0

        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        # Boundary values should be approximately preserved
        assert abs(x_deq[0, 0] - 1.0) < 0.2
        assert abs(x_deq[127, 127] - (-1.0)) < 0.2

    def test_negative_values(self):
        """All-negative tensor should round-trip correctly."""
        x = -torch.rand(128, 128, device="cuda") * 10
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        # Signs should be preserved
        assert (x_deq <= 0).all()

    def test_alternating_sign_pattern(self):
        """Checkerboard sign pattern should be preserved."""
        x = torch.ones(128, 128, device="cuda")
        x[::2, ::2] = -1
        x[1::2, 1::2] = -1

        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        sign_match = (x.sign() == x_deq.sign()).float().mean()
        assert sign_match > 0.95
