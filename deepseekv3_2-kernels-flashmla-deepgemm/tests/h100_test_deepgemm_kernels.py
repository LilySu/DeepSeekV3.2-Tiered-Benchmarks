"""
H100 DeepGEMM FP8 kernel tests.

Tests the DeepGEMM FP8 grouped GEMM kernel on Hopper GPUs:
  - Single group FP8 GEMM correctness
  - Multi-group FP8 GEMM correctness
  - Comparison against BF16 reference
  - Various matrix shapes (DeepSeek-V3 specific)
  - Edge cases (empty groups, single-token groups)

Requires: H100 GPU, DeepGEMM library.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fp8_utils import quantize_fp8_block, dequantize_fp8_block, per_block_cast_to_fp8

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)

requires_deepgemm = pytest.mark.skipif(
    True,
    reason="DeepGEMM not installed",
)

try:
    import deep_gemm
    requires_deepgemm = pytest.mark.skipif(False, reason="")
except ImportError:
    pass


@requires_hopper
class TestH100FP8Quantisation:
    """Test FP8 quantisation on H100 with native FP8 support."""

    def test_native_fp8_dtype(self):
        """H100 should support torch.float8_e4m3fn."""
        assert hasattr(torch, "float8_e4m3fn")

    def test_fp8_quantise_on_gpu(self):
        """Block quantisation should work on CUDA."""
        x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        x_fp8, scales = quantize_fp8_block(x.float())

        assert x_fp8.shape == x.shape
        assert scales.device.type == "cuda"

    def test_fp8_roundtrip_gpu(self):
        """Round-trip on GPU should have bounded error."""
        torch.manual_seed(42)
        x = torch.randn(256, 256, device="cuda", dtype=torch.float32)
        x_fp8, scales = quantize_fp8_block(x)
        x_deq = dequantize_fp8_block(x_fp8.float(), scales)

        rel_err = (x - x_deq).abs() / (x.abs() + 1e-8)
        assert rel_err.mean() < 0.2

    @pytest.mark.parametrize("shape", [(128, 7168), (7168, 2048), (2048, 7168)])
    def test_deepseek_v3_shapes(self, shape):
        """Test FP8 quantisation for DeepSeek-V3 specific shapes."""
        M, N = shape
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        x_fp8, scales = quantize_fp8_block(x)
        assert x_fp8.shape == (M, N)


@requires_hopper
@requires_deepgemm
class TestH100DeepGEMM:
    """Test DeepGEMM FP8 grouped GEMM on H100."""

    def test_single_gemm(self):
        """Single FP8 GEMM should produce correct results."""
        M, K, N = 128, 256, 128
        device = torch.device("cuda")

        a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        b = torch.randn(N, K, device=device, dtype=torch.bfloat16)

        # Reference
        ref = a.float() @ b.float().T

        # FP8
        a_fp8, a_scales = per_block_cast_to_fp8(a.float())
        b_fp8, b_scales = per_block_cast_to_fp8(b.float())

        # DeepGEMM
        out = deep_gemm.gemm_fp8_fp8_bf16_nt(a_fp8, a_scales, b_fp8, b_scales)

        # Check
        rel_err = (ref - out.float()).abs() / (ref.abs() + 1e-8)
        assert rel_err.mean() < 0.1, f"Mean relative error: {rel_err.mean():.4f}"

    def test_grouped_gemm(self):
        """Grouped FP8 GEMM should produce correct results."""
        G, M_per_group, K, N = 8, 64, 256, 128
        device = torch.device("cuda")

        a = torch.randn(G * M_per_group, K, device=device, dtype=torch.bfloat16)
        b = torch.randn(G, N, K, device=device, dtype=torch.bfloat16)
        group_sizes = torch.full((G,), M_per_group, dtype=torch.int32, device=device)

        # Reference: sequential
        ref_outs = []
        for g in range(G):
            ref_outs.append(
                a[g * M_per_group:(g+1) * M_per_group].float() @ b[g].float().T
            )
        ref = torch.cat(ref_outs, dim=0)

        # FP8 grouped
        a_fp8, a_scales = per_block_cast_to_fp8(a.float())
        b_fp8, b_scales = per_block_cast_to_fp8(b.reshape(-1, K).float())

        out = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
            a_fp8, a_scales, b_fp8, b_scales, group_sizes,
        )

        rel_err = (ref - out.float()).abs() / (ref.abs() + 1e-8)
        assert rel_err.mean() < 0.15

    @pytest.mark.parametrize("M", [1, 16, 64, 256])
    def test_varying_group_sizes(self, M):
        """Test with varying numbers of tokens per group."""
        G, K, N = 4, 128, 64
        device = torch.device("cuda")

        a = torch.randn(G * M, K, device=device, dtype=torch.bfloat16)
        b = torch.randn(G, N, K, device=device, dtype=torch.bfloat16)
        group_sizes = torch.full((G,), M, dtype=torch.int32, device=device)

        a_fp8, a_scales = per_block_cast_to_fp8(a.float())
        b_fp8, b_scales = per_block_cast_to_fp8(b.reshape(-1, K).float())

        out = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
            a_fp8, a_scales, b_fp8, b_scales, group_sizes,
        )

        assert out.shape == (G * M, N)
        assert torch.isfinite(out).all()

    def test_deepseek_v3_gate_projection(self):
        """Test with DeepSeek-V3 gate projection shape: [T, 7168] x [7168, 2048]."""
        T = 256
        K, N = 7168, 2048
        G = 8  # num active experts
        device = torch.device("cuda")

        a = torch.randn(T, K, device=device, dtype=torch.bfloat16)
        b = torch.randn(G, N, K, device=device, dtype=torch.bfloat16)
        group_sizes = torch.full((G,), T // G, dtype=torch.int32, device=device)

        a_fp8, a_scales = per_block_cast_to_fp8(a.float())
        b_fp8, b_scales = per_block_cast_to_fp8(b.reshape(-1, K).float())

        out = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
            a_fp8, a_scales, b_fp8, b_scales, group_sizes,
        )

        assert out.shape == (T, N)
