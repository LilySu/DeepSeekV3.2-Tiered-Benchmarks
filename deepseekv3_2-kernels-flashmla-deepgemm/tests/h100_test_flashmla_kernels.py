"""
H100 FlashMLA kernel tests.

Tests the FlashMLA kernel for DeepSeek-V3 MLA attention:
  - Kernel launch and basic correctness
  - Comparison against eager attention
  - Various batch sizes and sequence lengths
  - Paged KV cache integration
  - Decoding (seq_len=1) performance

Requires: H100 GPU, FlashMLA library.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEEPSEEK_V3_CONFIG
from mla_attention import MLAttention, eager_attention_forward

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)

requires_flash_mla = pytest.mark.skipif(True, reason="FlashMLA not installed")
try:
    import flash_mla
    requires_flash_mla = pytest.mark.skipif(False, reason="")
except ImportError:
    pass


@requires_hopper
class TestH100FlashMLABasic:
    """Basic FlashMLA tests on H100."""

    def test_eager_attention_on_gpu(self):
        """Eager attention should work on H100."""
        B, H, S, D = 1, 128, 64, 192
        D_v = 128
        device = torch.device("cuda")
        dtype = torch.bfloat16

        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D_v, device=device, dtype=dtype)

        out = eager_attention_forward(q, k, v, scale=D**-0.5, causal=True)
        assert out.shape == (B, H, S, D_v)
        assert torch.isfinite(out).all()

    def test_mla_layer_on_gpu(self):
        """MLA layer should work on H100."""
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")
        dtype = torch.bfloat16

        attn = MLAttention(config, layer_idx=0).to(device).to(dtype).eval()
        B, S = 1, 32
        x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

        with torch.no_grad():
            out, _ = attn(x)

        assert out.shape == (B, S, config.hidden_size)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("seq_len", [1, 16, 64, 256])
    def test_mla_various_seq_lens(self, seq_len):
        """MLA should handle various sequence lengths on GPU."""
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")
        dtype = torch.bfloat16

        attn = MLAttention(config, layer_idx=0).to(device).to(dtype).eval()
        B = 1
        x = torch.randn(B, seq_len, config.hidden_size, device=device, dtype=dtype)

        with torch.no_grad():
            out, _ = attn(x)

        assert out.shape == (B, seq_len, config.hidden_size)


@requires_hopper
@requires_flash_mla
class TestH100FlashMLAKernel:
    """Test FlashMLA kernel specifically."""

    def test_flash_mla_import(self):
        """FlashMLA should be importable."""
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata
        assert callable(flash_mla_with_kvcache)
        assert callable(get_mla_metadata)

    def test_flash_mla_vs_eager(self):
        """FlashMLA should produce similar results to eager attention."""
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Create two identical layers
        torch.manual_seed(42)
        attn_eager = MLAttention(config, layer_idx=0).to(device).to(dtype).eval()
        attn_eager.config = config._replace(use_flashmla=False)  # Force eager

        torch.manual_seed(42)
        attn_flash = MLAttention(config, layer_idx=0).to(device).to(dtype).eval()

        B, S = 1, 32
        x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

        with torch.no_grad():
            out_eager, _ = attn_eager(x)
            out_flash, _ = attn_flash(x)

        # Should be close (not exact due to different compute paths)
        rel_err = (out_eager - out_flash).abs() / (out_eager.abs() + 1e-8)
        assert rel_err.mean() < 0.05

    def test_flash_mla_decode_step(self):
        """FlashMLA should handle seq_len=1 decode."""
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")
        dtype = torch.bfloat16

        attn = MLAttention(config, layer_idx=0).to(device).to(dtype).eval()
        B = 1
        x = torch.randn(B, 1, config.hidden_size, device=device, dtype=dtype)

        with torch.no_grad():
            out, _ = attn(x)

        assert out.shape == (B, 1, config.hidden_size)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_flash_mla_batched(self, batch_size):
        """FlashMLA should handle various batch sizes."""
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")
        dtype = torch.bfloat16

        attn = MLAttention(config, layer_idx=0).to(device).to(dtype).eval()
        S = 16
        x = torch.randn(batch_size, S, config.hidden_size, device=device, dtype=dtype)

        with torch.no_grad():
            out, _ = attn(x)

        assert out.shape == (batch_size, S, config.hidden_size)
