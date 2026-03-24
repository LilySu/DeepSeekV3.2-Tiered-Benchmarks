"""
YaRN Rotary Position Embedding for DeepSeek-V3.

DeepSeek-V3 uses *partial* RoPE: only the ``qk_rope_head_dim=64`` slice of
query/key receives rotary encoding.  The remaining ``qk_nope_head_dim=128``
dimensions are position-independent.

The YaRN extension (Yet another RoPE extensioN) enables context-length
extrapolation beyond the original 4096 training window up to 163840 tokens.

Reference: arXiv 2412.19437, Section 3.2.1 & Appendix B.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import DeepSeekV3Config, DEEPSEEK_V3_CONFIG


# ---------------------------------------------------------------------------
# YaRN helpers
# ---------------------------------------------------------------------------

def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 4096,
) -> float:
    """Find the correction dimension for YaRN interpolation."""
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 4096,
) -> Tuple[int, int]:
    """Find the range of dimensions that need correction."""
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    low = max(low, 0)
    high = min(high, dim - 1)
    return low, high


def _yarn_linear_ramp_mask(
    low: int, high: int, dim: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a linear ramp mask for YaRN interpolation blending."""
    if low == high:
        high += 0.001  # prevent division by zero
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear_func, 0.0, 1.0)


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Compute the magnitude scaling factor."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class YaRNRotaryEmbedding(nn.Module):
    """YaRN Rotary Position Embedding for DeepSeek-V3.

    Only applied to the ``qk_rope_head_dim`` (64) dimensions of Q and K.
    The remaining ``qk_nope_head_dim`` (128) dimensions are left untouched.

    Parameters
    ----------
    config : DeepSeekV3Config
        Model configuration (uses rope_scaling sub-config).
    device : torch.device, optional
        Device for the frequency buffer.
    """

    def __init__(
        self,
        config: DeepSeekV3Config = DEEPSEEK_V3_CONFIG,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.max_position_embeddings = config.max_position_embeddings  # 163840
        self.base = config.rope_theta  # 10000.0

        rope_cfg = config.rope_scaling
        self.factor = rope_cfg.factor  # 40
        self.original_max_pos = rope_cfg.original_max_position_embeddings  # 4096
        self.beta_fast = rope_cfg.beta_fast  # 32
        self.beta_slow = rope_cfg.beta_slow  # 1
        self.mscale = rope_cfg.mscale  # 1.0
        self.mscale_all_dim = rope_cfg.mscale_all_dim  # 1.0

        # Build frequency tensor
        self._build_freqs(device or torch.device("cpu"))

    def _build_freqs(self, device: torch.device) -> None:
        """Compute YaRN-adjusted inverse frequencies."""
        dim = self.rope_dim
        base = self.base

        # Standard inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        # YaRN correction range
        low, high = _yarn_find_correction_range(
            self.beta_slow,
            self.beta_fast,
            dim,
            base,
            self.original_max_pos,
        )

        # Interpolation mask
        inv_freq_interpolation = inv_freq / self.factor
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, dim // 2, dtype=torch.float32
        ).to(device)

        # Blend between original and interpolated frequencies
        inv_freq = inv_freq * (1.0 - inv_freq_mask) + inv_freq_interpolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Magnitude scaling (HF reference: ratio of two mscale values)
        _mscale = _yarn_get_mscale(self.factor, self.mscale)
        _mscale_all = _yarn_get_mscale(self.factor, self.mscale_all_dim)
        self.register_buffer(
            "attention_scale",
            torch.tensor(_mscale / _mscale_all, dtype=torch.float32, device=device),
            persistent=False,
        )

        # Cache cos/sin (built lazily on first forward)
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._cached_seq_len = 0

    def _extend_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Extend the cos/sin cache to cover ``seq_len`` positions."""
        if seq_len <= self._cached_seq_len and self._cos_cached is not None:
            return
        self._cached_seq_len = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq.to(device))  # [seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply YaRN rotary embedding to the RoPE slices of Q and K.

        Parameters
        ----------
        q_rope : Tensor  [batch, heads, seq_len, qk_rope_head_dim]
        k_rope : Tensor  [batch, 1, seq_len, qk_rope_head_dim]
        position_ids : Tensor [batch, seq_len], optional
        seq_len : int, optional  (inferred from q_rope if not given)

        Returns
        -------
        q_rope_out, k_rope_out : same shapes as inputs
        """
        if seq_len is None:
            seq_len = q_rope.shape[-2]

        device = q_rope.device
        dtype = q_rope.dtype

        self._extend_cache(
            max(seq_len, (position_ids.max().item() + 1) if position_ids is not None else seq_len),
            device,
            dtype,
        )

        if position_ids is not None:
            # Gather cos/sin for each position
            cos = self._cos_cached[position_ids].unsqueeze(1)  # [B, 1, S, D]
            sin = self._sin_cached[position_ids].unsqueeze(1)
        else:
            cos = self._cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
            sin = self._sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # Apply rotary: x * cos + rotate_half(x) * sin
        q_out = q_rope * cos + self._rotate_half(q_rope) * sin
        k_out = k_rope * cos + self._rotate_half(k_rope) * sin

        return q_out, k_out

    def get_attention_scale(self) -> torch.Tensor:
        """Return the YaRN magnitude scaling factor."""
        return self.attention_scale


def apply_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_module: YaRNRotaryEmbedding,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: split Q/K, apply RoPE, re-concatenate.

    Parameters
    ----------
    q : [B, H, S, qk_nope + qk_rope]
    k : [B, 1, S, qk_nope + qk_rope]

    Returns
    -------
    q_out, k_out with RoPE applied to the rope slice.
    """
    q_nope = q[..., :qk_nope_head_dim]
    q_rope = q[..., qk_nope_head_dim : qk_nope_head_dim + qk_rope_head_dim]
    k_nope = k[..., :qk_nope_head_dim]
    k_rope = k[..., qk_nope_head_dim : qk_nope_head_dim + qk_rope_head_dim]

    q_rope_out, k_rope_out = rope_module(q_rope, k_rope, position_ids=position_ids)

    q_out = torch.cat([q_nope, q_rope_out], dim=-1)
    k_out = torch.cat([k_nope, k_rope_out], dim=-1)
    return q_out, k_out
