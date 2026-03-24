# Decoupled partial-dim RoPE for DeepSeek-V3 MLA with YaRN scaling.
#
# DeepSeek-V3's MLA applies RoPE only to a 64-dim slice (qk_rope_head_dim) out of
# the full 192-dim head (qk_head_dim = qk_nope_head_dim + qk_rope_head_dim).
# This is fundamentally different from standard RoPE which rotates all dims.
#
# Key difference from GLM-5: DeepSeek-V3 uses YaRN (Yet Another RoPE extensioN)
# scaling with factor=40 to extend context to 128K tokens from a 4K base.
#
# YaRN interpolation (arXiv 2309.00071):
#   - Frequency-dependent interpolation: low frequencies are interpolated more
#   - High frequencies are kept intact (they encode local patterns)
#   - Attention scaling compensates for the distribution shift
#
# Ported from: deepseekv3_2-raw-decoupled-from-hf/model.py
#
# STATUS: Pure PyTorch reference implementation. No Triton kernel yet.
# The 64-dim RoPE is memory-bandwidth-bound at this size, so PyTorch is
# fast enough. A fused kernel would be warranted only if profiling shows
# the slice/cat overhead is significant at scale.
#
# Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 2.1 -- "Multi-head Latent Attention"

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embedding to a single tensor.

    unsqueeze_dim=1 for [B, H, S, D] (BHSD), =2 for [B, S, H, D] (BSHD).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# YaRN helpers
# ---------------------------------------------------------------------------

def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=4096):
    """Find the correction dimension for YaRN interpolation."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=4096):
    """Find the correction range for YaRN interpolation."""
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low, high, dim, dtype=torch.float32):
    """Create a linear ramp mask for smooth interpolation between NTK and non-NTK regions."""
    if low == high:
        high += 0.001  # Prevent division by zero
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return linear_func.clamp(0, 1)


def _yarn_get_mscale(scale=1.0, mscale=1.0):
    """Compute the attention scaling factor for YaRN."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ---------------------------------------------------------------------------
# RotaryEmbedding -- precomputes inv_freq with YaRN scaling
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputes and caches inv_freq with YaRN scaling; returns (cos, sin) per forward call.

    DeepSeek-V3 config:
        qk_rope_head_dim = 64   (the dim this operates on)
        rope_theta = 10000.0
        max_position_embeddings = 163840
        rope_scaling = {"type": "yarn", "factor": 40, ...}
    """

    def __init__(self, cfg, device=None):
        super().__init__()
        dim = cfg["qk_rope_head_dim"]  # 64
        base = cfg["rope_theta"]       # 10000.0
        max_position_embeddings = cfg["max_position_embeddings"]
        rope_scaling = cfg.get("rope_scaling")

        # Compute base inv_freq
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )

        # Apply YaRN scaling if configured
        self.attention_scaling = 1.0
        if rope_scaling is not None and rope_scaling.get("type") == "yarn":
            factor = rope_scaling.get("factor", 40)
            original_max_pos = rope_scaling.get("original_max_position_embeddings", 4096)
            beta_fast = rope_scaling.get("beta_fast", 32)
            beta_slow = rope_scaling.get("beta_slow", 1)
            mscale = rope_scaling.get("mscale", 1.0)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.707)

            # Frequency-dependent interpolation
            freq_extra = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
            )
            freq_inter = 1.0 / (
                factor * base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
            )

            low, high = _yarn_find_correction_range(
                beta_slow, beta_fast, dim, base, original_max_pos
            )
            inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2, dtype=torch.float32).to(device)

            inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

            # Attention scaling for YaRN (HF reference: ratio of two mscale values)
            self.attention_scaling = _yarn_get_mscale(factor, mscale) / _yarn_get_mscale(
                factor, mscale_all_dim
            )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [B, S, D] -- only used for dtype/device
        # position_ids: [B, S]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Partial-dim RoPE application patterns used by MLA
# ---------------------------------------------------------------------------

def apply_rope_to_query(query_states, cos, sin, qk_nope_head_dim, qk_rope_head_dim):
    """Split query into nope/rope parts, apply RoPE to rope part, recombine.

    Args:
        query_states: [B, H, S, qk_head_dim] where qk_head_dim = nope + rope
        cos, sin: [B, S, rope_dim] from RotaryEmbedding
        qk_nope_head_dim: 128 (no-position-embedding dims)
        qk_rope_head_dim: 64 (rotary-embedded dims)

    Returns:
        query_states: [B, H, S, qk_head_dim] with RoPE applied to last 64 dims
    """
    q_nope, q_pe = torch.split(query_states, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)
    return torch.cat([q_nope, q_pe], dim=-1)


def apply_rope_to_compressed_kv_key(k_pe_raw, cos, sin, num_heads):
    """Apply RoPE to the single-head rope stream from compressed KV, expand to all heads.

    Args:
        k_pe_raw: [B, S, qk_rope_head_dim] -- the rope portion split from kv_a_proj output
        cos, sin: [B, S, rope_dim] from RotaryEmbedding
        num_heads: 128

    Returns:
        k_pe: [B, H, S, qk_rope_head_dim] -- rotated and expanded to all heads
    """
    batch_size, seq_length, rope_dim = k_pe_raw.shape
    k_pe = k_pe_raw.view(batch_size, 1, seq_length, rope_dim)
    k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
    k_pe = k_pe.expand(-1, num_heads, -1, -1)
    return k_pe
