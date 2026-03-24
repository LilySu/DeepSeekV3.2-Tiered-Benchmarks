# Decoupled partial-dim RoPE with YaRN for DeepSeek-V3 MLA.
#
# DeepSeek-V3's MLA applies RoPE only to a 64-dim slice (qk_rope_head_dim) out of
# the full 192-dim head (qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64).
# This is fundamentally different from standard RoPE which rotates all dims.
#
# Additionally, DeepSeek-V3 uses YaRN (Yet another RoPE extensioN) scaling to extend
# the context window from 4096 to 163840 tokens (factor=40). This is a KEY DIFFERENCE
# from GLM5 which uses standard RoPE without YaRN scaling.
#
# Ported from: deepseekv3_2-raw-decoupled-from-hf/model.py (lines 30-200)
#
# STATUS: Pure PyTorch reference implementation. No Triton kernel yet.
# The 64-dim RoPE is memory-bandwidth-bound at this size, so PyTorch is
# fast enough. A fused kernel would be warranted only if profiling shows
# the slice/cat overhead is significant at scale.
#
# Key shapes (DeepSeek-V3 config):
#   hidden_size:       7168
#   num_heads:         128
#   q_lora_rank:       1536  (Q compression bottleneck)
#   kv_lora_rank:      512   (KV compression bottleneck)
#   qk_rope_head_dim:  64    (RoPE-applied portion)
#   qk_nope_head_dim:  128   (non-RoPE portion)
#   qk_head_dim:       192   (128 + 64)
#   v_head_dim:        128
#
# YaRN config:
#   factor: 40, original_max_position_embeddings: 4096
#   beta_fast: 32, beta_slow: 1
#   mscale: 1.0, mscale_all_dim: 0.707
#
# Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 2.1 -- "Multi-head Latent Attention"
#            and Section 3.5 -- "Long Context Extension"

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
# YaRN RoPE Scaling Helpers
# ---------------------------------------------------------------------------
# These implement the YaRN (Yet another RoPE extensioN) method for extending
# the context window of rotary position embeddings. DeepSeek-V3 uses YaRN
# to extend from 4096 to 163840 tokens (factor=40).
#
# The key idea: instead of uniformly scaling all frequencies, YaRN blends
# between original and scaled frequencies based on a correction range
# determined by the fast/slow rotation thresholds (beta_fast=32, beta_slow=1).
# Low-frequency components are scaled by the factor, high-frequency ones
# are kept as-is, and mid-range ones are linearly interpolated.

def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=4096):
    """Find the correction dimension for YaRN."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=4096):
    """Find the correction range for YaRN."""
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_val, max_val, dim):
    """Create a linear ramp mask for YaRN blending.

    Returns a tensor of shape [dim] where values transition linearly
    from 0 to 1 between min_val and max_val. Values below min_val are 0
    (keep original frequency), values above max_val are 1 (use scaled frequency).
    """
    if min_val == max_val:
        max_val = min_val + 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def _yarn_get_mscale(scale=1, mscale=1):
    """Get the mscale factor for YaRN attention scaling.

    When scale > 1, this produces a scaling factor that compensates for
    the change in attention entropy caused by the longer context. The
    formula is: 0.1 * mscale * ln(scale) + 1.0
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ---------------------------------------------------------------------------
# RotaryEmbedding with YaRN — precomputes inv_freq for the 64-dim rope portion
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputes and caches inv_freq; returns (cos, sin) per forward call.

    Supports YaRN (Yet another RoPE extensioN) for extended context.
    DeepSeek-V3 uses YaRN with factor=40, extending from 4096 to 163840 tokens.

    DeepSeek-V3 config:
        qk_rope_head_dim = 64   (the dim this operates on)
        rope_theta = 10000.0
        max_position_embeddings = 163840
        rope_scaling: {type: "yarn", factor: 40, ...}
    """

    def __init__(self, cfg, device=None):
        super().__init__()
        dim = cfg["qk_rope_head_dim"]  # 64
        base = cfg["rope_theta"]       # 10000.0
        rope_scaling = cfg.get("rope_scaling")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )

        # Apply YaRN scaling if configured
        # DeepSeek-V3 uses: factor=40, original_max_pos=4096, beta_fast=32, beta_slow=1
        if rope_scaling is not None and rope_scaling.get("type") == "yarn":
            factor = rope_scaling["factor"]
            original_max_pos = rope_scaling.get("original_max_position_embeddings", 4096)
            beta_fast = rope_scaling.get("beta_fast", 32)
            beta_slow = rope_scaling.get("beta_slow", 1)
            mscale = rope_scaling.get("mscale", 1.0)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.707)

            # Find correction range based on fast/slow rotation thresholds
            low, high = _yarn_find_correction_range(
                beta_slow, beta_fast, dim, base, original_max_pos
            )

            # Create blending mask between original and scaled frequencies
            # inv_freq_mask=1 means "keep original", inv_freq_mask=0 means "use scaled"
            inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2).to(inv_freq.device)
            inv_freq_scaled = inv_freq / factor
            inv_freq = inv_freq_scaled * (1 - inv_freq_mask) + inv_freq * inv_freq_mask

            # Compute attention scaling from mscale parameters
            self.attention_scaling = _yarn_get_mscale(factor, mscale) / _yarn_get_mscale(
                factor, mscale_all_dim
            )
        else:
            self.attention_scaling = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """Compute cos and sin embeddings for the given positions.

        Args:
            x: [B, S, D] -- only used for dtype/device
            position_ids: [B, S] -- position indices

        Returns:
            cos: [B, S, rope_dim] -- cosine embeddings scaled by attention_scaling
            sin: [B, S, rope_dim] -- sine embeddings scaled by attention_scaling
        """
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
        qk_nope_head_dim: 128 (no-position-embedding dims) for DeepSeek-V3
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
        num_heads: 128 for DeepSeek-V3

    Returns:
        k_pe: [B, H, S, qk_rope_head_dim] -- rotated and expanded to all heads
    """
    batch_size, seq_length, rope_dim = k_pe_raw.shape
    k_pe = k_pe_raw.view(batch_size, 1, seq_length, rope_dim)
    k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
    k_pe = k_pe.expand(-1, num_heads, -1, -1)
    return k_pe
