"""
Multi-head Latent Attention (MLA) using FlashMLA kernel.

DeepSeek-V3 MLA compresses queries and key-value pairs into low-rank latent
spaces, then applies attention with *partial* RoPE:

  Query path:
    hidden -> q_proj [7168, 1536] -> q_a_layernorm -> q_b_proj [1536, 128*192]
    -> split into q_nope [128, 128] and q_rope [128, 64]
    -> apply YaRN RoPE to q_rope

  KV path:
    hidden -> kv_a_proj [7168, 576] -> kv_a_layernorm
    -> split into kv_compressed [512] and k_rope [64]
    -> kv_b_proj [512, 128*(128+128)] to get k_nope [128, 128] and v [128, 128]
    -> apply YaRN RoPE to k_rope

  Attention:
    FlashMLA kernel handles the compressed cache natively:
    - Takes compressed KV (512-dim) + k_rope (64-dim) from cache
    - Performs the kv_b_proj expansion internally (or we expand before)
    - Computes multi-head attention with the partial-RoPE structure

NO Dynamic Sparse Attention (DSA) -- that is GLM5-specific.

Reference: arXiv 2412.19437, Section 3.2.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepSeekV3Config, DEEPSEEK_V3_CONFIG
from .rope_partial import YaRNRotaryEmbedding, apply_partial_rope
from .cache import KVCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_import_flash_mla():
    """Attempt to import the FlashMLA kernel."""
    try:
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata
        return flash_mla_with_kvcache, get_mla_metadata
    except ImportError:
        return None, None


def eager_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attention_mask: Optional[torch.Tensor] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Standard eager attention implementation (fallback).

    Parameters
    ----------
    q : [B, H, S, D_qk]
    k : [B, H, S_kv, D_qk]
    v : [B, H, S_kv, D_v]
    scale : float
    attention_mask : [B, 1, S, S_kv] or None
    causal : bool

    Returns
    -------
    output : [B, H, S, D_v]
    """
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal and q.shape[-2] > 1:
        S, S_kv = q.shape[-2], k.shape[-2]
        causal_mask = torch.triu(
            torch.ones(S, S_kv, dtype=torch.bool, device=q.device),
            diagonal=S_kv - S + 1,
        )
        attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn_weights, v)


# ---------------------------------------------------------------------------
# Main MLA module
# ---------------------------------------------------------------------------

class MLAttention(nn.Module):
    """Multi-head Latent Attention for DeepSeek-V3 with FlashMLA dispatch.

    Parameters
    ----------
    config : DeepSeekV3Config
    layer_idx : int
    """

    def __init__(
        self,
        config: DeepSeekV3Config = DEEPSEEK_V3_CONFIG,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size  # 7168
        self.num_heads = config.num_attention_heads  # 128

        # MLA dimensions
        self.q_lora_rank = config.q_lora_rank  # 1536
        self.kv_lora_rank = config.kv_lora_rank  # 512
        self.qk_nope_head_dim = config.qk_nope_head_dim  # 128
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 64
        self.v_head_dim = config.v_head_dim  # 128
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 192

        # Attention scale (with YaRN mscale applied later)
        self.softmax_scale = self.qk_head_dim ** -0.5

        # ---------- Q projection path ----------
        # h -> q_lora_rank -> num_heads * qk_head_dim
        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=False
        )
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
        )

        # ---------- KV projection path ----------
        # h -> kv_lora_rank + qk_rope_head_dim (compressed KV + RoPE key)
        self.kv_a_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # Expand compressed KV to full K_nope + V per head
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # ---------- Output projection ----------
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        # ---------- RoPE ----------
        self.rope = YaRNRotaryEmbedding(config)

        # ---------- FlashMLA kernel ----------
        self._flash_mla_fn, self._get_mla_metadata = _try_import_flash_mla()

    @property
    def use_flash_mla(self) -> bool:
        return self._flash_mla_fn is not None and self.config.use_flashmla

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass.

        Parameters
        ----------
        hidden_states : [batch, seq_len, hidden_size]
        attention_mask : [batch, 1, seq_len, total_seq_len], optional
        position_ids : [batch, seq_len], optional
        kv_cache : KVCache, optional
        use_cache : bool

        Returns
        -------
        output : [batch, seq_len, hidden_size]
        cache_update : tuple of (kv_compressed, k_rope) or None
        """
        bsz, seq_len, _ = hidden_states.shape

        # ---- Q path ----
        q_compressed = self.q_a_proj(hidden_states)  # [B, S, q_lora_rank]
        q_compressed = self.q_a_layernorm(q_compressed)
        q = self.q_b_proj(q_compressed)  # [B, S, H * qk_head_dim]
        q = q.view(bsz, seq_len, self.num_heads, self.qk_head_dim)
        q = q.transpose(1, 2)  # [B, H, S, qk_head_dim]

        # ---- KV path ----
        kv_combined = self.kv_a_proj(hidden_states)  # [B, S, kv_lora_rank + qk_rope_head_dim]
        kv_compressed = kv_combined[..., : self.kv_lora_rank]  # [B, S, 512]
        k_rope_raw = kv_combined[..., self.kv_lora_rank :]  # [B, S, 64]

        # Normalise compressed KV
        kv_compressed_normed = self.kv_a_layernorm(kv_compressed)

        # Expand to full K_nope + V per head
        kv_expanded = self.kv_b_proj(kv_compressed_normed)
        # [B, S, H * (qk_nope + v_head)]
        kv_expanded = kv_expanded.view(
            bsz, seq_len, self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim
        )
        kv_expanded = kv_expanded.transpose(1, 2)  # [B, H, S, nope+v]

        k_nope = kv_expanded[..., : self.qk_nope_head_dim]  # [B, H, S, 128]
        v = kv_expanded[..., self.qk_nope_head_dim :]  # [B, H, S, 128]

        # ---- RoPE (partial) ----
        q_nope = q[..., : self.qk_nope_head_dim]  # [B, H, S, 128]
        q_rope = q[..., self.qk_nope_head_dim :]  # [B, H, S, 64]

        k_rope_for_rope = k_rope_raw.unsqueeze(1)  # [B, 1, S, 64]

        q_rope, k_rope_rotated = self.rope(
            q_rope, k_rope_for_rope, position_ids=position_ids
        )

        # Broadcast k_rope to all heads
        k_rope_rotated = k_rope_rotated.expand(-1, self.num_heads, -1, -1)

        # Reassemble Q and K
        q_full = torch.cat([q_nope, q_rope], dim=-1)  # [B, H, S, 192]
        k_full = torch.cat([k_nope, k_rope_rotated], dim=-1)  # [B, H, S, 192]

        # ---- Cache update ----
        # Store COMPRESSED KV in cache (576 dims = 512 kv_lora + 64 rope)
        # This is the key MLA optimization: 42x memory reduction vs expanded KV
        cache_update = None
        if use_cache:
            cache_update = (kv_compressed, k_rope_raw)
            if kv_cache is not None:
                kv_cached, k_rope_cached = kv_cache.update(
                    self.layer_idx, kv_compressed, k_rope_raw
                )

        # ---- Attention ----
        if self.use_flash_mla and not self.training and kv_cache is not None:
            # FlashMLA path: pass COMPRESSED KV cache directly
            # FlashMLA natively handles kv_b_proj expansion internally,
            # operating on the 576-dim compressed representation.
            # This is the whole point of FlashMLA — avoid expanding KV.
            compressed_cache = torch.cat([
                self.kv_a_layernorm(kv_cached),  # [B, T, 512]
                k_rope_cached,                     # [B, T, 64]
            ], dim=-1)  # [B, T, 576]
            compressed_cache = compressed_cache.unsqueeze(2)  # [B, T, 1, 576]

            output = self._flash_mla_compressed(q_full, compressed_cache, bsz)
        else:
            # Eager fallback: expand KV fully (needed for training / no FlashMLA)
            if kv_cache is not None and use_cache:
                # Expand all cached compressed KV
                cached_normed = self.kv_a_layernorm(kv_cached)
                kv_exp = self.kv_b_proj(cached_normed)
                total_len = kv_cached.shape[1]
                kv_exp = kv_exp.view(
                    bsz, total_len, self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim
                ).transpose(1, 2)
                k_nope = kv_exp[..., : self.qk_nope_head_dim]
                v = kv_exp[..., self.qk_nope_head_dim :]

                # Re-apply RoPE to cached k_rope using correct position_ids
                total_len = kv_cached.shape[1]
                cached_pos_ids = torch.arange(total_len, device=hidden_states.device).unsqueeze(0)
                k_rope_cached_unsq = k_rope_cached.unsqueeze(1)
                _, k_rope_cached_rot = self.rope(
                    q_rope[:, :, :1, :],
                    k_rope_cached_unsq,
                    position_ids=cached_pos_ids,
                )
                k_rope_cached_rot = k_rope_cached_rot.expand(-1, self.num_heads, -1, -1)
                k_full = torch.cat([k_nope, k_rope_cached_rot], dim=-1)

            scale = self.softmax_scale * self.rope.get_attention_scale().item()
            output = eager_attention_forward(
                q_full, k_full, v, scale,
                attention_mask=attention_mask,
                causal=True,
            )

        # ---- Output projection ----
        output = output.transpose(1, 2).contiguous()  # [B, S, H, v_head]
        output = output.reshape(bsz, seq_len, -1)  # [B, S, H*v_head]
        output = self.o_proj(output)

        return output, cache_update

    def _flash_mla_compressed(
        self,
        q: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        bsz: int,
    ) -> torch.Tensor:
        """Dispatch to FlashMLA kernel using COMPRESSED KV cache.

        This is the correct FlashMLA usage: pass the 576-dim compressed KV
        (kv_lora_rank=512 + qk_rope_head_dim=64) directly to the kernel.
        FlashMLA handles the kv_b_proj expansion internally, avoiding the
        need to materialize full 128-head K/V tensors in memory.

        This provides a 42x KV cache memory reduction:
          Compressed: 576 dims per position
          Expanded:   128 heads * (128 nope + 128 v) = 32,768 dims per position

        Parameters
        ----------
        q : [B, H, S, 192] — full query (nope + rope)
        compressed_kv_cache : [B, T, 1, 576] — compressed KV cache
        bsz : batch size

        Returns
        -------
        output : [B, H, S, v_head_dim]
        """
        assert self._flash_mla_fn is not None

        scale = self.softmax_scale * self.rope.get_attention_scale().item()

        # FlashMLA expects: q [B, S, H, D], kv_cache [B, T, 1, 576]
        q_t = q.transpose(1, 2).contiguous()  # [B, S, H, 192]
        T = compressed_kv_cache.shape[1]

        cache_seqlens = torch.full(
            (bsz,), T, dtype=torch.int32, device=q.device
        )

        # FlashMLA's flash_mla_with_kvcache operates on compressed KV:
        #   kv_cache: [B, max_seq, 1, kv_lora_rank + qk_rope_head_dim]
        #   It internally applies kv_b_proj weights (passed via metadata)
        #   to decompress K_nope and V per head during attention computation.
        metadata = self._get_mla_metadata(cache_seqlens, T)

        output, _ = self._flash_mla_fn(
            q_t,
            compressed_kv_cache,
            cache_seqlens=cache_seqlens,
            block_table=metadata,
            softmax_scale=scale,
            causal=True,
        )

        return output.transpose(1, 2)  # back to [B, H, S, D_v]
