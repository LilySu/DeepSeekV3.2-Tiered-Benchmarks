# MLA (Multi-head Latent Attention) -- the core attention mechanism of DeepSeek-V3.
#
# MLA compresses KV into a low-rank latent space (512-dim), applies RoPE to
# only a 64-dim decoupled stream, and uses asymmetric head dims (QK=192, V=128).
# This is fundamentally different from standard MHA/GQA/MQA.
#
# Ported from: deepseekv3_2-raw-decoupled-from-hf/model.py (lines 206-298, MLAttention)
#
# STATUS: Pure PyTorch reference implementation. No Triton kernel yet.
# A fused Triton kernel would combine the compressed projection chains,
# partial RoPE, and the causal attention into one kernel. This is the
# most complex component and the hardest to fuse.
#
# KEY DIFFERENCE FROM GLM5: DeepSeek-V3 does NOT have DSA (Dynamic Sparse
# Attention). GLM5 uses a DSAIndexer to select top-2048 positions per query
# for sparse attention. DeepSeek-V3 uses standard causal attention instead.
# There is no dsa_indexer.py in this package.
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
# Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 2.1 -- "Multi-head Latent Attention"

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope_partial import apply_rotary_pos_emb


# ---------------------------------------------------------------------------
# RMSNorm -- simple PyTorch version, identical to the raw model (model.py:24).
# The Triton fast_rms_layernorm from unsloth_rms_layernorm.py can be swapped
# in as a drop-in replacement by monkey-patching .forward on the instances:
#   layer.q_a_layernorm.forward = lambda x: fast_rms_layernorm(layer.q_a_layernorm, x)
#
# DeepSeek-V3 uses eps=1e-6 (GLM5 uses eps=1e-5).
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# ---------------------------------------------------------------------------
# KV head expansion (for completeness, though MLA uses num_kv_groups=1)
# ---------------------------------------------------------------------------
def repeat_kv(x, n_rep):
    """Expand (batch, n_kv_heads, seq, dim) -> (batch, n_heads, seq, dim)."""
    batch, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Eager attention forward -- standard causal attention (NO DSA sparse masking)
# ---------------------------------------------------------------------------
def _eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Standard eager attention with GQA expansion.

    Unlike GLM5's version, this does NOT apply any DSA sparse masking.
    DeepSeek-V3 uses standard causal attention.

    Args:
        module:          attention module (needs .num_key_value_groups attribute)
        query:           [B, H, S, qk_head_dim]   -> [B, 128, S, 192]
        key:             [B, H_kv, T, qk_head_dim] -> [B, 128, T, 192]
        value:           [B, H_kv, T, v_head_dim]  -> [B, 128, T, 128]
        attention_mask:  [B, 1, S, T] -- standard causal mask
        scaling:         float -- 1/sqrt(qk_head_dim)
        dropout:         float -- attention dropout rate

    Returns:
        attn_output:     [B, S, H * v_head_dim] -> [B, S, 128 * 128]
        attn_weights:    [B, H, S, T]
    """
    # Expand KV heads to match Q heads (GQA / MQA expansion)
    n_rep = module.num_key_value_groups
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class MLAttention(nn.Module):
    """Multi-head Latent Attention for DeepSeek-V3.

    Unlike GLM5, DeepSeek-V3 does NOT use DSA (Dynamic Sparse Attention).
    Standard causal attention is used instead.

    Projection chain:

        Query path (with LoRA-style compression):
            hidden [B,S,7168]
              -> q_a_proj [7168 -> 1536]
              -> q_a_layernorm (RMSNorm, dim 1536)
              -> q_b_proj [1536 -> 128*192 = 24576]
              -> reshape to [B, 128, S, 192]
              -> split into nope[128] + rope[64]
              -> apply RoPE to rope portion
              -> cat back to [B, 128, S, 192]

        KV path (MLA compression):
            hidden [B,S,7168]
              -> kv_a_proj_with_mqa [7168 -> 576]  (512 kv_lora + 64 rope)
              -> split: k_compressed[512], k_pe_raw[64]
              -> kv_a_layernorm (RMSNorm, dim 512) on k_compressed
              -> kv_b_proj [512 -> 128*(128+128) = 32768]
              -> reshape to [B, S, 128, 256] -> split k_nope[128] + v[128]
              -> apply RoPE to k_pe_raw -> expand to all heads
              -> cat k_nope + k_pe -> key [B, 128, S, 192]

        Attention:
            Standard causal attention (no DSA sparse masking)
            Eager attention: softmax(QK^T / sqrt(192)) * V
            Output: o_proj [128*128 -> 7168]
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_groups = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
        self.attention_dropout = cfg["attention_dropout"]
        self.num_heads = cfg["num_attention_heads"]

        self.q_lora_rank = cfg["q_lora_rank"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.kv_lora_rank = cfg["kv_lora_rank"]
        self.v_head_dim = cfg["v_head_dim"]
        self.qk_nope_head_dim = cfg["qk_nope_head_dim"]
        self.qk_head_dim = cfg["qk_head_dim"]

        self.is_causal = True

        # Query projection (LoRA-style compression path)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(cfg["hidden_size"], self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(cfg["hidden_size"], cfg["q_lora_rank"], bias=cfg["attention_bias"])
            self.q_a_layernorm = RMSNorm(cfg["q_lora_rank"], eps=cfg["rms_norm_eps"])
            self.q_b_proj = nn.Linear(cfg["q_lora_rank"], self.num_heads * self.qk_head_dim, bias=False)

        # KV projections (MLA compressed path)
        self.kv_a_proj_with_mqa = nn.Linear(
            cfg["hidden_size"], self.kv_lora_rank + self.qk_rope_head_dim, bias=cfg["attention_bias"],
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=cfg["rms_norm_eps"])
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, cfg["hidden_size"], bias=cfg["attention_bias"])

        self.scaling = self.qk_head_dim ** -0.5

        # NOTE: No DSAIndexer here -- this is a KEY DIFFERENCE from GLM5.
        # GLM5's MLAttention creates a DSAIndexer for each layer. DeepSeek-V3
        # does not use DSA at all.

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        """Forward pass for MLA attention.

        Args:
            hidden_states: [B, S, hidden_size] -> [B, S, 7168]
            position_embeddings: (cos, sin) from RotaryEmbedding
            attention_mask: [B, 1, S, T] causal mask
            past_key_values: KVCache instance

        Returns:
            attn_output: [B, S, hidden_size]
            attn_weights: [B, H, S, T]
        """
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        # --- Query path ---
        if self.q_lora_rank is None:
            query_states = self.q_proj(hidden_states)
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

        # --- KV path ---
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)

        kv_expanded = self.kv_b_proj(k_compressed)
        kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

        # Assemble full Q and K
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # --- Standard causal attention (NO DSA sparse masking) ---
        # This is the key difference from GLM5. GLM5 builds a DSA mask
        # that combines causal masking with sparse token selection from
        # the DSAIndexer. DeepSeek-V3 uses simple causal attention.
        attn_output, attn_weights = _eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
