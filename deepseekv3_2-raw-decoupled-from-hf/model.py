"""DeepSeek-V3 model — standalone pure-PyTorch implementation.

Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)
671B total parameters, ~37B active per token.

Key architectural features:
  - Multi-head Latent Attention (MLA) with compressed KV
  - Mixture of Experts (MoE) with auxiliary-loss-free load balancing
  - YaRN RoPE for extended context (163840 tokens)
  - Multi-Token Prediction (MTP) with 1 prediction layer
  - No DSA (Dynamic Sparse Attention) — this is a GLM5 feature, not present in DeepSeek-V3
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
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
# RoPE helpers
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
# KV head expansion
# ---------------------------------------------------------------------------

def repeat_kv(x, n_rep):
    """Expand (batch, n_kv_heads, seq, dim) -> (batch, n_heads, seq, dim)."""
    batch, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Eager attention forward
# ---------------------------------------------------------------------------

def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len, past_len, dtype, device):
    """Create [1, 1, seq_len, total_len] causal mask."""
    total_len = past_len + seq_len
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(total_len, device=device).unsqueeze(0)
    causal = cols <= (rows + past_len)
    mask = torch.where(causal, 0.0, torch.finfo(dtype).min)
    return mask.to(dtype).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# YaRN RoPE Scaling Helpers
# ---------------------------------------------------------------------------

def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=4096):
    """Find the correction dimension for YaRN."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=4096):
    """Find the correction range for YaRN."""
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_val, max_val, dim):
    """Create a linear ramp mask for YaRN blending."""
    if min_val == max_val:
        max_val = min_val + 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def _yarn_get_mscale(scale=1, mscale=1):
    """Get the mscale factor for YaRN attention scaling."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ---------------------------------------------------------------------------
# Rotary Embedding with YaRN
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputes and caches inv_freq; returns (cos, sin) per forward call.

    Supports YaRN (Yet another RoPE extensioN) for extended context.
    DeepSeek-V3 uses YaRN with factor=40, extending from 4096 to 163840 tokens.
    """

    def __init__(self, cfg, device=None):
        super().__init__()
        dim = cfg["qk_rope_head_dim"]
        base = cfg["rope_theta"]
        rope_scaling = cfg.get("rope_scaling")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )

        # Apply YaRN scaling if configured
        if rope_scaling is not None and rope_scaling.get("type") == "yarn":
            factor = rope_scaling["factor"]
            original_max_pos = rope_scaling.get("original_max_position_embeddings", 4096)
            beta_fast = rope_scaling.get("beta_fast", 32)
            beta_slow = rope_scaling.get("beta_slow", 1)
            mscale = rope_scaling.get("mscale", 1.0)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.707)

            # Find correction range
            low, high = _yarn_find_correction_range(
                beta_slow, beta_fast, dim, base, original_max_pos
            )

            # Create blending mask between original and scaled freqs
            inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2).to(inv_freq.device)
            inv_freq_scaled = inv_freq / factor
            inv_freq = inv_freq_scaled * (1 - inv_freq_mask) + inv_freq * inv_freq_mask

            # Compute attention scaling from mscale
            self.attention_scaling = _yarn_get_mscale(factor, mscale) / _yarn_get_mscale(
                factor, mscale_all_dim
            )
        else:
            self.attention_scaling = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# MLA Attention (Multi-head Latent Attention) — NO DSA
# ---------------------------------------------------------------------------

class MLAttention(nn.Module):
    """Multi-head Latent Attention for DeepSeek-V3.

    Unlike GLM5, DeepSeek-V3 does NOT use DSA (Dynamic Sparse Attention).
    Standard causal attention is used instead.
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

        # Query projection (LoRA path)
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

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
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

        # Standard causal attention (no DSA sparse masking)
        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Feed-Forward (SwiGLU)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, cfg, intermediate_size=None):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.intermediate_size = cfg["intermediate_size"] if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# TopkRouter — sigmoid-based with auxiliary-loss-free load balancing
# ---------------------------------------------------------------------------

class TopkRouter(nn.Module):
    """MoE routing with sigmoid scoring and noaux_tc method.

    DeepSeek-V3 uses auxiliary-loss-free load balancing:
    - Sigmoid activation (not softmax)
    - Additive bias (e_score_correction_bias) for selection only, not weight computation
    - Group-based routing: n_group=8, topk_group=4
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.n_routed_experts = cfg["n_routed_experts"]

        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    def forward(self, x):
        x = x.view(-1, self.hidden_size)
        return F.linear(x.float(), self.weight.float())


# ---------------------------------------------------------------------------
# MoeExperts (collection of expert weights as 3D tensors)
# ---------------------------------------------------------------------------

class MoeExperts(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg["n_routed_experts"]
        self.hidden_dim = cfg["hidden_size"]
        self.intermediate_dim = cfg["moe_intermediate_size"]
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


# ---------------------------------------------------------------------------
# MoE (routed experts + shared experts)
# ---------------------------------------------------------------------------

class MoE(nn.Module):
    """Mixture of Experts with DeepSeek-V3 group-based routing.

    Key differences from GLM5:
    - n_group=8: experts are divided into 8 groups of 32
    - topk_group=4: select top-4 groups first, then top-k within those
    - topk_method="noaux_tc": auxiliary-loss-free load balancing with bias correction
    """

    def __init__(self, cfg):
        super().__init__()
        self.experts = MoeExperts(cfg)
        self.gate = TopkRouter(cfg)
        self.shared_experts = FeedForward(
            cfg, intermediate_size=cfg["moe_intermediate_size"] * cfg["n_shared_experts"],
        )
        self.n_routed_experts = cfg["n_routed_experts"]
        self.n_group = cfg["n_group"]
        self.topk_group = cfg["topk_group"]
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]
        self.top_k = cfg["num_experts_per_tok"]

    def route_tokens_to_experts(self, router_logits):
        """Select top-k experts using sigmoid + group-based routing.

        DeepSeek-V3 routing:
        1. Sigmoid on router logits
        2. Add e_score_correction_bias for selection (not weights)
        3. Group experts into n_group groups, score each group by top-2 sum
        4. Select topk_group groups
        5. Within selected groups, pick top_k experts
        6. Normalize weights and scale by routed_scaling_factor
        """
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias

        # Group-based selection
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


# ---------------------------------------------------------------------------
# Multi-Token Prediction (MTP) Head
# ---------------------------------------------------------------------------

class MTPLayer(nn.Module):
    """Single MTP prediction layer for DeepSeek-V3.

    DeepSeek-V3 uses 1 MTP layer that predicts the next token beyond
    the standard LM head. During training, this provides an auxiliary loss.
    During inference, it can serve as a draft model for speculative decoding.

    Architecture per the paper (arXiv 2412.19437, Section 2.2):
      1. Embedding projection M_k: concat(hidden[i], embed(token[i+1])) -> hidden_size
      2. RMSNorm
      3. A full shared transformer decoder layer T_k (self-attention + FFN)
      4. Shared output head (reuses main model's lm_head weights)

    The transformer layer can optionally share weights with one of the
    main model's decoder layers for parameter efficiency.
    """

    def __init__(self, cfg):
        super().__init__()
        hidden_size = cfg["hidden_size"]

        # M_k: projection from concatenated (hidden + embedding) to hidden_size
        self.embed_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.proj_norm = RMSNorm(hidden_size, eps=cfg["rms_norm_eps"])

        # T_k: a full decoder layer (self-attention + FFN) for contextual processing.
        # In the real DeepSeek-V3, this shares weights with a main model layer.
        # Here we create a dedicated layer; call share_decoder_layer() to share.
        self.decoder_layer = DecoderLayer(cfg, layer_idx=0)

        # Output head — will be replaced with shared lm_head in set_shared_head()
        self.head = nn.Linear(hidden_size, cfg["vocab_size"], bias=False)
        self.enorm = RMSNorm(hidden_size, eps=cfg["rms_norm_eps"])

    def share_decoder_layer(self, layer):
        """Share weights with a main model decoder layer (parameter efficient)."""
        self.decoder_layer = layer

    def set_shared_head(self, lm_head):
        """Share the output head with the main model's lm_head."""
        self.head = lm_head

    def forward(self, hidden_states, embed_tokens, input_ids=None, labels=None,
                position_embeddings=None, attention_mask=None):
        """
        Args:
            hidden_states: [B, S, hidden_size] from the main model's last layer
            embed_tokens: nn.Embedding module for token lookup
            input_ids: [B, S] original input token ids (for shifted embedding)
            labels: [B, S] target token ids for MTP loss
            position_embeddings: (cos, sin) from RotaryEmbedding
            attention_mask: [1, 1, S, T] causal mask

        Returns:
            mtp_logits: [B, S-1, vocab_size]
            mtp_loss: scalar loss (None if labels not provided)
        """
        if input_ids is None:
            return None, None

        batch_size, seq_len, _ = hidden_states.shape
        if seq_len <= 1:
            return None, None

        # Step 1: Embedding projection M_k
        # For position i, combine hidden_states[i] with embedding of input[i+1]
        shifted_embeds = embed_tokens(input_ids[:, 1:])  # [B, S-1, hidden_size]
        truncated_hidden = hidden_states[:, :-1, :]       # [B, S-1, hidden_size]
        combined = torch.cat([truncated_hidden, shifted_embeds], dim=-1)
        mtp_hidden = self.proj_norm(self.embed_proj(combined))  # [B, S-1, hidden_size]

        # Step 2: Process through shared decoder layer T_k (self-attention + FFN)
        if position_embeddings is not None:
            # Trim position embeddings to match S-1 length
            cos, sin = position_embeddings
            pe = (cos[:, :seq_len-1, :], sin[:, :seq_len-1, :])
        else:
            pe = None
        if attention_mask is not None:
            mask = attention_mask[:, :, :seq_len-1, :seq_len-1]
        else:
            mask = None
        mtp_hidden = self.decoder_layer(
            mtp_hidden, attention_mask=mask, position_embeddings=pe,
        )

        # Step 3: Output head (shared with main model's lm_head)
        mtp_hidden = self.enorm(mtp_hidden)
        mtp_logits = self.head(mtp_hidden)  # [B, S-1, vocab_size]

        # Step 4: Compute MTP loss
        mtp_loss = None
        if labels is not None:
            # MTP predicts token at position i+2 from combined context at i
            shifted_labels = labels[:, 2:] if labels.shape[1] > 2 else labels[:, 1:]
            mtp_logits_for_loss = mtp_logits[:, :shifted_labels.shape[1], :]
            mtp_loss = F.cross_entropy(
                mtp_logits_for_loss.reshape(-1, mtp_logits_for_loss.shape[-1]),
                shifted_labels.reshape(-1),
            )

        return mtp_logits, mtp_loss


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.self_attn = MLAttention(cfg, layer_idx)

        if cfg["mlp_layer_types"][layer_idx] == "sparse":
            self.mlp = MoE(cfg)
        else:
            self.mlp = FeedForward(cfg)

        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.gradient_checkpointing = False

    def _forward(self, hidden_states, attention_mask, position_embeddings, past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, position_embeddings, attention_mask=attention_mask,
            past_key_values=past_key_values, **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None, **kwargs):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, hidden_states, attention_mask, position_embeddings,
                past_key_values, use_reentrant=False, **kwargs,
            )
        return self._forward(hidden_states, attention_mask, position_embeddings, past_key_values, **kwargs)


# ---------------------------------------------------------------------------
# DeepSeekV3Model (base model)
# ---------------------------------------------------------------------------

class DeepSeekV3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.padding_idx = cfg["pad_token_id"]
        self.vocab_size = cfg["vocab_size"]

        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.rotary_emb = RotaryEmbedding(cfg)

        self._init_weights()

    def _init_weights(self):
        std = self.cfg["initializer_range"]
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, TopkRouter):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                nn.init.zeros_(module.e_score_correction_bias)
            elif isinstance(module, MoeExperts):
                nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
                nn.init.normal_(module.down_proj, mean=0.0, std=std)

    def set_gradient_checkpointing(self, enable=True):
        for layer in self.layers:
            layer.gradient_checkpointing = enable

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            from cache import KVCache
            past_key_values = KVCache(self.cfg["num_hidden_layers"])

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = make_causal_mask(
            seq_len=inputs_embeds.shape[1],
            past_len=past_key_values.get_seq_length() if past_key_values is not None else 0,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values, **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values, position_embeddings, causal_mask


# ---------------------------------------------------------------------------
# DeepSeekV3ForCausalLM (model + LM head + MTP)
# ---------------------------------------------------------------------------

class DeepSeekV3ForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = DeepSeekV3Model(cfg)
        self.vocab_size = cfg["vocab_size"]
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        # Multi-Token Prediction layers
        num_mtp = cfg.get("num_nextn_predict_layers", 0)
        if num_mtp > 0:
            self.mtp_layers = nn.ModuleList([MTPLayer(cfg) for _ in range(num_mtp)])
        else:
            self.mtp_layers = None

        if cfg.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, **kwargs):
        hidden_states, past_key_values, position_embeddings, causal_mask = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1),
            )

            # Add MTP auxiliary loss (with full decoder layer per paper Section 2.2)
            if self.mtp_layers is not None and input_ids is not None:
                for mtp_layer in self.mtp_layers:
                    _, mtp_loss = mtp_layer(
                        hidden_states, self.model.embed_tokens,
                        input_ids=input_ids, labels=labels,
                        position_embeddings=position_embeddings,
                        attention_mask=causal_mask,
                    )
                    if mtp_loss is not None:
                        loss = loss + mtp_loss

        return loss, logits, past_key_values
