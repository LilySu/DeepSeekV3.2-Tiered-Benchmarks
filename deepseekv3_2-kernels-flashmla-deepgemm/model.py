"""
DeepSeek-V3 full model with FlashMLA attention and DeepGEMM MoE.

This module assembles the complete DeepSeek-V3 671B transformer:

  - 61 transformer layers
    - First 3 layers: dense FFN (no MoE)
    - Layers 3-60: MLA attention + MoE FFN (256 experts, top-8)
  - Multi-head Latent Attention (MLA) with FlashMLA kernel dispatch
  - YaRN RoPE for context extension to 163840 tokens
  - FP8 DeepGEMM grouped GEMM for MoE expert dispatch
  - Multi-Token Prediction (MTP) head (1 prediction layer)
  - RMSNorm throughout (eps=1e-6)

Reference: arXiv 2412.19437 (DeepSeek-V3 Technical Report).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepSeekV3Config, DEEPSEEK_V3_CONFIG
from .mla_attention import MLAttention
from .moe_grouped_gemm import MoEGroupedGEMM, ExpertFFN
from .cache import KVCache


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DeepSeekRMSNorm(nn.Module):
    """RMSNorm with eps=1e-6 (DeepSeek-V3 default)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class DeepSeekV3DecoderLayer(nn.Module):
    """Single decoder layer: MLA attention + (MoE or Dense) FFN.

    Parameters
    ----------
    config : DeepSeekV3Config
    layer_idx : int
    """

    def __init__(
        self,
        config: DeepSeekV3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Attention
        self.input_layernorm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MLAttention(config, layer_idx=layer_idx)

        # FFN (MoE or Dense)
        self.post_attention_layernorm = DeepSeekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if layer_idx < config.moe.first_k_dense_replace:
            # Dense FFN for first k layers
            self.mlp = ExpertFFN(
                config.hidden_size,
                config.intermediate_size,
            )
            self.is_moe = False
        else:
            # MoE FFN
            self.mlp = MoEGroupedGEMM(config, layer_idx=layer_idx)
            self.is_moe = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """
        Returns
        -------
        hidden_states : [B, S, D]
        router_logits : [B*S, num_experts] or None
        cache_update : tuple or None
        """
        # --- Self-attention ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, cache_update = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # --- FFN ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        router_logits = None
        if self.is_moe:
            hidden_states, router_logits = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, router_logits, cache_update


# ---------------------------------------------------------------------------
# MTP (Multi-Token Prediction) head
# ---------------------------------------------------------------------------

class MTPPredictionHead(nn.Module):
    """Multi-Token Prediction head for DeepSeek-V3.

    DeepSeek-V3 uses 1 MTP layer that predicts the next token from a
    lightweight projection of the last hidden state.

    Reference: arXiv 2412.19437, Section 3.5.
    """

    def __init__(self, config: DeepSeekV3Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Lightweight projection for the MTP prediction
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Shares the embedding weight (set externally)
        self.lm_head: Optional[nn.Linear] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedding_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : [B, S, D]
        embedding_weight : [vocab_size, D], optional

        Returns
        -------
        logits : [B, S, vocab_size]
        """
        h = self.norm(self.proj(hidden_states))
        if self.lm_head is not None:
            return self.lm_head(h)
        elif embedding_weight is not None:
            return F.linear(h, embedding_weight)
        else:
            raise ValueError("MTP head requires either lm_head or embedding_weight")


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DeepSeekV3Model(nn.Module):
    """Complete DeepSeek-V3 671B model.

    Parameters
    ----------
    config : DeepSeekV3Config
    """

    def __init__(self, config: DeepSeekV3Config = DEEPSEEK_V3_CONFIG) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            DeepSeekV3DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MTP head(s)
        self.mtp_heads = nn.ModuleList()
        for _ in range(config.mtp.num_mtp_layers):
            head = MTPPredictionHead(config)
            self.mtp_heads.append(head)

        # Tie MTP heads to main embedding if configured
        if config.mtp.share_embedding:
            for head in self.mtp_heads:
                head.lm_head = self.lm_head

        # Model parallel info
        self.gradient_checkpointing = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
        output_router_logits: bool = False,
        output_mtp_logits: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        input_ids : [B, S], optional
        attention_mask : [B, S] or [B, 1, S, S_kv], optional
        position_ids : [B, S], optional
        inputs_embeds : [B, S, D], optional
        kv_cache : KVCache, optional
        use_cache : bool
        output_router_logits : bool
        output_mtp_logits : bool

        Returns
        -------
        dict with keys:
          - logits: [B, S, vocab_size]
          - mtp_logits: list of [B, S, vocab_size] (if output_mtp_logits)
          - router_logits: list of router logit tensors (if output_router_logits)
          - cache_updates: list of cache update tuples (if use_cache)
        """
        # Embeddings
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        bsz, seq_len, _ = hidden_states.shape

        # Position IDs
        if position_ids is None:
            cache_len = 0
            if kv_cache is not None:
                cache_len = kv_cache.seq_lens[:bsz].max().item()
            position_ids = torch.arange(
                cache_len, cache_len + seq_len,
                dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0).expand(bsz, -1)

        # Prepare causal attention mask
        if attention_mask is not None and attention_mask.ndim == 2:
            # Convert [B, S] padding mask to [B, 1, S, S_kv] additive mask
            total_len = seq_len
            if kv_cache is not None:
                total_len += kv_cache.seq_lens[:bsz].max().item()
            attn_mask = self._prepare_attention_mask(
                attention_mask, seq_len, total_len, hidden_states.device, hidden_states.dtype
            )
        else:
            attn_mask = attention_mask

        # Layer-by-layer forward
        all_router_logits = []
        all_cache_updates = []

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, router_logits, cache_update = (
                    torch.utils.checkpoint.checkpoint(
                        layer,
                        hidden_states,
                        attn_mask,
                        position_ids,
                        kv_cache,
                        use_cache,
                        use_reentrant=False,
                    )
                )
            else:
                hidden_states, router_logits, cache_update = layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                    use_cache=use_cache,
                )

            if output_router_logits and router_logits is not None:
                all_router_logits.append(router_logits)
            if use_cache and cache_update is not None:
                all_cache_updates.append(cache_update)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # MTP logits
        mtp_logits = []
        if output_mtp_logits:
            for head in self.mtp_heads:
                mtp_logits.append(head(hidden_states))

        return {
            "logits": logits,
            "mtp_logits": mtp_logits if output_mtp_logits else None,
            "router_logits": all_router_logits if output_router_logits else None,
            "cache_updates": all_cache_updates if use_cache else None,
            "hidden_states": hidden_states,
        }

    @staticmethod
    def _prepare_attention_mask(
        mask: torch.Tensor,
        query_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert a [B, S] padding mask to a [B, 1, S_q, S_kv] additive mask."""
        bsz = mask.shape[0]
        # Causal mask
        causal = torch.triu(
            torch.ones(query_len, kv_len, device=device, dtype=dtype) * float("-inf"),
            diagonal=kv_len - query_len + 1,
        )
        causal = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, S_kv]

        # Padding mask
        if mask.shape[-1] < kv_len:
            # Pad mask to kv_len
            pad_len = kv_len - mask.shape[-1]
            mask = F.pad(mask, (pad_len, 0), value=1)

        padding_mask = mask[:, None, None, :kv_len].to(dtype)
        padding_mask = (1.0 - padding_mask) * float("-inf")

        return causal + padding_mask

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Count model parameters."""
        return sum(
            p.numel() for p in self.parameters()
            if not only_trainable or p.requires_grad
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.LongTensor:
        """Simple autoregressive generation loop.

        Parameters
        ----------
        input_ids : [B, S]
        max_new_tokens : int
        temperature : float
        top_k : int
        top_p : float

        Returns
        -------
        generated : [B, S + max_new_tokens]
        """
        bsz = input_ids.shape[0]
        device = input_ids.device

        cache = KVCache(
            self.config,
            max_batch_size=bsz,
            max_seq_len=input_ids.shape[1] + max_new_tokens,
            device=device,
            dtype=next(self.parameters()).dtype,
        )

        # Prefill
        outputs = self.forward(input_ids, kv_cache=cache, use_cache=True)
        cache.advance(bsz, input_ids.shape[1])
        next_logits = outputs["logits"][:, -1, :]  # [B, V]

        generated = input_ids
        for _ in range(max_new_tokens):
            # Sample
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                if top_k > 0:
                    topk_probs, topk_ids = torch.topk(probs, top_k, dim=-1)
                    # Top-p filtering on top-k
                    sorted_probs, sorted_idx = topk_probs.sort(descending=True, dim=-1)
                    cum_probs = sorted_probs.cumsum(dim=-1)
                    mask = cum_probs - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    sample_idx = torch.multinomial(sorted_probs, 1)
                    next_token = topk_ids.gather(-1, sorted_idx.gather(-1, sample_idx))
                else:
                    next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Decode step
            outputs = self.forward(
                next_token, kv_cache=cache, use_cache=True
            )
            cache.advance(bsz, 1)
            next_logits = outputs["logits"][:, -1, :]

        return generated
