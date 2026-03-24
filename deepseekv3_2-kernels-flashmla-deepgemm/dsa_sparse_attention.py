"""
DSA Sparse Attention -- STUB with eager attention passthrough.

Dynamic Sparse Attention is a **GLM5-specific** mechanism and is NOT used
in DeepSeek-V3.  This module provides an ``eager_attention_forward``
passthrough that implements standard dense causal attention, maintaining
interface compatibility with the shared kernel directory layout.

DeepSeek-V3 attention is handled by :mod:`mla_attention` using FlashMLA.

Reference: arXiv 2412.19437 -- DeepSeek-V3 does not employ DSA.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    causal: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Standard eager multi-head attention (no sparsity).

    This is the DSA-stub passthrough: it performs **full** causal (or
    non-causal) attention, since DeepSeek-V3 does not use dynamic sparse
    patterns.

    Parameters
    ----------
    query : Tensor [B, H, S_q, D]
    key : Tensor [B, H, S_kv, D]
    value : Tensor [B, H, S_kv, D_v]
    attention_mask : Tensor [B, 1, S_q, S_kv], optional
        Additive mask (0 = attend, -inf = mask).
    scale : float, optional
        Softmax scaling factor.  Defaults to 1/sqrt(D).
    causal : bool
        If True, applies a causal (lower-triangular) mask.
    dropout_p : float
        Attention dropout probability.
    training : bool
        Whether we are in training mode (for dropout).

    Returns
    -------
    output : Tensor [B, H, S_q, D_v]
    """
    D = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # QK^T
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Causal mask
    if causal and query.shape[-2] > 1:
        S_q = query.shape[-2]
        S_kv = key.shape[-2]
        causal_mask = torch.triu(
            torch.ones(S_q, S_kv, dtype=torch.bool, device=query.device),
            diagonal=S_kv - S_q + 1,
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    # Additive attention mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # Dropout
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # Weighted sum
    output = torch.matmul(attn_weights, value)
    return output


class DSASparseAttention:
    """Stub sparse attention module.

    Always delegates to :func:`eager_attention_forward` (full attention).
    No sparse patterns are computed or applied.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._reason = "DSA not applicable to DeepSeek-V3"

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return eager_attention_forward(query, key, value, **kwargs)

    @staticmethod
    def is_applicable() -> bool:
        return False

    def __repr__(self) -> str:
        return f"DSASparseAttention(stub=True, reason='{self._reason}')"
