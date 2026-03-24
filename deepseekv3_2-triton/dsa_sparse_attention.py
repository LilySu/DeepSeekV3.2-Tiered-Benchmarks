# DSA Sparse Attention -- NOT APPLICABLE to DeepSeek-V3.
#
# DSA (Dynamic Sparse Attention) is a GLM5-specific feature (arXiv 2602.15763v2,
# Section 2.1.1) that uses a "Lightning Indexer" to select top-k tokens for
# sparse attention. Each query position scores all key positions and attends
# to only the top-2048 most relevant ones, reducing attention from O(n^2) to O(n*k).
#
# DeepSeek-V3 (arXiv 2412.19437) does NOT use DSA. It uses standard causal
# attention with full sequence attention. The extended context (163840 tokens)
# is handled through YaRN RoPE scaling rather than sparse attention.
#
# This file provides stub/passthrough functions for API compatibility,
# in case any code references these functions by name. The functions
# are no-ops that delegate to standard causal attention behavior.
#
# STATUS: Stub -- NOT applicable to DeepSeek-V3.
#
# For GLM5's DSA implementation, see:
#   glm5-triton/dsa_indexer.py    -- selects top-k tokens per query position
#   glm5-triton/dsa_sparse_attention.py -- builds combined causal + sparse mask

import torch
import torch.nn.functional as F


def build_dsa_mask(topk_indices, attention_mask, query_states, total_len):
    """Build DSA sparse + causal attention mask.

    NOTE: This is a NO-OP passthrough for DeepSeek-V3. DSA is a GLM5-specific
    feature. This function simply returns the standard causal attention mask
    unmodified.

    In GLM5, this function would:
    1. Start with an all-inf mask (block everything)
    2. Allow only the top-k positions selected by the DSA indexer
    3. Combine with the causal mask

    For DeepSeek-V3, the causal mask is already sufficient.

    Args:
        topk_indices:    unused (would be [B, S, topk] indices from DSAIndexer)
        attention_mask:  [B, 1, S, T] -- standard causal mask
        query_states:    unused (only needed for device/dtype in GLM5)
        total_len:       unused

    Returns:
        attention_mask:  [B, 1, S, T] -- the input mask, unchanged
    """
    # DeepSeek-V3 does not use DSA. Return the causal mask as-is.
    return attention_mask


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Standard eager attention with GQA expansion.

    This is the same attention implementation used by DeepSeek-V3's MLA.
    In GLM5, this function also handles the DSA-combined mask, but since
    DeepSeek-V3 does not use DSA, this is just standard causal attention.

    Args:
        module:          attention module (needs .num_key_value_groups attribute)
        query:           [B, H, S, qk_head_dim]
        key:             [B, H_kv, T, qk_head_dim]
        value:           [B, H_kv, T, v_head_dim]
        attention_mask:  [B, 1, S, T] -- standard causal mask (0=attend, -inf=mask)
        scaling:         float -- 1/sqrt(qk_head_dim)
        dropout:         float -- attention dropout rate

    Returns:
        attn_output:     [B, S, H * v_head_dim]
        attn_weights:    [B, H, S, T]
    """
    # Expand KV heads to match Q heads (GQA / MQA expansion)
    n_rep = module.num_key_value_groups
    if n_rep > 1:
        batch, n_kv_heads, slen, head_dim = key.shape
        key = key[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
        key = key.reshape(batch, n_kv_heads * n_rep, slen, head_dim)
        batch, n_kv_heads, slen, head_dim = value.shape
        value = value[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
        value = value.reshape(batch, n_kv_heads * n_rep, slen, head_dim)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
