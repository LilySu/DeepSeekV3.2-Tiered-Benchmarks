# DSA Sparse Attention -- STUB for DeepSeek-V3.
#
# DeepSeek-V3 does NOT use Dynamic Sparse Attention (DSA). Unlike GLM-5 which
# uses DSA to select top-2048 tokens per query position, DeepSeek-V3 relies on
# standard causal attention for all layers.
#
# This module provides passthrough implementations of build_dsa_mask and
# eager_attention_forward for API compatibility with code that expects the
# DSA interface (e.g., shared test harnesses, unified model pipelines).
#
# The functions are fully functional but the mask-building path is never
# triggered in normal DeepSeek-V3 operation since no DSAIndexer produces
# topk_indices.
#
# Paper ref: DeepSeek-V3 (arXiv 2412.19437) -- no mention of DSA/sparse attention

import torch
import torch.nn.functional as F


def build_dsa_mask(topk_indices, attention_mask, query_states, total_len):
    """Build combined DSA sparse + causal attention mask.

    NOTE: This is a passthrough stub for DeepSeek-V3. DSA is not applicable.
    If topk_indices covers all positions (full selection), the mask reduces
    to the standard causal mask. This function is provided for API
    compatibility with code paths that expect a DSA mask builder.

    Args:
        topk_indices: [B, S, K] selected token indices (passthrough)
        attention_mask: [B, 1, S, T] causal attention mask
        query_states: query tensor (used for dtype/device)
        total_len: total KV sequence length

    Returns:
        combined_mask: attention mask tensor
    """
    # If no topk_indices provided, just return the causal mask
    if topk_indices is None:
        return attention_mask

    batch_size = topk_indices.shape[0]
    seq_length = topk_indices.shape[1]

    index_mask = torch.full(
        (batch_size, seq_length, total_len), float("-inf"),
        device=query_states.device, dtype=query_states.dtype,
    )
    index_mask.scatter_(-1, topk_indices, 0.0)
    index_mask = index_mask.unsqueeze(1)

    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask[..., :total_len]
        combined_mask = index_mask + causal_mask
    else:
        combined_mask = (
            attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
            if attention_mask is not None else index_mask
        )

    return combined_mask


def eager_attention_forward(query, key, value, attention_mask, scaling,
                            num_key_value_groups=1, dropout=0.0, training=False):
    """Standard eager attention with GQA expansion (fallback path).

    This is the same implementation used in the GLM-5 path. For DeepSeek-V3,
    this serves as the primary eager fallback when FlashInfer is not available.
    No DSA-specific logic is needed.
    """
    n_rep = num_key_value_groups
    if n_rep > 1:
        b, h_kv, t, d = key.shape
        key = key[:, :, None, :, :].expand(b, h_kv, n_rep, t, d).reshape(b, h_kv * n_rep, t, d)
        b, h_kv, t, d = value.shape
        value = value[:, :, None, :, :].expand(b, h_kv, n_rep, t, d).reshape(b, h_kv * n_rep, t, d)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
