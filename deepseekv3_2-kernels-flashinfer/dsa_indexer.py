# DSA Lightning Indexer -- STUB for DeepSeek-V3.
#
# DeepSeek-V3 does NOT use Dynamic Sparse Attention (DSA). This module
# provides a no-op stub that implements the DSAIndexer interface for API
# compatibility with unified model code that conditionally uses DSA.
#
# GLM-5 uses DSA with index_topk=2048, index_head_dim=128, index_n_heads=32.
# DeepSeek-V3 omits this entirely and uses standard causal attention.
#
# The stub returns all-positions indices (equivalent to full attention
# selection), ensuring that any downstream code consuming DSA indices
# produces correct results by selecting all available tokens.
#
# Paper ref: DeepSeek-V3 (arXiv 2412.19437) -- no DSA component

import torch
import torch.nn as nn


class DSAIndexerStub(nn.Module):
    """No-op DSA indexer stub for DeepSeek-V3.

    Returns indices selecting ALL available positions (equivalent to no sparsity).
    This ensures API compatibility with code paths that expect a DSA indexer.

    For DeepSeek-V3, attention is always dense causal -- no sparse token
    selection is performed.
    """

    def __init__(self, cfg=None, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        # No learnable parameters -- this is a passthrough

    @torch.no_grad()
    def forward(self, hidden_states, q_resid=None, position_embeddings=None,
                attention_mask=None, use_cache=False):
        """Return indices for all positions (full attention).

        Args:
            hidden_states: [B, S, D] input hidden states
            q_resid: unused (no query compression for indexer)
            position_embeddings: unused (no RoPE for indexer)
            attention_mask: unused (causal mask applied downstream)
            use_cache: unused (no indexer cache)

        Returns:
            indices: [B, S, T] where T = total KV length, selecting all positions
        """
        batch_size, seq_len, _ = hidden_states.shape

        # In a full-attention setting, each query attends to all positions
        # up to its own position (causality is enforced by the attention mask).
        # Return arange indices as a stand-in.
        indices = torch.arange(seq_len, device=hidden_states.device)
        indices = indices.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        return indices


# Alias for backward compatibility
DSAIndexer = DSAIndexerStub
