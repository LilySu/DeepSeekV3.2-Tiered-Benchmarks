# DSA Indexer — NOT APPLICABLE to DeepSeek-V3.
#
# Dynamic Sparse Attention (DSA) is a GLM-5 innovation (arXiv 2602.15763v2)
# that selects top-k tokens for sparse attention via a lightweight scoring
# network. This component does NOT exist in DeepSeek-V3.
#
# DeepSeek-V3 (arXiv 2412.19437) uses standard causal attention through
# Multi-head Latent Attention (MLA). The attention mask is purely causal
# (lower-triangular), with no sparse token selection.
#
# This stub file exists only for structural parity with the GLM-5 codebase.
# It provides a no-op DSAIndexer class for API compatibility.
#
# Architecture difference:
#   GLM-5:      MLA + DSA indexer (top-2048 sparse attention)
#   DeepSeek-V3: MLA only (standard causal attention, no sparsity)

import torch
import torch.nn as nn


class DSAIndexer(nn.Module):
    """Stub: DSA is not used in DeepSeek-V3.

    DeepSeek-V3 uses standard causal attention without sparse token selection.
    This class exists only for API compatibility with code that expects a
    DSAIndexer attribute on attention layers.
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        # No parameters — DSA is not applicable

    @torch.no_grad()
    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask=None, use_cache=False):
        """Returns None — no sparse token selection in DeepSeek-V3."""
        return None
