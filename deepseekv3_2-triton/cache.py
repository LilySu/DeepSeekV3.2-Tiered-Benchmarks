# KV cache for autoregressive generation.
# Stores key/value tensors per layer, concatenating new entries along the
# sequence dimension (dim=2) for each decode step.
#
# Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)
#
# This is identical to the raw model implementation. In the triton package,
# it's imported by model.py and mla_attention.py.
#
# STATUS: Pure PyTorch reference implementation. No Triton kernel needed.
# The KV cache is a simple list of tensor pairs; there's no GPU kernel to
# accelerate here. The memory layout (BHSD) is chosen to be compatible
# with both the PyTorch eager attention and potential FlashAttention
# integration.
#
# Key shapes for DeepSeek-V3 671B:
#   key:   [B, 128, T, 192]  (qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64)
#   value: [B, 128, T, 128]  (v_head_dim)

import torch


class KVCache:
    """Dynamic KV cache for autoregressive decoding.

    Each layer's cache holds (key, value) tensors in [B, H, T, D] format.
    On each step, new key/value states are concatenated to the existing cache.

    Args:
        num_layers: Number of decoder layers to allocate cache slots for.
                    DeepSeek-V3 has 61 layers.
    """

    def __init__(self, num_layers: int):
        # One slot per layer; None until first update
        self._cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers

    def update(
        self,
        key_states: torch.Tensor,    # [B, H, S_new, D]
        value_states: torch.Tensor,  # [B, H, S_new, D]
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value states to this layer's cache.

        Returns the full (past + new) key and value tensors.
        """
        if self._cache[layer_idx] is not None:
            prev_k, prev_v = self._cache[layer_idx]
            key_states = torch.cat([prev_k, key_states], dim=2)
            value_states = torch.cat([prev_v, value_states], dim=2)
        self._cache[layer_idx] = (key_states, value_states)
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current cached sequence length for a given layer."""
        if self._cache[layer_idx] is None:
            return 0
        return self._cache[layer_idx][0].shape[2]

    def reset(self):
        """Clear all cached states."""
        self._cache = [None] * len(self._cache)
