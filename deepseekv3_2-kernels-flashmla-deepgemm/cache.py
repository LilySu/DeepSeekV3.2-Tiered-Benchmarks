"""
KV Cache optimised for FlashMLA.

DeepSeek-V3 MLA compresses K and V into a single latent vector of dimension
``kv_lora_rank`` (512).  Additionally, the RoPE portion of K (dimension 64)
is stored separately so that FlashMLA can apply the positional encoding
directly within the kernel.

The cache therefore stores **two** tensors per layer:

  1. ``kv_cache``  -- compressed latent  [batch, max_seq, kv_lora_rank]
  2. ``k_rope_cache`` -- RoPE keys       [batch, max_seq, qk_rope_head_dim]

FlashMLA reads these in paged / blocked layout for efficient decoding.

Reference: arXiv 2412.19437, Section 3.2.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import DeepSeekV3Config, DEEPSEEK_V3_CONFIG


class KVCache(nn.Module):
    """Compressed KV cache for DeepSeek-V3 MLA + FlashMLA.

    Parameters
    ----------
    config : DeepSeekV3Config
        Model configuration.
    max_batch_size : int
        Maximum batch size the cache is allocated for.
    max_seq_len : int
        Maximum sequence length the cache is allocated for.
    device : torch.device
        Device to allocate tensors on.
    dtype : torch.dtype
        Data type (typically float16 or bfloat16).
    """

    def __init__(
        self,
        config: DeepSeekV3Config = DEEPSEEK_V3_CONFIG,
        max_batch_size: int = 1,
        max_seq_len: int = 4096,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers  # 61
        self.kv_lora_rank = config.kv_lora_rank  # 512
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 64
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self._device = device or torch.device("cpu")

        # Allocate per-layer caches (list of tuples)
        # Each entry: (kv_compressed, k_rope)
        self._kv: list[Optional[torch.Tensor]] = [None] * self.num_layers
        self._k_rope: list[Optional[torch.Tensor]] = [None] * self.num_layers

        # Sequence-length tracker per batch element
        self.register_buffer(
            "seq_lens",
            torch.zeros(max_batch_size, dtype=torch.int32, device=self._device),
            persistent=False,
        )

        # FlashMLA page/block metadata
        self.page_size: int = 64  # tokens per page (FlashMLA default)
        self._num_pages = (max_seq_len + self.page_size - 1) // self.page_size
        # Page table: [batch, num_pages] mapping logical page -> physical page
        self.register_buffer(
            "page_table",
            torch.arange(
                max_batch_size * self._num_pages,
                dtype=torch.int32,
                device=self._device,
            ).view(max_batch_size, self._num_pages),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate_layer(self, layer_idx: int) -> None:
        """Pre-allocate cache tensors for a single layer."""
        if self._kv[layer_idx] is not None:
            return
        self._kv[layer_idx] = torch.zeros(
            self.max_batch_size,
            self.max_seq_len,
            self.kv_lora_rank,
            dtype=self.dtype,
            device=self._device,
        )
        self._k_rope[layer_idx] = torch.zeros(
            self.max_batch_size,
            self.max_seq_len,
            self.qk_rope_head_dim,
            dtype=self.dtype,
            device=self._device,
        )

    def allocate_all(self) -> None:
        """Pre-allocate cache tensors for all layers."""
        for i in range(self.num_layers):
            self.allocate_layer(i)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        layer_idx: int,
        kv_compressed: torch.Tensor,
        k_rope: torch.Tensor,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write new tokens into the cache and return the full cache slice.

        Parameters
        ----------
        layer_idx : int
        kv_compressed : [batch, new_seq, kv_lora_rank]
        k_rope : [batch, new_seq, qk_rope_head_dim]
        start_pos : [batch] int tensor, optional
            Per-sequence write position.  If ``None``, uses ``self.seq_lens``.

        Returns
        -------
        kv_out : [batch, total_seq, kv_lora_rank]
        k_rope_out : [batch, total_seq, qk_rope_head_dim]
        """
        if self._kv[layer_idx] is None:
            self.allocate_layer(layer_idx)

        bsz, new_len, _ = kv_compressed.shape
        if start_pos is None:
            start_pos = self.seq_lens[:bsz]

        # Scatter new tokens into pre-allocated buffer
        for b in range(bsz):
            sp = start_pos[b].item()
            self._kv[layer_idx][b, sp : sp + new_len] = kv_compressed[b]
            self._k_rope[layer_idx][b, sp : sp + new_len] = k_rope[b]

        # Compute total lengths after update
        total_lens = start_pos[:bsz] + new_len

        # Return valid slices
        max_total = total_lens.max().item()
        kv_out = self._kv[layer_idx][:bsz, :max_total]
        k_rope_out = self._k_rope[layer_idx][:bsz, :max_total]

        return kv_out, k_rope_out

    # ------------------------------------------------------------------
    # FlashMLA-specific accessors
    # ------------------------------------------------------------------

    def get_paged_kv(
        self, layer_idx: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cache in paged layout expected by FlashMLA kernel.

        Returns
        -------
        kv_data : [batch, max_seq, kv_lora_rank]
        k_rope_data : [batch, max_seq, qk_rope_head_dim]
        page_table : [batch, num_pages]
        seq_lens : [batch]
        """
        assert self._kv[layer_idx] is not None, f"Layer {layer_idx} not allocated"
        return (
            self._kv[layer_idx][:batch_size],
            self._k_rope[layer_idx][:batch_size],
            self.page_table[:batch_size],
            self.seq_lens[:batch_size],
        )

    def get_flat_kv(
        self, layer_idx: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return flat (non-paged) cache tensors for eager attention fallback.

        Returns
        -------
        kv_compressed : [batch, cached_seq, kv_lora_rank]
        k_rope : [batch, cached_seq, qk_rope_head_dim]
        """
        assert self._kv[layer_idx] is not None, f"Layer {layer_idx} not allocated"
        max_len = self.seq_lens[:batch_size].max().item()
        return (
            self._kv[layer_idx][:batch_size, :max_len],
            self._k_rope[layer_idx][:batch_size, :max_len],
        )

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def advance(self, batch_size: int, num_tokens: int = 1) -> None:
        """Advance the sequence-length counters after writing ``num_tokens``."""
        self.seq_lens[:batch_size] += num_tokens

    def reset(self, batch_indices: Optional[torch.Tensor] = None) -> None:
        """Reset cache for the given batch indices (or all if ``None``)."""
        if batch_indices is None:
            self.seq_lens.zero_()
            for i in range(self.num_layers):
                if self._kv[i] is not None:
                    self._kv[i].zero_()
                    self._k_rope[i].zero_()
        else:
            self.seq_lens[batch_indices] = 0
            for i in range(self.num_layers):
                if self._kv[i] is not None:
                    self._kv[i][batch_indices].zero_()
                    self._k_rope[i][batch_indices].zero_()

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        elem_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        per_layer = (
            self.max_batch_size
            * self.max_seq_len
            * (self.kv_lora_rank + self.qk_rope_head_dim)
            * elem_size
        )
        allocated = sum(1 for x in self._kv if x is not None)
        return per_layer * allocated

    def __repr__(self) -> str:
        allocated = sum(1 for x in self._kv if x is not None)
        return (
            f"KVCache(layers={self.num_layers}, allocated={allocated}, "
            f"max_batch={self.max_batch_size}, max_seq={self.max_seq_len}, "
            f"kv_rank={self.kv_lora_rank}, rope_dim={self.qk_rope_head_dim}, "
            f"page_size={self.page_size}, "
            f"memory={self.memory_bytes / 1024**2:.1f}MB)"
        )
