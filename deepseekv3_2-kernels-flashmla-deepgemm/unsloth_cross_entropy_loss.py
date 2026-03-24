"""
Unsloth-optimised cross-entropy loss for DeepSeek-V3.

Provides a memory-efficient cross-entropy implementation that:
  - Processes the vocabulary in chunks to avoid materialising the full
    [batch*seq, vocab_size] logit tensor.
  - Supports MTP (Multi-Token Prediction) loss aggregation.
  - Compatible with LoRA fine-tuning via Unsloth.

Reference: Unsloth (https://github.com/unslothai/unsloth)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnslothCrossEntropyLoss(nn.Module):
    """Memory-efficient chunked cross-entropy loss.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    chunk_size : int
        Number of vocabulary entries processed per chunk.
        Smaller = less peak memory, slightly slower.
    ignore_index : int
        Token ID to ignore in loss computation.
    reduction : str
        ``'mean'`` or ``'sum'`` or ``'none'``.
    label_smoothing : float
        Label smoothing factor (0.0 = no smoothing).
    """

    def __init__(
        self,
        vocab_size: int = 129280,
        chunk_size: int = 8192,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        lm_head_weight: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Can operate in two modes:
          1. **Logits mode**: ``logits`` is [B, S, V] pre-computed.
          2. **Chunked mode**: ``hidden_states`` is [B, S, D] and
             ``lm_head_weight`` is [V, D].  Logits are computed in chunks
             to save memory.

        Parameters
        ----------
        logits : [B, S, V] or None
        labels : [B, S]
        lm_head_weight : [V, D], optional
        hidden_states : [B, S, D], optional

        Returns
        -------
        loss : scalar tensor
        """
        if hidden_states is not None and lm_head_weight is not None:
            return self._chunked_forward(hidden_states, lm_head_weight, labels)
        else:
            return self._standard_forward(logits, labels)

    def _standard_forward(
        self, logits: torch.Tensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        """Standard CE loss on pre-computed logits."""
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        return F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

    def _chunked_forward(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        """Chunked CE loss -- never materialises full logits."""
        # Shift
        h = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        B, S, D = h.shape
        h = h.view(-1, D)  # [T, D]
        shift_labels = shift_labels.view(-1)  # [T]

        # Valid mask
        valid = shift_labels != self.ignore_index
        if not valid.any():
            return torch.tensor(0.0, device=h.device, dtype=h.dtype, requires_grad=True)

        total_loss = torch.zeros(1, device=h.device, dtype=torch.float32)
        num_valid = valid.sum().float()

        # Process vocabulary in chunks
        V = lm_head_weight.shape[0]
        for v_start in range(0, V, self.chunk_size):
            v_end = min(v_start + self.chunk_size, V)
            w_chunk = lm_head_weight[v_start:v_end]  # [chunk, D]
            logit_chunk = h @ w_chunk.T  # [T, chunk]

            # Adjust labels for this chunk
            chunk_labels = shift_labels.clone()
            in_range = (chunk_labels >= v_start) & (chunk_labels < v_end)
            chunk_labels[~in_range] = self.ignore_index
            chunk_labels[in_range] -= v_start

            if in_range.any():
                loss_chunk = F.cross_entropy(
                    logit_chunk,
                    chunk_labels,
                    ignore_index=self.ignore_index,
                    reduction="sum",
                    label_smoothing=self.label_smoothing,
                )
                total_loss += loss_chunk

        if self.reduction == "mean":
            return total_loss / num_valid.clamp(min=1.0)
        elif self.reduction == "sum":
            return total_loss
        else:
            return total_loss  # 'none' not fully supported in chunked mode


class MTPLoss(nn.Module):
    """Aggregated loss for Multi-Token Prediction.

    Combines the main next-token loss with MTP auxiliary losses.

    Parameters
    ----------
    num_mtp_layers : int
        Number of MTP prediction layers.
    mtp_weight : float
        Weight for MTP loss relative to main loss.
    """

    def __init__(
        self,
        num_mtp_layers: int = 1,
        mtp_weight: float = 0.1,
        vocab_size: int = 129280,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.num_mtp_layers = num_mtp_layers
        self.mtp_weight = mtp_weight
        self.ce = UnslothCrossEntropyLoss(
            vocab_size=vocab_size,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        main_logits: torch.Tensor,
        mtp_logits_list: list,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        main_logits : [B, S, V]
        mtp_logits_list : list of [B, S, V] (one per MTP layer)
        labels : [B, S]

        Returns
        -------
        total_loss : scalar
        """
        main_loss = self.ce(main_logits, labels)
        total = main_loss

        for i, mtp_logits in enumerate(mtp_logits_list):
            # MTP layer i predicts token at position t+i+2
            if labels.shape[1] > i + 2:
                shifted_labels = labels[:, i + 2 :].contiguous()
                truncated_logits = mtp_logits[:, : shifted_labels.shape[1], :].contiguous()
                mtp_loss = self.ce._standard_forward(
                    torch.cat(
                        [truncated_logits, truncated_logits[:, -1:, :]], dim=1
                    ),
                    torch.cat(
                        [shifted_labels, shifted_labels[:, -1:]], dim=1
                    ),
                )
                total = total + self.mtp_weight * mtp_loss

        return total
