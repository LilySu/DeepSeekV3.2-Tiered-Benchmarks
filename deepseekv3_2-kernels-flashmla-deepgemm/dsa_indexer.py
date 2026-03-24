"""
DSA (Dynamic Sparse Attention) Indexer -- STUB.

Dynamic Sparse Attention is a **GLM5-specific** mechanism and is NOT used
in DeepSeek-V3.  This module exists solely for interface compatibility with
the shared kernel directory layout.

DeepSeek-V3 uses standard causal attention with FlashMLA acceleration.
No sparse attention patterns, no dynamic block selection.

Reference: arXiv 2412.19437 -- no DSA mentioned in DeepSeek-V3 architecture.
"""

from __future__ import annotations

from typing import Optional

import torch


class DSAIndexer:
    """Stub DSA indexer. All methods return None / no-ops.

    This class satisfies any code path that conditionally checks for a DSA
    indexer without introducing runtime errors.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def build_index(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        seq_len: int = 0,
        **kwargs,
    ) -> None:
        """No-op: DSA is not applicable to DeepSeek-V3."""
        return None

    def get_sparse_mask(
        self,
        batch_size: int = 1,
        seq_len: int = 1,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Returns None -- DeepSeek-V3 uses full causal attention."""
        return None

    def get_block_indices(self, *args, **kwargs) -> Optional[torch.Tensor]:
        """Returns None."""
        return None

    @staticmethod
    def is_applicable() -> bool:
        """DSA is never applicable for DeepSeek-V3."""
        return False

    def __repr__(self) -> str:
        return "DSAIndexer(stub=True, reason='DSA not applicable to DeepSeek-V3')"
