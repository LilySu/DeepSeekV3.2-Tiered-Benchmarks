"""
Unsloth fast LoRA utilities for DeepSeek-V3.

Provides optimised LoRA (Low-Rank Adaptation) routines tailored for the
DeepSeek-V3 MLA architecture:

  - Efficient LoRA application for q_a_proj, q_b_proj, kv_a_proj, kv_b_proj
  - Fused LoRA forward that avoids materialising full adapter matrices
  - MoE-aware LoRA: only applies adapters to attention and shared expert
    (not routed experts by default)

Reference: Unsloth (https://github.com/unslothai/unsloth)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer.

    Parameters
    ----------
    base_layer : nn.Linear
        Original linear layer (frozen).
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling factor.
    dropout : float
        Dropout on LoRA path.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        if dropout > 0:
            self.lora_dropout = nn.Dropout(dropout)
        else:
            self.lora_dropout = nn.Identity()

        # Initialise
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_out + self.scaling * lora_out

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base layer for inference."""
        with torch.no_grad():
            merged = self.base_layer
            delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            merged.weight.add_(delta)
        return merged


class FastLoRAConfig:
    """Configuration for which layers to apply LoRA to.

    Parameters
    ----------
    target_modules : set of str
        Names of module types or specific modules to adapt.
    rank : int
        LoRA rank.
    alpha : float
        LoRA alpha.
    dropout : float
        LoRA dropout.
    lora_on_moe_experts : bool
        Whether to apply LoRA to routed MoE experts (expensive -- default False).
    """

    def __init__(
        self,
        target_modules: Optional[Set[str]] = None,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        lora_on_moe_experts: bool = False,
    ) -> None:
        if target_modules is None:
            # Default: MLA projections + output + shared expert
            self.target_modules = {
                "q_a_proj", "q_b_proj",
                "kv_a_proj", "kv_b_proj",
                "o_proj",
                "gate_proj", "up_proj", "down_proj",  # shared expert only
            }
        else:
            self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_on_moe_experts = lora_on_moe_experts


def apply_fast_lora(
    model: nn.Module,
    config: FastLoRAConfig,
) -> nn.Module:
    """Apply LoRA adapters to the model in-place.

    Parameters
    ----------
    model : nn.Module
        The DeepSeek-V3 model.
    config : FastLoRAConfig
        LoRA configuration.

    Returns
    -------
    model with LoRA adapters applied.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        # Skip routed experts unless explicitly enabled
        if not config.lora_on_moe_experts and ".experts." in name:
            continue

        if isinstance(module, nn.Linear):
            # Check if this module's local name matches target
            local_name = name.split(".")[-1]
            if local_name in config.target_modules:
                # Replace with LoRA version
                lora_layer = LoRALinear(
                    module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                # Set in parent
                parent = model
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], lora_layer)
                replaced += 1

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only the LoRA adapter parameters (for optimizer)."""
    params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            params.append(param)
    return params


def merge_all_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA adapters into base weights for inference."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            merged = module.merge()
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], merged)
    return model


def count_lora_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable vs total parameters.

    Returns
    -------
    trainable : int
    total : int
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
