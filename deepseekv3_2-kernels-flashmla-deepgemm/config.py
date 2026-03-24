"""
DeepSeek-V3 model configuration.

Full configuration for the DeepSeek-V3 671B model (arXiv 2412.19437).
This mirrors the HuggingFace ``DeepseekV3Config`` layout so that state-dict
loading is straightforward, while also exposing kernel-specific fields
consumed by FlashMLA and DeepGEMM.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# YaRN RoPE sub-config
# ---------------------------------------------------------------------------

@dataclass
class YaRNRoPEConfig:
    """YaRN (Yet another RoPE extensioN) parameters for DeepSeek-V3."""

    type: str = "yarn"
    factor: float = 40.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 0.707


# ---------------------------------------------------------------------------
# MoE sub-config
# ---------------------------------------------------------------------------

@dataclass
class MoEConfig:
    """Mixture-of-Experts configuration for DeepSeek-V3."""

    num_experts: int = 256
    num_experts_per_tok: int = 8  # top-k
    n_group: int = 8
    topk_group: int = 4
    num_shared_experts: int = 1
    # DeepSeek-V3 uses sigmoid routing with no auxiliary loss (noaux_tc)
    routing_type: str = "sigmoid"
    aux_loss_type: str = "noaux_tc"
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    # Expert intermediate size (per expert)
    expert_intermediate_size: int = 2048
    shared_expert_intermediate_size: int = 2048
    # First k dense layers (no MoE)
    first_k_dense_replace: int = 3
    # FP8 DeepGEMM settings
    use_fp8_gemm: bool = True
    fp8_block_size: int = 128  # 128x128 tile quantisation


# ---------------------------------------------------------------------------
# MTP sub-config
# ---------------------------------------------------------------------------

@dataclass
class MTPConfig:
    """Multi-Token Prediction configuration for DeepSeek-V3."""

    num_mtp_layers: int = 1
    mtp_hidden_size: int = 7168
    # MTP shares the main embedding and uses a lightweight projection
    share_embedding: bool = True


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class DeepSeekV3Config:
    """Complete DeepSeek-V3 671B configuration.

    Attribute names follow the HuggingFace ``config.json`` convention so that
    ``model.load_state_dict(...)`` works out of the box.

    Reference: arXiv 2412.19437, Table 1 & Appendix A.
    """

    # ---- general ----
    model_type: str = "deepseek_v3"
    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128  # Same as num_attention_heads (MLA expands via kv_b_proj)
    max_position_embeddings: int = 163840
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    attention_bias: bool = False

    # ---- MLA (Multi-head Latent Attention) ----
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # Derived (set in __post_init__)
    qk_head_dim: int = field(init=False, default=0)

    # ---- RoPE ----
    rope_theta: float = 10000.0
    rope_scaling: YaRNRoPEConfig = field(default_factory=YaRNRoPEConfig)

    # ---- MoE ----
    moe: MoEConfig = field(default_factory=MoEConfig)

    # ---- MTP ----
    mtp: MTPConfig = field(default_factory=MTPConfig)

    # ---- Kernel hints (not part of HF config, consumed by this package) ----
    use_flashmla: bool = True
    use_deepgemm: bool = True
    # DSA is *not* applicable to DeepSeek-V3
    use_dsa: bool = False

    def __post_init__(self) -> None:
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # Sanity checks
        assert self.qk_head_dim == 192, (
            f"Expected qk_head_dim=192, got {self.qk_head_dim}"
        )
        assert self.moe.num_experts == 256
        assert self.moe.num_experts_per_tok == 8

    # ---- helpers ----

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeepSeekV3Config":
        d = copy.deepcopy(d)
        if "rope_scaling" in d and isinstance(d["rope_scaling"], dict):
            d["rope_scaling"] = YaRNRoPEConfig(**d["rope_scaling"])
        if "moe" in d and isinstance(d["moe"], dict):
            d["moe"] = MoEConfig(**d["moe"])
        if "mtp" in d and isinstance(d["mtp"], dict):
            d["mtp"] = MTPConfig(**d["mtp"])
        # Remove derived field if present
        d.pop("qk_head_dim", None)
        return cls(**d)

    @property
    def head_dim(self) -> int:
        """Alias used by FlashMLA dispatch."""
        return self.v_head_dim

    @property
    def kv_compressed_dim(self) -> int:
        """Total compressed KV dimension stored per token."""
        return self.kv_lora_rank + self.qk_rope_head_dim  # 512 + 64 = 576

    @property
    def num_local_experts(self) -> int:
        return self.moe.num_experts

    @property
    def num_experts_per_tok(self) -> int:
        return self.moe.num_experts_per_tok


# ---------------------------------------------------------------------------
# Singleton default config
# ---------------------------------------------------------------------------

DEEPSEEK_V3_CONFIG = DeepSeekV3Config()
