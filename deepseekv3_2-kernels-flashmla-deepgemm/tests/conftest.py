"""
Pytest fixtures for DeepSeek-V3 FlashMLA + DeepGEMM test suite.

Provides reusable fixtures for model configuration, small model instances,
random tensors, and KV cache objects -- all CPU-runnable.
"""

from __future__ import annotations

import sys
import os
from typing import Optional

import pytest
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config, YaRNRoPEConfig, MoEConfig, MTPConfig
from cache import KVCache
from mla_attention import MLAttention
from moe_router import MoERouter
from moe_grouped_gemm import MoEGroupedGEMM, ExpertFFN
from rope_partial import YaRNRotaryEmbedding
from fp8_utils import quantize_fp8_block, dequantize_fp8_block
from model import DeepSeekV3Model


# ---------------------------------------------------------------------------
# Small config for testing (reduced dimensions for CPU speed)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config() -> DeepSeekV3Config:
    """A tiny DeepSeek-V3 config for fast CPU tests."""
    return DeepSeekV3Config(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_scaling=YaRNRoPEConfig(
            factor=40.0,
            original_max_position_embeddings=64,
            beta_fast=32.0,
            beta_slow=1.0,
        ),
        moe=MoEConfig(
            num_experts=16,
            num_experts_per_tok=4,
            n_group=4,
            topk_group=2,
            num_shared_experts=1,
            expert_intermediate_size=128,
            shared_expert_intermediate_size=128,
            first_k_dense_replace=1,
            use_fp8_gemm=False,
        ),
        mtp=MTPConfig(num_mtp_layers=1),
        use_flashmla=False,
        use_deepgemm=False,
    )


@pytest.fixture
def full_config() -> DeepSeekV3Config:
    """Full DeepSeek-V3 config (for parameter counting, not forward pass)."""
    return DeepSeekV3Config()


@pytest.fixture
def device() -> torch.device:
    """Test device (always CPU for unit tests)."""
    return torch.device("cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    """Default test dtype."""
    return torch.float32


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def seq_len() -> int:
    return 16


@pytest.fixture
def hidden_states(small_config, batch_size, seq_len) -> torch.Tensor:
    """Random hidden states [B, S, D]."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, small_config.hidden_size)


@pytest.fixture
def input_ids(small_config, batch_size, seq_len) -> torch.LongTensor:
    """Random input IDs [B, S]."""
    torch.manual_seed(42)
    return torch.randint(0, small_config.vocab_size, (batch_size, seq_len))


@pytest.fixture
def position_ids(batch_size, seq_len) -> torch.LongTensor:
    """Position IDs [B, S]."""
    return torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model(small_config) -> DeepSeekV3Model:
    """Small DeepSeek-V3 model for CPU testing."""
    torch.manual_seed(42)
    model = DeepSeekV3Model(small_config)
    model.eval()
    return model


@pytest.fixture
def kv_cache(small_config, batch_size) -> KVCache:
    """KV cache for testing."""
    return KVCache(
        small_config,
        max_batch_size=batch_size,
        max_seq_len=128,
        dtype=torch.float32,
    )


@pytest.fixture
def rope_module(small_config) -> YaRNRotaryEmbedding:
    """YaRN RoPE module for testing."""
    return YaRNRotaryEmbedding(small_config)


@pytest.fixture
def moe_router(small_config) -> MoERouter:
    """MoE router for testing."""
    torch.manual_seed(42)
    return MoERouter(small_config)


@pytest.fixture
def attention_layer(small_config) -> MLAttention:
    """MLA attention layer for testing."""
    torch.manual_seed(42)
    return MLAttention(small_config, layer_idx=0)


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="Hopper GPU (SM90+) required",
)

requires_flash_mla = pytest.mark.skipif(
    True,  # Will be dynamically set based on import
    reason="FlashMLA not installed",
)

requires_deepgemm = pytest.mark.skipif(
    True,  # Will be dynamically set based on import
    reason="DeepGEMM not installed",
)

# Try to update skip conditions
try:
    import flash_mla
    requires_flash_mla = pytest.mark.skipif(False, reason="")
except ImportError:
    pass

try:
    import deep_gemm
    requires_deepgemm = pytest.mark.skipif(False, reason="")
except ImportError:
    pass
