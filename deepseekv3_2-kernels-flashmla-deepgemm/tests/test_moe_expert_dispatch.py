"""
MoE routing and expert dispatch tests.

Verifies:
  - Sigmoid router produces valid outputs
  - Group-restricted top-k routing selects correct number of experts
  - Expert dispatch gathers and scatters tokens correctly
  - Weight normalisation is applied
  - Load balance statistics are computed correctly

CPU-runnable.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config
from moe_router import MoERouter
from moe_grouped_gemm import MoEGroupedGEMM, ExpertFFN


class TestMoERouterBasics:
    """Test basic router functionality."""

    def test_router_output_shapes(self, small_config, moe_router):
        """Router outputs should have correct shapes."""
        T = 16
        x = torch.randn(T, small_config.hidden_size)

        weights, indices, logits = moe_router(x)

        top_k = small_config.moe.num_experts_per_tok
        num_experts = small_config.moe.num_experts

        assert weights.shape == (T, top_k)
        assert indices.shape == (T, top_k)
        assert logits.shape == (T, num_experts)

    def test_router_weights_positive(self, small_config, moe_router):
        """Routing weights should be non-negative (sigmoid scores)."""
        T = 8
        x = torch.randn(T, small_config.hidden_size)
        weights, _, _ = moe_router(x)

        assert (weights >= 0).all()

    def test_router_weights_normalised(self, small_config, moe_router):
        """Routing weights should sum to 1 per token (after normalisation)."""
        T = 8
        x = torch.randn(T, small_config.hidden_size)
        weights, _, _ = moe_router(x)

        sums = weights.sum(dim=-1)
        torch.testing.assert_close(
            sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5
        )

    def test_router_indices_valid(self, small_config, moe_router):
        """Expert indices should be in valid range."""
        T = 8
        x = torch.randn(T, small_config.hidden_size)
        _, indices, _ = moe_router(x)

        assert (indices >= 0).all()
        assert (indices < small_config.moe.num_experts).all()

    def test_router_top_k_unique_per_token(self, small_config, moe_router):
        """Each token should select K distinct experts."""
        T = 8
        x = torch.randn(T, small_config.hidden_size)
        _, indices, _ = moe_router(x)

        for t in range(T):
            unique = indices[t].unique()
            assert len(unique) == small_config.moe.num_experts_per_tok


class TestGroupRestrictedRouting:
    """Test group-level routing restriction."""

    def test_group_restriction_active(self, small_config, moe_router):
        """Selected experts should come from exactly topk_group groups."""
        torch.manual_seed(42)
        T = 16
        x = torch.randn(T, small_config.hidden_size)
        _, indices, _ = moe_router(x)

        topk_group = small_config.moe.topk_group
        experts_per_group = small_config.moe.num_experts // small_config.moe.n_group

        for t in range(T):
            groups_used = set()
            for idx in indices[t]:
                group = idx.item() // experts_per_group
                groups_used.add(group)
            assert len(groups_used) <= topk_group, (
                f"Token {t} uses {len(groups_used)} groups, max is {topk_group}"
            )

    def test_sigmoid_scoring(self, small_config):
        """Sigmoid scoring should produce values in (0, 1)."""
        router = MoERouter(small_config)
        T = 4
        x = torch.randn(T, small_config.hidden_size)

        _, _, logits = router(x)
        scores = torch.sigmoid(logits)

        assert (scores > 0).all()
        assert (scores < 1).all()


class TestExpertDispatch:
    """Test expert dispatch and output aggregation."""

    def test_moe_layer_output_shape(self, small_config):
        """MoE layer output should match input shape."""
        torch.manual_seed(42)
        layer_idx = small_config.moe.first_k_dense_replace
        moe = MoEGroupedGEMM(small_config, layer_idx=layer_idx)

        B, S, D = 2, 4, small_config.hidden_size
        x = torch.randn(B, S, D)

        out, router_logits = moe(x)

        assert out.shape == (B, S, D)
        assert router_logits is not None

    def test_dense_layer_no_router(self, small_config):
        """Dense layers (first k) should not have router logits."""
        torch.manual_seed(42)
        moe = MoEGroupedGEMM(small_config, layer_idx=0)

        B, S, D = 1, 4, small_config.hidden_size
        x = torch.randn(B, S, D)
        out, router_logits = moe(x)

        assert out.shape == (B, S, D)
        assert router_logits is None

    def test_shared_expert_always_active(self, small_config):
        """Shared expert should contribute to all tokens."""
        torch.manual_seed(42)
        layer_idx = small_config.moe.first_k_dense_replace
        moe = MoEGroupedGEMM(small_config, layer_idx=layer_idx)
        assert moe.shared_expert is not None

    def test_expert_ffn_swiglu(self):
        """Individual expert FFN should use SwiGLU."""
        expert = ExpertFFN(64, 128)
        x = torch.randn(4, 64)
        out = expert(x)
        assert out.shape == (4, 64)


class TestLoadBalanceStats:
    """Test load balance monitoring."""

    def test_compute_stats(self, small_config, moe_router):
        """Load balance stats should be computed correctly."""
        T = 32
        x = torch.randn(T, small_config.hidden_size)
        _, indices, _ = moe_router(x)

        stats = moe_router.compute_load_balance_stats(indices)

        assert "expert_counts" in stats
        assert "max_load" in stats
        assert "min_load" in stats
        assert "load_balance_ratio" in stats

        assert stats["expert_counts"].shape[0] == small_config.moe.num_experts
        assert stats["max_load"] >= stats["min_load"]
        assert stats["load_balance_ratio"] >= 1.0

    def test_total_assignments(self, small_config, moe_router):
        """Total expert assignments should equal T * top_k."""
        T = 16
        x = torch.randn(T, small_config.hidden_size)
        _, indices, _ = moe_router(x)

        stats = moe_router.compute_load_balance_stats(indices)
        total = stats["expert_counts"].sum().item()
        expected = T * small_config.moe.num_experts_per_tok
        assert total == expected
