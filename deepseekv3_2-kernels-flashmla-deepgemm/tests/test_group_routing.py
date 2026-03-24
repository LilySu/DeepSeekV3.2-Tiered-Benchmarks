"""
Group-based routing tests (n_group=8, topk_group=4).

Verifies the group-restricted top-k routing mechanism specific to DeepSeek-V3:
  - Expert groups are correctly partitioned
  - Only topk_group groups are selected per token
  - Top-k experts are drawn from selected groups
  - Different tokens can select different group combinations

CPU-runnable.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config, MoEConfig
from moe_router import MoERouter


class TestGroupPartitioning:
    """Test that experts are correctly partitioned into groups."""

    def test_experts_per_group(self, small_config):
        """Each group should have num_experts / n_group experts."""
        router = MoERouter(small_config)
        expected = small_config.moe.num_experts // small_config.moe.n_group
        assert router.experts_per_group == expected

    def test_full_config_groups(self):
        """Full config: 256 experts / 8 groups = 32 per group."""
        config = DeepSeekV3Config()
        router = MoERouter(config)
        assert router.experts_per_group == 32
        assert router.n_group == 8
        assert router.topk_group == 4


class TestGroupSelection:
    """Test group-level selection."""

    def test_at_most_topk_group_selected(self, small_config):
        """Each token should use at most topk_group groups."""
        torch.manual_seed(42)
        router = MoERouter(small_config)
        T = 32
        x = torch.randn(T, small_config.hidden_size)

        _, indices, _ = router(x)

        epg = router.experts_per_group
        topk_group = small_config.moe.topk_group

        for t in range(T):
            groups = set()
            for idx in indices[t]:
                groups.add(idx.item() // epg)
            assert len(groups) <= topk_group, (
                f"Token {t}: used {len(groups)} groups, limit is {topk_group}"
            )

    def test_different_tokens_different_groups(self, small_config):
        """Different tokens should be able to select different groups."""
        torch.manual_seed(42)
        router = MoERouter(small_config)
        T = 64
        x = torch.randn(T, small_config.hidden_size)

        _, indices, _ = router(x)

        epg = router.experts_per_group
        all_group_sets = set()
        for t in range(T):
            groups = frozenset(idx.item() // epg for idx in indices[t])
            all_group_sets.add(groups)

        # With enough tokens, multiple group combinations should appear
        assert len(all_group_sets) > 1

    def test_group_scores_determine_selection(self, small_config):
        """Groups with higher aggregate scores should be selected."""
        torch.manual_seed(42)
        router = MoERouter(small_config)

        # Create input that strongly activates one group
        x = torch.zeros(1, small_config.hidden_size)
        # Manually set gate weights to bias one group
        with torch.no_grad():
            router.gate.weight.zero_()
            # Make experts 0..epg-1 (group 0) have high scores
            epg = router.experts_per_group
            router.gate.weight[:epg, 0] = 10.0

        x[0, 0] = 1.0
        _, indices, _ = router(x)

        # All selected experts should be from group 0
        for idx in indices[0]:
            assert idx.item() < epg, f"Expert {idx.item()} not in group 0"


class TestGroupRoutingEdgeCases:
    """Test edge cases in group routing."""

    def test_single_token(self, small_config):
        """Routing should work with a single token."""
        router = MoERouter(small_config)
        x = torch.randn(1, small_config.hidden_size)
        weights, indices, logits = router(x)

        assert weights.shape[0] == 1
        assert indices.shape[0] == 1

    def test_all_groups_equal(self, small_config):
        """When all groups have equal scores, selection should still work."""
        torch.manual_seed(42)
        router = MoERouter(small_config)

        # Uniform input -> somewhat uniform scores
        x = torch.zeros(4, small_config.hidden_size)
        weights, indices, _ = router(x)

        # Should still select the correct number of experts
        assert indices.shape[1] == small_config.moe.num_experts_per_tok

    def test_custom_group_config(self):
        """Test with different group configurations."""
        config = DeepSeekV3Config(
            hidden_size=128,
            vocab_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            moe=MoEConfig(
                num_experts=32,
                num_experts_per_tok=4,
                n_group=4,
                topk_group=2,
                expert_intermediate_size=64,
                shared_expert_intermediate_size=64,
                first_k_dense_replace=0,
            ),
        )
        router = MoERouter(config)
        T = 8
        x = torch.randn(T, config.hidden_size)
        weights, indices, _ = router(x)

        epg = 32 // 4  # 8
        for t in range(T):
            groups = set(idx.item() // epg for idx in indices[t])
            assert len(groups) <= 2  # topk_group = 2


class TestGroupRoutingGradients:
    """Test gradient flow through group routing."""

    def test_router_gate_gradient(self, small_config):
        """Gate weight should receive gradient through group routing."""
        router = MoERouter(small_config)
        router.train()

        x = torch.randn(8, small_config.hidden_size, requires_grad=True)
        weights, _, logits = router(x)
        loss = weights.sum()
        loss.backward()

        assert router.gate.weight.grad is not None
        assert x.grad is not None

    def test_gradient_through_sigmoid(self, small_config):
        """Sigmoid scoring should have smooth gradients."""
        router = MoERouter(small_config)
        x = torch.randn(4, small_config.hidden_size, requires_grad=True)
        _, _, logits = router(x)

        scores = torch.sigmoid(logits)
        loss = scores.sum()
        loss.backward()

        # Sigmoid gradients should be well-behaved
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().max() < 1e4
