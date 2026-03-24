"""
Multi-Token Prediction (MTP) tests -- DeepSeek-V3 specific.

DeepSeek-V3 uses 1 MTP layer that predicts an additional future token
from a lightweight projection of the last hidden state.

Tests:
  - MTP head produces valid logits
  - MTP loss computation
  - MTP head shares embedding with main LM head
  - Gradient flow through MTP head

CPU-runnable.

Reference: arXiv 2412.19437, Section 3.5.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config, MTPConfig
from model import DeepSeekV3Model, MTPPredictionHead
from unsloth_cross_entropy_loss import MTPLoss


class TestMTPHead:
    """Test MTP prediction head."""

    def test_mtp_head_output_shape(self, small_config):
        """MTP head should produce [B, S, vocab_size] logits."""
        head = MTPPredictionHead(small_config)
        # Simulate sharing the lm_head
        head.lm_head = nn.Linear(small_config.hidden_size, small_config.vocab_size, bias=False)

        B, S, D = 2, 8, small_config.hidden_size
        h = torch.randn(B, S, D)
        logits = head(h)

        assert logits.shape == (B, S, small_config.vocab_size)

    def test_mtp_head_with_embedding_weight(self, small_config):
        """MTP head should work with passed embedding weight."""
        head = MTPPredictionHead(small_config)
        emb_weight = torch.randn(small_config.vocab_size, small_config.hidden_size)

        B, S, D = 1, 4, small_config.hidden_size
        h = torch.randn(B, S, D)
        logits = head(h, embedding_weight=emb_weight)

        assert logits.shape == (B, S, small_config.vocab_size)

    def test_mtp_head_requires_weight(self, small_config):
        """MTP head without weight or lm_head should raise error."""
        head = MTPPredictionHead(small_config)
        h = torch.randn(1, 4, small_config.hidden_size)

        with pytest.raises(ValueError, match="requires either"):
            head(h)

    def test_mtp_head_finite_output(self, small_config):
        """MTP head output should be finite."""
        head = MTPPredictionHead(small_config)
        head.lm_head = nn.Linear(small_config.hidden_size, small_config.vocab_size, bias=False)

        h = torch.randn(1, 4, small_config.hidden_size)
        logits = head(h)
        assert torch.isfinite(logits).all()


class TestMTPInModel:
    """Test MTP integration in full model."""

    def test_model_has_mtp_heads(self, small_model, small_config):
        """Model should have the configured number of MTP heads."""
        assert len(small_model.mtp_heads) == small_config.mtp.num_mtp_layers

    def test_mtp_shares_lm_head(self, small_model, small_config):
        """MTP heads should share the main LM head (when configured)."""
        if small_config.mtp.share_embedding:
            for head in small_model.mtp_heads:
                assert head.lm_head is small_model.lm_head

    def test_mtp_logits_shape(self, small_model, small_config):
        """MTP logits from model forward should have correct shape."""
        B, S = 1, 8
        input_ids = torch.randint(0, small_config.vocab_size, (B, S))

        with torch.no_grad():
            out = small_model(input_ids, output_mtp_logits=True)

        mtp_logits = out["mtp_logits"]
        assert len(mtp_logits) == small_config.mtp.num_mtp_layers
        for ml in mtp_logits:
            assert ml.shape == (B, S, small_config.vocab_size)

    def test_mtp_and_main_logits_differ(self, small_model, small_config):
        """MTP logits should differ from main logits (different projection)."""
        B, S = 1, 8
        input_ids = torch.randint(0, small_config.vocab_size, (B, S))

        with torch.no_grad():
            out = small_model(input_ids, output_mtp_logits=True)

        main_logits = out["logits"]
        mtp_logits = out["mtp_logits"][0]

        # They share lm_head but MTP has an extra proj + norm, so should differ
        assert not torch.allclose(main_logits, mtp_logits, atol=1e-6)


class TestMTPLoss:
    """Test MTP loss computation."""

    def test_mtp_loss_computes(self, small_config):
        """MTP loss should compute without errors."""
        loss_fn = MTPLoss(
            num_mtp_layers=1,
            mtp_weight=0.1,
            vocab_size=small_config.vocab_size,
        )

        B, S, V = 1, 8, small_config.vocab_size
        main_logits = torch.randn(B, S, V)
        mtp_logits = [torch.randn(B, S, V)]
        labels = torch.randint(0, V, (B, S))

        loss = loss_fn(main_logits, mtp_logits, labels)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_mtp_loss_greater_than_main_only(self, small_config):
        """MTP loss should be >= main loss (adds non-negative MTP term)."""
        from unsloth_cross_entropy_loss import UnslothCrossEntropyLoss

        B, S, V = 1, 8, small_config.vocab_size
        main_logits = torch.randn(B, S, V)
        mtp_logits = [torch.randn(B, S, V)]
        labels = torch.randint(0, V, (B, S))

        main_loss_fn = UnslothCrossEntropyLoss(vocab_size=V)
        main_loss = main_loss_fn(main_logits, labels)

        mtp_loss_fn = MTPLoss(
            num_mtp_layers=1, mtp_weight=0.1, vocab_size=V
        )
        total_loss = mtp_loss_fn(main_logits, mtp_logits, labels)

        # Total should be >= main (MTP adds a positive term)
        assert total_loss >= main_loss - 0.01  # small tolerance

    def test_mtp_loss_zero_weight(self, small_config):
        """With mtp_weight=0, total loss should equal main loss."""
        B, S, V = 1, 8, small_config.vocab_size
        main_logits = torch.randn(B, S, V)
        mtp_logits = [torch.randn(B, S, V)]
        labels = torch.randint(0, V, (B, S))

        loss_fn = MTPLoss(num_mtp_layers=1, mtp_weight=0.0, vocab_size=V)
        total_loss = loss_fn(main_logits, mtp_logits, labels)

        from unsloth_cross_entropy_loss import UnslothCrossEntropyLoss
        main_fn = UnslothCrossEntropyLoss(vocab_size=V)
        main_loss = main_fn(main_logits, labels)

        torch.testing.assert_close(total_loss, main_loss, atol=1e-5, rtol=1e-5)


class TestMTPGradients:
    """Test gradient flow through MTP head."""

    def test_mtp_gradient_to_model(self, small_model, small_config):
        """MTP loss should propagate gradients back through the model."""
        model = small_model
        model.train()

        B, S = 1, 8
        input_ids = torch.randint(0, small_config.vocab_size, (B, S))
        labels = torch.randint(0, small_config.vocab_size, (B, S))

        out = model(input_ids, output_mtp_logits=True)

        # Compute MTP loss
        loss_fn = MTPLoss(
            num_mtp_layers=1, mtp_weight=0.1,
            vocab_size=small_config.vocab_size,
        )
        loss = loss_fn(out["logits"], out["mtp_logits"], labels)
        loss.backward()

        # MTP projection should have gradients
        for head in model.mtp_heads:
            assert head.proj.weight.grad is not None

    def test_mtp_head_standalone_gradient(self, small_config):
        """Standalone MTP head should propagate gradients."""
        head = MTPPredictionHead(small_config)
        head.lm_head = nn.Linear(small_config.hidden_size, small_config.vocab_size, bias=False)
        head.train()

        h = torch.randn(1, 4, small_config.hidden_size, requires_grad=True)
        logits = head(h)
        loss = logits.sum()
        loss.backward()

        assert h.grad is not None
        assert head.proj.weight.grad is not None
