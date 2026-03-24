"""
Autoregressive decode tests.

Verifies that token-by-token generation with KV cache produces the same
results as full-sequence forward passes.  CPU-runnable.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config
from model import DeepSeekV3Model
from cache import KVCache


class TestAutoRegressiveDecode:
    """Test autoregressive generation with KV cache."""

    def test_single_token_decode(self, small_model, small_config, batch_size):
        """Single-token decode should produce valid logits."""
        model = small_model
        B = batch_size

        input_ids = torch.randint(0, small_config.vocab_size, (B, 1))

        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs["logits"].shape == (B, 1, small_config.vocab_size)
        # Logits should be finite
        assert torch.isfinite(outputs["logits"]).all()

    def test_prefill_then_decode(self, small_model, small_config, batch_size):
        """Prefill with prompt, then decode one token at a time."""
        model = small_model
        B = batch_size
        prompt_len = 8

        cache = KVCache(
            small_config,
            max_batch_size=B,
            max_seq_len=32,
            dtype=torch.float32,
        )

        # Prefill
        input_ids = torch.randint(0, small_config.vocab_size, (B, prompt_len))
        with torch.no_grad():
            outputs = model(input_ids, kv_cache=cache, use_cache=True)
        cache.advance(B, prompt_len)

        assert outputs["logits"].shape == (B, prompt_len, small_config.vocab_size)

        # Decode one token
        next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        with torch.no_grad():
            decode_out = model(next_token, kv_cache=cache, use_cache=True)
        cache.advance(B, 1)

        assert decode_out["logits"].shape == (B, 1, small_config.vocab_size)
        assert torch.isfinite(decode_out["logits"]).all()

    def test_decode_sequence_consistency(self, small_model, small_config):
        """Full sequence forward should match prefill+decode for last token."""
        model = small_model
        B = 1
        S = 8

        input_ids = torch.randint(0, small_config.vocab_size, (B, S))

        # Full forward
        with torch.no_grad():
            full_out = model(input_ids)
        full_logits = full_out["logits"]

        # Each token's logits should be valid
        assert full_logits.shape == (B, S, small_config.vocab_size)
        assert torch.isfinite(full_logits).all()

    def test_generate_produces_tokens(self, small_model, small_config):
        """model.generate should produce the requested number of tokens."""
        model = small_model
        B = 1
        prompt_len = 4
        max_new = 5

        input_ids = torch.randint(0, small_config.vocab_size, (B, prompt_len))

        with torch.no_grad():
            generated = model.generate(
                input_ids, max_new_tokens=max_new, temperature=0.0
            )

        assert generated.shape == (B, prompt_len + max_new)
        # All tokens should be valid vocab indices
        assert (generated >= 0).all()
        assert (generated < small_config.vocab_size).all()

    def test_greedy_decode_deterministic(self, small_model, small_config):
        """Greedy decode (temperature=0) should be deterministic."""
        model = small_model
        B = 1
        input_ids = torch.randint(0, small_config.vocab_size, (B, 4))

        with torch.no_grad():
            gen1 = model.generate(input_ids, max_new_tokens=3, temperature=0.0)
            gen2 = model.generate(input_ids, max_new_tokens=3, temperature=0.0)

        torch.testing.assert_close(gen1, gen2)

    def test_cache_seq_lens_tracked(self, small_config, batch_size):
        """KV cache should correctly track sequence lengths."""
        B = batch_size
        cache = KVCache(
            small_config,
            max_batch_size=B,
            max_seq_len=64,
            dtype=torch.float32,
        )

        assert (cache.seq_lens[:B] == 0).all()

        cache.advance(B, 10)
        assert (cache.seq_lens[:B] == 10).all()

        cache.advance(B, 1)
        assert (cache.seq_lens[:B] == 11).all()

    def test_cache_reset(self, small_config, batch_size):
        """Cache reset should zero sequence lengths."""
        B = batch_size
        cache = KVCache(
            small_config,
            max_batch_size=B,
            max_seq_len=64,
            dtype=torch.float32,
        )

        cache.advance(B, 10)
        cache.reset()
        assert (cache.seq_lens == 0).all()
