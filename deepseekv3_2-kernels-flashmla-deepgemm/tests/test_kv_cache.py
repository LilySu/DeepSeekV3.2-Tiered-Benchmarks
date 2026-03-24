"""
KV cache operation tests.

Tests the compressed KV cache implementation for FlashMLA, including:
  - Allocation and deallocation
  - Sequential writes
  - Page table management
  - Memory tracking
  - Reset operations

CPU-runnable.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config
from cache import KVCache


class TestKVCacheAllocation:
    """Test cache allocation and memory management."""

    def test_lazy_allocation(self, small_config):
        """Cache should not allocate layer tensors until first use."""
        cache = KVCache(small_config, max_batch_size=2, max_seq_len=64)
        # Initially no layers allocated
        assert cache.memory_bytes == 0

    def test_allocate_single_layer(self, small_config):
        """Allocating a single layer should increase memory."""
        cache = KVCache(small_config, max_batch_size=2, max_seq_len=64)
        cache.allocate_layer(0)
        assert cache.memory_bytes > 0

    def test_allocate_all_layers(self, small_config):
        """Allocating all layers should match expected memory."""
        cache = KVCache(small_config, max_batch_size=1, max_seq_len=32)
        cache.allocate_all()

        expected_per_layer = (
            1 * 32 * (small_config.kv_lora_rank + small_config.qk_rope_head_dim) * 4
        )
        expected_total = expected_per_layer * small_config.num_hidden_layers
        assert cache.memory_bytes == expected_total

    def test_double_allocate_no_effect(self, small_config):
        """Allocating the same layer twice should be idempotent."""
        cache = KVCache(small_config, max_batch_size=1, max_seq_len=32)
        cache.allocate_layer(0)
        mem1 = cache.memory_bytes
        cache.allocate_layer(0)
        mem2 = cache.memory_bytes
        assert mem1 == mem2


class TestKVCacheUpdate:
    """Test cache write and read operations."""

    def test_update_writes_data(self, small_config):
        """Update should write compressed KV and rope data."""
        B = 2
        cache = KVCache(small_config, max_batch_size=B, max_seq_len=64, dtype=torch.float32)

        seq_len = 8
        kv = torch.randn(B, seq_len, small_config.kv_lora_rank)
        k_rope = torch.randn(B, seq_len, small_config.qk_rope_head_dim)

        kv_out, k_rope_out = cache.update(0, kv, k_rope)

        assert kv_out.shape == (B, seq_len, small_config.kv_lora_rank)
        assert k_rope_out.shape == (B, seq_len, small_config.qk_rope_head_dim)

        # Data should match what we wrote
        torch.testing.assert_close(kv_out, kv)
        torch.testing.assert_close(k_rope_out, k_rope)

    def test_sequential_updates(self, small_config):
        """Sequential updates should append data correctly."""
        B = 1
        cache = KVCache(small_config, max_batch_size=B, max_seq_len=64, dtype=torch.float32)

        # First write: 4 tokens
        kv1 = torch.ones(B, 4, small_config.kv_lora_rank)
        k_rope1 = torch.ones(B, 4, small_config.qk_rope_head_dim)
        cache.update(0, kv1, k_rope1)
        cache.advance(B, 4)

        # Second write: 2 tokens
        kv2 = torch.ones(B, 2, small_config.kv_lora_rank) * 2
        k_rope2 = torch.ones(B, 2, small_config.qk_rope_head_dim) * 2
        kv_out, k_rope_out = cache.update(0, kv2, k_rope2)

        # Should contain all 6 tokens
        assert kv_out.shape[1] == 6
        # First 4 tokens should be 1s, next 2 should be 2s
        torch.testing.assert_close(kv_out[0, :4], kv1[0])
        torch.testing.assert_close(kv_out[0, 4:6], kv2[0])


class TestKVCachePageTable:
    """Test paged KV cache layout."""

    def test_page_table_shape(self, small_config):
        """Page table should have correct shape."""
        B = 4
        max_seq = 128
        cache = KVCache(small_config, max_batch_size=B, max_seq_len=max_seq)

        num_pages = (max_seq + cache.page_size - 1) // cache.page_size
        assert cache.page_table.shape == (B, num_pages)

    def test_get_paged_kv(self, small_config):
        """get_paged_kv should return correct tensors."""
        B = 2
        cache = KVCache(small_config, max_batch_size=B, max_seq_len=64, dtype=torch.float32)
        cache.allocate_layer(0)

        kv_data, k_rope_data, page_table, seq_lens = cache.get_paged_kv(0, B)

        assert kv_data.shape[0] == B
        assert k_rope_data.shape[0] == B
        assert page_table.shape[0] == B
        assert seq_lens.shape[0] == B


class TestKVCacheFlat:
    """Test flat (non-paged) cache access."""

    def test_get_flat_kv(self, small_config):
        """get_flat_kv should return valid-length slices."""
        B = 1
        cache = KVCache(small_config, max_batch_size=B, max_seq_len=64, dtype=torch.float32)

        kv = torch.randn(B, 8, small_config.kv_lora_rank)
        k_rope = torch.randn(B, 8, small_config.qk_rope_head_dim)
        cache.update(0, kv, k_rope)
        cache.advance(B, 8)

        kv_flat, k_rope_flat = cache.get_flat_kv(0, B)
        assert kv_flat.shape == (B, 8, small_config.kv_lora_rank)
        assert k_rope_flat.shape == (B, 8, small_config.qk_rope_head_dim)


class TestKVCacheReset:
    """Test cache reset operations."""

    def test_reset_all(self, small_config):
        """Full reset should zero everything."""
        B = 2
        cache = KVCache(small_config, max_batch_size=B, max_seq_len=32, dtype=torch.float32)

        kv = torch.ones(B, 4, small_config.kv_lora_rank)
        k_rope = torch.ones(B, 4, small_config.qk_rope_head_dim)
        cache.update(0, kv, k_rope)
        cache.advance(B, 4)

        cache.reset()

        assert (cache.seq_lens == 0).all()

    def test_reset_specific_batch(self, small_config):
        """Resetting specific batch indices should only affect those."""
        B = 2
        cache = KVCache(small_config, max_batch_size=B, max_seq_len=32, dtype=torch.float32)
        cache.advance(B, 5)

        cache.reset(torch.tensor([0]))
        assert cache.seq_lens[0] == 0
        assert cache.seq_lens[1] == 5

    def test_repr(self, small_config):
        """Cache repr should be informative."""
        cache = KVCache(small_config, max_batch_size=2, max_seq_len=64)
        r = repr(cache)
        assert "KVCache" in r
        assert "layers=" in r
