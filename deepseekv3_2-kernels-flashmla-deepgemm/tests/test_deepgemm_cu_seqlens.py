"""
Tests for cu_seqlens (cumulative sequence lengths) format used by DeepGEMM
grouped GEMM and FlashMLA paged attention.

cu_seqlens is the standard format for variable-length batched operations:
  cu_seqlens = [0, s0, s0+s1, s0+s1+s2, ...]  (length = batch_size + 1)

DeepSeek-V3 uses cu_seqlens for:
  1. FlashMLA paged attention (qo_indptr, kv_indptr)
  2. DeepGEMM grouped GEMM (m_sizes derived from cu_seqlens)
  3. MoE expert dispatch (tokens per expert)

Reference: arXiv 2412.19437
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# cu_seqlens utility functions under test
# ---------------------------------------------------------------------------

def build_cu_seqlens(seq_lens: torch.Tensor) -> torch.Tensor:
    """Build cu_seqlens from a list of sequence lengths."""
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=seq_lens.device)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    return cu


def cu_seqlens_to_sizes(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Convert cu_seqlens back to individual sizes."""
    return cu_seqlens[1:] - cu_seqlens[:-1]


def build_moe_cu_seqlens(
    expert_indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Build cu_seqlens for MoE expert dispatch.

    Given expert_indices (n_tokens, top_k), build cumulative token counts
    per expert for use with DeepGEMM grouped GEMM.
    """
    expert_counts = torch.zeros(n_experts, dtype=torch.int32, device=expert_indices.device)
    for e in range(n_experts):
        expert_counts[e] = (expert_indices == e).sum().item()
    cu = torch.zeros(n_experts + 1, dtype=torch.int32, device=expert_indices.device)
    cu[1:] = torch.cumsum(expert_counts, dim=0)
    return cu


def validate_cu_seqlens(cu_seqlens: torch.Tensor, total_tokens: int) -> bool:
    """Validate cu_seqlens format."""
    if cu_seqlens[0] != 0:
        return False
    if cu_seqlens[-1] != total_tokens:
        return False
    if (cu_seqlens[1:] < cu_seqlens[:-1]).any():
        return False  # Must be monotonically non-decreasing
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCuSeqlensBasic:
    """Basic cu_seqlens construction and validation."""

    def test_build_from_uniform_lengths(self):
        """Uniform sequence lengths produce evenly spaced cu_seqlens."""
        seq_lens = torch.tensor([128, 128, 128, 128], dtype=torch.int32)
        cu = build_cu_seqlens(seq_lens)
        assert cu.tolist() == [0, 128, 256, 384, 512]
        assert cu.dtype == torch.int32

    def test_build_from_variable_lengths(self):
        """Variable sequence lengths produce correct cumulative sums."""
        seq_lens = torch.tensor([10, 50, 1, 200], dtype=torch.int32)
        cu = build_cu_seqlens(seq_lens)
        assert cu.tolist() == [0, 10, 60, 61, 261]

    def test_roundtrip_cu_seqlens_to_sizes(self):
        """cu_seqlens -> sizes -> cu_seqlens should be identity."""
        seq_lens = torch.tensor([32, 64, 128, 16], dtype=torch.int32)
        cu = build_cu_seqlens(seq_lens)
        recovered = cu_seqlens_to_sizes(cu)
        assert torch.equal(recovered, seq_lens)

    def test_empty_sequences_allowed(self):
        """Some sequences can have zero length (empty slots in batch)."""
        seq_lens = torch.tensor([0, 128, 0, 64], dtype=torch.int32)
        cu = build_cu_seqlens(seq_lens)
        assert cu.tolist() == [0, 0, 128, 128, 192]
        assert validate_cu_seqlens(cu, 192)

    def test_single_sequence(self):
        """Single sequence in batch."""
        seq_lens = torch.tensor([4096], dtype=torch.int32)
        cu = build_cu_seqlens(seq_lens)
        assert cu.tolist() == [0, 4096]

    def test_validation_correct(self):
        """Valid cu_seqlens passes validation."""
        cu = torch.tensor([0, 10, 20, 30], dtype=torch.int32)
        assert validate_cu_seqlens(cu, 30)

    def test_validation_wrong_start(self):
        """cu_seqlens not starting at 0 fails validation."""
        cu = torch.tensor([5, 15, 25], dtype=torch.int32)
        assert not validate_cu_seqlens(cu, 25)

    def test_validation_wrong_total(self):
        """cu_seqlens with wrong total fails validation."""
        cu = torch.tensor([0, 10, 20], dtype=torch.int32)
        assert not validate_cu_seqlens(cu, 100)

    def test_validation_non_monotonic(self):
        """Non-monotonic cu_seqlens fails validation."""
        cu = torch.tensor([0, 20, 10, 30], dtype=torch.int32)
        assert not validate_cu_seqlens(cu, 30)


class TestMoECuSeqlens:
    """cu_seqlens for MoE expert dispatch (DeepSeek-V3 specific)."""

    def test_moe_dispatch_uniform(self):
        """With random routing, all experts should get some tokens."""
        torch.manual_seed(42)
        n_tokens, n_experts, top_k = 1024, 16, 4
        expert_indices = torch.randint(0, n_experts, (n_tokens, top_k))
        cu = build_moe_cu_seqlens(expert_indices, n_experts)

        assert cu[0] == 0
        assert cu[-1] == n_tokens * top_k
        sizes = cu_seqlens_to_sizes(cu)
        assert (sizes >= 0).all()
        # With 1024 * 4 = 4096 assignments across 16 experts, each should get ~256
        assert sizes.sum() == n_tokens * top_k

    def test_moe_dispatch_deepseekv3_dims(self):
        """Test with DeepSeek-V3 dimensions: 256 experts, top-8."""
        torch.manual_seed(42)
        n_tokens = 2048
        n_experts = 256
        top_k = 8

        expert_indices = torch.randint(0, n_experts, (n_tokens, top_k))
        cu = build_moe_cu_seqlens(expert_indices, n_experts)

        assert cu.shape[0] == n_experts + 1
        assert cu[0] == 0
        assert cu[-1] == n_tokens * top_k
        assert validate_cu_seqlens(cu, n_tokens * top_k)

    def test_moe_dispatch_grouped_routing(self):
        """Test with DeepSeek-V3 grouped routing: 8 groups, top-4 groups, top-2 per group."""
        torch.manual_seed(42)
        n_tokens = 512
        n_experts = 256
        n_group = 8
        topk_group = 4
        experts_per_group = n_experts // n_group  # 32
        top_k = 8  # topk_group * 2

        # Simulate grouped routing
        all_indices = []
        for t in range(n_tokens):
            # Select top-4 groups
            group_scores = torch.randn(n_group)
            _, top_groups = group_scores.topk(topk_group)

            # Select top-2 experts per selected group
            token_experts = []
            for g in top_groups:
                expert_offset = g.item() * experts_per_group
                expert_scores = torch.randn(experts_per_group)
                _, local_top = expert_scores.topk(2)
                token_experts.extend([expert_offset + local_top[0].item(),
                                      expert_offset + local_top[1].item()])
            all_indices.append(token_experts[:top_k])

        expert_indices = torch.tensor(all_indices, dtype=torch.long)
        cu = build_moe_cu_seqlens(expert_indices, n_experts)

        # Verify structure
        assert cu.shape[0] == n_experts + 1
        assert cu[-1] == n_tokens * top_k
        sizes = cu_seqlens_to_sizes(cu)

        # Only experts in selected groups should have tokens
        active_experts = (sizes > 0).sum().item()
        # With top-4 of 8 groups, ~128 out of 256 experts should be active
        assert active_experts > 50, f"Too few active experts: {active_experts}"

    def test_moe_cu_seqlens_empty_experts(self):
        """Some experts may receive zero tokens (empty segments in cu_seqlens)."""
        n_tokens = 32
        n_experts = 256
        top_k = 8
        # Route all tokens to first 8 experts only
        expert_indices = torch.arange(top_k).unsqueeze(0).expand(n_tokens, -1)
        cu = build_moe_cu_seqlens(expert_indices, n_experts)

        sizes = cu_seqlens_to_sizes(cu)
        # First 8 experts should have tokens, rest should be 0
        assert (sizes[:8] > 0).all()
        assert (sizes[8:] == 0).all()
        assert validate_cu_seqlens(cu, n_tokens * top_k)


class TestCuSeqlensGPU:
    """GPU-specific cu_seqlens tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cu_seqlens_on_gpu(self):
        """cu_seqlens should work correctly on CUDA tensors."""
        seq_lens = torch.tensor([128, 256, 64], dtype=torch.int32, device="cuda")
        cu = build_cu_seqlens(seq_lens)
        assert cu.device.type == "cuda"
        assert cu.tolist() == [0, 128, 384, 448]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_moe_cu_seqlens_on_gpu(self):
        """MoE cu_seqlens should work on GPU."""
        torch.manual_seed(42)
        expert_indices = torch.randint(0, 256, (1024, 8), device="cuda")
        cu = build_moe_cu_seqlens(expert_indices, 256)
        assert cu.device.type == "cuda"
        assert cu[-1] == 1024 * 8

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cu_seqlens_dtype_int32(self):
        """cu_seqlens must be int32 for FlashMLA and DeepGEMM compatibility."""
        seq_lens = torch.tensor([100, 200], dtype=torch.int32, device="cuda")
        cu = build_cu_seqlens(seq_lens)
        assert cu.dtype == torch.int32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_large_batch_cu_seqlens(self):
        """Large batch with many sequences."""
        batch_size = 256
        seq_lens = torch.randint(1, 1024, (batch_size,), dtype=torch.int32, device="cuda")
        cu = build_cu_seqlens(seq_lens)
        assert cu.shape[0] == batch_size + 1
        assert cu[0] == 0
        assert cu[-1] == seq_lens.sum().item()
        assert validate_cu_seqlens(cu, seq_lens.sum().item())
