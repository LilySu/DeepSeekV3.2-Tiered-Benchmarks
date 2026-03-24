"""
H100 MoE sparsity pattern tests (no DSA).

Tests MoE-related sparsity patterns on H100:
  - Expert activation sparsity (only 8/256 experts active per token)
  - Group selection patterns
  - Load balance across experts
  - Sparsity-aware memory access patterns
  - Token sorting efficiency for grouped GEMM

Note: DeepSeek-V3 does NOT use Dynamic Sparse Attention (DSA).
All sparsity here refers to MoE expert routing sparsity.

Requires: H100 GPU.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config, DEEPSEEK_V3_CONFIG, MoEConfig
from moe_router import MoERouter

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)


@requires_hopper
class TestH100ExpertSparsity:
    """Test MoE expert activation sparsity on H100."""

    def test_expert_activation_rate(self):
        """Only top_k/num_experts of experts should be active per token."""
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")
        router = MoERouter(config).to(device).to(torch.bfloat16).eval()

        T = 1024
        x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            _, indices, _ = router(x)

        # Each token activates exactly top_k experts
        assert indices.shape == (T, config.moe.num_experts_per_tok)

        # Activation rate = top_k / num_experts = 8/256 = 3.125%
        expected_rate = config.moe.num_experts_per_tok / config.moe.num_experts
        print(f"\nExpected activation rate: {expected_rate*100:.1f}%")

    def test_expert_load_distribution(self):
        """Expert load should be reasonably balanced with group routing."""
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")
        router = MoERouter(config).to(device).to(torch.bfloat16).eval()

        T = 4096
        x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            _, indices, _ = router(x)

        stats = router.compute_load_balance_stats(indices)
        ratio = stats["load_balance_ratio"]

        print(f"\nLoad balance ratio: {ratio:.2f} (1.0 = perfect)")
        print(f"Max load: {stats['max_load']}, Min load: {stats['min_load']}")

        # With sigmoid routing and group restriction, balance should be reasonable
        assert ratio < 10.0  # not catastrophically imbalanced

    def test_group_selection_distribution(self):
        """Group selection should be distributed across all groups."""
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")
        router = MoERouter(config).to(device).to(torch.bfloat16).eval()

        T = 2048
        x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            _, indices, _ = router(x)

        epg = config.moe.num_experts // config.moe.n_group
        group_counts = torch.zeros(config.moe.n_group, device=device)

        for t in range(T):
            groups_used = set()
            for idx in indices[t]:
                groups_used.add(idx.item() // epg)
            for g in groups_used:
                group_counts[g] += 1

        print(f"\nGroup selection counts: {group_counts.tolist()}")
        # All groups should be selected at least sometimes
        assert (group_counts > 0).all(), "Some groups never selected"


@requires_hopper
class TestH100TokenSorting:
    """Test token sorting efficiency for grouped GEMM dispatch."""

    def test_sort_by_expert_speed(self):
        """Sorting tokens by expert should be fast on GPU."""
        T = 8192
        K = 8  # top_k
        E = 256
        device = torch.device("cuda")

        indices = torch.randint(0, E, (T, K), device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        flat = indices.reshape(-1)
        flat.argsort()
        torch.cuda.synchronize()

        start.record()
        for _ in range(100):
            flat = indices.reshape(-1)
            sorted_order = flat.argsort()
            group_sizes = torch.bincount(flat[sorted_order], minlength=E)
        end.record()

        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 100

        print(f"\nToken sorting (T={T}, K={K}, E={E}): {ms:.3f}ms")
        assert ms < 10  # should be fast on GPU

    def test_group_sizes_sum_correctly(self):
        """Group sizes should sum to T*K."""
        T, K, E = 1024, 8, 256
        device = torch.device("cuda")

        indices = torch.randint(0, E, (T, K), device=device)
        group_sizes = torch.bincount(indices.reshape(-1), minlength=E)

        assert group_sizes.sum().item() == T * K

    def test_sorted_tokens_contiguous(self):
        """Tokens sorted by expert should form contiguous groups."""
        T, K, E = 64, 4, 16
        device = torch.device("cuda")

        indices = torch.randint(0, E, (T, K), device=device)
        flat = indices.reshape(-1)
        sorted_order = flat.argsort(stable=True)
        sorted_experts = flat[sorted_order]

        # Check contiguity: each expert's tokens should be consecutive
        for e in range(E):
            mask = sorted_experts == e
            if mask.any():
                positions = mask.nonzero(as_tuple=True)[0]
                if len(positions) > 1:
                    diffs = positions[1:] - positions[:-1]
                    assert (diffs == 1).all(), f"Expert {e} tokens not contiguous"


@requires_hopper
class TestH100SparsityPatterns:
    """Test MoE sparsity patterns (structural, not DSA)."""

    def test_expert_overlap_between_tokens(self):
        """Tokens often share experts -- measure overlap."""
        config = DEEPSEEK_V3_CONFIG
        device = torch.device("cuda")
        router = MoERouter(config).to(device).to(torch.bfloat16).eval()

        T = 128
        x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            _, indices, _ = router(x)

        # Compute pairwise Jaccard similarity of expert sets
        overlaps = []
        for i in range(min(T, 32)):
            for j in range(i + 1, min(T, 32)):
                set_i = set(indices[i].tolist())
                set_j = set(indices[j].tolist())
                jaccard = len(set_i & set_j) / len(set_i | set_j)
                overlaps.append(jaccard)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        print(f"\nAverage expert set Jaccard similarity: {avg_overlap:.3f}")

    def test_no_dsa_patterns(self):
        """Verify DSA is not used (stubs return None)."""
        from dsa_indexer import DSAIndexer
        from dsa_sparse_attention import DSASparseAttention

        indexer = DSAIndexer()
        assert not indexer.is_applicable()
        assert indexer.get_sparse_mask() is None

        dsa = DSASparseAttention()
        assert not dsa.is_applicable()
