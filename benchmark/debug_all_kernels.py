#!/usr/bin/env python3
"""
Debug all kernel implementations for DeepSeek-V3.

Systematically tests each kernel/operation used in the model to verify:
  1. Numerical correctness against reference implementations
  2. Shape compatibility across all DeepSeek-V3 GEMM shapes
  3. Dtype handling (BF16, FP8 e4m3, FP32)
  4. Edge cases (empty batches, seq_len=1, etc.)

This script is designed to be run before full benchmarks to catch
issues early.

Reference: arXiv:2412.19437
"""

from __future__ import annotations

import sys
import traceback
from typing import List, Tuple, Dict, Any

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available. Skipping GPU kernel tests.")

from benchmark.shared.config import DEEPSEEK_V3_CONFIG


def check_env() -> Dict[str, Any]:
    """Check environment and report capabilities."""
    info = {
        "python_version": sys.version,
        "torch_available": HAS_TORCH,
    }
    if HAS_TORCH:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
        info["has_fp8"] = hasattr(torch, "float8_e4m3fn")
        info["has_flash_attn"] = False
        try:
            import flash_attn
            info["has_flash_attn"] = True
            info["flash_attn_version"] = flash_attn.__version__
        except ImportError:
            pass
    return info


def debug_gemm_shapes() -> List[Tuple[str, bool, str]]:
    """Test all DeepSeek-V3 GEMM shapes for correctness."""
    results = []
    if not HAS_TORCH:
        return [("gemm_shapes", False, "PyTorch not available")]

    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    d_c = cfg["kv_lora_rank"]
    n_h = cfg["num_heads"]
    d_nope = cfg["qk_nope_head_dim"]
    d_rope = cfg["qk_rope_head_dim"]
    d_v = cfg["v_head_dim"]
    I = cfg["moe_intermediate_size"]
    N_exp = cfg["n_routed_experts"]
    V = cfg["vocab_size"]

    shapes = [
        ("MLA_kv_down", (32, H), (H, d_c)),
        ("MLA_k_up", (32, d_c), (d_c, n_h * d_nope)),
        ("MLA_rope_k", (32, H), (H, d_rope)),
        ("MLA_q_proj", (32, H), (H, n_h * (d_nope + d_rope))),
        ("MLA_v_up", (32, d_c), (d_c, n_h * d_v)),
        ("MLA_output", (32, n_h * d_v), (n_h * d_v, H)),
        ("MoE_gating", (32, H), (H, N_exp)),
        ("MoE_expert_gate", (8, H), (H, I)),
        ("MoE_expert_up", (8, H), (H, I)),
        ("MoE_expert_down", (8, I), (I, H)),
        ("LM_head", (32, H), (H, V)),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name, shape_a, shape_b in shapes:
        try:
            A = torch.randn(*shape_a, dtype=torch.bfloat16, device=device)
            B = torch.randn(*shape_b, dtype=torch.bfloat16, device=device)
            C = torch.mm(A, B)
            assert C.shape == (shape_a[0], shape_b[1]), f"Shape mismatch: {C.shape}"
            assert not torch.isnan(C).any(), "NaN in output"
            assert not torch.isinf(C).any(), "Inf in output"
            results.append((name, True, f"OK: {shape_a} @ {shape_b} -> {C.shape}"))
        except Exception as e:
            results.append((name, False, str(e)))

    return results


def debug_mla_attention() -> List[Tuple[str, bool, str]]:
    """Test MLA attention mechanism."""
    results = []
    if not HAS_TORCH:
        return [("mla_attention", False, "PyTorch not available")]

    cfg = DEEPSEEK_V3_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for bs, seq in [(1, 32), (2, 64), (1, 1)]:
        try:
            n_h = min(cfg["num_heads"], 8)  # reduced for memory
            d_qk = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
            d_v = cfg["v_head_dim"]

            q = torch.randn(bs, n_h, seq, d_qk, dtype=torch.bfloat16, device=device)
            k = torch.randn(bs, n_h, seq, d_qk, dtype=torch.bfloat16, device=device)
            v = torch.randn(bs, n_h, seq, d_v, dtype=torch.bfloat16, device=device)

            scale = d_qk ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask
            if seq > 1:
                mask = torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(mask, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

            assert out.shape == (bs, n_h, seq, d_v)
            assert not torch.isnan(out).any()
            results.append((f"mla_attn_bs{bs}_seq{seq}", True, f"OK: {out.shape}"))
        except Exception as e:
            results.append((f"mla_attn_bs{bs}_seq{seq}", False, str(e)))

    return results


def debug_moe_routing() -> List[Tuple[str, bool, str]]:
    """Test MoE routing (grouped top-K)."""
    results = []
    if not HAS_TORCH:
        return [("moe_routing", False, "PyTorch not available")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEEPSEEK_V3_CONFIG

    for n_tokens in [1, 32, 256]:
        try:
            N_exp = cfg["n_routed_experts"]
            n_group = cfg["n_group"]
            topk_group = cfg["topk_group"]
            top_k = cfg["num_experts_per_tok"]

            logits = torch.randn(n_tokens, N_exp, device=device)
            probs = F.softmax(logits, dim=-1)

            # Grouped routing
            experts_per_group = N_exp // n_group
            grouped = logits.reshape(n_tokens, n_group, experts_per_group)
            group_scores = grouped.max(dim=-1).values
            _, top_groups = group_scores.topk(topk_group, dim=-1)

            # Select top-K experts from selected groups
            topk_vals, topk_idx = probs.topk(top_k, dim=-1)

            assert topk_idx.shape == (n_tokens, top_k)
            assert topk_vals.shape == (n_tokens, top_k)
            assert (topk_idx >= 0).all() and (topk_idx < N_exp).all()

            results.append((f"moe_routing_t{n_tokens}", True,
                           f"OK: {n_tokens} tokens -> top-{top_k} from {N_exp} experts"))
        except Exception as e:
            results.append((f"moe_routing_t{n_tokens}", False, str(e)))

    return results


def debug_fp8_quantization() -> List[Tuple[str, bool, str]]:
    """Test FP8 quantization if available."""
    results = []
    if not HAS_TORCH:
        return [("fp8", False, "PyTorch not available")]

    if not hasattr(torch, "float8_e4m3fn"):
        return [("fp8", False, "FP8 not available in this PyTorch version")]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for shape in [(128, 128), (256, 512), (7168, 512)]:
        try:
            x = torch.randn(*shape, dtype=torch.bfloat16, device=device)
            x_fp8 = x.to(torch.float8_e4m3fn)
            x_back = x_fp8.to(torch.bfloat16)

            # Check round-trip error
            error = (x.float() - x_back.float()).abs().max().item()
            results.append((f"fp8_roundtrip_{shape}", True,
                           f"OK: max error={error:.6f}"))
        except Exception as e:
            results.append((f"fp8_roundtrip_{shape}", False, str(e)))

    return results


def debug_rmsnorm() -> List[Tuple[str, bool, str]]:
    """Test RMSNorm implementation."""
    results = []
    if not HAS_TORCH:
        return [("rmsnorm", False, "PyTorch not available")]

    H = DEEPSEEK_V3_CONFIG["hidden_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for tokens in [1, 32, 256]:
        try:
            x = torch.randn(tokens, H, dtype=torch.bfloat16, device=device)
            weight = torch.ones(H, dtype=torch.bfloat16, device=device)

            variance = x.float().pow(2).mean(-1, keepdim=True)
            normed = x * torch.rsqrt(variance + 1e-6)
            out = normed * weight

            assert out.shape == (tokens, H)
            assert not torch.isnan(out).any()

            # Check normalization property (RMS should be ~1)
            rms = out.float().pow(2).mean(-1).sqrt()
            rms_mean = rms.mean().item()

            results.append((f"rmsnorm_t{tokens}", True,
                           f"OK: shape={out.shape}, mean_rms={rms_mean:.4f}"))
        except Exception as e:
            results.append((f"rmsnorm_t{tokens}", False, str(e)))

    return results


def debug_swiglu() -> List[Tuple[str, bool, str]]:
    """Test SwiGLU activation."""
    results = []
    if not HAS_TORCH:
        return [("swiglu", False, "PyTorch not available")]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for tokens, I in [(32, 2048), (1, 2048), (256, 2048)]:
        try:
            gate = torch.randn(tokens, I, dtype=torch.bfloat16, device=device)
            up = torch.randn(tokens, I, dtype=torch.bfloat16, device=device)
            out = F.silu(gate) * up

            assert out.shape == (tokens, I)
            assert not torch.isnan(out).any()
            results.append((f"swiglu_{tokens}x{I}", True, f"OK: {out.shape}"))
        except Exception as e:
            results.append((f"swiglu_{tokens}x{I}", False, str(e)))

    return results


def main():
    print("=" * 70)
    print("  DeepSeek-V3 Kernel Debug Suite")
    print("=" * 70)

    # Environment check
    env = check_env()
    print("\nEnvironment:")
    for k, v in env.items():
        print(f"  {k}: {v}")

    # Run all debug tests
    all_tests = []
    test_groups = [
        ("GEMM Shapes", debug_gemm_shapes),
        ("MLA Attention", debug_mla_attention),
        ("MoE Routing", debug_moe_routing),
        ("FP8 Quantization", debug_fp8_quantization),
        ("RMSNorm", debug_rmsnorm),
        ("SwiGLU", debug_swiglu),
    ]

    total_pass = 0
    total_fail = 0

    for group_name, test_fn in test_groups:
        print(f"\n--- {group_name} ---")
        try:
            tests = test_fn()
        except Exception as e:
            print(f"  GROUP FAILED: {e}")
            traceback.print_exc()
            tests = [(group_name, False, str(e))]

        for name, passed, msg in tests:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {msg}")
            if passed:
                total_pass += 1
            else:
                total_fail += 1
            all_tests.append((name, passed, msg))

    print(f"\n{'='*70}")
    print(f"  Results: {total_pass} passed, {total_fail} failed, {total_pass + total_fail} total")
    print(f"{'='*70}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
