#!/usr/bin/env python3
"""
Debug a single DeepSeek-V3 layer execution.

Runs a single Transformer layer (MLA + MoE FFN or MLA + Dense FFN)
with detailed output at each stage to verify correctness.

Prints intermediate tensor shapes, norms, and value ranges
at each sublayer to catch numerical issues early.

Usage:
    python -m benchmark.debug_single_layer --layer-type moe --batch 2 --seq 64
    python -m benchmark.debug_single_layer --layer-type dense --batch 1 --seq 128
"""

from __future__ import annotations

import argparse
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG


def print_tensor_info(name: str, t: "torch.Tensor"):
    """Print diagnostic info about a tensor."""
    if t is None:
        print(f"    {name:<30s}: None")
        return

    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    norm = t.float().norm().item()
    mean = t.float().mean().item()
    std_val = t.float().std().item()
    min_val = t.min().item()
    max_val = t.max().item()

    warn = ""
    if has_nan:
        warn += " [NaN!]"
    if has_inf:
        warn += " [Inf!]"
    if norm > 1e6:
        warn += " [large norm]"

    print(f"    {name:<30s}: shape={list(t.shape):<20s} "
          f"norm={norm:10.4f} mean={mean:+8.4f} std={std_val:8.4f} "
          f"min={min_val:+8.4f} max={max_val:+8.4f}{warn}")


def debug_mla_layer(batch_size: int, seq_len: int, device: str, dtype: torch.dtype):
    """Debug MLA layer step by step."""
    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    d_c = cfg["kv_lora_rank"]
    n_h = min(cfg["num_heads"], 16)  # reduced for memory
    d_nope = cfg["qk_nope_head_dim"]
    d_rope = cfg["qk_rope_head_dim"]
    d_v = cfg["v_head_dim"]

    print(f"\n  --- MLA Layer Debug (n_h={n_h} of {cfg['num_heads']}) ---")

    x = torch.randn(batch_size, seq_len, H, dtype=dtype, device=device) * 0.02
    print_tensor_info("input", x)

    # RMSNorm
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + 1e-6)
    print_tensor_info("after_rmsnorm", x_norm)

    # KV compression
    W_dkv = torch.randn(H, d_c, dtype=dtype, device=device) / (H ** 0.5)
    c_kv = x_norm.reshape(-1, H) @ W_dkv
    c_kv = c_kv.reshape(batch_size, seq_len, d_c)
    print_tensor_info("c_kv (latent)", c_kv)

    # K nope up-projection
    W_uk = torch.randn(d_c, n_h * d_nope, dtype=dtype, device=device) / (d_c ** 0.5)
    k_nope = c_kv.reshape(-1, d_c) @ W_uk
    k_nope = k_nope.reshape(batch_size, seq_len, n_h, d_nope)
    print_tensor_info("k_nope", k_nope)

    # V up-projection
    W_uv = torch.randn(d_c, n_h * d_v, dtype=dtype, device=device) / (d_c ** 0.5)
    v = c_kv.reshape(-1, d_c) @ W_uv
    v = v.reshape(batch_size, seq_len, n_h, d_v)
    print_tensor_info("v", v)

    # Q projection
    W_q = torch.randn(H, n_h * (d_nope + d_rope), dtype=dtype, device=device) / (H ** 0.5)
    q = x_norm.reshape(-1, H) @ W_q
    q = q.reshape(batch_size, seq_len, n_h, d_nope + d_rope)
    print_tensor_info("q", q)

    # RoPE key (simplified)
    W_rope = torch.randn(H, d_rope, dtype=dtype, device=device) / (H ** 0.5)
    k_rope = x_norm.reshape(-1, H) @ W_rope
    k_rope = k_rope.reshape(batch_size, seq_len, 1, d_rope).expand(-1, -1, n_h, -1)
    print_tensor_info("k_rope", k_rope)

    # Concatenate K
    k = torch.cat([k_nope, k_rope], dim=-1)
    print_tensor_info("k_full", k)

    # Attention: (B, n_h, S, d) @ (B, n_h, d, S) -> (B, n_h, S, S)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    scale = (d_nope + d_rope) ** -0.5
    attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    print_tensor_info("attn_scores", attn_scores)

    # Causal mask
    if seq_len > 1:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

    attn_probs = F.softmax(attn_scores, dim=-1)
    print_tensor_info("attn_probs", attn_probs)

    attn_out = torch.matmul(attn_probs, v_t)
    print_tensor_info("attn_output", attn_out)

    # Output projection
    attn_out_flat = attn_out.transpose(1, 2).reshape(-1, n_h * d_v)
    W_o = torch.randn(n_h * d_v, H, dtype=dtype, device=device) / ((n_h * d_v) ** 0.5)
    output = attn_out_flat @ W_o
    output = output.reshape(batch_size, seq_len, H)
    print_tensor_info("output_proj", output)

    # Residual
    final = x + output
    print_tensor_info("residual_add", final)


def debug_moe_ffn(batch_size: int, seq_len: int, device: str, dtype: torch.dtype):
    """Debug MoE FFN layer step by step."""
    cfg = DEEPSEEK_V3_CONFIG
    H = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    N_exp = min(cfg["n_routed_experts"], 32)
    K = min(cfg["num_experts_per_tok"], 4)
    n_group = min(cfg["n_group"], N_exp // 4)

    print(f"\n  --- MoE FFN Debug (N_exp={N_exp}, K={K}, groups={n_group}) ---")

    x = torch.randn(batch_size, seq_len, H, dtype=dtype, device=device) * 0.02
    print_tensor_info("input", x)

    x_flat = x.reshape(-1, H)
    tokens = x_flat.shape[0]

    # Gating
    W_gate = torch.randn(H, N_exp, dtype=dtype, device=device) / (H ** 0.5)
    logits = x_flat @ W_gate
    print_tensor_info("gate_logits", logits)

    probs = F.softmax(logits.float(), dim=-1)
    print_tensor_info("gate_probs", probs)

    # Grouped routing
    if n_group > 1:
        epg = N_exp // n_group
        grouped = logits.float().reshape(tokens, n_group, epg)
        group_scores = grouped.max(dim=-1).values
        topk_groups = group_scores.topk(min(4, n_group), dim=-1).indices
        print_tensor_info("group_scores", group_scores)
        print(f"    topk_groups sample: {topk_groups[0].tolist()}")

    topk_vals, topk_idx = probs.topk(K, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    print_tensor_info("topk_vals", topk_vals)
    print(f"    topk_idx sample: {topk_idx[0].tolist()}")

    # Expert FFN (shared weights for debug)
    W_g = torch.randn(H, I, dtype=dtype, device=device) / (H ** 0.5)
    W_u = torch.randn(H, I, dtype=dtype, device=device) / (H ** 0.5)
    W_d = torch.randn(I, H, dtype=dtype, device=device) / (I ** 0.5)

    gate_out = F.silu(x_flat @ W_g)
    print_tensor_info("silu(gate)", gate_out)

    up_out = x_flat @ W_u
    print_tensor_info("up_proj", up_out)

    hidden = gate_out * up_out
    print_tensor_info("gate*up", hidden)

    expert_out = hidden @ W_d
    print_tensor_info("expert_output", expert_out)

    # Weighted combine
    weighted = expert_out * topk_vals.sum(dim=-1, keepdim=True).to(dtype)
    output = weighted.reshape(batch_size, seq_len, H)
    print_tensor_info("weighted_output", output)

    final = x + output
    print_tensor_info("residual_add", final)


def main():
    parser = argparse.ArgumentParser(description="Debug single DeepSeek-V3 layer")
    parser.add_argument("--layer-type", type=str, default="moe", choices=["moe", "dense"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=64)
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch is required")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("=" * 70)
    print(f"  DeepSeek-V3 Single Layer Debug")
    print(f"  layer_type={args.layer_type}, batch={args.batch}, seq={args.seq}")
    print(f"  device={device}, dtype={dtype}")
    print("=" * 70)

    with torch.no_grad():
        debug_mla_layer(args.batch, args.seq, device, dtype)

        if args.layer_type == "moe":
            debug_moe_ffn(args.batch, args.seq, device, dtype)
        else:
            # Dense FFN debug (simpler)
            H = DEEPSEEK_V3_CONFIG["hidden_size"]
            I = 4 * H  # Dense FFN uses 4x
            print(f"\n  --- Dense FFN Debug (I={I}) ---")
            x = torch.randn(args.batch, args.seq, H, dtype=dtype, device=device) * 0.02
            W_g = torch.randn(H, I, dtype=dtype, device=device) / (H ** 0.5)
            W_u = torch.randn(H, I, dtype=dtype, device=device) / (H ** 0.5)
            W_d = torch.randn(I, H, dtype=dtype, device=device) / (I ** 0.5)
            h = x.reshape(-1, H)
            out = (F.silu(h @ W_g) * (h @ W_u)) @ W_d
            print_tensor_info("dense_ffn_output", out.reshape(args.batch, args.seq, H))

    print("\n  Debug complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
