"""
FLOP calculations, MFU, and bandwidth metrics for DeepSeek-V3.

All formulas are derived from the DeepSeek-V3 architecture (arXiv: 2412.19437).
Key differences from standard Transformer FLOP counting:
  - MLA uses compressed KV (kv_lora_rank=512) instead of full multi-head KV.
  - MoE activates top-8 of 256 experts per token, plus 1 shared expert.
  - Grouped routing: 8 groups, top-4 groups selected first.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from benchmark.shared.config import DEEPSEEK_V3_CONFIG, H100_SPECS


# ---------------------------------------------------------------------------
# MLA (Multi-head Latent Attention) FLOPs
# ---------------------------------------------------------------------------

def compute_mla_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int = DEEPSEEK_V3_CONFIG["hidden_size"],
    num_heads: int = DEEPSEEK_V3_CONFIG["num_heads"],
    kv_lora_rank: int = DEEPSEEK_V3_CONFIG["kv_lora_rank"],
    qk_rope_head_dim: int = DEEPSEEK_V3_CONFIG["qk_rope_head_dim"],
    qk_nope_head_dim: int = DEEPSEEK_V3_CONFIG["qk_nope_head_dim"],
    v_head_dim: int = DEEPSEEK_V3_CONFIG["v_head_dim"],
    causal: bool = True,
) -> Dict[str, float]:
    """
    Compute FLOPs for one MLA layer.

    MLA compresses KV into a low-rank latent (kv_lora_rank=512).
    The Q projection can also be compressed (q_lora_rank=1536).

    Per-token FLOPs breakdown:
      1. Down-projection: x -> c_kv  (hidden_size -> kv_lora_rank)  : 2 * H * d_c
      2. Up-project K_nope from c_kv: (kv_lora_rank -> num_heads * qk_nope_head_dim) : 2 * d_c * n_h * d_nope
      3. RoPE K: c_kv -> k_rope (kv_lora_rank -> qk_rope_head_dim) : 2 * d_c * d_rope (shared across heads)
      4. Q projection: x -> q  (hidden_size -> num_heads * (qk_nope + qk_rope)) : 2 * H * n_h * (d_nope + d_rope)
      5. Attention scores (nope + rope parts combined): 2 * n_h * seq * (d_nope + d_rope)
      6. Attention over V: 2 * n_h * seq * v_head_dim
      7. Output projection: 2 * n_h * v_head_dim * H
    """
    tokens = batch_size * seq_len
    H = hidden_size
    n_h = num_heads
    d_c = kv_lora_rank
    d_rope = qk_rope_head_dim
    d_nope = qk_nope_head_dim
    d_v = v_head_dim

    # Average attention span (causal masking halves the effective seq)
    avg_attn_span = seq_len / 2 if causal else seq_len

    # Projection FLOPs (per token, multiply by tokens)
    kv_down_proj = 2 * H * d_c                          # compress to latent
    k_nope_up = 2 * d_c * n_h * d_nope                  # up-project K nope
    k_rope_proj = 2 * H * d_rope                         # shared RoPE key
    q_proj = 2 * H * n_h * (d_nope + d_rope)            # full Q projection
    out_proj = 2 * n_h * d_v * H                         # output projection

    proj_flops = tokens * (kv_down_proj + k_nope_up + k_rope_proj + q_proj + out_proj)

    # Attention FLOPs (QK^T + softmax + AV, per token)
    attn_qk = 2 * n_h * avg_attn_span * (d_nope + d_rope)
    attn_av = 2 * n_h * avg_attn_span * d_v

    attn_flops = tokens * (attn_qk + attn_av)

    total = proj_flops + attn_flops

    return {
        "total_flops": total,
        "proj_flops": proj_flops,
        "attn_flops": attn_flops,
        "flops_per_token": total / tokens if tokens > 0 else 0,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# MoE FLOPs
# ---------------------------------------------------------------------------

def compute_moe_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int = DEEPSEEK_V3_CONFIG["hidden_size"],
    n_routed_experts: int = DEEPSEEK_V3_CONFIG["n_routed_experts"],
    num_experts_per_tok: int = DEEPSEEK_V3_CONFIG["num_experts_per_tok"],
    moe_intermediate_size: int = DEEPSEEK_V3_CONFIG["moe_intermediate_size"],
    n_shared_experts: int = DEEPSEEK_V3_CONFIG.get("n_shared_experts", 1),
    n_group: int = DEEPSEEK_V3_CONFIG["n_group"],
    topk_group: int = DEEPSEEK_V3_CONFIG["topk_group"],
    include_gating: bool = True,
) -> Dict[str, float]:
    """
    Compute FLOPs for one MoE FFN layer (DeepSeek-V3 style).

    Each MoE layer has:
      - A gating network that routes tokens to experts (grouped routing).
      - 256 routed experts, top-8 activated per token.
      - 1 shared expert that processes all tokens.
      - Each expert is a SwiGLU FFN: gate_proj (H->I), up_proj (H->I), down_proj (I->H).

    Grouped routing: tokens are first assigned to top-4 of 8 groups, then
    top-2 experts within each selected group (4 groups * 2 experts = 8 total).
    """
    tokens = batch_size * seq_len
    H = hidden_size
    I = moe_intermediate_size
    K = num_experts_per_tok

    # SwiGLU FFN per expert: gate(2*H*I) + up(2*H*I) + element_mul(I) + down(2*I*H) = 6*H*I + I
    expert_ffn_flops_per_token = 6 * H * I + I

    # Routed experts: each token activates K experts
    routed_flops = tokens * K * expert_ffn_flops_per_token

    # Shared expert: all tokens go through the shared expert
    # Shared expert has larger intermediate (typically moe_intermediate_size * n_shared_experts * scale)
    shared_intermediate = moe_intermediate_size * 2  # shared expert is 2x wider in DeepSeek-V3
    shared_ffn_flops_per_token = 6 * H * shared_intermediate + shared_intermediate
    shared_flops = tokens * n_shared_experts * shared_ffn_flops_per_token

    # Gating FLOPs: linear projection to n_routed_experts + group selection overhead
    gating_flops = 0
    if include_gating:
        # Gate projection: H -> n_routed_experts
        gating_flops = tokens * 2 * H * n_routed_experts
        # Group scoring (lightweight, typically top-k ops, negligible vs matmuls)

    total = routed_flops + shared_flops + gating_flops

    return {
        "total_flops": total,
        "routed_expert_flops": routed_flops,
        "shared_expert_flops": shared_flops,
        "gating_flops": gating_flops,
        "flops_per_token": total / tokens if tokens > 0 else 0,
        "active_params_per_token": K * (3 * H * I + I) + n_shared_experts * (3 * H * shared_intermediate + shared_intermediate),
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Dense FFN FLOPs (for the 3 dense layers)
# ---------------------------------------------------------------------------

def compute_dense_ffn_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int = DEEPSEEK_V3_CONFIG["hidden_size"],
    intermediate_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute FLOPs for one dense SwiGLU FFN layer.

    The 3 dense layers in DeepSeek-V3 use standard SwiGLU with a larger
    intermediate size (typically 4x hidden for dense layers = 18432).
    """
    tokens = batch_size * seq_len
    H = hidden_size
    I = intermediate_size or (4 * H)  # 4x hidden for dense layers

    # SwiGLU: gate(2HI) + up(2HI) + mul(I) + down(2IH) = 6HI + I
    flops_per_token = 6 * H * I + I
    total = tokens * flops_per_token

    return {
        "total_flops": total,
        "flops_per_token": flops_per_token,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# MTP (Multi-Token Prediction) FLOPs
# ---------------------------------------------------------------------------

def compute_mtp_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int = DEEPSEEK_V3_CONFIG["hidden_size"],
    vocab_size: int = DEEPSEEK_V3_CONFIG["vocab_size"],
    mtp_layers: int = DEEPSEEK_V3_CONFIG["mtp_layers"],
) -> Dict[str, float]:
    """
    Compute FLOPs for the MTP (Multi-Token Prediction) module.

    MTP in DeepSeek-V3 uses 1 additional prediction layer that shares the
    main model's embedding and output head. The MTP module takes the last
    hidden state and predicts the next token beyond the standard next-token.

    FLOPs: projection (H->H) + output head (H->vocab)
    """
    tokens = batch_size * seq_len
    H = hidden_size
    V = vocab_size

    # MTP layer: linear projection + shared output head
    proj_flops = tokens * 2 * H * H  # hidden -> hidden projection
    head_flops = tokens * 2 * H * V  # output head projection

    total = mtp_layers * (proj_flops + head_flops)

    return {
        "total_flops": total,
        "proj_flops": proj_flops * mtp_layers,
        "head_flops": head_flops * mtp_layers,
        "flops_per_token": total / tokens if tokens > 0 else 0,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Full model FLOPs (forward pass)
# ---------------------------------------------------------------------------

def compute_full_model_flops(
    batch_size: int,
    seq_len: int,
) -> Dict[str, float]:
    """
    Compute total forward-pass FLOPs for DeepSeek-V3 671B.

    Layers:
      - 61 MLA layers (all layers use MLA)
      - 3 dense FFN layers (layers 0, 1, 2)
      - 58 MoE FFN layers (layers 3-60)
      - 1 MTP prediction head
      - Embedding + final LM head
    """
    cfg = DEEPSEEK_V3_CONFIG
    tokens = batch_size * seq_len

    # Attention across all 61 layers
    mla = compute_mla_flops(batch_size, seq_len)
    total_mla = mla["total_flops"] * cfg["num_layers"]

    # Dense FFN (3 layers)
    dense = compute_dense_ffn_flops(batch_size, seq_len)
    total_dense = dense["total_flops"] * cfg["num_dense_layers"]

    # MoE FFN (58 layers)
    moe = compute_moe_flops(batch_size, seq_len)
    total_moe = moe["total_flops"] * cfg["num_moe_layers"]

    # MTP
    mtp = compute_mtp_flops(batch_size, seq_len)
    total_mtp = mtp["total_flops"]

    # Embedding + LM head
    embed_flops = tokens * 2 * cfg["hidden_size"] * cfg["vocab_size"]  # both embed + lm_head

    total = total_mla + total_dense + total_moe + total_mtp + embed_flops

    return {
        "total_flops": total,
        "mla_flops": total_mla,
        "dense_ffn_flops": total_dense,
        "moe_ffn_flops": total_moe,
        "mtp_flops": total_mtp,
        "embed_flops": embed_flops,
        "flops_per_token": total / tokens if tokens > 0 else 0,
        "tokens": tokens,
        "breakdown_pct": {
            "mla": 100 * total_mla / total,
            "dense_ffn": 100 * total_dense / total,
            "moe_ffn": 100 * total_moe / total,
            "mtp": 100 * total_mtp / total,
            "embed": 100 * embed_flops / total,
        },
    }


# ---------------------------------------------------------------------------
# MFU (Model FLOPs Utilization)
# ---------------------------------------------------------------------------

def compute_mfu(
    achieved_flops: float,
    elapsed_seconds: float,
    dtype: str = "bfloat16",
    hardware: Dict[str, Any] = H100_SPECS,
) -> float:
    """
    Compute Model FLOPs Utilization (MFU).

    MFU = achieved_flops / (peak_flops * elapsed_time)

    Args:
        achieved_flops: Total FLOPs performed.
        elapsed_seconds: Wall-clock time in seconds.
        dtype: Data type used ("fp8", "bfloat16", "float16").
        hardware: Hardware specs dict.

    Returns:
        MFU as a percentage (0-100).
    """
    if elapsed_seconds <= 0:
        return 0.0

    dtype_to_key = {
        "fp8": "fp8_tflops",
        "float8_e4m3fn": "fp8_tflops",
        "bfloat16": "bf16_tflops",
        "float16": "fp16_tflops",
    }
    key = dtype_to_key.get(dtype, "bf16_tflops")
    peak_tflops = hardware[key]
    peak_flops_per_sec = peak_tflops * 1e12

    achieved_flops_per_sec = achieved_flops / elapsed_seconds
    mfu = 100.0 * achieved_flops_per_sec / peak_flops_per_sec

    return mfu


# ---------------------------------------------------------------------------
# Bandwidth utilization
# ---------------------------------------------------------------------------

def compute_bandwidth_utilization(
    bytes_transferred: float,
    elapsed_seconds: float,
    hardware: Dict[str, Any] = H100_SPECS,
) -> float:
    """
    Compute HBM bandwidth utilization.

    Args:
        bytes_transferred: Total bytes read/written from HBM.
        elapsed_seconds: Wall-clock time in seconds.
        hardware: Hardware specs dict.

    Returns:
        Bandwidth utilization as a percentage (0-100).
    """
    if elapsed_seconds <= 0:
        return 0.0

    peak_bw = hardware["hbm_bandwidth_tb_s"] * 1e12  # bytes/sec
    achieved_bw = bytes_transferred / elapsed_seconds
    return 100.0 * achieved_bw / peak_bw


# ---------------------------------------------------------------------------
# Arithmetic intensity
# ---------------------------------------------------------------------------

def compute_arithmetic_intensity(
    flops: float,
    bytes_transferred: float,
) -> float:
    """
    Compute arithmetic intensity (FLOPs / byte).

    Useful for roofline model analysis. DeepSeek-V3 MLA is compute-bound
    during prefill and memory-bound during decode due to KV cache access.
    """
    if bytes_transferred <= 0:
        return float("inf")
    return flops / bytes_transferred


def roofline_bound(
    flops: float,
    bytes_transferred: float,
    hardware: Dict[str, Any] = H100_SPECS,
    dtype: str = "bfloat16",
) -> str:
    """
    Determine if a kernel is compute-bound or memory-bound on the roofline.

    Returns "compute" or "memory".
    """
    ai = compute_arithmetic_intensity(flops, bytes_transferred)

    dtype_to_key = {
        "fp8": "fp8_tflops",
        "bfloat16": "bf16_tflops",
        "float16": "fp16_tflops",
    }
    key = dtype_to_key.get(dtype, "bf16_tflops")
    peak_flops = hardware[key] * 1e12
    peak_bw = hardware["hbm_bandwidth_tb_s"] * 1e12

    ridge_point = peak_flops / peak_bw  # FLOPs/byte

    return "compute" if ai >= ridge_point else "memory"
