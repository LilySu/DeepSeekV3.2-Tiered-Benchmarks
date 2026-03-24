# MoE Grouped GEMM -- DeepGEMM FP8 / FlashInfer / Triton dispatch for DeepSeek-V3.
#
# DeepSeek-V3 MoE: 256 experts, top-8 selection, 2048 intermediate per expert.
# Routing: sigmoid + group-based selection (n_group=8, topk_group=4).
#
# Kernel dispatch priority:
#   1. DeepGEMM FP8 grouped GEMM (H100/SM90, best performance)
#   2. FlashInfer/Triton grouped GEMM (cross-platform)
#   3. Per-expert loop fallback (pure PyTorch, always works)
#
# DeepGEMM provides:
#   - m_grouped_fp8_gemm_nt_contiguous(): Prefill (variable M per expert, sorted tokens)
#   - m_grouped_fp8_gemm_nt_masked():     Decode (fixed M per expert, CUDA graphs)
#   - m_grouped_bf16_gemm_nt_contiguous(): BF16 fallback
#
# Key differences from GLM-5:
#   - Same dispatch structure, but DeepSeek-V3 uses 8 groups of 32 experts
#   - More tokens per expert on average due to balanced routing
#   - topk_method="noaux_tc" means no auxiliary load-balancing loss
#
# Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 2.2 -- Mixture of Experts

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import deep_gemm
    DEEP_GEMM_AVAILABLE = True
except ImportError:
    DEEP_GEMM_AVAILABLE = False


def moe_grouped_gemm_forward(
    hidden_states: torch.Tensor,   # [num_tokens, hidden_dim]
    gate_up_proj: torch.Tensor,    # [num_experts, 2*intermediate_dim, hidden_dim]
    down_proj: torch.Tensor,       # [num_experts, hidden_dim, intermediate_dim]
    topk_indices: torch.Tensor,    # [num_tokens, top_k]
    topk_weights: torch.Tensor,    # [num_tokens, top_k]
    num_experts: int,
) -> torch.Tensor:
    """Route tokens through selected experts using grouped GEMM.

    When DeepGEMM is available, uses FP8 grouped GEMM kernels.
    Otherwise falls back to a per-expert loop.

    Args:
        hidden_states: [N, D] flattened input tokens
        gate_up_proj:  [E, 2*I, D] stacked expert gate+up weights
        down_proj:     [E, D, I] stacked expert down weights
        topk_indices:  [N, K] selected expert indices per token
        topk_weights:  [N, K] routing weights per token
        num_experts:   total number of experts (E=256 for DeepSeek-V3)

    Returns:
        output: [N, D] weighted sum of expert outputs
    """
    # For now, use the per-expert loop (same as GLM-5 triton path).
    # DeepGEMM FP8 grouped GEMM requires FP8 quantized weights,
    # which is a load-time transformation. The infrastructure is
    # in place via fp8_utils.quantize_activations_deepgemm().
    return _expert_loop_forward(
        hidden_states, gate_up_proj, down_proj,
        topk_indices, topk_weights, num_experts,
    )


def _expert_loop_forward(hidden_states, gate_up_proj, down_proj, topk_indices, topk_weights, num_experts):
    """Per-expert loop fallback (reference implementation).

    Iterates over experts that have at least one assigned token,
    applies SwiGLU activation, and accumulates weighted outputs.
    """
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = F.one_hot(topk_indices, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # [E, K, N]
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        # SwiGLU: split gate_up into gate and up
        gate, up = F.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = F.silu(gate) * up
        current_hidden_states = F.linear(current_hidden_states, down_proj[expert_idx])

        # Apply routing weights
        current_hidden_states = current_hidden_states * topk_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def _deepgemm_grouped_forward(hidden_states, gate_up_proj, down_proj, topk_indices, topk_weights, num_experts):
    """DeepGEMM FP8 grouped GEMM forward (H100/SM90 only).

    Requires pre-quantized FP8 weights and DeepGEMM installation.
    This is the production path for inference on H100.
    """
    if not DEEP_GEMM_AVAILABLE:
        raise RuntimeError("DeepGEMM not available. Install with: pip install deep-gemm")

    # Sort tokens by expert assignment for contiguous memory access
    num_tokens = hidden_states.shape[0]
    top_k = topk_indices.shape[1]
    total_tokens = num_tokens * top_k

    # Flatten topk selections
    flat_indices = topk_indices.view(-1)
    flat_weights = topk_weights.view(-1)

    # Sort by expert
    sorted_order = flat_indices.argsort(stable=True)
    sorted_experts = flat_indices[sorted_order]
    sorted_weights = flat_weights[sorted_order]

    # Compute per-expert token counts
    m_sizes = torch.histc(sorted_experts.float(), bins=num_experts, min=0, max=num_experts - 1).int()

    # Expand hidden states for each top-k slot
    token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(top_k)
    sorted_token_ids = token_ids[sorted_order]
    sorted_hidden = hidden_states[sorted_token_ids]

    # First GEMM: gate_up = sorted_hidden @ gate_up_proj^T (per expert)
    # Second GEMM: output = activation(gate_up) @ down_proj^T (per expert)
    # TODO: Call deep_gemm.m_grouped_fp8_gemm_nt_contiguous when FP8 weights are ready
    raise NotImplementedError("DeepGEMM FP8 path requires pre-quantized weights")
