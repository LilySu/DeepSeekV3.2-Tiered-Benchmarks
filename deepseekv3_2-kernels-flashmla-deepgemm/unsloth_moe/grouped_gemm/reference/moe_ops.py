"""
Reference MoE operations for testing and validation.

Low-level operations used in MoE dispatch/gather, implemented in pure
PyTorch for correctness checking against optimised kernels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def reference_expert_dispatch(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> Tuple[list, list]:
    """Dispatch tokens to experts based on routing decisions.

    Parameters
    ----------
    hidden_states : [T, D]
        Token representations.
    topk_indices : [T, top_k]
        Expert assignments.
    num_experts : int

    Returns
    -------
    expert_inputs : list of [n_e, D] tensors (one per expert)
    token_maps : list of [n_e] int tensors (original token indices)
    """
    T, K = topk_indices.shape

    expert_inputs = []
    token_maps = []

    for e in range(num_experts):
        # Find all (token, slot) pairs assigned to expert e
        mask = (topk_indices == e)  # [T, K]
        token_ids = mask.any(dim=-1).nonzero(as_tuple=True)[0]

        if len(token_ids) == 0:
            expert_inputs.append(
                torch.zeros(0, hidden_states.shape[-1],
                            dtype=hidden_states.dtype,
                            device=hidden_states.device)
            )
            token_maps.append(torch.zeros(0, dtype=torch.long, device=hidden_states.device))
        else:
            expert_inputs.append(hidden_states[token_ids])
            token_maps.append(token_ids)

    return expert_inputs, token_maps


def reference_expert_gather(
    expert_outputs: list,
    token_maps: list,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    total_tokens: int,
    hidden_size: int,
    num_experts: int,
) -> torch.Tensor:
    """Gather and weight expert outputs back to token positions.

    Parameters
    ----------
    expert_outputs : list of [n_e, D] tensors
    token_maps : list of [n_e] int tensors
    topk_weights : [T, top_k]
    topk_indices : [T, top_k]
    total_tokens : int
    hidden_size : int
    num_experts : int

    Returns
    -------
    output : [T, D]
    """
    device = topk_weights.device
    dtype = expert_outputs[0].dtype if len(expert_outputs) > 0 and expert_outputs[0].numel() > 0 else topk_weights.dtype
    output = torch.zeros(total_tokens, hidden_size, dtype=dtype, device=device)

    for e in range(num_experts):
        if expert_outputs[e].numel() == 0:
            continue

        token_ids = token_maps[e]
        e_out = expert_outputs[e]

        # For each token assigned to this expert, find the routing weight
        for i, tid in enumerate(token_ids):
            tid = tid.item()
            # Find which top-k slot(s) point to expert e
            for k in range(topk_indices.shape[1]):
                if topk_indices[tid, k] == e:
                    w = topk_weights[tid, k]
                    output[tid] += w * e_out[i]

    return output


def reference_swiglu_expert(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply a single SwiGLU expert.

    Parameters
    ----------
    x : [N, D]
    gate_weight : [I, D]
    up_weight : [I, D]
    down_weight : [D, I]

    Returns
    -------
    output : [N, D]
    """
    gate = F.silu(x @ gate_weight.T)  # [N, I]
    up = x @ up_weight.T  # [N, I]
    hidden = gate * up  # [N, I]
    return hidden @ down_weight.T  # [N, D]


def reference_grouped_gemm(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    """Reference grouped GEMM: C[g] = A[g] @ B[g]^T.

    Parameters
    ----------
    inputs : [sum(group_sizes), K]
    weights : [G, N, K]
    group_sizes : [G]

    Returns
    -------
    output : [sum(group_sizes), N]
    """
    outputs = []
    offset = 0
    for g, size in enumerate(group_sizes.tolist()):
        size = int(size)
        if size == 0:
            continue
        inp = inputs[offset : offset + size]
        out = inp @ weights[g].T
        outputs.append(out)
        offset += size

    if outputs:
        return torch.cat(outputs, dim=0)

    N = weights.shape[1]
    return torch.zeros(0, N, dtype=inputs.dtype, device=inputs.device)


def compute_expert_load(
    topk_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute the number of tokens assigned to each expert.

    Parameters
    ----------
    topk_indices : [T, top_k]
    num_experts : int

    Returns
    -------
    counts : [num_experts]
    """
    return torch.bincount(
        topk_indices.reshape(-1),
        minlength=num_experts,
    )


def expert_load_balance_loss(
    router_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute load-balance auxiliary loss.

    Note: DeepSeek-V3 uses ``noaux_tc`` (no auxiliary loss), so this is
    provided for reference only and is NOT used in the default config.

    Parameters
    ----------
    router_logits : [T, num_experts]
    topk_indices : [T, top_k]
    num_experts : int

    Returns
    -------
    loss : scalar
    """
    T = router_logits.shape[0]
    top_k = topk_indices.shape[1]

    # Fraction of tokens routed to each expert
    counts = compute_expert_load(topk_indices, num_experts).float()
    f = counts / (T * top_k)

    # Average routing probability per expert
    probs = torch.sigmoid(router_logits).mean(dim=0)

    # Load balance loss: num_experts * sum(f * p)
    return num_experts * (f * probs).sum()
