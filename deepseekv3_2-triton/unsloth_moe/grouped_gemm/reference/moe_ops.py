# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.
#
# Reference MoE operations for DeepSeek-V3 grouped GEMM testing.
#
# Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)
# MoE routing: sigmoid scoring (not softmax), n_group=8, topk_group=4
# KEY DIFFERENCE from Qwen3/Llama4: DeepSeek-V3 uses auxiliary-loss-free
# load balancing with e_score_correction_bias.

import torch
import torch.nn.functional as F


def permute(X: torch.Tensor, gather_indices: torch.Tensor, topk: int):
    """Scatter X to expert-grouped order using gather_indices.

    For DeepSeek-V3: X is [num_tokens, 7168], gather_indices is [num_tokens * 8]
    """
    assert gather_indices.ndim == 1
    X = X.view(-1, X.shape[-1])
    if topk == 1:
        return X[gather_indices]
    return X[gather_indices // topk]


def unpermute(X: torch.Tensor, gather_indices: torch.Tensor):
    """Restore X from expert-grouped order to token order."""
    X = X.view(-1, X.shape[-1]) if X.ndim > 2 else X
    unpermuted = torch.empty_like(X)
    unpermuted.index_copy_(0, gather_indices, X)
    return unpermuted.view_as(X)


def calculate_topk(
    gating_output: torch.Tensor, top_k: int, use_sigmoid: bool,
    renormalize: bool, pre_act: bool = True, post_act: bool = False,
):
    """Calculate top-k routing weights.

    DeepSeek-V3 uses pre_act=True with sigmoid (not softmax).
    """
    assert pre_act ^ post_act

    def _activation(gating_output):
        if use_sigmoid:
            scores = torch.sigmoid(gating_output.to(torch.float32)).to(gating_output.dtype)
        else:
            scores = F.softmax(gating_output.to(torch.float32), dim=1).to(gating_output.dtype)
        return scores

    if pre_act:
        scores = _activation(gating_output)
    else:
        scores = gating_output

    topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=1)

    if post_act:
        topk_weights = _activation(topk_weights)

    if renormalize:
        topk_weights /= torch.sum(topk_weights, dim=-1, keepdim=True).to(gating_output.dtype)

    return topk_weights, topk_ids


@torch.no_grad()
def get_routing_indices(selected_experts, num_experts, return_scatter_indices=False):
    """Compute token counts per expert and gather indices for expert-grouped ordering.

    For DeepSeek-V3: num_experts=256, selected_experts shape [num_tokens, 8]
    """
    token_counts_by_expert = torch.histc(
        selected_experts.view(-1), bins=num_experts, min=0, max=num_experts,
    )
    gather_indices = torch.argsort(selected_experts.view(-1), stable=True)
    if return_scatter_indices:
        scatter_indices = gather_indices.argsort()
        return token_counts_by_expert, gather_indices, scatter_indices
    else:
        return token_counts_by_expert, gather_indices


def torch_grouped_gemm(X, W, m_sizes, transpose=True):
    """Reference torch-native grouped GEMM.

    For DeepSeek-V3 forward:
      X: [total_tokens, 7168], W: [256, 4096, 7168] -> Y: [total_tokens, 4096]
    """
    X = X.view(-1, X.shape[-1])
    M, K = X.shape
    assert m_sizes.ndim == 1
    E = m_sizes.shape[0]
    assert W.ndim == 3 and W.shape[0] == E
    N = W.shape[1]

    result = torch.zeros((M, N), dtype=X.dtype, device=X.device)
    m_start = 0
    for g in range(E):
        m_size = m_sizes[g]
        if m_size > 0:
            m_end = m_start + m_size
            X_g = X[m_start:m_end]
            W_g = W[g]
            W_g = W_g.T if transpose else W_g
            Y_g = X_g @ W_g
            result[m_start:m_end] = Y_g
            m_start = m_end
    return result
