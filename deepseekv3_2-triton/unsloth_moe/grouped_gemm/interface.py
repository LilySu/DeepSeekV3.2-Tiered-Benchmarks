# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.
#
# High-level grouped GEMM interface for DeepSeek-V3 MoE dispatch.
#
# Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)
# MoE config: 256 routed experts, hidden=7168, intermediate=2048, top_k=8
# Expert weight shapes:
#   gate_up_proj: [256, 4096, 7168]  (2 * 2048 fused gate+up)
#   down_proj:    [256, 7168, 2048]
#
# STATUS: Triton kernel -- mirrors GLM5 interface adapted for DeepSeek-V3 dimensions.

import logging
import warnings
from dataclasses import asdict

import torch
import triton

from .kernels.backward import (
    _autotuned_grouped_gemm_dW_kernel,
    _autotuned_grouped_gemm_dX_kernel,
    _grouped_gemm_dW_kernel,
    _grouped_gemm_dX_kernel,
)
from .kernels.forward import (
    _autotuned_grouped_gemm_forward_kernel,
    _grouped_gemm_forward_kernel,
)
from .kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "%(asctime)s::%(levelname)s,%(pathname)s:%(lineno)d:: %(message)s"
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def _check_tma_support():
    """Check if current hardware+software supports TMA (Hopper+ GPUs with compatible Triton)."""
    try:
        import triton.language as tl
        gpu_supports_tma = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
        triton_has_tma_api = hasattr(tl, "make_tensor_descriptor") or hasattr(
            tl, "_experimental_make_tensor_descriptor"
        )
        return gpu_supports_tma and triton_has_tma_api
    except Exception:
        return False


_SUPPORTS_TMA = _check_tma_support()
_HAS_SET_ALLOCATOR = hasattr(triton, "set_allocator")


def supports_tma():
    return _SUPPORTS_TMA


try:
    from torch.compiler import allow_in_graph
except ImportError:
    from torch._dynamo import allow_in_graph


def _is_tracing(*tensors):
    """Check if tensors are fake tensors used during torch.compile tracing."""
    for t in tensors:
        name = type(t).__name__
        if name in ("FakeTensor", "FunctionalTensor", "FunctionalTensorWrapper"):
            return True
    return False


_per_device_alloc_fns = {}


def get_per_device_per_stream_alloc_fn(device):
    if device not in _per_device_alloc_fns:
        _per_stream_tensors = {}
        def alloc_fn(size: int, alignment: int, stream):
            assert alignment == 128
            if stream not in _per_stream_tensors or _per_stream_tensors[stream].numel() < size:
                _per_stream_tensors[stream] = torch.empty(size, device=device, dtype=torch.int8)
                _per_stream_tensors[stream].__hibernate__ = {"type": "ignore"}
            return _per_stream_tensors[stream]
        _per_device_alloc_fns[device] = alloc_fn
    return _per_device_alloc_fns[device]


@allow_in_graph
def grouped_gemm_forward(
    X: torch.Tensor, W: torch.Tensor, topk: int,
    m_sizes: torch.Tensor, gather_indices: torch.Tensor = None,
    topk_weights: torch.Tensor = None,
    permute_x: bool = False, permute_y: bool = False,
    fuse_mul_post: bool = False,
    autotune: bool = False,
    BLOCK_SIZE_M: int = 32, BLOCK_SIZE_N: int = 32, BLOCK_SIZE_K: int = 32,
    num_warps: int = 4, num_stages: int = 2,
    use_tma_load_w: bool = False, use_tma_load_x: bool = False, use_tma_store: bool = False,
    flatten: bool = True, debug: bool = False,
) -> torch.Tensor:
    """Grouped GEMM forward pass for DeepSeek-V3 MoE expert dispatch.

    For DeepSeek-V3:
    - First GEMM: X[num_tokens, 7168] @ gate_up_proj[256, 4096, 7168].T -> Y[total_tokens, 4096]
    - Second GEMM: X[total_tokens, 2048] @ down_proj[256, 7168, 2048].T -> Y[total_tokens, 7168]
    """
    assert X.device.type == "cuda", "X and W must be on CUDA"
    assert m_sizes.device.type == "cuda", "m_sizes must be on CUDA"

    X = X.contiguous()
    W = W.contiguous()
    m_sizes = m_sizes.contiguous()

    assert not (permute_x and permute_y), "Cannot permute both X and Y"

    num_experts, N, K = W.shape
    num_tokens = X.shape[0]
    total_tokens = num_tokens * topk if permute_x else num_tokens

    Y = torch.empty((total_tokens, N), dtype=X.dtype, device=X.device)

    from .kernels.tuning import get_device_properties
    NUM_SMS = get_device_properties().NUM_SM

    kernel = _autotuned_grouped_gemm_forward_kernel if autotune else _grouped_gemm_forward_kernel

    kernel[(NUM_SMS,)](
        X, W, Y, m_sizes, gather_indices,
        topk_weights if topk_weights is not None else X,  # placeholder
        NUM_EXPERTS=num_experts, NUM_TOKENS=num_tokens, TOPK=topk,
        N=N, K=K, NUM_SMS=NUM_SMS,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        PERMUTE_X=permute_x, PERMUTE_Y=permute_y,
        FUSE_MUL_POST=fuse_mul_post,
        num_warps=num_warps, num_stages=num_stages,
    )
    return Y


@allow_in_graph
def grouped_gemm_dX(
    dY: torch.Tensor, W: torch.Tensor,
    gather_indices: torch.Tensor, m_sizes: torch.Tensor, topk: int,
    permute_x: bool = False, permute_y: bool = False,
    autotune: bool = False,
    BLOCK_SIZE_M: int = 32, BLOCK_SIZE_N: int = 32, BLOCK_SIZE_K: int = 32,
    num_warps: int = 4, num_stages: int = 2,
    use_tma_load_dy: bool = False, use_tma_load_w: bool = False, use_tma_store: bool = False,
) -> torch.Tensor:
    """Grouped GEMM backward dX for DeepSeek-V3 MoE."""
    dY = dY.contiguous()
    W = W.contiguous()

    num_experts, N, K = W.shape
    total_tokens = dY.shape[0]
    num_tokens = total_tokens // topk

    dX = torch.empty((total_tokens, K), dtype=dY.dtype, device=dY.device)

    from .kernels.tuning import get_device_properties
    NUM_SMS = get_device_properties().NUM_SM

    kernel = _autotuned_grouped_gemm_dX_kernel if autotune else _grouped_gemm_dX_kernel
    kernel[(NUM_SMS,)](
        dY, W, dX, gather_indices, m_sizes,
        NUM_EXPERTS=num_experts, NUM_TOKENS=num_tokens, TOPK=topk,
        N=N, K=K, NUM_SMS=NUM_SMS,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        PERMUTE_X=permute_x, PERMUTE_Y=permute_y,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dX


@allow_in_graph
def grouped_gemm_dW(
    X: torch.Tensor, dY: torch.Tensor,
    m_sizes: torch.Tensor, gather_indices: torch.Tensor, topk: int,
    permute_x: bool = False, permute_y: bool = False,
    autotune: bool = False,
    BLOCK_SIZE_M: int = 32, BLOCK_SIZE_N: int = 32, BLOCK_SIZE_K: int = 32,
    num_warps: int = 4, num_stages: int = 2,
    use_tma_load_dy: bool = False, use_tma_load_x: bool = False, use_tma_store: bool = False,
) -> torch.Tensor:
    """Grouped GEMM backward dW for DeepSeek-V3 MoE."""
    X = X.contiguous()
    dY = dY.contiguous()

    num_tokens = X.shape[0]
    num_experts = m_sizes.shape[0]
    N = dY.shape[1]
    K = X.shape[1]

    dW = torch.zeros((num_experts, N, K), dtype=X.dtype, device=X.device)

    from .kernels.tuning import get_device_properties
    NUM_SMS = get_device_properties().NUM_SM

    kernel = _autotuned_grouped_gemm_dW_kernel if autotune else _grouped_gemm_dW_kernel
    kernel[(NUM_SMS,)](
        X, dY, dW, m_sizes, gather_indices,
        NUM_TOKENS=num_tokens, TOPK=topk, NUM_EXPERTS=num_experts,
        N=N, K=K, NUM_SMS=NUM_SMS,
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_M=BLOCK_SIZE_M,
        PERMUTE_X=permute_x, PERMUTE_Y=permute_y,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dW
