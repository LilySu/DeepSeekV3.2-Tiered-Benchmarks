# Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the GNU Affero General Public License v3.0.
#
# Autotuning configuration generation and pruning for grouped GEMM kernels.
# Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)
# MoE dimensions: 256 experts, hidden=7168, intermediate=2048, top_k=8

import logging
from itertools import product
from typing import List

import torch
import triton

logger = logging.getLogger(__name__)

DEFAULT_M_BLOCK_SIZES = [64, 128]
DEFAULT_N_BLOCK_SIZES = [64, 128, 256]
DEFAULT_K_BLOCK_SIZES = [64, 128, 256]
DEFAULT_NUM_CTAS = 1
DEFAULT_NUM_WARPS = [4, 8]
DEFAULT_NUM_STAGES = [3, 4, 5]
BOOLS = [True, False]


def val_to_list(val):
    if val is None:
        return None
    elif isinstance(val, list):
        return val
    else:
        return [val]


def convert_args_to_list(args):
    return [val_to_list(arg) for arg in args]


def _triton_supports_tma():
    import triton.language as tl
    return hasattr(tl, "make_tensor_descriptor") or hasattr(tl, "_experimental_make_tensor_descriptor")


_TRITON_HAS_TMA = _triton_supports_tma()  # Auto-detect Hopper TMA support


def get_forward_configs(
    BLOCK_M=DEFAULT_M_BLOCK_SIZES, BLOCK_N=DEFAULT_N_BLOCK_SIZES, BLOCK_K=DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_X=None, TMA_LOAD_W=None, TMA_STORE=False,
    num_warps=DEFAULT_NUM_WARPS, num_stages=DEFAULT_NUM_STAGES, num_ctas=DEFAULT_NUM_CTAS,
):
    if TMA_LOAD_X is None:
        TMA_LOAD_X = _TRITON_HAS_TMA
    if TMA_LOAD_W is None:
        TMA_LOAD_W = _TRITON_HAS_TMA

    (BLOCK_M, BLOCK_N, BLOCK_K, TMA_LOAD_X, TMA_LOAD_W, TMA_STORE, num_warps, num_stages, num_ctas) = \
        convert_args_to_list([BLOCK_M, BLOCK_N, BLOCK_K, TMA_LOAD_X, TMA_LOAD_W, TMA_STORE, num_warps, num_stages, num_ctas])

    kernel_configs = []
    for block_m, block_n, block_k, w, s, tma_load_x, tma_load_w, tma_store, nc in product(
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, TMA_LOAD_X, TMA_LOAD_W, TMA_STORE, num_ctas,
    ):
        kernel_configs.append(triton.Config(
            dict(BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
                 USE_TMA_LOAD_X=tma_load_x, USE_TMA_LOAD_W=tma_load_w, USE_TMA_STORE=tma_store),
            num_warps=w, num_stages=s, num_ctas=nc,
        ))
    return kernel_configs


def get_dX_kernel_configs(
    BLOCK_M=DEFAULT_M_BLOCK_SIZES, BLOCK_N=DEFAULT_N_BLOCK_SIZES, BLOCK_K=DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_dY=None, TMA_LOAD_W=None, TMA_STORE=False,
    num_warps=DEFAULT_NUM_WARPS, num_stages=DEFAULT_NUM_STAGES, num_ctas=DEFAULT_NUM_CTAS,
):
    if TMA_LOAD_dY is None:
        TMA_LOAD_dY = _TRITON_HAS_TMA
    if TMA_LOAD_W is None:
        TMA_LOAD_W = _TRITON_HAS_TMA

    (BLOCK_M, BLOCK_N, BLOCK_K, TMA_LOAD_dY, TMA_LOAD_W, TMA_STORE, num_warps, num_stages, num_ctas) = \
        convert_args_to_list([BLOCK_M, BLOCK_N, BLOCK_K, TMA_LOAD_dY, TMA_LOAD_W, TMA_STORE, num_warps, num_stages, num_ctas])

    kernel_configs = []
    for block_m, block_n, block_k, w, s, tma_load_dy, tma_load_w, tma_store, nc in product(
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, TMA_LOAD_dY, TMA_LOAD_W, TMA_STORE, num_ctas,
    ):
        kernel_configs.append(triton.Config(
            dict(BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
                 USE_TMA_LOAD_dY=tma_load_dy, USE_TMA_LOAD_W=tma_load_w, USE_TMA_STORE=tma_store),
            num_warps=w, num_stages=s, num_ctas=nc,
        ))
    return kernel_configs


def get_dW_kernel_configs(
    BLOCK_M=DEFAULT_M_BLOCK_SIZES, BLOCK_N=DEFAULT_N_BLOCK_SIZES, BLOCK_K=DEFAULT_K_BLOCK_SIZES,
    num_warps=DEFAULT_NUM_WARPS, num_stages=DEFAULT_NUM_STAGES, num_ctas=DEFAULT_NUM_CTAS,
    TMA_LOAD_dY=None, TMA_LOAD_X=None, TMA_STORE=False,
):
    if TMA_LOAD_dY is None:
        TMA_LOAD_dY = _TRITON_HAS_TMA
    if TMA_LOAD_X is None:
        TMA_LOAD_X = _TRITON_HAS_TMA

    (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, num_ctas, TMA_LOAD_dY, TMA_LOAD_X, TMA_STORE) = \
        convert_args_to_list([BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, num_ctas, TMA_LOAD_dY, TMA_LOAD_X, TMA_STORE])

    kernel_configs = []
    for block_m, block_n, block_k, w, s, tma_load_dy, tma_load_x, tma_store, nc in product(
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, TMA_LOAD_dY, TMA_LOAD_X, TMA_STORE, num_ctas,
    ):
        kernel_configs.append(triton.Config(
            dict(BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
                 USE_TMA_LOAD_dY=tma_load_dy, USE_TMA_LOAD_X=tma_load_x, USE_TMA_STORE=tma_store),
            num_warps=w, num_stages=s, num_ctas=nc,
        ))
    return kernel_configs


def estimate_smem_reqs(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype):
    num_bytes = dtype.itemsize
    return (num_stages * BLOCK_SIZE_K * (BLOCK_SIZE_M + BLOCK_SIZE_N) + BLOCK_SIZE_M * BLOCK_SIZE_N) * num_bytes


def exceeds_smem_capacity(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, smem_size, slack=50000):
    smem_reqs = estimate_smem_reqs(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)
    return smem_reqs > smem_size + slack


def common_prune_criteria(config, kwargs, dtype):
    from ..interface import supports_tma
    from .tuning import get_device_properties
    smem_size = get_device_properties().SIZE_SMEM
    num_stages = config.num_stages
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]
    num_tokens = kwargs["NUM_TOKENS"]
    num_experts = kwargs["NUM_EXPERTS"]
    permute_x = kwargs["PERMUTE_X"]
    permute_y = kwargs["PERMUTE_Y"]
    tokens_per_expert = num_tokens // num_experts
    MIN_BLOCK_SIZE_M = DEFAULT_M_BLOCK_SIZES[0]
    if exceeds_smem_capacity(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, smem_size):
        return True
    if BLOCK_SIZE_M > tokens_per_expert * 2 and tokens_per_expert > MIN_BLOCK_SIZE_M:
        return True
    if permute_x and permute_y:
        return True
    return False


def maybe_disable_tma(config):
    from ..interface import supports_tma
    tma_keys = [k for k in config.kwargs.keys() if k.startswith("USE_TMA_")]
    if not supports_tma():
        for k in tma_keys:
            config.kwargs[k] = False


def prune_kernel_configs_fwd(configs, args, **kwargs):
    x = kwargs["x_ptr"]
    dtype = x.dtype
    pruned_configs = []
    for config in configs:
        maybe_disable_tma(config)
        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_X"] and kwargs["PERMUTE_X"]:
            config.kwargs["USE_TMA_LOAD_X"] = False
        if config.kwargs["USE_TMA_STORE"] and kwargs["PERMUTE_Y"]:
            continue
        pruned_configs.append(config)
    return pruned_configs


def prune_dX_configs(configs, args, **kwargs):
    dtype = kwargs["w_ptr"].dtype
    pruned_configs = []
    for config in configs:
        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_dY"] and kwargs["PERMUTE_Y"]:
            config.kwargs["USE_TMA_LOAD_dY"] = False
        if config.kwargs["USE_TMA_STORE"] and kwargs["PERMUTE_X"]:
            continue
        pruned_configs.append(config)
    return pruned_configs


def prune_kernel_configs_backward_dW(configs, args, **kwargs):
    dtype = kwargs["x_ptr"].dtype
    pruned_configs = []
    for config in configs:
        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_dY"] and kwargs["PERMUTE_Y"]:
            config.kwargs["USE_TMA_LOAD_dY"] = False
        if config.kwargs["USE_TMA_LOAD_X"] and kwargs["PERMUTE_X"]:
            config.kwargs["USE_TMA_LOAD_X"] = False
        pruned_configs.append(config)
    return pruned_configs
