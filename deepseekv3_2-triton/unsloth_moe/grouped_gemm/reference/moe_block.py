# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.
#
# Reference implementation of MoE block using grouped GEMM for DeepSeek-V3.
#
# Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)
# MoE config: 256 experts, hidden=7168, intermediate=2048, top_k=8
# Routing: sigmoid scoring, n_group=8, topk_group=4, noaux_tc method
#
# NOTE: This is a reference implementation for testing, NOT for production.

import torch
from ..interface import grouped_gemm_forward
from ..kernels.tuning import KernelConfigForward, KernelConfigBackward_dW, KernelConfigBackward_dX
from .moe_ops import permute, unpermute, get_routing_indices
