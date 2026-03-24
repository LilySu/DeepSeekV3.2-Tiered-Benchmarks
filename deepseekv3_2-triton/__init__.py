# deepseekv3_2-triton: Self-contained DeepSeek-V3 model with Triton-accelerated kernels.
#
# This package contains everything needed to instantiate and run the DeepSeek-V3
# model with zero external dependencies beyond torch and triton.
#
# Files prefixed with "unsloth_" contain Triton kernels extracted from
# unsloth (https://github.com/unslothai/unsloth) -- optimized GPU ops.
#
# Files WITHOUT the prefix are pure-PyTorch implementations ported from
# the standalone DeepSeek-V3 reference (deepseekv3_2-raw-decoupled-from-hf/model.py).
#
# Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)
# 671B total parameters, ~37B active per token.
# 61 transformer layers (3 dense + 58 MoE), MLA attention, YaRN RoPE,
# 256 routed experts, auxiliary-loss-free load balancing,
# multi-token prediction (1 MTP layer).
#
# KEY DIFFERENCES FROM GLM5:
# - DeepSeek-V3 does NOT have DSA (Dynamic Sparse Attention) -- GLM5-specific
# - DeepSeek-V3 uses YaRN RoPE scaling (factor=40, 4096 -> 163840)
# - Different MoE routing: n_group=8, topk_group=4, topk_method="noaux_tc"
# - DeepSeek-V3 has MTP with 1 layer (GLM5 has 3 shared MTP layers)
# - Different dimensions: hidden=7168, heads=128, layers=61, qk_nope=128, v_head=128
#
# +----------------------------------+------------------------+-------------------+
# | Component                        | File                   | Status            |
# +----------------------------------+------------------------+-------------------+
# | RMSNorm fwd+bwd                  | unsloth_rms_layernorm  | Triton kernel     |
# | SwiGLU fwd+bwd                   | unsloth_swiglu         | Triton kernel     |
# | Cross-Entropy Loss (chunked)     | unsloth_cross_entropy  | Triton kernel     |
# | LoRA MLP/QKV/W                   | unsloth_fast_lora      | Triton kernel     |
# | MoE Grouped GEMM                 | unsloth_moe/           | Triton kernel     |
# | Utilities                        | unsloth_utils          | Triton support    |
# +----------------------------------+------------------------+-------------------+
# | Config                           | config                 | PyTorch (dict)    |
# | KV Cache                         | cache                  | PyTorch           |
# | Decoupled partial-dim RoPE+YaRN  | rope_partial           | PyTorch           |
# | DSA Sparse Attention             | dsa_sparse_attention   | Stub (N/A)        |
# | MLA (Multi-Latent Attention)     | mla_attention          | PyTorch           |
# | FeedForward / MoE / Router       | model                  | PyTorch           |
# | DecoderLayer / Base / CausalLM   | model                  | PyTorch           |
# | MTP (Multi-Token Prediction)     | mtp                    | PyTorch           |
# +----------------------------------+------------------------+-------------------+

# --- Triton kernels (from unsloth) ---
from .unsloth_rms_layernorm import (
    fast_rms_layernorm,
    Fast_RMS_Layernorm,
)
from .unsloth_swiglu import (
    swiglu_fg_kernel,
    swiglu_DWf_DW_dfg_kernel,
)
from .unsloth_cross_entropy_loss import (
    fast_cross_entropy_loss,
    Fast_CrossEntropyLoss,
)
from .unsloth_fast_lora import (
    LoRA_MLP,
    LoRA_QKV,
    LoRA_W,
    apply_lora_mlp_swiglu,
    apply_lora_qkv,
    apply_lora_o,
    get_lora_parameters,
    get_lora_parameters_bias,
)
from .unsloth_utils import (
    calculate_settings,
    torch_gpu_device,
    DEVICE_TYPE,
    MAX_FUSED_SIZE,
    is_cdna,
    is_rdna,
    matmul_lora,
    fast_linear_forward,
)

# --- Config ---
from .config import DEEPSEEK_V3_CONFIG, load_config_from_hf

# --- KV Cache ---
from .cache import KVCache

# --- PyTorch reference implementations ---
from .rope_partial import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rope_to_query,
    apply_rope_to_compressed_kv_key,
    rotate_half,
)

# --- DSA stub (NOT applicable to DeepSeek-V3) ---
from .dsa_sparse_attention import eager_attention_forward, build_dsa_mask

# --- MLA Attention ---
from .mla_attention import MLAttention

# --- Full model scaffolding ---
from .model import (
    make_causal_mask,
    FeedForward,
    TopkRouter,
    MoeExperts,
    MoE,
    DecoderLayer,
    DeepSeekV3Model,
    DeepSeekV3ForCausalLM,
    MTPLayer,
)
