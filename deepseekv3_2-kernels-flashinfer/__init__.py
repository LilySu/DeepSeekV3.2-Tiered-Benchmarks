# deepseekv3_2-kernels-flashinfer: DeepSeek-V3 model with FlashInfer kernel acceleration.
#
# Key architecture (arXiv 2412.19437):
#   MLA Attention:  FlashInfer BatchMLAPagedAttentionWrapper (FA3) — native MLA support
#   FP8 KV Cache:   FlashInfer native [num_pages, page_size, 576] contiguous FP8
#   No DSA:         DeepSeek-V3 does NOT use Dynamic Sparse Attention (standard causal)
#   MoE Router:     Sigmoid with group-based selection (n_group=8, topk_group=4, noaux_tc)
#   MoE GEMM:      DeepGEMM FP8 grouped GEMM / Triton fallback
#   RMSNorm/SwiGLU: Unsloth Triton kernels
#   RoPE:          YaRN-extended (factor=40) for 128K context
#   MTP:           Multi-Token Prediction with 1 additional layer
#
# Why FlashInfer for DeepSeek-V3:
#   1. FlashInfer has native MLA support (page attention with latent heads)
#   2. FlashInfer supports variable-length sequences with paged KV cache
#   3. DeepSeek-V3 does NOT need DSA sparse masks — FlashInfer standard causal works
#   4. FlashInfer handles the compressed KV cache format efficiently
#   5. CUDA graph native support via use_cuda_graph=True
#
# Key dimensions (DeepSeek-V3 671B):
#   hidden_size=7168, num_attention_heads=128, num_hidden_layers=61
#   q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64
#   qk_nope_head_dim=128, v_head_dim=128
#   n_routed_experts=256, num_experts_per_tok=8, n_group=8, topk_group=4
#   moe_intermediate_size=2048, n_shared_experts=1
#   YaRN RoPE scaling factor=40, rope_theta=10000.0
#
# Dependencies:
#   pip install flashinfer  (CUDA 12.0+)
#   pip install deep-gemm   (build from source, CUDA 12.8+, SM90)

from .config import DEEPSEEK_V3_CONFIG, load_config_from_hf
from .mla_attention import MLAttention
from .dsa_indexer import DSAIndexerStub
from .dsa_sparse_attention import build_dsa_mask, eager_attention_forward
from .moe_router import sigmoid_topk_route
from .moe_grouped_gemm import moe_grouped_gemm_forward
from .fp8_utils import quantize_kv_flashinfer, quantize_activations_deepgemm
from .cache import KVCache
from .rope_partial import RotaryEmbedding, apply_rotary_pos_emb
from .model import (
    make_causal_mask, FeedForward, TopkRouter, MoeExperts, MoE,
    DecoderLayer, DeepSeekV3Model, DeepSeekV3ForCausalLM,
)
