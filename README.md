# DeepSeek-V3 Decoupled Implementation

Standalone PyTorch implementation of **DeepSeek-V3** (671B parameters, ~37B active) decoupled from HuggingFace transformers dependencies, with multiple kernel backends and comprehensive benchmarking.

**Paper:** [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

## Architecture

DeepSeek-V3 is a Mixture-of-Experts (MoE) language model with:

- **Multi-head Latent Attention (MLA):** Compresses KV into 512-dim latent space, applies RoPE to decoupled 64-dim stream
- **Mixture of Experts:** 256 routed experts (top-8 per token), 1 shared expert, group-based routing (n_group=8, topk_group=4)
- **Auxiliary-loss-free load balancing:** Sigmoid routing with correction bias (topk_method="noaux_tc")
- **YaRN RoPE:** Extended context to 163,840 tokens (factor=40)
- **Multi-Token Prediction (MTP):** 1 prediction layer for speculative decoding
- **61 layers:** 3 dense + 58 MoE

### Key Dimensions (671B variant)

| Parameter | Value |
|-----------|-------|
| hidden_size | 7168 |
| num_attention_heads | 128 |
| num_hidden_layers | 61 |
| q_lora_rank | 1536 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128 |
| qk_rope_head_dim | 64 |
| v_head_dim | 128 |
| n_routed_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 2048 |
| vocab_size | 129,280 |

## Directory Structure

```
deepseekv3_2/
  deepseekv3_2-raw-decoupled-from-hf/   # Pure PyTorch, no dependencies
  deepseekv3_2-triton/                    # Triton kernels + unsloth
  deepseekv3_2-kernels-flashinfer/        # FlashInfer attention backend
  deepseekv3_2-kernels-flashmla-deepgemm/ # FlashMLA + DeepGEMM (highest perf)
  benchmark/                               # Comprehensive benchmarking suite
  data/                                    # Shared sample data
  research/                                # Research directions
  viz/                                     # Architecture visualizations
```

## Kernel Selection

| Kernel | Component | Status | Notes |
|--------|-----------|--------|-------|
| FlashMLA | MLA Attention | Recommended | Built by DeepSeek specifically for MLA |
| DeepGEMM | MoE Expert GEMM | Recommended | FP8 grouped GEMM on Hopper |
| FlashInfer | MLA Attention | Alternative | Good MLA support, wider hardware compat |
| Triton | All components | Fallback | Custom fused kernels |
| Unsloth | RMSNorm, SwiGLU, CE, LoRA | All variants | Fast fused operations |

## Architecture Differences from GLM5

| Feature | DeepSeek-V3 | GLM5 |
|---------|------------|------|
| DSA (Dynamic Sparse Attention) | No | Yes |
| MTP | 1 layer | 3 shared layers |
| RoPE | YaRN (factor=40) | Standard |
| MoE routing | Group-based (n_group=8) | Flat (n_group=1) |
| hidden_size | 7168 | 6144 |
| num_layers | 61 | 78 |
| num_heads | 128 | 64 |
| qk_nope_head_dim | 128 | 192 |
| v_head_dim | 128 | 256 |
| rms_norm_eps | 1e-6 | 1e-5 |

## Quick Start

```python
from deepseekv3_2_raw_decoupled_from_hf.config import DEEPSEEK_V3_CONFIG
from deepseekv3_2_raw_decoupled_from_hf.model import DeepSeekV3ForCausalLM

model = DeepSeekV3ForCausalLM(DEEPSEEK_V3_CONFIG)
```

## Tests

```bash
# Standalone tests (CPU, no GPU required)
python deepseekv3_2-raw-decoupled-from-hf/test_standalone.py

# Data flow integration tests
python test_data_flow_unsloth_pytorch.py

# Validation with shared data
python deepseekv3_2-raw-decoupled-from-hf/validate.py
```
# DeepSeekV3.2-Tiered-Benchmarks
